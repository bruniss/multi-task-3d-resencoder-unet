# lots borrowed from discord user @Mojonero, who kindly shared his s2 starter here: https://discord.com/channels/1079907749569237093/1204133327083147264/1204133327083147264
from typing import Tuple, Union, List

import volumentations
import zarr
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from skimage.morphology import dilation, ball
from pytorch3dunet.augment.transforms import (Compose, LabelToAffinities, Standardize,
                                              RandomFlip, RandomRotate90)

import albumentations as A
from pytorch3dunet.augment.transforms import Standardize
from helpers import _find_valid_patches
from transforms.geometric.geometry import RandomFlipWithNormals, RandomRotate90WithNormals

from volumentations import Compose as vCompose
from volumentations import ElasticTransform

class ZarrSegmentationDataset3D(Dataset):
    def __init__(self,
                 volume_paths: list,
                 tasks=dict,
                 patch_size=(128, 128, 128),
                 min_labeled_ratio=0.9,
                 min_bbox_percent=95,
                 normalization=Standardize(channelwise=False),
                 dilate_label=False,
                 transforms=None,
                 use_cache=True,
                 cache_file: Path = "./valid_cache.json",
                 ):

        self.volume_paths = volume_paths
        self.tasks = tasks
        self.patch_size = patch_size
        self.min_labeled_ratio = min_labeled_ratio
        self.min_bbox_percent = min_bbox_percent
        self.normalization = normalization
        self.dilate_label = dilate_label
        self.transforms_list = transforms
        self.use_cache = use_cache
        self.cache_file = cache_file

        self.volumes = []
        for vol_idx, vol_info in enumerate(volume_paths):
            # open input
            input_zarr = zarr.open(vol_info["input"], mode='r')

            # open each target for the tasks we care about
            target_arrays = {}
            for task_name in tasks.keys():
                if task_name in vol_info:
                    t_arr = zarr.open(vol_info[task_name], mode='r')
                    print(f"Opened zarr {(vol_info[task_name])}. Volume shape is: {t_arr.shape}")
                    target_arrays[task_name] = t_arr
                else:
                    raise ValueError(f"Volume {vol_idx} is missing the path for task '{task_name}'")

            # also store which key to use for valid patches
            # e.g. "sheet" or "normals" or something else
            ref_label_key = vol_info.get("ref_label", "sheet")

            self.volumes.append({
                "input": input_zarr,
                "targets": target_arrays,
                "ref_label_key": ref_label_key
            })

        # scan through each reference volume , and slide through it by the overlap * patch size,
        #   for each candidate patch, compute how large the bounding box containing _all the labels_
        #   would be in relation to the size of patch. then , compute how many pixels within that
        #   bounding box are labeled. then save the results to a json since this takes forever
        self.all_valid_patches = []
        if self.use_cache and self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self.all_valid_patches = json.load(f)
            print(f"Loaded {len(self.all_valid_patches)} patches from cache.")
        else:
            print("Computing valid patches from scratch...")
            for vol_idx, vol_dict in enumerate(self.volumes):
                # pick the reference label to find valid patches from
                ref_label_key = vol_dict["ref_label_key"]
                ref_label_zarr = vol_dict["targets"][ref_label_key]

                # and find the patches
                vol_patches = _find_valid_patches(
                    ref_label_zarr,
                    patch_size=self.patch_size,
                    bbox_threshold=self.min_bbox_percent,
                    label_threshold=self.min_labeled_ratio
                )

                # tag each patches volume so we get the right one
                for p in vol_patches:
                    p["volume_idx"] = vol_idx
                self.all_valid_patches.extend(vol_patches)

            # save to cache if desired
            if self.use_cache:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.all_valid_patches, f)
                print(f"Saved {len(self.all_valid_patches)} patches to cache.")


    def __len__(self):
            return len(self.all_valid_patches)

    def __getitem__(self, idx):
        patch_info = self.all_valid_patches[idx]
        vol_idx = patch_info["volume_idx"]

        z0, y0, x0 = patch_info["start_pos"]
        dz, dy, dx = self.patch_size
        patch_slice = np.s_[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]

        # get the correct volume
        vol_dict = self.volumes[vol_idx]
        # convert image data to float32, note that it still retains the original values
        input_data = vol_dict["input"][patch_slice].astype(np.float32)

        # input data here should be shape (z, y, x) or (c, z, y, x)
        # and dtype of float32, with original pixel values

        # scaling input values from 255 or 65000 to 0 to 1
        # scaling input data for uint8
        if input_data.dtype == np.uint8:
            input_data /= 255.0

        # scaling input data for uint16
        elif input_data.dtype == np.uint16:
            input_data /= 65535.0

        # input data is still in the same shape (z, y, x) or (c, z, y, x)
        # input data here is float32
        # all values are now properly scaled between 0 and 1

        # apply z-score normalization to the _input data only_
        input_data = self.normalization(input_data)

        # enumerate through our target dictionary and gather the patches
        data_dict = {"image": input_data}
        for task_name, task_cfg in self.tasks.items():
            t_arr = vol_dict["targets"][task_name]
            # convert to float32, still in original values
            t_patch = t_arr[patch_slice].astype(np.float32)

            if task_name.lower() == "normals":
                # Handle normals stored in uint16 format from your mesh processing
                if t_arr.dtype == np.uint16:
                    # Convert directly from uint16 to [-1, 1] range using your original scaling
                    t_patch = (t_patch.astype(np.float32) / 32767.5) - 1.0
                else:
                    # Keep your existing handling for normals stored in other formats
                    t_patch = (t_patch * 2.0) - 1.0
                # reorder if channels-last
                if t_patch.ndim == 4:  # e.g. (Z, Y, X, C)
                    t_patch = t_patch.transpose(3, 0, 1, 2).copy() # (z, y, x, c) => (c, z, y, x)

            else:
                # scale image values from 0 to 255/65k => 0 to 1
                if t_arr.dtype == np.uint8:
                    t_patch /= 255.0
                elif t_arr.dtype == np.uint16:
                    t_patch /= 65535.0

            # apply an optional label dilation you can set in the json
            if (task_name.lower() == "sheet") and self.dilate_label:
                t_patch = (t_patch > 0).astype(np.float32)
                t_patch = dilation(t_patch, ball(5))

            # all _target_ data is now shape (z, y, x) or (c, z, y, x) and type float32
            # with all values scaled between 0 and 1 (except for normals)
            # send it to the dict
            data_dict[task_name] = t_patch

        # i have to split up 2d and 3d augs because albumentations hates us.
        # i should find a more volumetric based aug library, but i have not done this yet
        # input AND target data now all scaled to 0 to 1 and in (z, y, x) or (c, z, y, x)
        # and of type float32

        # compose our augmentation list, note that these stack
        img_transform = A.Compose([

            # illumination
            A.OneOf([
                A.RandomBrightnessContrast(),
                #A.AutoContrast(),
                A.Illumination(),
            ], p=0.3),

            # noise
            A.OneOf([
                A.MultiplicativeNoise(),
                A.GaussNoise()
            ], p=0.3),

            # blur
            A.OneOf([
                A.Blur(),
                A.Downscale()
            ], p=0.3),
        ],
            p=1.0,
        )

        # dropout
        vol_transform = A.Compose([
                A.CoarseDropout3D(fill=0.5,
                                  num_holes_range=(1, 4),
                                  hole_depth_range=(0.1, 0.5),
                                  hole_height_range=(0.1, 0.5),
                                  hole_width_range=(0.1, 0.5))
        ],
            p=0.5
        )

        # apply the 2d augs to an "image" key only, these apply per slice
        img_augmented = img_transform(image=data_dict["image"])
        image_2d_aug = img_augmented["image"]

        # apply the volumetric ones to the 2d image augs (we have to target "volume")
        vol_augmented = vol_transform(volume=image_2d_aug)
        data_dict["image"] = vol_augmented["volume"]

        # i have to use different rotations/flips here because of normal vectors
        # this still applies if you do not have a key called normals, it just wont do the sign flipping/rotations
        rotate = RandomRotate90WithNormals(axes=('z',), p_transform=0.3)
        flip = RandomFlipWithNormals(p_transform=0.3)
        data_dict = rotate(data_dict)
        data_dict = flip(data_dict)

        #if "normals" not in data_dict and "normal" not in data_dict:
            #v_transform = vCompose([
              #ElasticTransform(p=1.0)
            #], p=0.15)
            #data_dict = v_transform(data_dict)

        # convert image to tensors, adding an additional channel for single channel input data
        # input -> [C, D, H, W]
        if data_dict["image"].ndim == 3:
            data_dict["image"] = data_dict["image"][None, ...]
        data_dict["image"] = torch.from_numpy(np.ascontiguousarray(data_dict["image"]))


        # convert each target label to tensors, adding an additional channel for single channel target data
        for task_name in self.tasks.keys():
            tgt = data_dict[task_name]
            if tgt.ndim == 3 and task_name.lower() != "normals":
                tgt = tgt[None, ...]
            data_dict[task_name] = torch.from_numpy(np.ascontiguousarray(tgt))

        # our data now looks like this:
        # input data is shape (c, z, y, x) and values from 0 to 1
        # target data is shape (c, z, y, x) and values are from 0 to 1
        # all data and targets are float32
        # all data and targets are torch tensors
        # send 'er to the trainer

        return data_dict

    def close(self):
        """Close all Zarr stores if needed."""
        for vol_dict in self.volumes:
            vol_dict["input"].store.close()
            for t_name, t_arr in vol_dict["targets"].items():
                t_arr.store.close()

