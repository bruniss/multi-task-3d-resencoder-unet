import os
import json
import argparse
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
import zarr
from numcodecs import Blosc
import cv2

import torch
from torch.utils.data import DataLoader

from pytorch3dunet.unet3d.model import MultiTaskResidualUNetSE3D

from dataloading.inference_dataset import InferenceDataset

class ZarrInferenceHandler:

    def __init__(self, config_file: str, write_layers:bool):
        with open(config_file, "r") as f:
            config = json.load(f)

        tr_params = SimpleNamespace(**config["tr_params"])
        model_config = SimpleNamespace(**config["model_config"])
        dataset_config = SimpleNamespace(**config["dataset_config"])
        inference_params = SimpleNamespace(**config["inference_params"])

        # --- configs --- #
        self.patch_size = tuple(getattr(inference_params, "patch_size", [192, 192, 192]))
        self.batch_size = int(getattr(inference_params, "batch_size", 2))
        self.f_maps = list(getattr(model_config, "f_maps", [32, 64, 128, 256]))
        self.num_levels = int(getattr(model_config, "n_levels", 6))

        self.checkpoint_path = getattr(inference_params, "checkpoint_path", None)
        self.num_dataloader_workers = int(getattr(inference_params, "num_dataloader_workers", 4))
        self.normalization = str(getattr(dataset_config, "normalization", "ct_normalize"))

        self.input_path = str(getattr(inference_params, "input_path", None))
        self.input_format = str(getattr(inference_params, "input_format", "zarr"))
        self.output_format = str(getattr(inference_params, "output_format", "zarr"))
        self.load_all = bool(getattr(inference_params, "load_all", False))
        self.output_dtype= str(getattr(inference_params, "output_type", "np.uint8"))
        self.output_targets = list(getattr(inference_params, "output_targets", "all"))
        self.overlap = float(getattr(inference_params, "overlap", 0.25))
        self.targets = getattr(inference_params, "targets", {})


        self.output_dir = str(getattr(inference_params, "output_dir", "./"))
        self.write_layers = write_layers
        self.volume_paths = dataset_config.volume_paths
        self.inference_targets = inference_params.targets


        os.makedirs(self.output_dir, exist_ok=True)

    def infer(self):
        torch.set_float32_matmul_precision('high')
        model = MultiTaskResidualUNetSE3D(
            in_channels=1,
            tasks=self.inference_targets,
            f_maps=self.f_maps,
            num_levels=self.num_levels
        )

        device = torch.device('cuda')
        model = torch.compile(model)
        model = model.to(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

        dataset = InferenceDataset(
            input_path=  self.input_path,
            targets= self.output_targets,
            patch_size= self.patch_size,
            normalization= self.normalization,
            input_format= self.input_format,
            overlap = self.overlap,
            load_all = self.load_all
            )

        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_dataloader_workers,
                            prefetch_factor=4,
                            pin_memory=True,
                            persistent_workers=True,)

        z_max, y_max, x_max = dataset.input_shape
        output_arrays = {}
        count_arrays = {}

        store_path = os.path.join(self.output_dir, "predictions.zarr")
        zarr_store = zarr.open(store_path, mode='w')

        chunk_z = 128
        chunk_y = 128
        chunk_x = 128

        for tgt_name in self.output_targets:
            c = self.targets[tgt_name]["channels"]

            # If channels=1, shape is (z,y,x); else (c,z,y,x)
            if c == 1:
                out_shape = (z_max, y_max, x_max)
                chunks = (chunk_z, chunk_y, chunk_x)  # example chunk
            else:
                out_shape = (c, z_max, y_max, x_max)
                chunks = (c, chunk_z, chunk_y, chunk_x)

            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

            # "sum" array
            sum_ds = zarr_store.create_dataset(
                name=f"{tgt_name}_sum",
                shape=out_shape,
                chunks=chunks,
                dtype='float32',
                compressor=compressor,
                fill_value=0,
                write_empty_chunks=False
            )

            # "count" array (always z,y,x shape, ignoring channels)
            cnt_ds = zarr_store.create_dataset(
                name=f"{tgt_name}_count",
                shape=(z_max, y_max, x_max),
                chunks=(chunk_z, chunk_y, chunk_x),
                dtype='float32',
                compressor=compressor,
                fill_value=0,
                write_empty_chunks=False
            )

            output_arrays[tgt_name] = sum_ds
            count_arrays[tgt_name] = cnt_ds

        model.eval()
        with torch.no_grad():

            # ---- first pass , raw data => float32 preds ---- #
            for batch_idx, data in tqdm(enumerate(loader), total=len(loader), desc="Running inference on patches..."):
                patches = data["image"].to(device)  # (batch, in_channels, z, y, x)

                raw_outputs = model(patches)
                outputs = {}
                for t_name in self.output_targets:
                    # e.g. t_name = "ink"
                    t_conf = self.inference_targets[t_name]  # e.g. {"channels":1, "activation":"sigmoid", ...}
                    activation_str = t_conf.get("activation", "none").lower()

                    if activation_str == "sigmoid":
                        outputs[t_name] = torch.sigmoid(raw_outputs[t_name])
                    elif activation_str == "softmax":
                        outputs[t_name] = torch.softmax(raw_outputs[t_name], dim=1)
                    elif activation_str == "none":
                        # no activation
                        outputs[t_name] = raw_outputs[t_name]
                    else:
                        # fallback to no activation
                        outputs[t_name] = raw_outputs[t_name]

                for i_in_batch in range(patches.size(0)):
                    global_idx = batch_idx * self.batch_size + i_in_batch
                    # retrieve the patch origin (z,y,x) from the dataset
                    z0, y0, x0 = dataset.all_positions[global_idx]

                    # for each target (e.g. "sheet", "normals")
                    for tgt_name in self.output_targets:
                        pred_patch = outputs[tgt_name][i_in_batch].cpu().numpy()
                        c = self.targets[tgt_name]["channels"]

                        # single channel pred_patch comes out with addtl channel like (1, z, y, x)
                        # drop first channel so (1, z, y, x) => (z, y, x) to save space/sanity
                        if c == 1 and pred_patch.shape[0] == 1:
                            pred_patch = np.squeeze(pred_patch, axis=0)

                        # compute patch size
                        z_size = pred_patch.shape[-3]
                        y_size = pred_patch.shape[-2]
                        x_size = pred_patch.shape[-1]

                        # read current sums from zarr
                        sum_block = output_arrays[tgt_name][...,
                                    z0:z0 + z_size,
                                    y0:y0 + y_size,
                                    x0:x0 + x_size
                                    ]

                        # read current counts
                        cnt_block = count_arrays[tgt_name][
                                    z0:z0 + z_size,
                                    y0:y0 + y_size,
                                    x0:x0 + x_size
                                    ]

                        # update sum
                        sum_block += pred_patch
                        # increment count by 1 for all voxels in this patch
                        cnt_block += 1

                        # store them back to zarr
                        output_arrays[tgt_name][...,
                        z0:z0 + z_size,
                        y0:y0 + y_size,
                        x0:x0 + x_size
                        ] = sum_block

                        count_arrays[tgt_name][
                        z0:z0 + z_size,
                        y0:y0 + y_size,
                        x0:x0 + x_size
                        ] = cnt_block

        # ---- sum / averaging pass on the overlaps ---- #
        for tgt_name in self.output_targets:
            sum_ds = output_arrays[tgt_name]
            cnt_ds = count_arrays[tgt_name]

            # sum_ds might be shape (c, z, y, x) or (z, y, x)
            # cnt_ds is shape (z, y, x)

            z = sum_ds.shape[-3]
            y = sum_ds.shape[-2]
            x = sum_ds.shape[-1]

            chunk_size = chunk_z
            for z0 in tqdm(range(0, z, chunk_size), desc=f"Averaging {tgt_name} overlaps"):
                z1 = min(z0 + chunk_size, z)
                for y0 in range(0, y, chunk_size):
                    y1 = min(y0 + chunk_size, y)
                    for x0 in range(0, x, chunk_size):
                        x1 = min(x0 + chunk_size, x)

                        sum_block = sum_ds[..., z0:z1, y0:y1, x0:x1]
                        cnt_block = cnt_ds[z0:z1, y0:y1, x0:x1]

                        # avoid divide by zero
                        mask = (cnt_block > 0)

                        # for multi-channel, sum_block might have shape (c, z_chunk, y_chunk, x_chunk)
                        # broadcast cnt_block if needed
                        # e.g. if sum_block is (c,z,y,x), we might do:
                        sum_block[..., mask] /= cnt_block[mask]

                        # write result
                        sum_ds[..., z0:z1, y0:y1, x0:x1] = sum_block

        # delete the "count" arrays
        # for tgt_name in self.output_targets:
        #    arr_name = f"{tgt_name}_count"
        #    del zarr_store[arr_name]

        # ---- final pass ---- #
        # cast float32 arrays to uint8 or uint16
        for tgt_name in self.output_targets:
            sum_ds = output_arrays[tgt_name]  # float32 "sum" dataset (already averaged)
            c = self.targets[tgt_name]["channels"]

            # decide the final data type
            # we want "normals" in uint16, everything else in uint8
            if tgt_name.lower() == "normals":
                final_dtype = "uint16"
            else:
                final_dtype = "uint8"

            # shape is (z_max, y_max, x_max) or (c, z_max, y_max, x_max)
            out_shape = sum_ds.shape
            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

            # create the final dataset to store scaled integer values
            final_ds = zarr_store.create_dataset(
                name=f"{tgt_name}_final",
                shape=out_shape,
                chunks=sum_ds.chunks,
                dtype=final_dtype,
                compressor=compressor,
                fill_value=0,
                write_empty_chunks=False
            )

            z = sum_ds.shape[-3]
            y = sum_ds.shape[-2]
            x = sum_ds.shape[-1]
            chunk_size_z = sum_ds.chunks[-3]
            chunk_size_y = sum_ds.chunks[-2]
            chunk_size_x = sum_ds.chunks[-1]

            for z0 in tqdm(range(0, z, chunk_size_z), desc=f"Casting {tgt_name} to int"):
                z1 = min(z0 + chunk_size_z, z)
                for y0 in range(0, y, chunk_size_y):
                    y1 = min(y0 + chunk_size_y, y)
                    for x0 in range(0, x, chunk_size_x):
                        x1 = min(x0 + chunk_size_x, x)

                        # read the float32 block from sum_ds
                        float_block = sum_ds[..., z0:z1, y0:y1, x0:x1]

                        # scale & cast depending on target
                        if tgt_name.lower() == "normals":
                            # normals are in [-1,1].
                            # scale them into [0,65000] for uint16.
                            int_block = (float_block + 1.0) / 2.0  # map [-1..1] => [0..1]
                            int_block *= 65535.0  # map [0..1] => [0..65000]
                            np.clip(int_block, 0, 65535, out=int_block)
                            int_block = int_block.astype(np.uint16)

                        else:
                            # everything else is already in [0,1].
                            # Scale into [0,255] for uint8.
                            int_block = float_block * 255.0
                            np.clip(int_block, 0, 255, out=int_block)
                            int_block = int_block.astype(np.uint8)

                        # write the scaled uint8 block back out
                        final_ds[..., z0:z1, y0:y1, x0:x1] = int_block


        if self.write_layers:
            slices_dir = os.path.join(self.output_dir, "z_slices")
            os.makedirs(slices_dir, exist_ok=True)

            for tgt_name in self.output_targets:
                target_dir = os.path.join(slices_dir, tgt_name)
                os.makedirs(target_dir, exist_ok=True)

                final_ds = zarr_store[f"{tgt_name}_final"]
                if len(final_ds.shape) == 4:
                    for z in tqdm(range(final_ds.shape[1]), desc=f"Writing {tgt_name} z-slices"):
                        slice_data = final_ds[:, z, :, :]
                        slice_data = slice_data.astype(np.uint8)
                        slice_path = os.path.join(target_dir, f"{z}.jpg")
                        cv2.imwrite(slice_path, slice_data)
                else:
                    for z in tqdm(range(final_ds.shape[0]), desc=f"Writing {tgt_name} z-slices"):
                        slice_data = final_ds[z, :, :]
                        slice_data = slice_data.astype(np.uint8)
                        slice_path = os.path.join(target_dir, f"{z}.jpg")
                        cv2.imwrite(slice_path, slice_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for MultiTaskResidualUNetSE3D.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to your config file. Use the same one you used for training!")
    parser.add_argument("--write_layers", action="store_true", help="Write the sliced z layers to disk")

    args = parser.parse_args()

    inference_handler = ZarrInferenceHandler(config_file=args.config_path, write_layers=args.write_layers)
    inference_handler.infer()
