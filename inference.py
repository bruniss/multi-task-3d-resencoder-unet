import os
import json
import argparse
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
import zarr
from numcodecs import Blosc
import cv2
import yaml
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from pytorch3dunet.unet3d.model import MultiTaskResidualUNetSE3D
from helpers import get_overlapping_chunks

from dataloading.inference_dataset import InferenceDataset

class ZarrInferenceHandler:

    def __init__(self, config_file: str, write_layers: bool):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

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
        self.output_dtype = str(getattr(inference_params, "output_type", "np.uint8"))
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
            input_path=self.input_path,
            targets=self.output_targets,
            patch_size=self.patch_size,
            normalization=self.normalization,
            input_format=self.input_format,
            overlap=self.overlap,
            load_all=self.load_all
        )

        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_dataloader_workers,
                            prefetch_factor=8,
                            pin_memory=True,
                            persistent_workers=True)

        z_max, y_max, x_max = dataset.input_shape
        output_arrays = {}
        count_arrays = {}

        store_path = os.path.join(self.output_dir, "predictions.zarr")
        zarr_store = zarr.open(store_path, mode='w')

        chunk_z = 128
        chunk_y = 128
        chunk_x = 128

        # -- Create sum/count datasets for each target --
        for tgt_name in self.output_targets:
            c = self.targets[tgt_name]["channels"]

            if c == 1:
                out_shape = (z_max, y_max, x_max)
                chunks = (chunk_z, chunk_y, chunk_x)
            else:
                out_shape = (c, z_max, y_max, x_max)
                chunks = (c, chunk_z, chunk_y, chunk_x)

            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

            sum_ds = zarr_store.create_dataset(
                name=f"{tgt_name}_sum",
                shape=out_shape,
                chunks=chunks,
                dtype='float32',
                compressor=compressor,
                fill_value=0
            )

            cnt_ds = zarr_store.create_dataset(
                name=f"{tgt_name}_count",
                shape=(z_max, y_max, x_max),
                chunks=(chunk_z, chunk_y, chunk_x),
                dtype='float32',
                compressor=compressor,
                fill_value=0
            )

            output_arrays[tgt_name] = sum_ds
            count_arrays[tgt_name] = cnt_ds

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            # ---- Inference pass (accumulate sums) ---- #
            for batch_idx, data in tqdm(enumerate(loader), total=len(loader), desc="Running inference on patches..."):
                patches = data["image"].to(device)  # (batch, in_channels, z, y, x)

                raw_outputs = model(patches)
                outputs = {}
                for t_name in self.output_targets:
                    t_conf = self.inference_targets[t_name]
                    activation_str = t_conf.get("activation", "none").lower()

                    if activation_str == "sigmoid":
                        outputs[t_name] = torch.sigmoid(raw_outputs[t_name])
                    elif activation_str == "softmax":
                        outputs[t_name] = torch.softmax(raw_outputs[t_name], dim=1)
                    else:
                        # "none" or fallback
                        outputs[t_name] = raw_outputs[t_name]

                # Write each patch's result into the sum / count zarr arrays
                for i_in_batch in range(patches.size(0)):
                    global_idx = batch_idx * self.batch_size + i_in_batch
                    z0, y0, x0 = dataset.all_positions[global_idx]

                    for tgt_name in self.output_targets:
                        pred_patch = outputs[tgt_name][i_in_batch].cpu().numpy()
                        c = self.targets[tgt_name]["channels"]

                        # If single-channel, remove the extra leading channel dim
                        if c == 1 and pred_patch.shape[0] == 1:
                            pred_patch = np.squeeze(pred_patch, axis=0)

                        z_size = pred_patch.shape[-3]
                        y_size = pred_patch.shape[-2]
                        x_size = pred_patch.shape[-1]

                        sum_block = output_arrays[tgt_name][..., z0:z0 + z_size,
                                                             y0:y0 + y_size,
                                                             x0:x0 + x_size]
                        cnt_block = count_arrays[tgt_name][z0:z0 + z_size,
                                                           y0:y0 + y_size,
                                                           x0:x0 + x_size]

                        sum_block += pred_patch
                        cnt_block += 1

                        output_arrays[tgt_name][..., z0:z0 + z_size,
                                                y0:y0 + y_size,
                                                x0:x0 + x_size] = sum_block

                        count_arrays[tgt_name][z0:z0 + z_size,
                                               y0:y0 + y_size,
                                               x0:x0 + x_size] = cnt_block

        # ---- Post-processing of overlaps ----
        #  1) For non-normals => average
        #  2) For normals => renormalize (only sum, do not average)
        for tgt_name in self.output_targets:
            sum_ds = output_arrays[tgt_name]
            cnt_ds = count_arrays[tgt_name]
            c = self.targets[tgt_name]["channels"]

            z = sum_ds.shape[-3]
            y = sum_ds.shape[-2]
            x = sum_ds.shape[-1]
            chunk_size = chunk_z

            # If this is the normals target, we do a vector normalization instead of averaging:
            is_normals = (tgt_name.lower() == "normals")

            for z0 in tqdm(range(0, z, chunk_size), desc=f"Processing overlaps for {tgt_name}"):
                z1 = min(z0 + chunk_size, z)
                for y0 in range(0, y, chunk_size):
                    y1 = min(y0 + chunk_size, y)
                    for x0 in range(0, x, chunk_size):
                        x1 = min(x0 + chunk_size, x)

                        sum_block = sum_ds[..., z0:z1, y0:y1, x0:x1]
                        cnt_block = cnt_ds[z0:z1, y0:y1, x0:x1]

                        mask = (cnt_block > 0)

                        # If "normals", treat sum_block as vector data
                        if is_normals:
                            # sum_block shape -> (c, z_chunk, y_chunk, x_chunk), c=3 expected
                            # We'll re-normalize each voxel's 3D vector where count>0
                            # NOTE: We do NOT average, we just keep the sum and do v = v / |v|
                            # But if you prefer "average then normalize", you could also do sum_block[...,mask] /= cnt_block[mask].
                            # For pure 'sum then normalize', skip dividing by count:
                            # -> sum_block = sum of overlap patches => now normalize
                            # We'll do the normalization channel-by-channel
                            if c == 3:  # typical for a 3D normal
                                # small epsilon to avoid /0
                                eps = 1e-8
                                # compute magnitude along axis=0 => shape (z_chunk, y_chunk, x_chunk)
                                mag = np.sqrt(
                                    sum_block[0] ** 2 + sum_block[1] ** 2 + sum_block[2] ** 2
                                ) + eps

                                # only do for positions where mask == True
                                sum_block[0][mask] = sum_block[0][mask] / mag[mask]
                                sum_block[1][mask] = sum_block[1][mask] / mag[mask]
                                sum_block[2][mask] = sum_block[2][mask] / mag[mask]

                            else:
                                print(f"Warning: normals target has c={c}, expected 3. Skipping normalization.")

                            # write back
                            sum_ds[..., z0:z1, y0:y1, x0:x1] = sum_block

                        else:
                            # For non-normals => do standard average: sum / count
                            # sum_block might be (c, z_chunk, y_chunk, x_chunk) or (z_chunk, y_chunk, x_chunk)
                            sum_block[..., mask] /= cnt_block[mask]
                            sum_ds[..., z0:z1, y0:y1, x0:x1] = sum_block

        # ---- Final pass: cast float32 to uint8/uint16, with scaling ----
        for tgt_name in self.output_targets:
            sum_ds = output_arrays[tgt_name]  # float32
            c = self.targets[tgt_name]["channels"]

            # decide final dtype
            if tgt_name.lower() == "normals":
                final_dtype = "uint16"
            else:
                final_dtype = "uint8"

            out_shape = sum_ds.shape
            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

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

                        float_block = sum_ds[..., z0:z1, y0:y1, x0:x1]

                        if tgt_name.lower() == "normals":
                            # Map [-1..1] -> [0..65535]
                            int_block = (float_block + 1.0) / 2.0
                            int_block *= 65535.0
                            np.clip(int_block, 0, 65535, out=int_block)
                            int_block = int_block.astype(np.uint16)
                        else:
                            # Map [0..1] -> [0..255]
                            int_block = float_block * 255.0
                            np.clip(int_block, 0, 255, out=int_block)
                            int_block = int_block.astype(np.uint8)

                        final_ds[..., z0:z1, y0:y1, x0:x1] = int_block

        # Optionally write out .jpg slices
        if self.write_layers:
            slices_dir = os.path.join(self.output_dir, "z_slices")
            os.makedirs(slices_dir, exist_ok=True)

            for tgt_name in self.output_targets:
                target_dir = os.path.join(slices_dir, tgt_name)
                os.makedirs(target_dir, exist_ok=True)

                final_ds = zarr_store[f"{tgt_name}_final"]
                # shape => either (c, z, y, x) or (z, y, x)
                if len(final_ds.shape) == 4:
                    # example: normals => (3, z, y, x)
                    for z in tqdm(range(final_ds.shape[1]), desc=f"Writing {tgt_name} z-slices"):
                        # final_ds[:, z, :, :] => shape (c, y, x)
                        slice_data = final_ds[:, z, :, :]
                        slice_data = slice_data.astype(np.uint8)

                        # If c=3 for color, OpenCV expects (y, x, 3)
                        if slice_data.shape[0] == 3:
                            slice_data = np.transpose(slice_data, (1, 2, 0))  # (3,y,x)->(y,x,3)

                        slice_path = os.path.join(target_dir, f"{z}.jpg")
                        cv2.imwrite(slice_path, slice_data)
                else:
                    # single channel => (z, y, x)
                    for z in tqdm(range(final_ds.shape[0]), desc=f"Writing {tgt_name} z-slices"):
                        slice_data = final_ds[z, :, :].astype(np.uint8)
                        slice_path = os.path.join(target_dir, f"{z}.jpg")
                        cv2.imwrite(slice_path, slice_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for MultiTaskResidualUNetSE3D.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to your config file. Use the same one you used for training!")
    parser.add_argument("--write_layers", action="store_true", help="Write the sliced z layers to disk")

    args = parser.parse_args()

    inference_handler = ZarrInferenceHandler(config_file=args.config_path, write_layers=args.write_layers)
    inference_handler.infer()
