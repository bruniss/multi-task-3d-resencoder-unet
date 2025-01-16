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
from functools import lru_cache
from scipy.ndimage import gaussian_filter

from builders.build_network_from_config import BuildNetworkFromConfig
from dataloading.inference_dataset import InferenceDataset


class ZarrInferenceHandler:
    def __init__(self, config_file: str, write_layers: bool, postprocess_only: bool = False):
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

        self.checkpoint_path = getattr(inference_params, "checkpoint_path", None)
        self.load_strict = bool(getattr(inference_params, "load_strict", True))
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
        self.vram_max = float(getattr(tr_params, "vram_max", 12.0))
        self.autoconfigure = bool(getattr(tr_params, "autoconfigure", False))
        self.output_dir = str(getattr(inference_params, "output_dir", "./"))
        self.write_layers = write_layers
        self.volume_paths = dataset_config.volume_paths
        self.inference_targets = inference_params.targets
        self.in_channels = int(getattr(dataset_config, "in_channels", 1))

        os.makedirs(self.output_dir, exist_ok=True)
        self.tasks = dataset_config.targets
        self.out_channels = ()
        for task_name, task_info in self.tasks.items():
            self.out_channels += (task_info["channels"],)

        self.model_kwargs = vars(model_config).copy()
        self.postprocess_only = postprocess_only

    def _build_model(self, model_kwargs):
        builder = BuildNetworkFromConfig(
            tasks=self.tasks,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=self.batch_size,
            vram_target=self.vram_max,
            autoconfigure=self.autoconfigure,
            **model_kwargs,
        )

        vram = builder.estimate_vram_usage()
        print(f"Estimated vram usage for this model with your configs: {vram} MB")
        if vram > self.vram_max:
            print(f"Estimated vram use of {vram} is greater than the vram max set in your config. "
                  "Exiting. Please adjust your config accordingly.")
            return None

        model = builder.build()
        model.print_config()
        return model

    _gauss_map_cache = {}
    def infer(self):
        store_path = os.path.join(self.output_dir, "predictions.zarr")

        # If we are NOT only postprocessing, run normal inference pass
        if not self.postprocess_only:
            model = self._build_model(self.model_kwargs)
            if model is None:
                return  # vram check failed

            torch.set_float32_matmul_precision('high')
            device = torch.device('cuda')
            model = torch.compile(model)
            model = model.to(device)
            checkpoint = torch.load(self.checkpoint_path, map_location=device)

            if not self.load_strict:
                model.load_state_dict(checkpoint['model'], strict=False)
            else:
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

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_dataloader_workers,
                prefetch_factor=8,
                pin_memory=True,
                persistent_workers=True
            )

            z_max, y_max, x_max = dataset.input_shape
            output_arrays = {}
            count_arrays = {}

            # Protect against overwriting an existing zarr store
            if os.path.isdir(store_path):
                raise FileExistsError(
                    f"Zarr store '{store_path}' already exists. Aborting to prevent overwrite."
                )

            zarr_store = zarr.open(store_path, mode='w')

            chunk_z = self.patch_size[0]
            chunk_y = self.patch_size[1]
            chunk_x = self.patch_size[2]

            # Create sum/count datasets for each target
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

            # Inference loop
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for batch_idx, data in tqdm(enumerate(loader), total=len(loader),
                                            desc="Running inference on patches..."):
                    patches = data["image"].to(device)
                    raw_outputs = model(patches)

                    # Apply final activation
                    outputs = {}
                    for t_name in self.output_targets:
                        t_conf = self.inference_targets[t_name]
                        act = t_conf.get("activation", "none").lower()
                        if act == "sigmoid":
                            outputs[t_name] = torch.sigmoid(raw_outputs[t_name])
                        elif act == "softmax":
                            outputs[t_name] = torch.softmax(raw_outputs[t_name], dim=1)
                        else:
                            outputs[t_name] = raw_outputs[t_name]

                    # Accumulate with Gaussian weighting
                    for i_in_batch in range(patches.size(0)):
                        global_idx = batch_idx * self.batch_size + i_in_batch
                        z0, y0, x0 = dataset.all_positions[global_idx]

                        for tgt_name in self.output_targets:
                            pred_patch = outputs[tgt_name][i_in_batch].cpu().numpy()
                            c = self.targets[tgt_name]["channels"]

                            # Retrieve or compute a suitable Gaussian
                            weight_patch = get_gaussian_map(
                                channels=c,
                                patch_size_3d=self.patch_size,  # (D,H,W)
                                sigma_scale=1./8,
                                device=torch.device('cpu')
                            )

                            # pred_patch shape: (c, D, H, W) or (D, H, W) if c=1
                            # sum_block shape: same (for the last 3 dims)
                            z_size = pred_patch.shape[-3]
                            y_size = pred_patch.shape[-2]
                            x_size = pred_patch.shape[-1]

                            sum_block = output_arrays[tgt_name][..., z0:z0 + z_size,
                                                                       y0:y0 + y_size,
                                                                       x0:x0 + x_size]
                            cnt_block = count_arrays[tgt_name][z0:z0 + z_size,
                                                               y0:y0 + y_size,
                                                               x0:x0 + x_size]

                            # Weighted accumulation
                            sum_block += pred_patch * weight_patch
                            # For count, we add only one channel’s weight if multi-channel
                            if c > 1:
                                cnt_block += weight_patch[0]
                            else:
                                cnt_block += weight_patch

                            # Write back
                            output_arrays[tgt_name][..., z0:z0 + z_size,
                                                            y0:y0 + y_size,
                                                            x0:x0 + x_size] = sum_block
                            count_arrays[tgt_name][z0:z0 + z_size,
                                                   y0:y0 + y_size,
                                                   x0:x0 + x_size] = cnt_block

        else:
            # If we only want to postprocess, open the existing store in read+ mode
            zarr_store = zarr.open(store_path, mode='r+')

        # ---------------------------------------------------------------------
        # Final overlap processing & casting
        # ---------------------------------------------------------------------
        for tgt_name in self.output_targets:
            sum_ds = zarr_store[f"{tgt_name}_sum"]
            cnt_ds = zarr_store[f"{tgt_name}_count"]
            c = self.targets[tgt_name]["channels"]

            z = sum_ds.shape[-3]
            y = sum_ds.shape[-2]
            x = sum_ds.shape[-1]
            chunk_size = sum_ds.chunks[-3]  # or self.patch_size[0], etc.

            # Check if it's a normals target
            is_normals = (tgt_name.lower() == "normals")

            # Overlap resolution (averaging or re-normalization)
            for z0 in tqdm(range(0, z, chunk_size), desc=f"Processing overlaps for {tgt_name}"):
                z1 = min(z0 + chunk_size, z)
                for y0 in range(0, y, chunk_size):
                    y1 = min(y0 + chunk_size, y)
                    for x0 in range(0, x, chunk_size):
                        x1 = min(x0 + chunk_size, x)

                        sum_block = sum_ds[..., z0:z1, y0:y1, x0:x1]
                        cnt_block = cnt_ds[z0:z1, y0:y1, x0:x1]

                        mask = (cnt_block > 0)

                        if is_normals:
                            # Re-normalize vector sums (not a normal average)
                            if c == 3:
                                eps = 1e-8
                                mag = np.sqrt(
                                    sum_block[0]**2 + sum_block[1]**2 + sum_block[2]**2
                                ) + eps

                                sum_block[0][mask] /= mag[mask]
                                sum_block[1][mask] /= mag[mask]
                                sum_block[2][mask] /= mag[mask]
                            else:
                                print(f"Warning: 'normals' target has c={c}, expected 3. Skipping normalization.")
                            sum_ds[..., z0:z1, y0:y1, x0:x1] = sum_block
                        else:
                            # Average (sum / count)
                            sum_block[..., mask] /= cnt_block[mask]
                            sum_ds[..., z0:z1, y0:y1, x0:x1] = sum_block

        # ---- Final pass: cast float32 -> int (uint8 or uint16) ---
        for tgt_name in self.output_targets:
            sum_ds = zarr_store[f"{tgt_name}_sum"]
            c = self.targets[tgt_name]["channels"]

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

            for z0 in tqdm(range(0, z, chunk_size_z), desc=f"Casting {tgt_name} to {final_dtype} for final write"):
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

        # ---- (Optional) write JPEG slices if requested ----
        if self.write_layers:
            slices_dir = os.path.join(self.output_dir, "z_slices")
            os.makedirs(slices_dir, exist_ok=True)

            for tgt_name in self.output_targets:
                target_dir = os.path.join(slices_dir, tgt_name)
                os.makedirs(target_dir, exist_ok=True)

                final_ds = zarr_store[f"{tgt_name}_final"]
                # shape => either (c, z, y, x) or (z, y, x)
                if len(final_ds.shape) == 4:
                    # e.g. normals => (3, z, y, x)
                    for z_idx in tqdm(range(final_ds.shape[1]), desc=f"Writing {tgt_name} z-slices"):
                        slice_data = final_ds[:, z_idx, :, :].astype(np.uint8)
                        if slice_data.shape[0] == 3:
                            slice_data = np.transpose(slice_data, (1, 2, 0))  # (H, W, C)
                        slice_path = os.path.join(target_dir, f"{z_idx}.jpg")
                        cv2.imwrite(slice_path, slice_data)
                else:
                    # single channel => (z, y, x)
                    for z_idx in tqdm(range(final_ds.shape[0]), desc=f"Writing {tgt_name} z-slices"):
                        slice_data = final_ds[z_idx, :, :].astype(np.uint8)
                        slice_path = os.path.join(target_dir, f"{z_idx}.jpg")
                        cv2.imwrite(slice_path, slice_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for MultiTaskResidualUNetSE3D.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to your config file. Use the same one you used for training!")
    parser.add_argument("--write_layers", action="store_true",
                        help="Write the sliced z layers to disk")
    parser.add_argument("--postprocess_only", action="store_true",
                        help="Skip the inference pass and only do final averaging + casting on existing sums/counts.")

    args = parser.parse_args()

    inference_handler = ZarrInferenceHandler(
        config_file=args.config_path,
        write_layers=args.write_layers,
        postprocess_only=args.postprocess_only
    )
    inference_handler.infer()
