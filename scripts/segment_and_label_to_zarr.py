import os
import numpy as np
import zarr
import cv2
import glob
import re
from tqdm import tqdm

def natural_sort_key(s):
    # Split the string into text and numeric parts
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def stack_images_to_zarr(input_folder,
                         start,
                         stop,
                         layers_only=False):
    # Get all image files and sort them naturally
    layers = glob.glob(os.path.join(f"{input_folder}/layers", '*.tif'))
    layers.sort(key=natural_sort_key)

    if not layers:
        raise ValueError(f"No .tif or .png files found in {input_folder}")

    if not layers_only:
        inklabels = glob.glob(os.path.join(f"{input_folder}/inklabels", '*.png'))
        inklabels.sort(key=natural_sort_key)
        if not inklabels:
            raise ValueError(f"No inklabels found in {input_folder}")
        print(f"Found {len(layers)} layers and {len(inklabels)} inklabels")
    else:
        print(f"Found {len(layers)} layers (layers only mode)")

    # Read first image to get shape
    layers_dtype, layers_shape = cv2.imread(layers[0]).dtype, cv2.imread(layers[0]).shape

    if not layers_only:
        inklabels_dtype, inklabels_shape = cv2.imread(inklabels[0]).dtype, cv2.imread(inklabels[0]).shape
        assert layers_shape == inklabels_shape, (
            f"layers and labels must have the same dimensions\n"
            f"first layer: {layers_shape}\n"
            f"first label: {inklabels_shape}"
        )

    # Create zarr array
    parent_folder = os.path.dirname(input_folder)
    z_name = os.path.basename(input_folder)
    zarr_path = os.path.join(parent_folder, f'{z_name}.zarr')
    print(f"Creating zarr array at {zarr_path}...")

    store = zarr.DirectoryStore(zarr_path)
    z_root = zarr.group(store=store, overwrite=True)

    z_chunks = stop - start

    # Create datasets
    z_root.create_dataset(
        'layers',
        shape=(stop - start, layers_shape[0], layers_shape[1]),
        chunks=(z_chunks, 128, 128),
        dtype=np.uint8
    )

    if not layers_only:
        z_root.create_dataset(
            'inklabels',
            shape=(stop - start, layers_shape[0], layers_shape[1]),
            chunks=(z_chunks, 128, 128),
            dtype=np.uint8
        )

    for idx, layer in enumerate(tqdm(layers[start:stop], desc="Loading and writing layers")):
        img = cv2.imread(layer, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        scaled_img = (img / 65535.0 * 255).astype(np.uint8)
        z_root['layers'][idx] = scaled_img

    if not layers_only:
        first_inklabel = cv2.imread(inklabels[0], cv2.IMREAD_GRAYSCALE)
        for idx in tqdm(range(stop - start), desc="Loading and writing inklabels"):
            if idx + start < len(inklabels):
                inklabel = cv2.imread(inklabels[idx + start], cv2.IMREAD_GRAYSCALE)
            else:
                inklabel = first_inklabel
            z_root['inklabels'][idx] = inklabel

    print("Done!")


if __name__ == "__main__":
    input_folder = "/home/sean/Desktop/s1_segments/1619/"
    stack_images_to_zarr(input_folder, start=20, stop=40, layers_only=True)  # Set layers_only=True to skip inklabels
