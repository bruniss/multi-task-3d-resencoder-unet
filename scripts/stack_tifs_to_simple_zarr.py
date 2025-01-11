import os
import numpy as np
import zarr
from PIL import Image
import glob
import re

Image.MAX_IMAGE_PIXELS = None


def natural_sort_key(s):
    # Split the string into text and numeric parts
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def stack_images_to_zarr(input_folder, output_name):
    # Get all image files and sort them naturally
    image_files = glob.glob(os.path.join(input_folder, '*.[tp][in][gf]'))  # matches .tif and .png
    image_files.sort(key=natural_sort_key)  # Sort numerically

    if not image_files:
        raise ValueError(f"No .tif or .png files found in {input_folder}")

    # Read first image to get dimensions
    first_img = np.array(Image.open(image_files[0]))
    height, width = first_img.shape

    # Create zarr array
    parent_folder = os.path.dirname(input_folder)
    zarr_path = os.path.join(parent_folder, f'{output_name}.zarr')

    # Initialize zarr array with known dimensions
    z = zarr.open(zarr_path, mode='w', shape=(len(image_files), height, width),
                  chunks=(1, height, width), dtype=first_img.dtype)

    # Load and stack images
    for idx, img_path in enumerate(image_files):
        try:
            # Open and convert image to numpy array
            img = np.array(Image.open(img_path))

            # Verify dimensions match
            if img.shape != (height, width):
                raise ValueError(f"Image {img_path} has different dimensions from first image")

            # Write to zarr array
            z[idx] = img
            print(f"Processed {idx + 1}/{len(image_files)}: {os.path.basename(img_path)}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            raise

    print(f"\nComplete! Zarr file saved to: {zarr_path}")
    print(f"Final array shape: {z.shape}")

# Example usage
if __name__ == "__main__":
    input_folder = "/mnt/raid_hdd/s1_segments/5753/inklabels"
    output_name = "inklabels"
    stack_images_to_zarr(input_folder.strip(), output_name)