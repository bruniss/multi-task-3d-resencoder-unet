import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import fsspec
import zarr

def _chunker(seq, chunk_size):
    """Yield successive 'chunk_size'-sized chunks from 'seq'."""
    for pos in range(0, len(seq), chunk_size):
        yield seq[pos:pos + chunk_size]

def compute_bounding_box_3d(mask):
    """
    Given a 3D boolean array (True where labeled, False otherwise),
    returns (minz, maxz, miny, maxy, minx, maxx).
    If there are no nonzero elements, returns None.
    """
    nonzero_coords = np.argwhere(mask)
    if nonzero_coords.size == 0:
        return None

    minz, miny, minx = nonzero_coords.min(axis=0)
    maxz, maxy, maxx = nonzero_coords.max(axis=0)
    return (minz, maxz, miny, maxy, minx, maxx)


def bounding_box_volume(bbox):
    """
    Given a bounding box (minz, maxz, miny, maxy, minx, maxx),
    returns the volume (number of voxels) inside the box.
    """
    minz, maxz, miny, maxy, minx, maxx = bbox
    return ((maxz - minz + 1) *
            (maxy - miny + 1) *
            (maxx - minx + 1))


def _check_patch_chunk(chunk, sheet_label, patch_size, bbox_threshold=0.5, label_threshold=0.05):
    """
    Worker function to check each patch in 'chunk' with both:
      - bounding box coverage >= bbox_threshold
      - overall labeled voxel ratio >= label_threshold
    """
    pD, pH, pW = patch_size
    valid_positions = []

    for (z, y, x) in chunk:
        patch = sheet_label[z:z + pD, y:y + pH, x:x + pW]
        # Compute bounding box of nonzero pixels in this patch
        bbox = compute_bounding_box_3d(patch > 0)
        if bbox is None:
            # No nonzero voxels at all -> skip
            continue

        # 1) Check bounding box coverage
        bb_vol = bounding_box_volume(bbox)
        patch_vol = patch.size  # pD * pH * pW
        if bb_vol / patch_vol < bbox_threshold:
            continue

        # 2) Check overall labeled fraction
        labeled_ratio = np.count_nonzero(patch) / patch_vol
        if labeled_ratio < label_threshold:
            continue

        # If we passed both checks, add to valid positions
        valid_positions.append((z, y, x))

    return valid_positions


def find_label_bounding_box(sheet_label_array,
                            chunk_shape=(192, 192, 192)) -> tuple:
    """
    Find the minimal bounding box (minz, maxz, miny, maxy, minx, maxx)
    that contains all non-zero voxels in `sheet_label_array`.

    :param sheet_label_array: a 3D zarr array of shape (D, H, W)
    :param chunk_shape: (chunk_z, chunk_y, chunk_x) to read from disk at once
    :return: (minz, maxz, miny, maxy, minx, maxx)
             such that sheet_label_array[minz:maxz+1, miny:maxy+1, minx:maxx+1]
             contains all non-zero voxels.
             If no non-zero voxel is found, returns (0, -1, 0, -1, 0, -1)
    """
    D, H, W = sheet_label_array.shape

    # Initialize bounding box to "empty"
    minz, miny, minx = D, H, W
    maxz = maxy = maxx = -1

    # We'll track total chunks for a TQDM progress bar
    # so we know how many chunk-reads we're doing
    num_chunks_z = (D + chunk_shape[0] - 1) // chunk_shape[0]
    num_chunks_y = (H + chunk_shape[1] - 1) // chunk_shape[1]
    num_chunks_x = (W + chunk_shape[2] - 1) // chunk_shape[2]
    total_chunks = num_chunks_z * num_chunks_y * num_chunks_x

    with tqdm(desc="Finding label bounding box", total=total_chunks) as pbar:
        for z_start in range(0, D, chunk_shape[0]):
            z_end = min(D, z_start + chunk_shape[0])
            for y_start in range(0, H, chunk_shape[1]):
                y_end = min(H, y_start + chunk_shape[1])
                for x_start in range(0, W, chunk_shape[2]):
                    x_end = min(W, x_start + chunk_shape[2])
                    # Read just this chunk from the zarr
                    chunk = sheet_label_array[z_start:z_end, y_start:y_end, x_start:x_end]

                    if chunk.any():  # means there's at least one non-zero voxel
                        # Find the local coords of non-zero voxels
                        nz_idx = np.argwhere(chunk > 0)  # shape (N, 3)
                        # Shift them by the chunk offset
                        nz_idx[:, 0] += z_start
                        nz_idx[:, 1] += y_start
                        nz_idx[:, 2] += x_start

                        cminz = nz_idx[:, 0].min()
                        cmaxz = nz_idx[:, 0].max()
                        cminy = nz_idx[:, 1].min()
                        cmaxy = nz_idx[:, 1].max()
                        cminx = nz_idx[:, 2].min()
                        cmaxx = nz_idx[:, 2].max()

                        # Update global bounding box
                        minz = min(minz, cminz)
                        maxz = max(maxz, cmaxz)
                        miny = min(miny, cminy)
                        maxy = max(maxy, cmaxy)
                        minx = min(minx, cminx)
                        maxx = max(maxx, cmaxx)

                    pbar.update(1)

    # If maxz remains -1, that means no non-zero voxel was found at all
    return (minz, maxz, miny, maxy, minx, maxx)

def _find_valid_patches(target_array,
                        patch_size,
                        bbox_threshold=0.97,  # bounding-box coverage fraction
                        label_threshold=0.10,  # minimum % of voxels labeled
                        num_workers=4):
    """
    Finds patches that contain:
      - a bounding box of labeled voxels >= bbox_threshold fraction of the patch volume
      - an overall labeled voxel fraction >= label_threshold
    """
    # Decide which volume to use as reference

    pZ, pY, pX = patch_size

    # find bounding box that contains all labels for a reference array
    minz, maxz, miny, maxy, minx, maxx = find_label_bounding_box(target_array)

    # generate possible start positions
    z_step = pZ // 2
    y_step = pY // 2
    x_step = pX // 2
    all_positions = []
    for z in range(minz, maxz - pZ + 2, z_step):
        for y in range(miny, maxy - pY + 2, y_step):
            for x in range(minx, maxx - pX + 2, x_step):
                all_positions.append((z, y, x))

    chunk_size = max(1, len(all_positions) // (num_workers * 2))
    position_chunks = list(_chunker(all_positions, chunk_size))

    print(
        f"Finding valid patches of size: {patch_size} "
        f"with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}."
    )

    valid_positions_ref = []
    with Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(
                _check_patch_chunk,
                (
                    chunk,
                    target_array,
                    patch_size,
                    bbox_threshold,  # pass bounding box threshold
                    label_threshold  # pass label fraction threshold
                )
            )
            for chunk in position_chunks
        ]
        for r in tqdm(results, desc="Checking patches", total=len(results)):
            valid_positions_ref.extend(r.get())

    valid_patches = []
    for (z, y, x) in valid_positions_ref:
        valid_patches.append({'volume_idx': 0, 'start_pos': [z, y, x]})

    print(
        f"Found {len(valid_positions_ref)} valid patches in reference volume. "
        f"Total {len(valid_patches)} across all volumes."
    )

    return valid_patches

def generate_positions(min_val, max_val, patch_size, step):
    """
    Returns a list of start indices (inclusive) for sliding-window patches,
    ensuring the final patch covers the end of the volume.
    """
    positions = []
    pos = min_val
    while pos + patch_size <= max_val:
        positions.append(pos)
        pos += step

    # Force the last patch if not already covered
    last_start = max_val - patch_size
    if last_start > positions[-1]:
        positions.append(last_start)

    return sorted(set(positions))


def get_overlapping_chunks(z0, y0, x0, patch_z, patch_y, patch_x,
                           chunk_z, chunk_y, chunk_x):
    """
    Given a patch at (z0, y0, x0) with shape (patch_z, patch_y, patch_x),
    returns a list of:
    [
      (cz, cy, cx,
       z_start_in_chunk, z_end_in_chunk,
       y_start_in_chunk, y_end_in_chunk,
       x_start_in_chunk, x_end_in_chunk,
       z_start_in_patch, z_end_in_patch,
       y_start_in_patch, y_end_in_patch,
       x_start_in_patch, x_end_in_patch),
      ...
    ]
    where (cz, cy, cx) is the chunk index.
    """
    results = []

    # Global patch range:
    z1 = z0 + patch_z
    y1 = y0 + patch_y
    x1 = x0 + patch_x

    # For each chunk index in Z:
    cz0 = z0 // chunk_z
    cz1 = (z1 - 1) // chunk_z  # inclusive
    for cz in range(cz0, cz1 + 1):
        # chunk's global z range
        chunk_z0_global = cz * chunk_z
        chunk_z1_global = chunk_z0_global + chunk_z

        # Overlap in Z:
        overlap_z0 = max(z0, chunk_z0_global)
        overlap_z1 = min(z1, chunk_z1_global)

        # local offset inside chunk:
        z_start_in_chunk = overlap_z0 - chunk_z0_global
        z_end_in_chunk = overlap_z1 - chunk_z0_global

        # offset inside patch:
        z_start_in_patch = overlap_z0 - z0
        z_end_in_patch = overlap_z1 - z0

        # Similar logic for Y:
        cy0 = y0 // chunk_y
        cy1 = (y1 - 1) // chunk_y
        for cy in range(cy0, cy1 + 1):
            chunk_y0_global = cy * chunk_y
            chunk_y1_global = chunk_y0_global + chunk_y
            overlap_y0 = max(y0, chunk_y0_global)
            overlap_y1 = min(y1, chunk_y1_global)
            y_start_in_chunk = overlap_y0 - chunk_y0_global
            y_end_in_chunk = overlap_y1 - chunk_y0_global
            y_start_in_patch = overlap_y0 - y0
            y_end_in_patch = overlap_y1 - y0

            # Similar logic for X:
            cx0 = x0 // chunk_x
            cx1 = (x1 - 1) // chunk_x
            for cx in range(cx0, cx1 + 1):
                chunk_x0_global = cx * chunk_x
                chunk_x1_global = chunk_x0_global + chunk_x
                overlap_x0 = max(x0, chunk_x0_global)
                overlap_x1 = min(x1, chunk_x1_global)
                x_start_in_chunk = overlap_x0 - chunk_x0_global
                x_end_in_chunk = overlap_x1 - chunk_x0_global
                x_start_in_patch = overlap_x0 - x0
                x_end_in_patch = overlap_x1 - x0

                results.append((
                    cz, cy, cx,
                    z_start_in_chunk, z_end_in_chunk,
                    y_start_in_chunk, y_end_in_chunk,
                    x_start_in_chunk, x_end_in_chunk,
                    z_start_in_patch, z_end_in_patch,
                    y_start_in_patch, y_end_in_patch,
                    x_start_in_patch, x_end_in_patch
                ))
    return results

def open_zarr_path(path: str, mode='r'):
    """
    Attempts to open a Zarr store from either a local path or a remote URL.

    Examples of possible `path` values:
      - '/local/path/to/data.zarr'
      - 's3://my-bucket/folder/data.zarr'
      - 'https://my-dataset-server.com/data.zarr'
    """
    # Check if it's likely a remote path
    if (
        path.startswith('http://') or path.startswith('https://')
        or path.startswith('s3://') or path.startswith('gs://')
        # Add more protocols here if needed
    ):
        store = fsspec.get_mapper(path)
        return zarr.open(store, mode=mode)
    else:
        # Assume it's a local path
        return zarr.open(path, mode=mode)
