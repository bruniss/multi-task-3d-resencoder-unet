import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import fsspec
import zarr

def check_patch_chunk_xyz(
    chunk,
    sheet_label,     # shape (Xdim, Ydim, Zdim)
    patch_size_xyz,  # (pX, pY, pZ)
    bbox_threshold=0.5,
    label_threshold=0.05
):
    pX, pY, pZ = patch_size_xyz
    valid_positions = []

    for (x, y, z) in chunk:
        # Extract the patch in X,Y,Z order
        patch = sheet_label[x:x + pX, y:y + pY, z:z + pZ]

        bbox = compute_bounding_box_3d_xyz(patch > 0)
        if bbox is None:
            continue

        bb_vol = bounding_box_volume_xyz(bbox)
        patch_vol = patch.size  # pX * pY * pZ

        # 1) bounding box coverage
        if bb_vol / patch_vol < bbox_threshold:
            continue

        # 2) fraction of labeled voxels
        labeled_ratio = np.count_nonzero(patch) / patch_vol
        if labeled_ratio < label_threshold:
            continue

        # Passed checks
        valid_positions.append((x, y, z))

    return valid_positions


def compute_bounding_box_3d_xyz(mask):
    """
    For a 3D boolean array in X,Y,Z order,
    returns (minx, maxx, miny, maxy, minz, maxz).
    """
    nonzero_coords = np.argwhere(mask)
    if nonzero_coords.size == 0:
        return None

    minx, miny, minz = nonzero_coords.min(axis=0)
    maxx, maxy, maxz = nonzero_coords.max(axis=0)
    return (minx, maxx, miny, maxy, minz, maxz)

def bounding_box_volume_xyz(bbox):
    """
    Given (minx, maxx, miny, maxy, minz, maxz),
    returns total voxels in that bounding box.
    """
    (minx, maxx, miny, maxy, minz, maxz) = bbox
    return ((maxx - minx + 1) *
            (maxy - miny + 1) *
            (maxz - minz + 1))


def chunker(seq, chunk_size):
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


def check_patch_chunk(chunk, sheet_label, patch_size, bbox_threshold=0.5, label_threshold=0.05):
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
def find_valid_patches_xyz(
    target_array,
    patch_size_xyz,         # (pX, pY, pZ)
    bbox_threshold=0.97,
    label_threshold=0.10,
    num_workers=4,
    overlap_fraction=0.25
):
    """
    Like find_valid_patches, but for data in X,Y,Z order.
    Returns valid patches in *zyx* positions by default,
    so it stays consistent with your final usage.
    """

    if target_array.ndim == 4:
        # let's assume shape is (C, X, Y, Z)
        target_array = target_array[0]

    pX, pY, pZ = patch_size_xyz
    Xdim, Ydim, Zdim = target_array.shape

    # Convert overlap fraction -> step (stride) along each axis
    x_stride = max(1, int(pX * (1 - overlap_fraction)))
    y_stride = max(1, int(pY * (1 - overlap_fraction)))
    z_stride = max(1, int(pZ * (1 - overlap_fraction)))

    # Generate all possible (x, y, z) starting positions
    all_positions = []
    for x in range(0, Xdim - pX + 1, x_stride):
        for y in range(0, Ydim - pY + 1, y_stride):
            for z in range(0, Zdim - pZ + 1, z_stride):
                all_positions.append((x, y, z))

    chunk_size = max(1, len(all_positions) // (num_workers * 2))
    position_chunks = list(chunker(all_positions, chunk_size))

    print(
        f"[XYZ] Finding valid patches of size: {patch_size_xyz} "
        f"(X,Y,Z) with bounding box >= {bbox_threshold} and labeled fraction >= {label_threshold}."
    )

    valid_positions_xyz = []
    with Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(
                check_patch_chunk_xyz,
                (
                    chunk,
                    target_array,
                    patch_size_xyz,
                    bbox_threshold,
                    label_threshold
                )
            )
            for chunk in position_chunks
        ]
        for r in tqdm(results, desc="Checking XYZ patches", total=len(results)):
            valid_positions_xyz.extend(r.get())

    # Convert valid (x,y,z) to a final list storing (z,y,x)
    valid_patches = []
    for (x, y, z) in valid_positions_xyz:
        valid_patches.append({
            'volume_idx': 0,
            'start_pos': [z, y, x]  # reorder to Z,Y,X if that’s your standard
        })

    print(f"[XYZ] Found {len(valid_positions_xyz)} valid patches (converted to Z,Y,X).")
    return valid_patches

def find_valid_patches(
    target_array,
    patch_size,
    bbox_threshold=0.97,  # bounding-box coverage fraction
    label_threshold=0.10, # minimum % of voxels labeled
    num_workers=4,
    overlap_fraction=0.25
):
    """
    Finds patches that have:
      - A bounding box of labeled voxels >= bbox_threshold fraction of the patch volume
      - Overall labeled voxel fraction >= label_threshold
    """

    # Patch dimensions
    pZ, pY, pX = patch_size
    Zdim, Ydim, Xdim = target_array.shape

    # Convert overlap fraction -> step (stride)
    z_stride = max(1, int(pZ * (1 - overlap_fraction)))
    y_stride = max(1, int(pY * (1 - overlap_fraction)))
    x_stride = max(1, int(pX * (1 - overlap_fraction)))

    # Generate all possible (z, y, x) starting positions
    all_positions = []
    for z in range(0, Zdim - pZ + 1, z_stride):
        for y in range(0, Ydim - pY + 1, y_stride):
            for x in range(0, Xdim - pX + 1, x_stride):
                all_positions.append((z, y, x))

    chunk_size = max(1, len(all_positions) // (num_workers * 2))
    position_chunks = list(chunker(all_positions, chunk_size))

    print(
        f"Finding valid patches of size: {patch_size} "
        f"with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}."
    )

    valid_positions_ref = []
    with Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(
                check_patch_chunk,
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

