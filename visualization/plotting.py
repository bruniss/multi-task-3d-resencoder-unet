import cv2
import imageio
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def apply_activation_if_needed(scalar_or_vector, activation_str):
    """
    scalar_or_vector: a NumPy array (e.g., shape [D, H, W] or [C, D, H, W])
    activation_str: e.g., "none", "sigmoid", "softmax", etc.

    Returns a NumPy array with the activation applied if appropriate.
    """
    if not activation_str or activation_str.lower() == "none":
        return scalar_or_vector

    if activation_str.lower() == "sigmoid":
        return 1.0 / (1.0 + np.exp(-scalar_or_vector))

    # If "softmax" is needed, implement as appropriate
    return scalar_or_vector

def convert_slice_to_bgr(
    slice_2d_or_3d: np.ndarray,
    show_magnitude: bool = False,
    dynamic_range: bool = True
) -> np.ndarray:
    """
    Converts a slice (either single-channel grayscale or 3-channel normal) into a BGR image
    by doing min..max scaling to the full range of values present in *that slice*.

    Args:
        slice_2d_or_3d (np.ndarray): shape [H, W] for single-channel, or [3, H, W] for 3-channel.
        show_magnitude (bool): If True and data is 3-channel, we horizontally stack an extra panel
                               showing that slice's magnitude, also min–max scaled to [0..255].
        dynamic_range (bool): If True, we scale by the slice's min and max. If False, we clamp
                              to [-1..1] for 3-channel (old approach) or [0..1] for single-channel.

    Returns:
        np.ndarray: BGR image of shape [H, W, 3], or shape [H, 2*W, 3] if magnitude panel is shown.
    """

    def minmax_scale_to_8bit(img: np.ndarray) -> np.ndarray:
        """Scales `img` so that its min -> 0 and max -> 255 for this slice only."""
        min_val = img.min()
        max_val = img.max()
        eps = 1e-6
        if (max_val - min_val) > eps:
            img_scaled = (img - min_val) / (max_val - min_val)
        else:
            # Instead of zeros, you could use a constant value:
            img_scaled = np.full_like(img, 0.5, dtype=np.float32)
        return (img_scaled * 255).astype(np.uint8)

    # -----------------------------------------
    # Case 1: Single-channel [H, W]
    # -----------------------------------------
    if slice_2d_or_3d.ndim == 2:
        if dynamic_range:
            slice_8u = minmax_scale_to_8bit(slice_2d_or_3d)
        else:
            # Old clamp approach
            slice_clamped = np.clip(slice_2d_or_3d, 0, 1)
            slice_8u = (slice_clamped * 255).astype(np.uint8)
        return cv2.cvtColor(slice_8u, cv2.COLOR_GRAY2BGR)

    # -----------------------------------------
    # Case 2: Three-channel [3, H, W]
    # -----------------------------------------
    elif (
        slice_2d_or_3d.ndim == 3
        and slice_2d_or_3d.shape[0] == 3
    ):
        # shape => [3, H, W]
        if dynamic_range:
            # Per-channel local min..max
            ch_list = []
            for ch_idx in range(3):
                ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[ch_idx])
                ch_list.append(ch_8u)
            mapped_normals = np.stack(ch_list, axis=0)  # shape => [3, H, W]
        else:
            # Old clamp approach for normals => [-1..1] mapped to [0..255]
            normal_slice = np.clip(slice_2d_or_3d, -1, 1)
            mapped_normals = ((normal_slice * 0.5) + 0.5) * 255
            mapped_normals = np.clip(mapped_normals, 0, 255).astype(np.uint8)

        # Reorder to [H, W, 3]
        bgr_normals = np.transpose(mapped_normals, (1, 2, 0))

        # Optionally add a separate magnitude panel
        if show_magnitude:
            # Magnitude of the original float channels
            mag = np.linalg.norm(slice_2d_or_3d, axis=0)  # => [H, W]
            if dynamic_range:
                mag_8u = minmax_scale_to_8bit(mag)
            else:
                # If you like, you could do the same [-1..1] clamp, but magnitude rarely is negative
                mag_8u = minmax_scale_to_8bit(mag)
            mag_bgr = cv2.cvtColor(mag_8u, cv2.COLOR_GRAY2BGR)
            # Combine horizontally: [H, W + W, 3]
            return np.hstack([mag_bgr, bgr_normals])

        return bgr_normals

    else:
        raise ValueError(
            f"convert_slice_to_bgr expects shape [H, W] or [3, H, W], got {slice_2d_or_3d.shape}"
        )



def log_3d_slices_as_images(
    writer: SummaryWriter,
    tag_prefix: str,
    volume_3d: torch.Tensor,       # shape [1, C, D, H, W]
    task_name: str = "",           # optional
    tasks_dict: dict = None,       # e.g. {"sheet": {"activation": "sigmoid"}}
    global_step: int = 0,
    max_slices: int = 8
):
    """
    Logs up to `max_slices` equally spaced 2D slices from a 3D volume to TensorBoard.

    Args:
        writer (SummaryWriter): The TensorBoard SummaryWriter.
        tag_prefix (str): Prefix for the logged images (e.g., "val_sheet_pred").
        volume_3d (torch.Tensor): shape [1, C, D, H, W].
        task_name (str): Name of the task, used to look up activation in tasks_dict.
        tasks_dict (dict): e.g. {"sheet": {"activation": "sigmoid"}}
        global_step (int): The global step or epoch.
        max_slices (int): Maximum number of slices to log.
    """
    # Make sure we have exactly one item in the batch
    b, c, d, h, w = volume_3d.shape
    assert b == 1, f"Expected batch=1, got shape={volume_3d.shape}"

    # Convert to NumPy
    vol_np = volume_3d.cpu().detach().numpy()[0]  # => [C, D, H, W]

    # Optional: apply activation if single-channel, based on tasks_dict
    if tasks_dict and task_name in tasks_dict:
        activation_str = tasks_dict[task_name].get("activation", "none")
        if c == 1:  # if single channel
            vol_np = apply_activation_if_needed(vol_np, activation_str)
    # vol_np = np.clip(vol_np, 0, 1)

    # Decide which slice indices to use
    if d > max_slices:
        slice_indices = np.linspace(0, d - 1, max_slices, dtype=int)
    else:
        slice_indices = range(d)

    # Log each slice to TensorBoard
    for slice_idx in slice_indices:
        # shape => [C, H, W]
        slice_2d = vol_np[:, slice_idx, :, :]

        # Convert back to torch for add_image
        slice_tensor = torch.tensor(slice_2d, dtype=torch.float32)

        # dataformats='CHW' because it's [C, H, W]
        writer.add_image(
            f"{tag_prefix}/slice_{slice_idx}",
            slice_tensor,
            global_step=global_step,
            dataformats='CHW'
        )


def save_debug_gif(
    input_volume: torch.Tensor,          # shape [1, C, Z, H, W]
    targets_dict: dict,                 # e.g. {"sheet": tensor([1, Z, H, W]), "normals": tensor([3, Z, H, W])}
    outputs_dict: dict,                 # same shape structure
    tasks_dict: dict,                   # e.g. {"sheet": {"activation":"sigmoid"}, "normals": {"activation":"none"}}
    epoch: int,
    save_path: str = "debug.gif",
    show_normal_magnitude: bool = True,
    fps: int = 5
):
    """
    Creates a multi-panel GIF for debugging.

    The top row will show: [Input slice] + [GT for each task].
    The bottom row will show: [Prediction for each task].
    If a task has 3 channels (normals), we visualize them as BGR. Otherwise, grayscale.

    Args:
        input_volume (torch.Tensor): shape [1, C, Z, H, W].
        targets_dict (dict): {task_name: Tensor of shape [1, C, Z, H, W]}.
        outputs_dict (dict): {task_name: Tensor of shape [1, C, Z, H, W]}.
        tasks_dict (dict): e.g. {"sheet":{"activation":"sigmoid"}, "normals":{"activation":"none"}}.
        epoch (int): The current epoch number for reference/logging.
        save_path (str): Where to save the resulting GIF.
        show_normal_magnitude (bool): If True, the bottom row for normals includes a magnitude sub-panel.
        fps (int): frames per second for the GIF.
    """
    # Convert input volume to NumPy
    inp_np = input_volume.cpu().numpy()[0]  # => shape [C, Z, H, W]
    if inp_np.shape[0] == 1:
        inp_np = inp_np[0]  # shape => [Z, H, W] if single-channel

    # Convert target & prediction volumes to NumPy (apply activation for single-channel tasks if needed)
    targets_np, preds_np = {}, {}
    for t_name, t_tensor in targets_dict.items():
        arr_np = t_tensor.cpu().numpy()[0]  # => [C, Z, H, W]
        targets_np[t_name] = arr_np

    for t_name, p_tensor in outputs_dict.items():
        arr_np = p_tensor.cpu().numpy()[0]  # => [C, Z, H, W]

        # If single-channel, apply any specified activation
        activation_str = tasks_dict[t_name].get("activation", "none")
        if arr_np.shape[0] == 1:
            arr_np = apply_activation_if_needed(arr_np, activation_str)
        preds_np[t_name] = arr_np

    # Optional: if your normal tasks need re-normalizing to [-1..1], do so here
    # or rely on your pipeline already having them in [-1..1].
    # e.g. you can do a “safe_normalize” if you prefer:
    # if "normals" in preds_np or ...
    # ...

    # Build frames
    frames = []
    z_dim = inp_np.shape[0] if inp_np.ndim == 3 else inp_np.shape[1]  # if input is single-channel, shape is [Z, H, W]
    task_names = sorted(list(targets_dict.keys()))

    for z_idx in range(z_dim):
        # Top row: input + GTs
        top_row_imgs = []

        # -- Input slice
        if inp_np.ndim == 3:
            # shape => [Z, H, W]
            inp_slice = inp_np[z_idx]
        else:
            # multi-channel input (rare), shape => [C, Z, H, W]
            # pick out each channel or just show the 0th?
            inp_slice = inp_np[:, z_idx, :, :]  # shape => [C, H, W]
        top_row_imgs.append(convert_slice_to_bgr(inp_slice))

        # -- Each GT
        for t_name in task_names:
            gt_slice = targets_np[t_name]
            if gt_slice.shape[0] == 1:
                # shape => [1, Z, H, W] => [Z, H, W]
                slice_2d = gt_slice[0, z_idx, :, :]
            else:
                # shape => [3, Z, H, W] for normals, etc.
                slice_2d = gt_slice[:, z_idx, :, :]
            top_row_imgs.append(convert_slice_to_bgr(slice_2d))

        top_row = np.hstack(top_row_imgs)

        # Bottom row: predictions
        bottom_row_imgs = []
        for t_name in task_names:
            pd_slice = preds_np[t_name]
            if pd_slice.shape[0] == 1:
                slice_2d = pd_slice[0, z_idx, :, :]
                bgr_pred = convert_slice_to_bgr(slice_2d)
                bottom_row_imgs.append(bgr_pred)
            else:
                # normals
                slice_3d = pd_slice[:, z_idx, :, :]
                bgr_normals = convert_slice_to_bgr(slice_3d, show_magnitude=show_normal_magnitude)
                bottom_row_imgs.append(bgr_normals)

        if bottom_row_imgs:
            bottom_row = np.hstack(bottom_row_imgs)
        else:
            # If no tasks exist, or something is empty
            bottom_row = np.zeros_like(top_row, dtype=np.uint8)

        final_img = np.vstack([top_row, bottom_row])
        frames.append(final_img)

    # Save frames as GIF
    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Epoch {epoch}] Saving GIF to: {save_path}")
    imageio.mimsave(save_path, frames, fps=fps)
