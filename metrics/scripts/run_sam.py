import argparse
import os
import sys


import imageio as iio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAM3_MODEL_PATH


def load_model(device):
    sys.path.insert(0, SAM3_MODEL_PATH)
    from sam3.model_builder import build_sam3_video_model
    sam3 = build_sam3_video_model(load_from_HF=True, device=device)
    predictor = sam3.tracker
    predictor.backbone = sam3.detector.backbone
    return predictor


def compute_predictions(model, video_path, masks, device):
    """
    Args:
        masks: list of (H, W) bool ndarrays — first-frame prompts, one per object
    Returns:
        all_masks: (N_obj, T, H, W) bool ndarray
    """
    mask_tensors = [torch.from_numpy(m.astype(bool)).to(device) for m in masks]

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = model.init_state(video_path=video_path)
        for obj_id, mask in enumerate(mask_tensors, start=1):
            model.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=mask)

        masks_per_obj = {i: [] for i in range(1, len(mask_tensors) + 1)}
        for _, obj_ids, _, video_res_masks, _ in model.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=state["num_frames"],
            reverse=False, propagate_preflight=True,
        ):
            for i, obj_id in enumerate(obj_ids):
                masks_per_obj[obj_id].append((video_res_masks[i] > 0.0).squeeze().cpu().numpy())

    return np.stack(
        [np.stack(masks_per_obj[k]) for k in sorted(masks_per_obj)], axis=0
    )  # (N_obj, T, H, W)


def save_predictions(all_masks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "segmentation_masks.npz"), masks=all_masks)


def _first_frame_masks(path):
    """Load mask file and return per-object (H, W) bool arrays from frame 0."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".jpg", ".jpeg", ".png"):
        img = iio.imread(path)
        if img.ndim == 2:
            return [(img > 127)]
        # Multi-channel image: treat each channel as a separate object
        return [img[..., i] > 127 for i in range(img.shape[-1]) if (img[..., i]>127).any()]
    data = np.load(path)
    mask = data[data.files[0]] if hasattr(data, "files") else data
    if mask.ndim == 2:
        mask = mask[:, :, None]
    elif mask.ndim == 4 and mask.shape[-1] <= 10:   # (T, H, W, N_obj) → frame 0
        mask = mask[0]
    elif mask.ndim == 4:                              # (N_obj, T, H, W) → t=0, transpose
        mask = mask[:, 0].transpose(1, 2, 0)
    return [mask[..., i] > 0.5 for i in range(mask.shape[-1]) if mask[..., i].any()]


def main():
    parser = argparse.ArgumentParser(description="Run SAM3 video segmentation.")
    parser.add_argument("--video",  required=True, help="Input .mp4 file")
    parser.add_argument("--mask",   required=True, help="First-frame mask (.npy/.npz, H×W×N_obj or .jpg)")
    parser.add_argument("--output", required=True, help="Output directory for segmentation_masks.npz")
    args = parser.parse_args()

    masks = _first_frame_masks(args.mask)
    print(f"Detected {len(masks)} object(s)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    all_masks = compute_predictions(model, args.video, masks, device)
    save_predictions(all_masks, args.output)
    print(f"Saved masks → {args.output}/segmentation_masks.npz  shape={all_masks.shape}")


if __name__ == "__main__":
    main()
