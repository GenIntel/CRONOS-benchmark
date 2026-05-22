import argparse
import os
import sys


import imageio.v3 as iio

import numpy as np
import torch
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COTRACKER_HUB, COTRACKER_GRID_SIZE


def load_model(device):
    model = torch.hub.load(*COTRACKER_HUB).to(device)
    return model.eval()


def _split_tracks_by_object(tracks, visibility, mask):
    """Split (T, N, 2) tracks into per-object lists using first-frame mask positions."""
    n_objects = mask.any(axis=(0, 1)).sum()
    ix = np.clip(np.round(tracks[0, :, 0]).astype(int), 0, mask.shape[1] - 1)
    iy = np.clip(np.round(tracks[0, :, 1]).astype(int), 0, mask.shape[0] - 1)
    per_obj_tracks, per_obj_vis = [], []
    for i in range(n_objects):
        sel = mask[:, :, i][iy, ix]
        per_obj_tracks.append(tracks[:, sel])
        per_obj_vis.append(visibility[:, sel])
    return per_obj_tracks, per_obj_vis


def compute_predictions(model, video_frames, mask, device):
    """
    Args:
        video_frames: (T, H, W, 3) uint8 ndarray
        mask: (H, W, N_obj) bool ndarray — first-frame mask
    Returns:
        tracks (T, N, 2), visibility (T, N),
        per_obj_tracks list[(T, Ni, 2)], per_obj_vis list[(T, Ni)]
    """
    H, W = video_frames.shape[1:3]
    if mask.shape[:2] != (H, W):
        mask = resize(mask.astype(float), (H, W), order=0,
                      preserve_range=True, anti_aliasing=False) > 0.5

    combined = torch.from_numpy(mask.any(axis=-1)).float().to(device)[None, None]
    video = torch.from_numpy(video_frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    with torch.no_grad():
        raw_tracks, raw_vis = model(video, grid_size=COTRACKER_GRID_SIZE, segm_mask=combined)

    tracks = raw_tracks.squeeze().cpu().numpy()
    visibility = raw_vis.squeeze().cpu().numpy()
    per_obj_tracks, per_obj_vis = _split_tracks_by_object(tracks, visibility, mask)
    return tracks, visibility, per_obj_tracks, per_obj_vis


def save_predictions(out_dir, tracks, visibility, per_obj_tracks, per_obj_vis):
    os.makedirs(out_dir, exist_ok=True)
    save_dict = {"tracks": tracks, "visibility": visibility}
    for i, (t, v) in enumerate(zip(per_obj_tracks, per_obj_vis)):
        save_dict[f"tracks_obj{i}"] = t
        save_dict[f"visibility_obj{i}"] = v
    np.savez_compressed(os.path.join(out_dir, "tracks.npz"), **save_dict)


def _load_mask(path):
    """Load mask file and return (H, W, N_obj) bool array."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".jpg", ".jpeg", ".png"):
        img = iio.imread(path)
        if img.ndim == 2:
            return img[:, :, None] > 0
        # Multi-channel: each channel is one object
        return img > 0
    data = np.load(path)
    mask = data[data.files[0]] if hasattr(data, "files") else data
    if mask.ndim == 2:
        mask = mask[:, :, None]
    elif mask.ndim == 4 and mask.shape[-1] <= 10:   # (T, H, W, N_obj) → take frame 0
        mask = mask[0]
    elif mask.ndim == 4:                              # (N_obj, T, H, W) → take t=0, transpose
        mask = mask[:, 0].transpose(1, 2, 0)
    return mask > 0.5


def main():
    parser = argparse.ArgumentParser(description="Run CoTracker on a video.")
    parser.add_argument("--video",  required=True, help="Input .mp4 file")
    parser.add_argument("--mask",   required=True, help="First-frame mask (.npy/.npz, H×W×N_obj)")
    parser.add_argument("--output", required=True, help="Output directory for tracks.npz")
    args = parser.parse_args()

    frames = iio.imread(args.video)
    if frames.ndim == 4 and frames.shape[-1] == 4:
        frames = frames[..., :3]

    mask = _load_mask(args.mask)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    results = compute_predictions(model, frames, mask, device)
    save_predictions(args.output, *results)
    print(f"Saved tracks → {args.output}/tracks.npz")


if __name__ == "__main__":
    main()
