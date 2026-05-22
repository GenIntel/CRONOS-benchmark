import argparse
import os
import sys

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DINO_HUB, DINO_MIN_MASK_SIZE


def load_model(device):
    model = torch.hub.load(*DINO_HUB).to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def _embed_frames(frames, model, preprocess, device):
    """Embed a list of H×W×3 uint8 arrays with DINOv2."""
    try:
        t = preprocess(
            torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        ).to(device)
        with torch.no_grad():
            return model(t).cpu().numpy()
    except Exception:
        embs = []
        for f in frames:
            t = preprocess(
                torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            )
            with torch.no_grad():
                embs.append(model(t))
        return torch.cat(embs).cpu().numpy()


def _mask_validity(mask):
    """Return (T,) bool: True where mask bounding box ≥ DINO_MIN_MASK_SIZE in both dims."""
    valid = np.zeros(len(mask), dtype=bool)
    for t, m in enumerate(mask):
        ys, xs = np.where(m)
        if (len(ys) and ys.max() - ys.min() >= DINO_MIN_MASK_SIZE
                and xs.max() - xs.min() >= DINO_MIN_MASK_SIZE):
            valid[t] = True
    return valid


def _crop_and_mask(frames, mask):
    """Zero background and crop to bounding box. Returns list of H'×W'×3 arrays."""
    out = []
    for f, m in zip(frames, mask):
        ys, xs = np.where(m)
        if len(ys) == 0:
            out.append(f)
            continue
        f = np.where(m[:, :, None], f, 0)
        out.append(f[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return out


def compute_predictions(model, preprocess, video_frames, masks_list, device):
    """
    Args:
        video_frames: (T, H, W, 3) uint8 ndarray
        masks_list: list of (T, H, W) bool ndarrays — one per object
    Returns:
        dict with keys emb_obj{i} (T, D) and valid_obj{i} (T,) per object
    """
    T = min(video_frames.shape[0], *(m.shape[0] for m in masks_list))
    frames = video_frames[:T]
    results = {}
    for i, mask in enumerate(masks_list):
        masked = _crop_and_mask(frames, mask[:T])
        results[f"emb_obj{i}"]   = _embed_frames(masked, model, preprocess, device)
        results[f"valid_obj{i}"] = _mask_validity(mask[:T])
    return results


def save_predictions(predictions, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **predictions)


def _load_masks(path):
    """Load segmentation_masks.npz → list of (T, H, W) bool arrays, one per object."""
    masks = np.load(path)["masks"]   # (N_obj, T, H, W)
    return [masks[i] > 0.5 for i in range(masks.shape[0])]


def main():
    parser = argparse.ArgumentParser(description="Compute DINOv2 embeddings per object.")
    parser.add_argument("--video",  required=True, help="Input .mp4 file")
    parser.add_argument("--mask",   required=True,
                        help="Segmentation masks (.npz with key 'masks', shape N_obj×T×H×W)")
    parser.add_argument("--output", required=True, help="Output .npz path (e.g. dino_embeddings.npz)")
    args = parser.parse_args()

    import imageio.v3 as iio
    frames = iio.imread(args.video)
    if frames.ndim == 4 and frames.shape[-1] == 4:
        frames = frames[..., :3]

    masks_list = _load_masks(args.mask)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model(device)
    predictions = compute_predictions(model, preprocess, frames, masks_list, device)
    save_predictions(predictions, args.output)
    print(f"Saved embeddings ({len(masks_list)} object(s)) → {args.output}")


if __name__ == "__main__":
    main()
