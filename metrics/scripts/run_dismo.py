import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISMO_HUB, DISMO_TIME_WINDOW


def load_model(device):
    model = torch.hub.load(*DISMO_HUB).to(device)
    return model


def compute_predictions(model, video_frames, device, time_window=DISMO_TIME_WINDOW):
    """
    Args:
        video_frames: (T, H, W, 3) uint8 ndarray
        time_window:  lookahead frames (output length = T - time_window)
    Returns:
        embeddings: (T - time_window, 128) float tensor
    """
    video = torch.from_numpy(np.asarray(video_frames)).float().div(127.5).sub(1.0)
    video = video.unsqueeze(0).to(device)   # (1, T, H, W, C)
    with torch.no_grad():
        embeddings = model.forward_sliding(video, lookahead=time_window)
    return embeddings.squeeze(0).cpu()       # (T - time_window, 128)


def save_predictions(embeddings, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, embeddings=embeddings.numpy())


def main():
    parser = argparse.ArgumentParser(description="Extract DisMo motion embeddings from a video.")
    parser.add_argument("--video",       required=True, help="Input .mp4 file")
    parser.add_argument("--output",      required=True, help="Output .npz path")
    parser.add_argument("--time_window", type=int, default=DISMO_TIME_WINDOW,
                        help=f"Lookahead frames (default: {DISMO_TIME_WINDOW})")
    args = parser.parse_args()

    import imageio.v3 as iio
    frames = iio.imread(args.video)
    if frames.ndim == 4 and frames.shape[-1] == 4:
        frames = frames[..., :3]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    embeddings = compute_predictions(model, frames, device, args.time_window)
    save_predictions(embeddings, args.output)
    print(f"Saved embeddings shape={tuple(embeddings.shape)} → {args.output}")


if __name__ == "__main__":
    main()
