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


def get_video_fps(path):
    """Return the frame rate stored in the video container."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Could not read fps from {path}")
    return float(fps)


def resample_frames(video_frames, src_fps, target_fps):
    """Nearest-neighbour temporal resampling of (T, H, W, 3) frames to target_fps."""
    if abs(src_fps - target_fps) < 1e-3:
        return video_frames
    duration = len(video_frames) / src_fps
    t = np.arange(int(np.floor(duration * target_fps + 1e-6))) / target_fps
    idx = np.clip(np.round(t * src_fps).astype(int), 0, len(video_frames) - 1)
    return video_frames[idx]


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


def save_predictions(embeddings, out_path, fps=None):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    extra = {"fps": np.float32(fps)} if fps is not None else {}
    np.savez_compressed(out_path, embeddings=embeddings.numpy(), **extra)


def main():
    parser = argparse.ArgumentParser(description="Extract DisMo motion embeddings from a video.")
    parser.add_argument("--video",       required=True, help="Input .mp4 file")
    parser.add_argument("--output",      required=True, help="Output .npz path")
    parser.add_argument("--time_window", type=int, default=DISMO_TIME_WINDOW,
                        help=f"Lookahead frames (default: {DISMO_TIME_WINDOW})")
    parser.add_argument("--fps",         type=float, default=None,
                        help="Resample the video to this fps before extraction")
    parser.add_argument("--align_to",    nargs="+", default=None, metavar="VIDEO",
                        help="Resample to the lowest fps among --video and these videos "
                             "(pass the same set for every video that will be compared)")
    args = parser.parse_args()

    import imageio.v3 as iio
    frames = iio.imread(args.video)
    if frames.ndim == 4 and frames.shape[-1] == 4:
        frames = frames[..., :3]

    src_fps = get_video_fps(args.video)
    target_fps = args.fps
    if target_fps is None and args.align_to:
        target_fps = min([src_fps] + [get_video_fps(p) for p in args.align_to])
    if target_fps is not None:
        frames = resample_frames(frames, src_fps, target_fps)
        print(f"Resampled {src_fps:.2f} fps → {target_fps:.2f} fps ({len(frames)} frames)")
    else:
        target_fps = src_fps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    embeddings = compute_predictions(model, frames, device, args.time_window)
    save_predictions(embeddings, args.output, fps=target_fps)
    print(f"Saved embeddings shape={tuple(embeddings.shape)} → {args.output}")


if __name__ == "__main__":
    main()
