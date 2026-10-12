import json
import os
import sys

import numpy as np
import imageio.v3 as iio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import METRIC_RANGES, METRIC_DECAY


# ── Math helpers ──────────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    """Per-row cosine similarity between (T, D) arrays. Returns (T,) float."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(a_n * b_n, axis=1)


def robust_min(values, fraction=0.05):
    """Mean of the lowest `fraction` of finite values. Returns NaN if empty."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(arr.size * fraction)))
    return float(np.mean(np.partition(arr, k - 1)[:k]))


def robust_max(values, fraction=0.05):
    """Mean of the highest `fraction` of finite values. Returns NaN if empty."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(arr.size * fraction)))
    return float(np.mean(np.partition(arr, -k)[-k:]))


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_video_frames(path):
    """Load video as (T, H, W, 3) uint8 array."""
    frames = iio.imread(path)
    if frames.ndim == 4 and frames.shape[-1] == 4:
        frames = frames[..., :3]
    return frames


def load_masks_npz(path):
    """Load segmentation_masks.npz. Returns (N_obj, T, H, W) bool array."""
    return np.load(path)["masks"].astype(bool)


def load_dino_embeddings(path):
    """Load dino_embeddings.npz. Returns (embeddings_list, validities_list)."""
    if not os.path.isfile(path):
        return None, None
    npz = np.load(path)
    embeddings, validities = [], []
    i = 0
    while f"emb_obj{i}" in npz:
        embeddings.append(npz[f"emb_obj{i}"])
        vkey = f"valid_obj{i}"
        validities.append(npz[vkey] if vkey in npz else None)
        i += 1
    return (embeddings, validities) if embeddings else (None, None)


def load_dismo_embeddings(path):
    """Load dismo_embeddings.npz. Returns ((T, D) float array, fps or None)."""
    npz = np.load(path)
    fps = float(npz["fps"]) if "fps" in npz else None
    return npz["embeddings"].astype(float), fps


def load_vlm_predictions(path):
    """Load vlm_predictions.json. Returns dict."""
    with open(path) as f:
        return json.load(f)


def load_tracks_npz(path):
    """Load tracks.npz written by run_cotracker.py.

    Returns (tracks_list, vis_list) where each element is a per-object array:
        tracks_list[i] : (T, Ni, 2)  float  — x, y coordinates
        vis_list[i]    : (T, Ni)     bool   — CoTracker visibility flag
    Returns (None, None) if the file is absent or contains no per-object keys.
    """
    if not os.path.isfile(path):
        return None, None
    npz = np.load(path)
    tracks_list, vis_list = [], []
    i = 0
    while f"tracks_obj{i}" in npz:
        tracks_list.append(npz[f"tracks_obj{i}"])
        vis_list.append(npz[f"visibility_obj{i}"])
        i += 1
    return (tracks_list, vis_list) if tracks_list else (None, None)


# ── Normalization ─────────────────────────────────────────────────────────────

def scale_metric(name, value):
    """Normalize a raw metric to [0, 1].

    Linear range from METRIC_RANGES; exponential decay exp(-λ·v) from
    METRIC_DECAY for metrics where lower raw value is better.
    Returns NaN for unknown or non-finite inputs.
    """
    if not np.isfinite(value):
        return float("nan")
    if name not in METRIC_RANGES:
        return float("nan")
    rng = METRIC_RANGES[name]
    if rng is not None:
        lo, hi = rng
        return float(np.clip((value - lo) / (hi - lo + 1e-8), 0.0, 1.0))
    lam = METRIC_DECAY.get(name, 1.0)
    return float(np.exp(-lam * value))
