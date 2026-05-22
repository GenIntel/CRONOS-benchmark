import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FAILURE_THRESHOLDS, MIN_OCCLUSION_FRACTION

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    cosine_similarity, robust_min, robust_max,
    load_video_frames, load_masks_npz, load_dino_embeddings,
    load_dismo_embeddings, load_vlm_predictions, load_tracks_npz, scale_metric,
)


def object_consistency(dino_path, masks_path=None):
    """Robust-min temporal DINO consistency across all objects.

    Per object: cosine similarity of each valid frame's embedding to the first
    valid frame.  Frames where the visible object area drops below
    MIN_OCCLUSION_FRACTION of its frame-0 area are excluded (matching
    causal-bench's occlusion masking).  Returns robust min of the per-object
    means.
    """
    embeddings, validities = load_dino_embeddings(dino_path)
    if embeddings is None:
        return float("nan")

    occ_fracs = None
    if masks_path is not None and os.path.isfile(masks_path):
        masks  = load_masks_npz(masks_path)                                   # (N_obj, T, H, W)
        counts = masks.reshape(masks.shape[0], masks.shape[1], -1).sum(axis=2).astype(np.float32)
        first  = counts[:, 0:1]
        with np.errstate(invalid="ignore", divide="ignore"):
            occ_fracs = np.where(first > 0, counts / first, 0.0).clip(0.0, 1.0)  # (N_obj, T)

    consistencies = []
    for i, (emb, valid) in enumerate(zip(embeddings, validities)):
        if occ_fracs is not None and i < occ_fracs.shape[0]:
            occ_valid = occ_fracs[i, :emb.shape[0]] >= MIN_OCCLUSION_FRACTION
            valid = (valid & occ_valid) if valid is not None else occ_valid
        idx = np.where(valid)[0] if valid is not None else np.arange(len(emb))
        if len(idx) == 0:
            continue
        ref  = emb[idx[0]:idx[0] + 1]
        sims = cosine_similarity(emb[idx], np.repeat(ref, len(idx), axis=0))
        consistencies.append(float(np.mean(sims)))
    return robust_min(consistencies) if consistencies else float("nan")


def dismo_similarity(dismo_path, gt_dismo_path):
    """Robust-min per-frame cosine similarity between generated and GT DisMo embeddings."""
    gen_emb = load_dismo_embeddings(dismo_path)
    gt_emb  = load_dismo_embeddings(gt_dismo_path)
    n = min(len(gen_emb), len(gt_emb))
    if n == 0:
        return float("nan")
    return robust_min(cosine_similarity(gen_emb[:n], gt_emb[:n]))


def chamfer_distance(output_dir, masks_path=None):
    """Robust-max per-frame Chamfer distance (each frame vs frame 0), over all objects.

    For each obj{i}/sam3D/mesh/ directory, aligns every mesh to the first mesh
    using the best axis-aligned 90° rotation + uniform scale (RANSAC-style),
    then computes symmetric Chamfer distance.  Frames where any object's visible
    area drops below MIN_OCCLUSION_FRACTION of its frame-0 area are excluded
    (matching causal-bench's occlusion masking).  Returns robust max over all
    per-frame distances across all objects (worst-case shape change).

    PLY filenames encode the source video frame index as a zero-padded integer
    (e.g. ``000000010.ply`` = frame 10), which is used to look up the occlusion
    fraction in the full-resolution mask array.
    """
    try:
        import trimesh
    except ImportError:
        print("trimesh not available; skipping Chamfer distance.")
        return float("nan")

    _ROTS = [
        np.linalg.matrix_power(base, k).astype(np.float32)
        for base in (
            np.array([[0., -1., 0.], [1.,  0., 0.], [0., 0.,  1.]], dtype=np.float32),
            np.array([[1.,  0., 0.], [0.,  0., -1.], [0., 1., 0.]], dtype=np.float32),
            np.array([[0.,  0., 1.], [0.,  1., 0.], [-1., 0., 0.]], dtype=np.float32),
        )
        for k in range(4)
    ]

    def _load_verts(path):
        try:
            return np.asarray(trimesh.load(str(path), process=False, force="mesh").vertices,
                              dtype=np.float32)
        except Exception:
            return None

    def _chamfer(a, b):
        d_ab, _ = cKDTree(b).query(a, workers=-1)
        d_ba, _ = cKDTree(a).query(b, workers=-1)
        return float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2))

    def _align_and_chamfer(ref, verts):
        """Best-fit axis-aligned rotation + scale, then Chamfer distance."""
        rng = np.random.default_rng(0)
        c_ref  = ref   - ref.mean(axis=0)
        c_vert = verts - verts.mean(axis=0)
        n_sc   = min(500, len(c_ref), len(c_vert))
        a_sc   = c_ref [rng.choice(len(c_ref),  n_sc, replace=False)]
        b_sc   = c_vert[rng.choice(len(c_vert), n_sc, replace=False)]
        rms_a  = float(np.sqrt(np.mean(np.sum(c_ref ** 2,  axis=1))))
        rms_b  = float(np.sqrt(np.mean(np.sum(c_vert ** 2, axis=1))))
        scale  = float(np.clip(rms_a / rms_b, 0.1, 10.0)) if rms_b > 1e-10 else 1.0
        best_cd, best_R = np.inf, np.eye(3, dtype=np.float32)
        for R in _ROTS:
            bt     = scale * (b_sc @ R.T)
            d1, _  = cKDTree(a_sc).query(bt, workers=-1)
            d2, _  = cKDTree(bt).query(a_sc, workers=-1)
            cd     = float(np.mean(d1 ** 2) + np.mean(d2 ** 2))
            if cd < best_cd:
                best_cd, best_R = cd, R
        aligned = (scale * c_vert @ best_R.T + ref.mean(axis=0)).astype(np.float32)
        return _chamfer(ref, aligned)

    # Occlusion fractions: (N_obj, T), a frame is valid when ALL objects are visible enough
    occ_fracs = None
    if masks_path is not None and os.path.isfile(masks_path):
        masks  = load_masks_npz(masks_path)
        counts = masks.reshape(masks.shape[0], masks.shape[1], -1).sum(axis=2).astype(np.float32)
        first  = counts[:, 0:1]
        with np.errstate(invalid="ignore", divide="ignore"):
            occ_fracs = np.where(first > 0, counts / first, 0.0).clip(0.0, 1.0)  # (N_obj, T)

    def _frame_visible(stem):
        """Return True if all objects are sufficiently visible at this video frame."""
        if occ_fracs is None:
            return True
        try:
            frame_idx = int(stem)
        except ValueError:
            return True
        if frame_idx >= occ_fracs.shape[1]:
            return True
        return bool(np.all(occ_fracs[:, frame_idx] >= MIN_OCCLUSION_FRACTION))

    all_distances = []
    obj_idx = 0
    while True:
        mesh_dir = Path(output_dir) / "sam3D" / f"obj{obj_idx}" / "sam3D" / "mesh"
        if not mesh_dir.is_dir():
            break
        ply_files  = sorted(f for f in os.listdir(mesh_dir) if f.endswith(".ply"))
        ply_files  = [f for f in ply_files if _frame_visible(Path(f).stem)]
        verts_list = [_load_verts(mesh_dir / f) for f in ply_files]
        ref_verts  = next((v for v in verts_list if v is not None), None)
        if ref_verts is not None:
            for verts in verts_list:
                if verts is not None and verts is not ref_verts:
                    all_distances.append(_align_and_chamfer(ref_verts, verts))
        obj_idx += 1

    return robust_max(all_distances) if all_distances else float("nan")


def background_mse(video_path, masks_path):
    """Robust-max per-frame background MSE vs frame 0, normalized to [0, 1].

    Background region: pixels belonging to no object in frame 0.
    """
    frames = load_video_frames(video_path)
    masks  = load_masks_npz(masks_path)       # (N_obj, T, H, W) bool
    T, H, W = frames.shape[:3]

    if masks.shape[2] != H or masks.shape[3] != W:
        from skimage.transform import resize as sk_resize
        resized = np.zeros((masks.shape[0], T, H, W), dtype=bool)
        for i in range(masks.shape[0]):
            for t in range(T):
                resized[i, t] = sk_resize(masks[i, t].astype(float), (H, W), order=0) > 0.5
        masks = resized

    obj_union = masks.any(axis=0)         # (T, H, W)
    bg0       = ~obj_union[0]             # background pixels in frame 0
    ref       = frames[0].astype(np.float32) / 255.0

    mse_per_frame = []
    for t in range(1, T):
        region = bg0 & ~obj_union[t]
        if region.sum() == 0:
            continue
        ft  = frames[t].astype(np.float32) / 255.0
        mse_per_frame.append(float(np.mean((ft - ref)[region] ** 2)))

    return robust_max(mse_per_frame) if mse_per_frame else float("nan")


def vlm_task_performance(vlm_path):
    """Return the VLM positive fraction from vlm_predictions.json."""
    data = load_vlm_predictions(vlm_path)
    val  = data.get("positive_fraction")
    return float(val) if val is not None and np.isfinite(val) else float("nan")


def disappearance_fraction(masks_path, tracks_path, edge_margin_frac=0.05):
    """Fraction of tracked objects that permanently vanish inside the frame.

    Occlusion is detected from segmentation masks: for each object, finds the
    earliest frame where it becomes permanently invisible (zero mask pixels for
    all remaining frames).  The centroid at that frame is taken from CoTracker
    per-object tracks (tracks_obj{i} in tracks.npz).  If the centroid is not
    near any border, the object is counted as a mid-frame disappearance.

    Returns 0.0 when no objects permanently vanish inside the frame, or nan if
    required files are missing.

    edge_margin_frac : fraction of each frame dimension used as the
        edge-proximity threshold (default 0.05, matching causal-bench).
    """
    masks = load_masks_npz(masks_path)  # (N_obj, T, H, W) bool
    N, T, H, W = masks.shape
    margin_x = edge_margin_frac * W
    margin_y = edge_margin_frac * H

    tracks_list, vis_list = load_tracks_npz(tracks_path)
    if tracks_list is None:
        return float("nan")

    n_evaluated  = 0
    n_disappeared = 0

    for obj_idx in range(min(N, len(tracks_list))):
        occ = masks[obj_idx].any(axis=(1, 2))  # (T,) — True when object has any visible pixel

        # Skip objects already invisible at frame 0 or visible throughout
        if not occ[0] or occ.all():
            continue

        # Earliest frame where the object is permanently invisible
        t_gone = None
        for t in range(T):
            if not occ[t] and not occ[t:].any():
                t_gone = t
                break
        if t_gone is None:
            continue

        n_evaluated += 1

        tr = tracks_list[obj_idx]  # (T_tr, Ni, 2) — columns: x, y
        vi = vis_list[obj_idx]     # (T_tr, Ni)
        T_tr = tr.shape[0]

        # Find the last frame with any visible track point up to t_gone
        last_vis_t = None
        for t in range(min(t_gone, T_tr - 1), -1, -1):
            if vi[t].any():
                last_vis_t = t
                break

        if last_vis_t is None:
            # Tracker never saw this object — treat as mid-frame disappearance
            n_disappeared += 1
            continue

        # Centroid from track positions at t_gone (matches causal-bench)
        pts = tr[min(t_gone, T_tr - 1)]  # (Ni, 2)
        cx  = float(np.mean(pts[:, 0]))
        cy  = float(np.mean(pts[:, 1]))

        at_edge = (
            cx < margin_x
            or cx > W - margin_x
            or cy < margin_y
            or cy > H - margin_y
        )
        if not at_edge:
            n_disappeared += 1

    return float(n_disappeared / n_evaluated) if n_evaluated > 0 else 0.0


def compute_success(metrics, thresholds=None):
    """Return True if every threshold criterion is satisfied; NaN values fail."""
    if thresholds is None:
        thresholds = FAILURE_THRESHOLDS
    for name, (direction, bound) in thresholds.items():
        val = metrics.get(name)
        if val is None or not np.isfinite(val):
            return False
        if direction == "min" and val < bound:
            return False
        if direction == "max" and val > bound:
            return False
    disp = metrics.get("disappearance_fraction")
    if disp is None or not np.isfinite(disp) or disp > 0.1:
        return False
    return True


def compute_all(video_path, ref_dir, output_dir):
    """Compute all metrics for one evaluated video.

    Returns a dict with raw metric values, scaled values, and a success flag.
    """
    dino_path   = os.path.join(output_dir, "dino_embeddings.npz")
    dismo_path  = os.path.join(output_dir, "dismo_embeddings.npz")
    gt_dismo    = os.path.join(output_dir, "gt_dismo_embeddings.npz")
    masks_path  = os.path.join(output_dir, "segmentation_masks.npz")
    vlm_path    = os.path.join(output_dir, "vlm_predictions.json")
    tracks_path = os.path.join(output_dir, "tracks.npz")

    raw = {}
    raw["object_consistency"]    = (object_consistency(dino_path, masks_path)
                                    if os.path.isfile(dino_path) else float("nan"))
    raw["motion_similarity"]     = (dismo_similarity(dismo_path, gt_dismo)
                                    if os.path.isfile(dismo_path) and os.path.isfile(gt_dismo)
                                    else float("nan"))
    raw["mean_chamfer_distance"] = chamfer_distance(output_dir, masks_path)
    raw["bg_mse"]                = (background_mse(video_path, masks_path)
                                    if os.path.isfile(masks_path) else float("nan"))
    raw["vlm_positive_fraction"] = (vlm_task_performance(vlm_path)
                                    if os.path.isfile(vlm_path) else float("nan"))
    raw["disappearance_fraction"] = (disappearance_fraction(masks_path, tracks_path)
                                     if os.path.isfile(masks_path) and os.path.isfile(tracks_path)
                                     else float("nan"))
    raw["success"]               = compute_success(raw)

    _no_scale = {"success", "disappearance_fraction"}
    scaled = {f"{k}_scaled": scale_metric(k, v)
              for k, v in raw.items() if k not in _no_scale}
    return {**raw, **scaled}


def save_metrics(metrics, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics from predictions.")
    parser.add_argument("--video",   required=True, help="Generated .mp4 file")
    parser.add_argument("--ref",     required=True, help="Reference dataset directory")
    parser.add_argument("--output",  required=True, help="Predictions output directory")
    parser.add_argument("--metrics", default=None,  help="Output .json path (default: output/metrics.json)")
    args = parser.parse_args()

    out_path = args.metrics or os.path.join(args.output, "metrics.json")
    metrics  = compute_all(args.video, args.ref, args.output)
    save_metrics(metrics, out_path)

    for name in ("object_consistency", "motion_similarity",
                 "mean_chamfer_distance", "bg_mse", "vlm_positive_fraction",
                 "disappearance_fraction"):
        val = metrics[name]
        print(f"  {name:<25} {val:.4f}" if np.isfinite(val) else f"  {name:<25} nan")
    print(f"  {'success':<25} {metrics['success']}")
    print(f"Metrics saved → {out_path}")


if __name__ == "__main__":
    main()
