import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import trimesh
from huggingface_hub import snapshot_download
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAM3D_OBJECTS_DIR, SAM3D_HF_REPO, SAM3D_HF_CACHE, SAM3D_SEED, SAM3D_STRIDE

# ── SAM3D repo setup ──────────────────────────────────────────────────────────
# inference.py sets CONDA_PREFIX = CONDA_PREFIX; stub it when absent so the
# assignment is a no-op on module-based clusters that only export CUDA_HOME.
if "CONDA_PREFIX" not in os.environ and "CUDA_HOME" in os.environ:
    os.environ["CONDA_PREFIX"] = os.environ["CUDA_HOME"]
# spconv asserts BOOST_ROOT points to a valid tree; clear it so it falls back
# to its bundled headers when the cluster's BOOST_ROOT is incompatible.
os.environ.pop("BOOST_ROOT", None)

_sam3d_dir = Path(SAM3D_OBJECTS_DIR)
sys.path.insert(0, str(_sam3d_dir / "notebook"))
sys.path.insert(0, str(_sam3d_dir))
os.chdir(_sam3d_dir)

from inference import Inference, load_image, load_single_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(compile_model=False):
    local_dir = snapshot_download(repo_id=SAM3D_HF_REPO, local_dir=SAM3D_HF_CACHE)
    config_path = os.path.join(local_dir, "checkpoints", "pipeline.yaml")
    return Inference(config_path, compile=compile_model)


def compute_predictions(model, input_dir, output_dir=None,
                        frames=None, stride=1, seed=SAM3D_SEED, skip_existing=True):
    """Process frames in input_dir/rgb/ with the pre-loaded model.

    Args:
        input_dir:  directory containing rgb/ and mask/ subdirs
        output_dir: where to write sam3D/ outputs (default: input_dir)
        frames:     explicit list of filenames; None = all images at given stride
        stride:     process every N-th frame when frames is None
        seed:       random seed for inference
        skip_existing: skip frames whose splat .ply already exists

    Returns:
        list of dicts with keys 'status' ('success'|'skipped'|'failed') and 'sample'
    """
    data_path = Path(input_dir)
    out_dir   = Path(output_dir) if output_dir else data_path
    image_dir = data_path / "rgb"

    if frames is None:
        frames = sorted(
            f.name for f in image_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if stride > 1:
            frames = frames[::stride]

    results = []
    for frame_name in tqdm(frames, desc="SAM3D"):
        sample    = Path(frame_name).stem
        splat_dir = out_dir / "sam3D" / "splat"
        meta_dir  = out_dir / "sam3D" / "meta"
        mesh_dir  = out_dir / "sam3D" / "mesh"

        if skip_existing and (splat_dir / f"{sample}.ply").exists():
            results.append({"status": "skipped", "sample": sample})
            continue

        try:
            image    = load_image(str(image_dir / frame_name))
            mask     = load_single_mask(str(data_path / "mask") + "/",
                                        index=sample, extension=Path(frame_name).suffix)
            mask_arr = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
            if not np.any(mask_arr > 0):
                results.append({"status": "skipped", "sample": sample})
                continue
            output = model(image, mask, seed=seed)
        except Exception as e:
            logger.error(f"Failed {sample}: {e}")
            results.append({"status": "failed", "sample": sample, "error": str(e)})
            continue

        meta_dir.mkdir(parents=True, exist_ok=True)
        splat_dir.mkdir(parents=True, exist_ok=True)
        mesh_dir.mkdir(parents=True, exist_ok=True)

        def _to_list(v):
            return v.cpu().numpy().tolist() if hasattr(v, "cpu") else v

        with open(meta_dir / f"{sample}.yaml", "w") as f:
            yaml.dump({
                "rotation_cam":    _to_list(output["rotation"]),
                "translation_cam": _to_list(output["translation"]),
                "scale_cam":       _to_list(output["scale"]),
            }, f, default_flow_style=False)

        gs = output["gs"]
        if isinstance(gs, list):
            gs = gs[0]
        gs.save_ply(str(splat_dir / f"{sample}.ply"))

        mesh = output.get("mesh")
        if isinstance(mesh, list):
            mesh = mesh[0] if mesh else None
        if mesh is not None:
            try:
                verts = mesh.vertices.float().cpu().numpy()
                faces = mesh.faces.cpu().numpy()
                m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                if hasattr(mesh, "vertex_attrs") and mesh.vertex_attrs is not None:
                    vc = mesh.vertex_attrs[:, :3].cpu().numpy()
                    m.visual.vertex_colors = (vc * 255).astype(np.uint8) if vc.max() <= 1.0 else vc
                m.export(str(mesh_dir / f"{sample}.ply"))
            except Exception as e:
                logger.warning(f"Mesh save failed for {sample}: {e}")

        results.append({"status": "success", "sample": sample})

    ok = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Done — success: {ok}, skipped: {sum(1 for r in results if r['status']=='skipped')}, "
                f"failed: {sum(1 for r in results if r['status']=='failed')}")
    return results


def _extract_frames(video_path, rgb_dir, stride=1):
    """Extract every stride-th frame from video_path into rgb_dir as PNGs.

    Returns:
        frame_names: ordered list of filenames written (e.g. ["000000000.png", ...])
        stride_indices: corresponding source frame indices
    """
    rgb_dir = Path(rgb_dir)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    video = iio.imread(str(video_path), plugin="pyav")   # (T, H, W, C)
    frame_names, stride_indices = [], []
    for i in range(0, video.shape[0], stride):
        fname = f"{i:09d}.png"
        iio.imwrite(str(rgb_dir / fname), video[i])
        frame_names.append(fname)
        stride_indices.append(i)
    logger.info(f"Extracted {len(frame_names)} frames → {rgb_dir}")
    return frame_names, stride_indices


def _prepare_object_dir(obj_dir, shared_rgb_dir, object_masks, frame_names):
    """Set up obj_dir with a rgb/ symlink and per-frame masks in mask/.

    Args:
        obj_dir:        directory for this object (created if needed)
        shared_rgb_dir: path to the shared rgb/ directory (symlinked in)
        object_masks:   (len(frame_names), H, W) bool/uint8 array
        frame_names:    filenames matching object_masks rows
    """
    obj_dir = Path(obj_dir)
    obj_dir.mkdir(parents=True, exist_ok=True)

    rgb_link = obj_dir / "rgb"
    if not rgb_link.exists():
        rgb_link.symlink_to(Path(shared_rgb_dir).resolve())

    mask_dir = obj_dir / "mask"
    mask_dir.mkdir(exist_ok=True)
    for fname, m in zip(frame_names, object_masks):
        iio.imwrite(str(mask_dir / fname), m.astype(np.uint8) * 255)


def save_predictions(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sam3d_predictions.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run SAM3D on a video + segmentation masks.")
    parser.add_argument("--video",   required=True, help="Input .mp4 file")
    parser.add_argument("--mask",    required=True,
                        help="segmentation_masks.npz from run_sam.py (key 'masks', N_obj×T×H×W)")
    parser.add_argument("--output",  required=True, help="Output directory")
    parser.add_argument("--stride",  type=int, default=SAM3D_STRIDE,
                        help=f"Process every N-th frame (default: {SAM3D_STRIDE}, from config)")
    parser.add_argument("--seed",    type=int, default=SAM3D_SEED)
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-process frames with existing outputs")
    parser.add_argument("--compile", action="store_true",
                        help="Compile model for faster inference")
    args = parser.parse_args()

    out_dir   = Path(args.output)
    sam3d_dir = out_dir / "sam3D"
    masks     = np.load(args.mask)["masks"]    # (N_obj, T, H, W)
    n_obj     = masks.shape[0]

    # Extract RGB frames once into sam3D/rgb/
    rgb_dir = sam3d_dir / "rgb"
    frame_names, stride_indices = _extract_frames(args.video, rgb_dir, args.stride)

    model = load_model(compile_model=args.compile)
    all_results = {}
    for i in range(n_obj):
        obj_dir = sam3d_dir / f"obj{i}"
        _prepare_object_dir(obj_dir, rgb_dir, masks[i][stride_indices], frame_names)
        results = compute_predictions(model, obj_dir, obj_dir,
                                      frames=frame_names, seed=args.seed,
                                      skip_existing=not args.no_skip)
        save_predictions(results, str(obj_dir))
        ok = sum(1 for r in results if r["status"] == "success")
        print(f"obj{i}: {ok}/{len(results)} frames succeeded")
        all_results[f"obj{i}"] = results

    save_predictions(all_results, str(sam3d_dir))


if __name__ == "__main__":
    main()
