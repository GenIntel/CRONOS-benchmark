# CRONOS: Benchmarking Counterfactual Physical Consistency in Video Models

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2605.23699)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://genintel.github.io/CRONOS/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/genintel/CRONOS-benchmark)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/GenIntel/CRONOS-benchmark)

Evaluation code for CRONOS-Benchmark. Given a generated video and a reference sequence from the dataset, this repo runs all predictions and computes the six benchmark metrics.

## Repository layout

```
CRONOS-Benchmark/
├── config.py                   # model paths and hyperparameters
├── requirements-cronos.txt     # Python deps for the cronos environment
├── constraints-cronos.txt      # torch version pins (prevents pip from upgrading)
├── setup_cronos.sh             # sets up the 'cronos' conda environment
├── setup_sam3d.sh              # sets up the 'cronos-sam3d' conda environment
├── scripts/
│   ├── run_predictions.sh  # orchestrates all 7 prediction steps
│   ├── run_sam.py          # SAM3 video segmentation
│   ├── run_cotracker.py    # CoTracker point tracking
│   ├── run_sam3d.py        # SAM3D per-object 3D reconstruction
│   ├── run_dino.py         # DINOv2 temporal embeddings
│   ├── run_dismo.py        # DisMo motion embeddings
│   └── run_qwen.py         # Qwen VLM task-performance evaluation
└── metrics/
    ├── metrics.py          # compute and aggregate all metrics
    └── utils.py            # shared loading and scaling utilities
```

## Models

| Model | Role | Source |
|---|---|---|
| [SAM3](https://github.com/facebookresearch/sam3) | Video object segmentation (prompt → per-frame masks) | Clone locally; set `SAM3_MODEL_PATH` in `config.py` |
| [SAM3D-Objects](https://github.com/facebookresearch/sam-3d-objects) | Per-object 3D mesh reconstruction from video | Clone locally for source code (`SAM3D_OBJECTS_DIR`); checkpoints downloaded automatically from HF Hub (`facebook/sam-3d-objects`) |
| [CoTracker3](https://github.com/facebookresearch/co-tracker) | Dense point tracking across frames | Loaded via PyTorch Hub (`facebookresearch/co-tracker`) |
| [DisMo](https://github.com/CompVis/DisMo) | Motion representation embeddings | Loaded via PyTorch Hub (`CompVis/DisMo`) |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Visual feature embeddings for object consistency | Loaded via PyTorch Hub (`facebookresearch/dinov2`) |
| [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | VLM task-performance evaluation | HuggingFace model ID or local path; set `QWEN_MODEL_PATH` in `config.py` |
| [CLIP](https://github.com/openai/CLIP) | Visual features used by SAM3D pipeline | Download locally; set `CLIP_MODEL_PATH` in `config.py` |

## Installation

### Prerequisites

- Linux x86_64 with an NVIDIA GPU (≥24 GB VRAM for Qwen3-VL-32B; ≥16 GB for all other models)
- CUDA 12.4 driver (`nvidia-smi` must work)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Mamba
- `nvcc` (CUDA 12.4 toolkit) in PATH — required only for the SAM3D environment
- ~80 GB of free disk space (model weights + environments)

### Step 1 — Clone model repositories

Clone the following repos to a common directory (e.g. `models/`) and update the paths in `config.py`:

```bash
MODELS_DIR=/path/to/models

# SAM3 (video segmentation)
git clone https://github.com/facebookresearch/sam3 "$MODELS_DIR/sam3"

# SAM3D-Objects (3D reconstruction — source code only; checkpoints are fetched from HF Hub automatically)
git clone https://github.com/facebookresearch/sam-3d-objects "$MODELS_DIR/sam-3d-objects"

# Qwen3-VL (VLM evaluation — includes qwen-vl-utils)
git clone https://github.com/QwenLM/Qwen3-VL "$MODELS_DIR/Qwen3-VL"
```

Edit `config.py` so the path variables point to your clones:

| Variable | Value |
|---|---|
| `SAM3_MODEL_PATH` | `$MODELS_DIR/sam3` |
| `SAM3D_OBJECTS_DIR` | `$MODELS_DIR/sam-3d-objects` |
| `SAM3D_HF_CACHE` | Local directory where SAM3D-Objects checkpoints are cached |
| `CLIP_MODEL_PATH` | Path to a local CLIP checkpoint (used by SAM3D) |
| `QWEN_MODEL_PATH` | `"Qwen/Qwen3-VL-32B-Instruct"` (HuggingFace) or a local path |

CoTracker, DisMo, and DINOv2 are downloaded automatically via PyTorch Hub at first run.  
SAM3D-Objects checkpoints are downloaded automatically from `facebook/sam-3d-objects` on HuggingFace Hub at first run and cached at `SAM3D_HF_CACHE`.

### Step 2 — `cronos` environment (all steps except SAM3D)

This environment covers `run_sam.py`, `run_cotracker.py`, `run_dino.py`, `run_dismo.py`, `run_qwen.py`, and `metrics/`.

```bash
cd CRONOS-benchmark

bash setup_cronos.sh \
    --sam3-path       /path/to/models/sam3 \
    --qwen-utils-path /path/to/models/Qwen3-VL/qwen-vl-utils

conda activate cronos
```

`setup_cronos.sh` installs PyTorch 2.5.1+cu124, all requirements from `requirements-cronos.txt`, and editable installs of SAM3 and `qwen-vl-utils`.

### Step 3 — `cronos-sam3d` environment (`run_sam3d.py` only)

SAM3D-Objects requires older versions of `timm`, `transformers`, and `bitsandbytes` that conflict with the `cronos` environment. It also needs several packages that must be compiled from source (`pytorch3d`, `gsplat`).

> **Note:** `nvcc` (CUDA 12.4 toolkit) must be available before running `setup_sam3d.sh`.

```bash
cd CRONOS-benchmark

bash setup_sam3d.sh --sam3d-path /path/to/models/sam-3d-objects

# Record the Python binary path printed at the end:
#   Python binary path:  /path/to/envs/cronos-sam3d/bin/python
```

> `pytorch3d` and `gsplat` are compiled from source and may take 10–20 minutes.

`setup_sam3d.sh` automatically removes the `dataclasses` backport package from site-packages after installation. This backport is pulled in as a transitive dependency and shadows the Python 3.11 stdlib version, breaking `imageio` at import time.

### Step 4 — Configure `config.py`

Edit `config.py` to point to your local model paths before running anything.

The CoTracker, DisMo, and DINOv2 models are loaded from PyTorch Hub at runtime using the identifiers in `config.py`. SAM3D-Objects checkpoints are fetched from HuggingFace Hub automatically by `run_sam3d.py` at first run.

## Running predictions

`run_predictions.sh` runs all seven steps in order for a single sequence.  
Because the SAM3D step runs in a separate environment, set `PYTHON_SAM3D` to the Python binary of the `cronos-sam3d` conda environment before calling the script:

```bash
# Set PYTHON_SAM3D to the path printed at the end of setup_sam3d.sh, then:
conda activate cronos
PYTHON_SAM3D=/path/to/envs/cronos-sam3d/bin/python bash scripts/run_predictions.sh \
    --video  <generated>.mp4 \
    --ref    <ref_sequence_dir> \
    --output <output_dir>
```

- `--video`: the generated `.mp4` file to evaluate
- `--ref`: a sequence directory from the CRONOS-Benchmark dataset; must contain `metadata.json`, `mask/frame_0000.jpg`, and `movies/complete.mp4`
- `--output`: directory where all prediction artefacts are written

> If `PYTHON_SAM3D` is not set, the script falls back to `python` and assumes a single environment that satisfies all dependencies.

### Output files

| File | Produced by |
|---|---|
| `segmentation_masks.npz` | `run_sam.py` |
| `tracks.npz` | `run_cotracker.py` |
| `sam3D/obj{i}/sam3D/mesh/*.ply` | `run_sam3d.py` |
| `dino_embeddings.npz` | `run_dino.py` |
| `dismo_embeddings.npz` | `run_dismo.py` (generated video) |
| `gt_dismo_embeddings.npz` | `run_dismo.py` (reference video) |
| `vlm_predictions.json` | `run_qwen.py` |

Individual scripts can also be run standalone; each accepts `--help`.

## Computing metrics

```bash
python metrics/metrics.py \
    --video   <generated>.mp4 \
    --ref     <dataset>/{event}/{scene}/{object}/{appearance}/{view} \
    --output  <output_dir> \
    --metrics <output_dir>/metrics.json   # optional, defaults to output/metrics.json
```

`--output` must point to the same directory used with `run_predictions.sh`.

### Metrics

| Metric | Description | Pass threshold |
|---|---|---|
| `bg_mse` | Background MSE vs frame 0 — detects unwanted scene changes | ≤ 0.25 |
| `motion_similarity` | DisMo cosine similarity between generated and reference motion | ≥ 0.70 |
| `object_consistency` | DINOv2 temporal consistency of tracked objects | ≥ 0.50 |
| `mean_chamfer_distance` | SAM3D mesh Chamfer distance across frames — measures shape stability | ≤ 0.80 |
| `vlm_positive_fraction` | Fraction of Qwen VLM task-specific questions answered positively | ≥ 0.35 |
| `success` | `True` only when every metric above meets its threshold simultaneously and no disappearance is detected| — |

Raw values, linearly- or exponentially-scaled values, and the `success` flag are all written to the output JSON.

## Acknowledgments

CRONOS builds on the following models and codebases:

- **[SAM3](https://github.com/facebookresearch/sam3)** — Ravi et al., *SAM 2: Segment Anything in Images and Videos*, Meta AI Research.
- **[SAM3D-Objects](https://github.com/facebookresearch/sam-3d-objects)** — El Banani et al., *SAM3D: Zero-Shot 3D Object Understanding with Vision Foundation Models*, Meta AI Research.
- **[CoTracker3](https://github.com/facebookresearch/co-tracker)** — Karaev et al., *CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos*, Meta AI Research.
- **[DisMo](https://github.com/CompVis/DisMo)** — Rühling Cachay et al., *DisMo: Disentangled Motion Representations*, CompVis.
- **[DINOv2](https://github.com/facebookresearch/dinov2)** — Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, Meta AI Research.
- **[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)** — Qwen Team, *Qwen3-VL*, Alibaba Cloud.
- **[CLIP](https://github.com/openai/CLIP)** — Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, OpenAI.

## Citation

If you use CRONOS-Benchmark in your research, please cite:

```bibtex
@misc{begiristain2026cronos,
      title={CRONOS: Benchmarking Counterfactual Physical Consistency in Video Models}, 
      author={Le{\'o}n Begiristain and Olaf D{\"u}nkel and Adam Kortylewski},
      year={2026},
      eprint={2605.23699},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.23699}, 
}
```
