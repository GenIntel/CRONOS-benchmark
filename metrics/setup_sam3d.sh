#!/usr/bin/env bash
# setup_sam3d.sh — Create the 'cronos-sam3d' conda environment for the
# SAM3D-Objects 3D reconstruction step (run_sam3d.py).
#
# Follows the official sam-3d-objects installation guide (doc/setup.md):
#   conda env create -f environments/default.yml  (Python 3.11 + CUDA 12.1 + nvcc)
#   pip install -e '.[dev]'                        (torch + core deps)
#   pip install -e '.[p3d]'                        (pytorch3d + flash_attn — compile from source)
#   pip install -e '.[inference]'                  (kaolin, gsplat, MoGe, gradio)
#   python patching/hydra                          (hydra patch)
#
# Requirements:
#   - conda (Miniconda or Mamba) on PATH
#   - sam-3d-objects repo cloned
#
# ⚠ The pytorch3d compilation step takes 20–30 min. Run this script inside a
#   screen or tmux session, or on a compute node, to prevent it being killed.
#
# Usage:
#   bash setup_sam3d.sh --sam3d-path PATH
#
# After setup, activate with:
#   conda activate cronos-sam3d

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="cronos-sam3d"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-$(conda info --base)/envs}"
ENV_PREFIX="$CONDA_ENVS_DIR/$ENV_NAME"

# ── Arguments ──────────────────────────────────────────────────────────────────
SAM3D_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sam3d-path)  SAM3D_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$SAM3D_PATH" ]]; then
    echo "Error: --sam3d-path is required." >&2
    echo "Usage: bash setup_sam3d.sh --sam3d-path PATH" >&2
    exit 1
fi

# ── Preflight checks ───────────────────────────────────────────────────────────
echo "=== CRONOS setup_sam3d.sh ==="

if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Install Miniconda or Mamba first." >&2
    exit 1
fi

if [[ ! -d "$SAM3D_PATH" ]]; then
    echo "Error: sam-3d-objects path does not exist: $SAM3D_PATH" >&2
    exit 1
fi

ENV_YML="$SAM3D_PATH/environments/default.yml"
if [[ ! -f "$ENV_YML" ]]; then
    echo "Error: $ENV_YML not found." >&2
    exit 1
fi

echo "SAM3D-Objects path: $SAM3D_PATH"
echo

# ── Create conda environment from the official pinned yml ─────────────────────
# The yml installs Python 3.11, CUDA 12.1 toolkit (nvcc included), and all
# system-level dependencies — no separate CUDA install needed.
if [[ -d "$ENV_PREFIX" ]]; then
    echo "Conda env already exists at $ENV_PREFIX; recreating from yml..."
    conda env remove --prefix "$ENV_PREFIX" -y
fi

echo "Creating conda env at $ENV_PREFIX from environments/default.yml..."
echo "(This downloads ~500 MB — may take a few minutes.)"
set +u  # conda env create references unset vars internally
conda env create --prefix "$ENV_PREFIX" -f "$ENV_YML"
set -u

# Use conda run to target the env directly — conda activate does not propagate
# pip changes into subshells when the script is run non-interactively.
RUN="conda run --no-capture-output --prefix $ENV_PREFIX"

# ── pip index configuration ────────────────────────────────────────────────────
# Required so pip can resolve torch==2.5.1+cu121 and kaolin.
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# ── Hatchling build backend (not in the yml) ───────────────────────────────────
# sam-3d-objects uses hatchling + hatch-requirements-txt; install before any
# editable install so the build backend is available in the env.
$RUN pip install hatchling hatch-requirements-txt editables -q

# ── Step 1: core sam3d-objects dependencies (torch, timm, transformers, ...) ──
echo "Installing sam3d-objects core dependencies (.[dev])..."
$RUN pip install -e "$SAM3D_PATH[dev]"

# ── Step 2: pytorch3d + flash_attn (compile from source — needs nvcc) ─────────
# The env yml provides nvcc via conda-forge cuda-nvcc.
# --no-build-isolation lets the build see already-installed torch and psutil.
echo "Building pytorch3d and flash-attn (.[p3d]) — expect 20–30 min..."
$RUN pip install psutil ninja packaging editables  # flash_attn + hatchling editable build requirements
$RUN pip install --no-build-isolation -e "$SAM3D_PATH[p3d]"

# ── Step 3: inference dependencies (kaolin, gsplat, MoGe, gradio) ─────────────
# gsplat compiles CUDA kernels from source; TORCH_CUDA_ARCH_LIST must be exported
# so it propagates through conda run into the build subprocess.
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0+PTX"
echo "Installing inference dependencies (.[inference])..."
PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html" \
    $RUN pip install -e "$SAM3D_PATH[inference]"

# ── Step 4: hydra patch ────────────────────────────────────────────────────────
echo "Applying hydra patch..."
$RUN python "$SAM3D_PATH/patching/hydra"

# ── Remove backport dataclasses.py that shadows stdlib on Python 3.11 ─────────
# Some transitive dependency installs the old 'dataclasses' backport package into
# site-packages, which shadows the stdlib version and breaks imageio on Python 3.11
# (typing._ClassVar no longer exists). Remove it so the stdlib copy is used instead.
SITE_PKGS="$ENV_PREFIX/lib/python3.11/site-packages"
if [[ -f "$SITE_PKGS/dataclasses.py" ]]; then
    echo "Removing stale dataclasses.py backport from site-packages..."
    rm "$SITE_PKGS/dataclasses.py"
fi

# ── Step 5: CRONOS-specific extras not in sam3d-objects deps ──────────────────
# These are imported by run_sam3d.py and sam3d_objects internals but are not
# listed in the sam-3d-objects requirements files.
echo "Installing CRONOS extras..."
$RUN pip install \
    utils3d \
    pyvista \
    plotly \
    plyfile \
    safetensors

# ── Verification ───────────────────────────────────────────────────────────────
echo
echo "=== Verification ==="
# sam3d_objects/__init__.py imports sam3d_objects.init which is a Meta-internal
# module not included in the open-source release. LIDRA_SKIP_INIT bypasses it.
LIDRA_SKIP_INIT=1 $RUN python - <<'EOF'
import sys
print(f"Python:       {sys.version.split()[0]}")

import torch
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:          {torch.cuda.get_device_name(0)}")

import timm
print(f"timm:         {timm.__version__}")

import transformers
print(f"transformers: {transformers.__version__}")

for pkg in ["spconv", "pytorch3d", "flash_attn", "kaolin", "gsplat", "sam3d_objects"]:
    try:
        m = __import__(pkg)
        ver = getattr(m, "__version__", "OK")
        print(f"{pkg:15}: {ver}")
    except ImportError as e:
        print(f"{pkg:15}: FAILED — {e}")
EOF

PYTHON_BIN=$($RUN python -c "import sys; print(sys.executable)")
echo
echo "Setup complete."
echo "Activate with:          conda activate $ENV_PREFIX"
echo "Python binary path:     ${PYTHON_BIN}"
echo "Use as PYTHON_SAM3D:    PYTHON_SAM3D=${PYTHON_BIN}"
