#!/usr/bin/env bash
# setup_cronos.sh — Create the 'cronos' conda environment for the CRONOS-Benchmark
# evaluation pipeline (all steps except SAM3D-Objects reconstruction).
#
# Requirements:
#   - conda (Miniconda or Mamba) on PATH
#   - NVIDIA GPU with CUDA 12.x driver (nvidia-smi must work)
#   - SAM3 repo cloned
#   - Qwen3-VL repo cloned
#
# Usage:
#   bash setup_cronos.sh --sam3-path PATH --qwen-utils-path PATH
#
# After setup, activate with:
#   conda activate cronos

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Arguments ──────────────────────────────────────────────────────────────────
SAM3_PATH=""
QWEN_UTILS_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sam3-path)        SAM3_PATH="$2";        shift 2 ;;
        --qwen-utils-path)  QWEN_UTILS_PATH="$2";  shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$SAM3_PATH" || -z "$QWEN_UTILS_PATH" ]]; then
    echo "Error: --sam3-path and --qwen-utils-path are required." >&2
    echo "Usage: bash setup_cronos.sh --sam3-path PATH --qwen-utils-path PATH" >&2
    exit 1
fi

# ── Preflight checks ───────────────────────────────────────────────────────────
echo "=== CRONOS setup_cronos.sh ==="

if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Install Miniconda or Mamba first." >&2
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "Warning: nvidia-smi not found — make sure a CUDA GPU is available."
fi

echo "SAM3 path:       ${SAM3_PATH:-(not set — skip SAM3 install)}"
echo "Qwen utils path: ${QWEN_UTILS_PATH:-(not set — skip qwen-vl-utils install)}"
echo

# ── Create conda environment ───────────────────────────────────────────────────
ENV_NAME="cronos"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Conda env '${ENV_NAME}' already exists. Updating..."
else
    echo "Creating conda env '${ENV_NAME}' (Python 3.11)..."
    conda create -n "$ENV_NAME" python=3.11 -y
fi

# Use conda run to target the env directly — conda activate does not propagate
# pip changes into subshells when the script is run non-interactively.
RUN="conda run --no-capture-output -n $ENV_NAME"

# ── PyTorch 2.5.1 + CUDA 12.4 ─────────────────────────────────────────────────
echo "Installing PyTorch 2.5.1+cu124..."
$RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Main requirements ──────────────────────────────────────────────────────────
echo "Installing requirements-cronos.txt..."
$RUN pip install -r "$SCRIPT_DIR/requirements-cronos.txt" \
    -c "$SCRIPT_DIR/constraints-cronos.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu124

# ── Local editable installs ────────────────────────────────────────────────────
if [[ -d "$SAM3_PATH" ]]; then
    echo "Installing SAM3 from $SAM3_PATH..."
    # sam3 uses pkg_resources (removed from setuptools>=72); pin before installing
    $RUN pip install "setuptools<72" -q
    $RUN pip install -e "$SAM3_PATH"
else
    echo "Error: SAM3 path does not exist: $SAM3_PATH" >&2
    exit 1
fi

if [[ -d "$QWEN_UTILS_PATH" ]]; then
    echo "Installing qwen-vl-utils from $QWEN_UTILS_PATH..."
    $RUN pip install -e "$QWEN_UTILS_PATH"
else
    echo "Error: qwen-vl-utils path does not exist: $QWEN_UTILS_PATH" >&2
    exit 1
fi

# ── Verification ───────────────────────────────────────────────────────────────
echo
echo "=== Verification ==="
$RUN python - <<'EOF'
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

try:
    import vllm
    print(f"vllm:         {vllm.__version__}")
except ImportError:
    print("vllm:         NOT INSTALLED")

try:
    import sam3
    print("sam3:         OK")
except ImportError:
    print("sam3:         NOT INSTALLED (run: pip install -e <SAM3_MODEL_PATH>)")

try:
    from qwen_vl_utils import process_vision_info
    print("qwen_vl_utils: OK")
except ImportError:
    print("qwen_vl_utils: NOT INSTALLED (run: pip install -e <Qwen3-VL/qwen-vl-utils>)")
EOF

echo
echo "Setup complete. Activate with:  conda activate ${ENV_NAME}"
