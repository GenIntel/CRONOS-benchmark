#!/bin/bash
# Run all predictions in order for a single video against a reference sequence.
#
# Usage:
#   run_predictions.sh --video VIDEO --ref REF_DIR --output OUTPUT_DIR [--ref_alternatives ALT_DIR]
#
# Arguments:
#   --video   Generated/input .mp4 file to evaluate
#   --ref     Reference dataset directory containing metadata.json, mask/, and movies/
#   --output  Directory where all predictions are stored
#   --ref_alternatives  Optional directory of alternative reference .mp4 files;
#             motion similarity is computed against each and the best is kept
#
# Output files in OUTPUT_DIR:
#   segmentation_masks.npz      (run_sam)
#   tracks.npz                  (run_cotracker)
#   sam3D/obj{i}/sam3D/...      (run_sam3d, SAM3D_STRIDE from config)
#   dino_embeddings.npz         (run_dino)
#   dismo_embeddings.npz        (run_dismo, generated video)
#   gt_dismo_embeddings.npz     (run_dismo, reference frames)
#   gt_dismo_embeddings/*.npz   (run_dismo, one per alternative; only with --ref_alternatives)
#   vlm_predictions.json        (run_qwen)

set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPTS")"



VIDEO=""
REF_DIR=""
OUTPUT=""
ALT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)   VIDEO="$2";   shift 2 ;;
        --ref)     REF_DIR="$2"; shift 2 ;;
        --output)  OUTPUT="$2";  shift 2 ;;
        --ref_alternatives) ALT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$VIDEO" || -z "$REF_DIR" || -z "$OUTPUT" ]]; then
    echo "Error: --video, --ref and --output are required." >&2
    exit 1
fi

REF_DIR="$(realpath "$REF_DIR")"
METADATA="$REF_DIR/metadata.json"
REF_MASK="$REF_DIR/mask/frame_0000.jpg"
REF_VIDEO="$REF_DIR/movies/complete.mp4"

if [[ ! -f "$METADATA" ]]; then
    echo "Error: metadata.json not found at $METADATA" >&2; exit 1
fi

if [[ ! -f "$REF_MASK" ]]; then
    echo "Error: mask not found at $REF_MASK" >&2; exit 1
fi
if [[ ! -f "$REF_VIDEO" ]]; then
    echo "Error: reference video not found at $REF_VIDEO" >&2; exit 1
fi

ALT_VIDEOS=()
if [[ -n "$ALT_DIR" ]]; then
    shopt -s nullglob
    ALT_VIDEOS=("$ALT_DIR"/*.mp4)
    shopt -u nullglob
    if [[ ${#ALT_VIDEOS[@]} -eq 0 ]]; then
        echo "Error: no .mp4 files found in $ALT_DIR" >&2; exit 1
    fi
fi

mkdir -p "$OUTPUT"
MASKS="$OUTPUT/segmentation_masks.npz"

echo "[1/7] SAM3 segmentation"
python "$SCRIPTS/run_sam.py" \
    --video  "$VIDEO" \
    --mask   "$REF_MASK" \
    --output "$OUTPUT"

echo "[2/7] CoTracker"
python "$SCRIPTS/run_cotracker.py" \
    --video  "$VIDEO" \
    --mask   "$REF_MASK" \
    --output "$OUTPUT"


echo "[3/7] DINOv2 embeddings"
python "$SCRIPTS/run_dino.py" \
    --video  "$VIDEO" \
    --mask   "$MASKS" \
    --output "$OUTPUT/dino_embeddings.npz"

# All DisMo inputs are resampled to the lowest fps among them so the
# lookahead window covers the same duration in every video.
DISMO_ALIGN=("$VIDEO" "$REF_VIDEO" ${ALT_VIDEOS[@]+"${ALT_VIDEOS[@]}"})

echo "[4/7] DisMo embeddings (generated)"
python "$SCRIPTS/run_dismo.py" \
    --video    "$VIDEO" \
    --output   "$OUTPUT/dismo_embeddings.npz" \
    --align_to "${DISMO_ALIGN[@]}"

echo "[5/7] DisMo embeddings (reference)"
python "$SCRIPTS/run_dismo.py" \
    --video    "$REF_VIDEO" \
    --output   "$OUTPUT/gt_dismo_embeddings.npz" \
    --align_to "${DISMO_ALIGN[@]}"

# Clear alternatives from previous runs so metrics.py never picks up stale ones.
rm -rf "$OUTPUT/gt_dismo_embeddings"
for ALT in ${ALT_VIDEOS[@]+"${ALT_VIDEOS[@]}"}; do
    python "$SCRIPTS/run_dismo.py" \
        --video    "$ALT" \
        --output   "$OUTPUT/gt_dismo_embeddings/$(basename "${ALT%.*}").npz" \
        --align_to "${DISMO_ALIGN[@]}"
done

echo "[6/7] Qwen VLM evaluation"
python "$SCRIPTS/run_qwen.py" \
    --video    "$VIDEO" \
    --metadata "$METADATA" \
    --output   "$OUTPUT/vlm_predictions.json"



# Python interpreter for run_sam3d.py.  SAM3D-Objects requires different package
# versions (timm 0.9.16, older transformers) that conflict with the main cronos
# environment.  Point this at the SAM3D venv's python when running both envs:
#
#   PYTHON_SAM3D=/path/to/cronos-sam3d/bin/python \
#       bash scripts/run_predictions.sh --video ... --ref ... --output ...
#
# If left unset, 'python' is used and the active environment must satisfy both.

PYTHON_SAM3D="${PYTHON_SAM3D:-python}"

# When PYTHON_SAM3D points to a different interpreter, prepend its site-packages
# to PYTHONPATH so it can find its own venv packages (spconv, pytorch3d, etc.)
# while still inheriting the caller's environment (e.g. CUDA libraries).
if [[ "$PYTHON_SAM3D" != "python" && -x "$PYTHON_SAM3D" ]]; then
    _sam3d_site="$("$PYTHON_SAM3D" -c \
        'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
    if [[ -n "$_sam3d_site" ]]; then
        export PYTHONPATH="${_sam3d_site}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
fi

echo "[7/7] SAM3D (all objects, stride from config)"
"$PYTHON_SAM3D" "$SCRIPTS/run_sam3d.py" \
    --video  "$VIDEO" \
    --mask   "$MASKS" \
    --output "$OUTPUT"
    
echo "Done. All predictions stored in: $OUTPUT"
