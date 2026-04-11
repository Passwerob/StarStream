#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Event-Based RGB Reconstruction Evaluation Benchmark
# ============================================================================
#
# Usage:
#   bash run_eval_benchmark.sh <pred_root> <gt_root> [output_dir] [device] [event_method]
#
# Examples:
#   # Evaluate checkpoint-2 against DL3DV GT
#   bash run_eval_benchmark.sh \
#     /share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT_mixed_FSDP/output/checkpoint-2 \
#     /share/magic_group/aigc/fcr/EventVGGT/data/DL3DV_data/DL3DV/screen-000167
#
#   # Evaluate all checkpoints
#   bash run_eval_benchmark.sh \
#     /share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT_mixed_FSDP/output \
#     /share/magic_group/aigc/fcr/EventVGGT/data/DL3DV_data/DL3DV/screen-000167 \
#     /share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT_mixed_FSDP/eval_output \
#     cuda \
#     all
# ============================================================================

PRED_ROOT="${1:?Usage: $0 <pred_root> <gt_root> [output_dir] [device] [event_method]}"
GT_ROOT="${2:-}"
OUTPUT_DIR="${3:-}"
DEVICE="${4:-}"
EVENT_METHOD="${5:-l1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD="python -m eval_benchmark --root \"${PRED_ROOT}\""

if [[ -n "$GT_ROOT" ]]; then
  CMD="${CMD} --gt_root \"${GT_ROOT}\""
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD="${CMD} --output \"${OUTPUT_DIR}\""
fi

if [[ -n "$DEVICE" ]]; then
  CMD="${CMD} --device \"${DEVICE}\""
fi

CMD="${CMD} --event_method \"${EVENT_METHOD}\""

CONDA_ENV="${CONDA_ENV:-StreamVGGT}"

echo "[eval_benchmark] Running from: ${SCRIPT_DIR}"
echo "[eval_benchmark] Conda env: ${CONDA_ENV}"
echo "[eval_benchmark] Command: ${CMD}"

cd "${SCRIPT_DIR}"

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)"
  conda activate "${CONDA_ENV}" 2>/dev/null || true
fi

eval "${CMD}"
