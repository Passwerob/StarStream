#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "用法: $0 <checkpoint_path> <frame_num> [port]"
  echo "示例: $0 /share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT_mixed_FSDP/checkpoints/vggt_train_dl3dv_fsdp_2026_03_14_dino/checkpoint-last.pth 30 8001"
  exit 1
fi

CKPT_PATH="$1"
FRAME_NUM="$2"
PORT="${3:-8001}"

DATA_ROOT="/share/magic_group/aigc/fcr/EventVGGT/data/DL3DV_data/DL3DV/screen-000167"
SRC_DIR="/share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT_mixed_FSDP_inject/src"
OUT_BASE="/share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT_mixed_FSDP_inject/output"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[ERROR] checkpoint不存在: $CKPT_PATH"
  exit 1
fi

if ! [[ "$FRAME_NUM" =~ ^[0-9]+$ ]] || [[ "$FRAME_NUM" -le 0 ]]; then
  echo "[ERROR] frame_num必须是正整数，当前: $FRAME_NUM"
  exit 1
fi

CKPT_NAME="$(basename "$CKPT_PATH")"
CKPT_NAME="${CKPT_NAME%.*}"
OUT_DIR="${OUT_BASE}/${CKPT_NAME}"

mkdir -p "$OUT_DIR"

echo "[INFO] checkpoint: $CKPT_PATH"
echo "[INFO] frame_num: $FRAME_NUM"
echo "[INFO] data_root: $DATA_ROOT"
echo "[INFO] output: $OUT_DIR"
echo "[INFO] port: $PORT"

python "${SRC_DIR}/inference_with_event.py" \
  --checkpoint "$CKPT_PATH" \
  --data_root "$DATA_ROOT" \
  --output "$OUT_DIR" \
  --max_frames "$FRAME_NUM" \
  --fusion crossattn \
  --event_in_chans 8

PLY_PATH="${OUT_DIR}/point_cloud/merged.ply"
if [[ ! -f "$PLY_PATH" ]]; then
  echo "[ERROR] 未找到 merged.ply: $PLY_PATH"
  exit 1
fi

python "${SRC_DIR}/see_pointcloud.py" "$PLY_PATH" "$PORT"