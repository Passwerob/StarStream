#!/usr/bin/env bash
set -euo pipefail

# Usage: bash train_resume.sh [GPU_IDS] [CONFIG_NAME]
GPU_IDS="${1:-0,1,2,3,4,5,6,7}"
CONFIG_NAME="${2:-train_M3ed_curriculum_0415_8gpu}"
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vggt

# WandB config — resume previous run
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://33.180.4.104}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
export WANDB_API_KEY="${WANDB_API_KEY:-local-a89761145abe2f8f9208ea20575fa351958b90be}"
export WANDB_PROJECT="${WANDB_PROJECT:-StarStream}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${CONFIG_NAME}}"
export WANDB_RESUME="must"
export WANDB_RUN_ID="417ef5b7"

# Fix cuDNN version
CUDNN_LIB="$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__)+'/lib')" 2>/dev/null)"
if [ -n "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH}"
fi

# GPU config
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# FSDP config
export ACCELERATE_USE_FSDP=true
export FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP
export FSDP_TRANSFORMER_CLS_TO_WRAP=Block
export FSDP_BACKWARD_PREFETCH=BACKWARD_PRE
export FSDP_SHARDING_STRATEGY=FULL_SHARD
export FSDP_STATE_DICT_TYPE=FULL_STATE_DICT
export FSDP_OFFLOAD_PARAMS=false
export FSDP_SYNC_MODULE_STATES=true
export FSDP_USE_ORIG_PARAMS=true
export FSDP_CPU_RAM_EFFICIENT_LOADING=false
export FSDP_FORWARD_PREFETCH=false
export FSDP_ACTIVATION_CHECKPOINTING=false

# Misc
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../checkpoints/${CONFIG_NAME}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_resume_$(date +%Y%m%d_%H%M%S).log"

# Resume checkpoint
RESUME_CKPT="${SCRIPT_DIR}/../checkpoints/vggt_train_M3ed_curriculum_v7/checkpoint-last.pth"

echo "============================================"
echo "  StreamVGGT RESUME Training (FSDP, ${NUM_GPUS}-GPU)"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  Config: ${CONFIG_NAME}"
echo "  Resume: ${RESUME_CKPT}"
echo "  WandB Run ID: ${WANDB_RUN_ID}"
echo "  Log: ${LOG_FILE}"
echo "============================================"

accelerate launch \
    --num_processes ${NUM_GPUS} \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port 29501 \
    "${SCRIPT_DIR}/train.py" \
    --config-name "${CONFIG_NAME}" \
    resume="${RESUME_CKPT}" \
    2>&1 | tee "${LOG_FILE}"
