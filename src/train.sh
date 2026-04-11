#!/usr/bin/env bash
set -euo pipefail

# Usage: bash train.sh [GPU_IDS] [CONFIG_NAME]
#   GPU_IDS     - comma-separated GPU list, default: 4,5
#   CONFIG_NAME - yaml config name (without .yaml), default: train_M3ed_curriculum_v7
#
# Examples:
#   bash train.sh                        # GPU 4,5, v7 config
#   bash train.sh 4,5                    # GPU 4,5, v7 config
#   bash train.sh 0,1,2,3               # GPU 0,1,2,3, v7 config
#   bash train.sh 4,5,7 train_M3ed_curriculum_v7

GPU_IDS="${1:-4,5}"
CONFIG_NAME="${2:-train_M3ed_curriculum_v7}"

# Count GPUs
NUM_GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)

# Activate conda
source /home/tzy/miniconda3/etc/profile.d/conda.sh
conda activate StreamVGGT

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
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  StreamVGGT Training (FSDP, ${NUM_GPUS}-GPU)"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  Config: ${CONFIG_NAME}"
echo "  Log: ${LOG_FILE}"
echo "============================================"

accelerate launch \
    --num_processes ${NUM_GPUS} \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port 29501 \
    "${SCRIPT_DIR}/train.py" \
    --config-name "${CONFIG_NAME}" \
    2>&1 | tee "${LOG_FILE}"
