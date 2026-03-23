#!/usr/bin/env bash
cd /share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT_mixed_FSDP_inject/src

eval "$(conda shell.bash hook)"
conda activate StreamVGGT

PORT=${MASTER_PORT:-$((29500 + RANDOM % 500))}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" HYDRA_FULL_ERROR=1 \
accelerate launch --use_fsdp --num_processes 4 --main_process_port "${PORT}" \
./train.py --config-name train_M3ed_curriculum_v2
