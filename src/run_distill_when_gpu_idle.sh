#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT="/share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT_mixed_FSDP_inject/src/train.sh"

CHECK_INTERVAL=5
REQUIRED_IDLE_CHECKS=4

GPU_GROUP_A=(0 1 2 3)
GPU_GROUP_B=(4 5 6 7)

MAX_USED_MEM_MB=1500
MAX_UTIL=3

ALL_GPU_IDS=("${GPU_GROUP_A[@]}" "${GPU_GROUP_B[@]}")
ALL_GPU_QUERY=$(IFS=,; echo "${ALL_GPU_IDS[*]}")

IDLE_STREAK_A=(0 0 0 0)
IDLE_STREAK_B=(0 0 0 0)

echo "[INFO] 监控两组 GPU: A=[${GPU_GROUP_A[*]}]  B=[${GPU_GROUP_B[*]}]"
echo "[INFO] 判定条件：util<=${MAX_UTIL}，used_mem<=${MAX_USED_MEM_MB}MiB，且无活跃 compute 进程"
echo "[INFO] 任意一组 4 卡全部空闲（连续 ${REQUIRED_IDLE_CHECKS} 次）后启动训练"
echo "[INFO] 训练脚本: ${TRAIN_SCRIPT}"

declare -A UUID_MAP
for gpu_id in "${ALL_GPU_IDS[@]}"; do
  UUID_MAP["$gpu_id"]=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${gpu_id}" | awk '{gsub(/[[:space:]]/, "", $0); print $0}')
done

declare -A UTIL_MAP
declare -A MEM_MAP
declare -A PROC_MAP

while true; do
  mapfile -t gpu_info < <(
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits \
      -i "${ALL_GPU_QUERY}" \
    | awk -F',' '{
        for (i=1; i<=NF; i++) gsub(/[[:space:]]/, "", $i);
        print $1 " " $2 " " $3
      }'
  )

  if [[ ${#gpu_info[@]} -ne ${#ALL_GPU_IDS[@]} ]]; then
    echo "[WARN] 未成功读取目标 GPU 信息，${CHECK_INTERVAL}s 后重试"
    sleep "${CHECK_INTERVAL}"
    continue
  fi

  for line in "${gpu_info[@]}"; do
    idx=$(awk '{print $1}' <<< "${line}")
    util=$(awk '{print $2}' <<< "${line}")
    mem=$(awk '{print $3}' <<< "${line}")

    UTIL_MAP["$idx"]="$util"
    MEM_MAP["$idx"]="$mem"
    PROC_MAP["$idx"]=0
  done

  proc_lines=()
  mapfile -t proc_lines < <(
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits 2>/dev/null || true
  )

  if [[ ${#proc_lines[@]} -gt 0 ]]; then
    for gpu_id in "${ALL_GPU_IDS[@]}"; do
      gpu_uuid="${UUID_MAP[$gpu_id]}"
      for pline in "${proc_lines[@]}"; do
        line_no_space=$(sed 's/[[:space:]]//g' <<< "${pline}")
        if [[ "${line_no_space}" == "${gpu_uuid},"* ]]; then
          PROC_MAP["$gpu_id"]=1
          break
        fi
      done
    done
  fi

  status_msg="[INFO] 状态:"
  for gpu_id in "${ALL_GPU_IDS[@]}"; do
    util="${UTIL_MAP[$gpu_id]:-NA}"
    mem="${MEM_MAP[$gpu_id]:-NA}"
    has_proc="${PROC_MAP[$gpu_id]:-NA}"
    status_msg+=" ${gpu_id}=util:${util},mem:${mem}MiB,proc:${has_proc}"
  done
  echo "${status_msg}"

  is_gpu_idle() {
    local gid="$1"
    local util="${UTIL_MAP[$gid]}"
    local mem="${MEM_MAP[$gid]}"
    local has_proc="${PROC_MAP[$gid]}"
    [[ "${util}" -le "${MAX_UTIL}" ]] && [[ "${mem}" -le "${MAX_USED_MEM_MB}" ]] && [[ "${has_proc}" -eq 0 ]]
  }

  for i in "${!GPU_GROUP_A[@]}"; do
    if is_gpu_idle "${GPU_GROUP_A[$i]}"; then
      IDLE_STREAK_A[$i]=$((IDLE_STREAK_A[$i] + 1))
    else
      IDLE_STREAK_A[$i]=0
    fi
  done

  for i in "${!GPU_GROUP_B[@]}"; do
    if is_gpu_idle "${GPU_GROUP_B[$i]}"; then
      IDLE_STREAK_B[$i]=$((IDLE_STREAK_B[$i] + 1))
    else
      IDLE_STREAK_B[$i]=0
    fi
  done

  echo "[INFO] idle streak A: 0=${IDLE_STREAK_A[0]}/${REQUIRED_IDLE_CHECKS} 1=${IDLE_STREAK_A[1]}/${REQUIRED_IDLE_CHECKS} 2=${IDLE_STREAK_A[2]}/${REQUIRED_IDLE_CHECKS} 3=${IDLE_STREAK_A[3]}/${REQUIRED_IDLE_CHECKS}"
  echo "[INFO] idle streak B: 4=${IDLE_STREAK_B[0]}/${REQUIRED_IDLE_CHECKS} 5=${IDLE_STREAK_B[1]}/${REQUIRED_IDLE_CHECKS} 6=${IDLE_STREAK_B[2]}/${REQUIRED_IDLE_CHECKS} 7=${IDLE_STREAK_B[3]}/${REQUIRED_IDLE_CHECKS}"

  check_group_ready() {
    local -n streaks=$1
    for i in "${!streaks[@]}"; do
      if [[ ${streaks[$i]} -lt ${REQUIRED_IDLE_CHECKS} ]]; then
        return 1
      fi
    done
    return 0
  }

  CHOSEN_GPUS=""
  if check_group_ready IDLE_STREAK_A; then
    CHOSEN_GPUS=$(IFS=,; echo "${GPU_GROUP_A[*]}")
    echo "[INFO] >>> 组 A [${CHOSEN_GPUS}] 全部空闲，选择该组 <<<"
  elif check_group_ready IDLE_STREAK_B; then
    CHOSEN_GPUS=$(IFS=,; echo "${GPU_GROUP_B[*]}")
    echo "[INFO] >>> 组 B [${CHOSEN_GPUS}] 全部空闲，选择该组 <<<"
  fi

  if [[ -n "${CHOSEN_GPUS}" ]]; then
    echo "[INFO] 即将启动训练，使用 GPU: ${CHOSEN_GPUS}"
    echo "[INFO] 命令: CUDA_VISIBLE_DEVICES=${CHOSEN_GPUS} bash ${TRAIN_SCRIPT}"

    cd "$(dirname "${TRAIN_SCRIPT}")"
    export CUDA_VISIBLE_DEVICES="${CHOSEN_GPUS}"
    exec bash "./$(basename "${TRAIN_SCRIPT}")"
  fi

  sleep "${CHECK_INTERVAL}"
done
