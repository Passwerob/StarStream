#!/usr/bin/env bash
set -euo pipefail

# ─── 等待指定 PID 退出后再开始 GPU 空闲检测 ───
WATCH_PID="${1:-2629948}"
PID_CHECK_INTERVAL=10

if [[ -z "${WATCH_PID}" ]]; then
  echo "[ERROR] 用法: $0 [PID]  (默认 PID=2629948)"
  exit 1
fi

if ! kill -0 "${WATCH_PID}" 2>/dev/null; then
  echo "[WARN] PID ${WATCH_PID} 当前不存在或已结束，直接进入 GPU 空闲检测阶段"
else
  echo "[INFO] 等待 PID ${WATCH_PID} 退出..."
  echo "[INFO] 进程命令行: $(cat /proc/${WATCH_PID}/cmdline 2>/dev/null | tr '\0' ' ' || echo '无法读取')"
  while kill -0 "${WATCH_PID}" 2>/dev/null; do
    echo "[INFO] PID ${WATCH_PID} 仍在运行，${PID_CHECK_INTERVAL}s 后重新检查"
    sleep "${PID_CHECK_INTERVAL}"
  done
  echo "[INFO] PID ${WATCH_PID} 已退出，开始 GPU 空闲检测"
fi

# ─── GPU 空闲检测配置 ───
DISTILL_SCRIPT="/share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT/src/distill.py"
DISTILL_ARGS="--gpu_ids 0,1,2 --batch_size 2 --epochs 100 --num_workers 4"

CHECK_INTERVAL=5
REQUIRED_IDLE_CHECKS=4

GPU_IDS=(4 5 7)
GPU_QUERY=$(IFS=,; echo "${GPU_IDS[*]}")

MAX_USED_MEM_MB=1500
MAX_UTIL=3

IDLE_STREAK=(0 0 0)

echo "[INFO] 监控 GPU: [${GPU_IDS[*]}]"
echo "[INFO] 判定条件：util<=${MAX_UTIL}，used_mem<=${MAX_USED_MEM_MB}MiB，且无活跃 compute 进程"
echo "[INFO] 3 卡全部空闲（连续 ${REQUIRED_IDLE_CHECKS} 次）后启动训练"
echo "[INFO] Distill 脚本: ${DISTILL_SCRIPT}"
echo "[INFO] Distill 参数: ${DISTILL_ARGS}"

declare -A UUID_MAP
for gpu_id in "${GPU_IDS[@]}"; do
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
      -i "${GPU_QUERY}" \
    | awk -F',' '{
        for (i=1; i<=NF; i++) gsub(/[[:space:]]/, "", $i);
        print $1 " " $2 " " $3
      }'
  )

  if [[ ${#gpu_info[@]} -ne ${#GPU_IDS[@]} ]]; then
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
    for gpu_id in "${GPU_IDS[@]}"; do
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
  for gpu_id in "${GPU_IDS[@]}"; do
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

  for i in "${!GPU_IDS[@]}"; do
    if is_gpu_idle "${GPU_IDS[$i]}"; then
      IDLE_STREAK[$i]=$((IDLE_STREAK[$i] + 1))
    else
      IDLE_STREAK[$i]=0
    fi
  done

  echo "[INFO] idle streak: 4=${IDLE_STREAK[0]}/${REQUIRED_IDLE_CHECKS} 5=${IDLE_STREAK[1]}/${REQUIRED_IDLE_CHECKS} 7=${IDLE_STREAK[2]}/${REQUIRED_IDLE_CHECKS}"

  all_ready=true
  for i in "${!IDLE_STREAK[@]}"; do
    if [[ ${IDLE_STREAK[$i]} -lt ${REQUIRED_IDLE_CHECKS} ]]; then
      all_ready=false
      break
    fi
  done

  if [[ "${all_ready}" == true ]]; then
    CHOSEN_GPUS="${GPU_QUERY}"
    echo "[INFO] >>> GPU [${CHOSEN_GPUS}] 全部空闲 <<<"
    echo "[INFO] 即将启动 distill，使用 GPU: ${CHOSEN_GPUS}"
    echo "[INFO] 命令: CUDA_VISIBLE_DEVICES=${CHOSEN_GPUS} python ${DISTILL_SCRIPT} ${DISTILL_ARGS}"

    cd "$(dirname "${DISTILL_SCRIPT}")"
    export CUDA_VISIBLE_DEVICES="${CHOSEN_GPUS}"
    exec python "${DISTILL_SCRIPT}" ${DISTILL_ARGS}
  fi

  sleep "${CHECK_INTERVAL}"
done
