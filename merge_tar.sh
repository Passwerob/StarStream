#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash merge_tar_parts.sh \
    --input-dir /path/to/parts \
    --prefix raw_data.tar.gz.part- \
    --output /path/to/raw_data.tar.gz \
    [--check] \
    [--extract /path/to/extract_dir]

Arguments:
  --input-dir   分片所在目录
  --prefix      分片前缀，例如 raw_data.tar.gz.part-
  --output      合并后的输出 tar.gz 路径
  --check       合并后执行 tar -tzf 完整性检查
  --extract     合并并检查后，直接解压到指定目录

Example:
  bash merge_tar_parts.sh \
    --input-dir /share/magic_group/aigc/fcr/EventVGGT/data/M3ED_parts \
    --prefix raw_data.tar.gz.part- \
    --output /share/magic_group/aigc/fcr/EventVGGT/data/M3ED_parts/raw_data.tar.gz \
    --check
EOF
}

INPUT_DIR=""
PREFIX=""
OUTPUT=""
DO_CHECK=0
EXTRACT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"; shift 2;;
    --prefix)
      PREFIX="$2"; shift 2;;
    --output)
      OUTPUT="$2"; shift 2;;
    --check)
      DO_CHECK=1; shift 1;;
    --extract)
      EXTRACT_DIR="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1;;
  esac
done

if [[ -z "$INPUT_DIR" || -z "$PREFIX" || -z "$OUTPUT" ]]; then
  echo "Error: --input-dir, --prefix, --output are required." >&2
  usage
  exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: input dir not found: $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

mapfile -t PARTS < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "${PREFIX}*" | sort)

if [[ ${#PARTS[@]} -eq 0 ]]; then
  echo "Error: no parts found in $INPUT_DIR with prefix $PREFIX" >&2
  exit 1
fi

echo "[INFO] Found ${#PARTS[@]} parts"

# 检查编号连续
expected=0
for part in "${PARTS[@]}"; do
  base="$(basename "$part")"
  suffix="${base#$PREFIX}"
  if [[ ! "$suffix" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid part suffix: $base" >&2
    exit 1
  fi
  idx=$((10#$suffix))
  if [[ $idx -ne $expected ]]; then
    printf 'Error: part index is not continuous. Expected %03d but got %03d (%s)\n' \
      "$expected" "$idx" "$base" >&2
    exit 1
  fi
  expected=$((expected + 1))
done

echo "[INFO] Part indices are continuous: 000 ~ $(printf "%03d" $((expected-1)))"

# 显示总大小
python3 - <<PY
import os
parts = ${PARTS@Q}.splitlines()
total = sum(os.path.getsize(p) for p in parts)
print(f"[INFO] Total parts size: {total/1024**3:.2f} GiB")
PY

echo "[INFO] Merging to: $OUTPUT"

# 合并
tmp_out="${OUTPUT}.tmp"
rm -f "$tmp_out"

cat "${PARTS[@]}" > "$tmp_out"
mv "$tmp_out" "$OUTPUT"

echo "[INFO] Merge done."

if [[ $DO_CHECK -eq 1 ]]; then
  echo "[INFO] Running tar integrity check: tar -tzf \"$OUTPUT\""
  tar -tzf "$OUTPUT" > /dev/null
  echo "[INFO] Tar integrity check passed."
fi

if [[ -n "$EXTRACT_DIR" ]]; then
  mkdir -p "$EXTRACT_DIR"
  echo "[INFO] Extracting to: $EXTRACT_DIR"
  tar -xzf "$OUTPUT" -C "$EXTRACT_DIR"
  echo "[INFO] Extraction done."
fi

echo "[INFO] Finished successfully."