#!/usr/bin/env bash
# 从训练图片中随机选取 RKNN 量化校准集，并生成容器内可用的 dataset.txt。
#
# 默认用法（construction-ppe 训练集随机选 200 张）：
#   bash prepare_calibration.sh
#
# 自定义示例：
#   SOURCE_DIR=../model_train/datasets/construction-ppe/images/val \
#   COUNT=300 \
#   CALIB_DIR=calib/construction-ppe-val-300 \
#   DATASET=dataset_val_300.txt \
#   bash prepare_calibration.sh
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

SOURCE_DIR="${SOURCE_DIR:-../model_train/datasets/construction-ppe/images/train}"
COUNT="${COUNT:-200}"
CALIB_DIR="${CALIB_DIR:-calib/construction-ppe-${COUNT}}"
DATASET="${DATASET:-dataset.txt}"

fail() {
  echo "[!] $*" >&2
  exit 1
}

[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || fail "COUNT 必须是正整数，当前值：$COUNT"
command -v find >/dev/null 2>&1 || fail "缺少 find 命令"
command -v shuf >/dev/null 2>&1 || fail "缺少 shuf 命令（Ubuntu 安装 coreutils）"
command -v realpath >/dev/null 2>&1 || fail "缺少 realpath 命令（Ubuntu 安装 coreutils）"

SOURCE_ABS="$(realpath -e -- "$SOURCE_DIR" 2>/dev/null)" \
  || fail "找不到图片目录：$SOURCE_DIR"
[[ -d "$SOURCE_ABS" ]] || fail "图片来源不是目录：$SOURCE_ABS"

CALIB_ABS="$(realpath -m -- "$ROOT/$CALIB_DIR")"
DATASET_ABS="$(realpath -m -- "$ROOT/$DATASET")"

# 校准图片与清单必须位于 model_convert 下，才能通过 /work 挂载进入转换容器。
case "$CALIB_ABS" in
  "$ROOT"/*) ;;
  *) fail "CALIB_DIR 必须位于 $ROOT 内：$CALIB_DIR" ;;
esac
case "$DATASET_ABS" in
  "$ROOT"/*) ;;
  *) fail "DATASET 必须位于 $ROOT 内：$DATASET" ;;
esac
[[ "$CALIB_ABS" != "$ROOT" ]] || fail "CALIB_DIR 不能是 model_convert 根目录"

if [[ -e "$CALIB_ABS" ]]; then
  [[ -d "$CALIB_ABS" ]] || fail "CALIB_DIR 已存在且不是目录：$CALIB_DIR"
  if find "$CALIB_ABS" -mindepth 1 -print -quit | grep -q .; then
    fail "目标目录非空：$CALIB_DIR。为避免混入旧图片，请改用新的 CALIB_DIR"
  fi
fi

# dataset.txt 只有空行/注释时可以替换；已有真实路径时要求换一个清单名。
if [[ -f "$DATASET_ABS" ]] \
  && grep -Ev '^[[:space:]]*(#|$)' "$DATASET_ABS" | grep -q .; then
  fail "清单已有有效内容：$DATASET。请改用新的 DATASET 文件名"
fi

mapfile -d '' -t SELECTED < <(
  find "$SOURCE_ABS" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    -print0 |
    shuf -z -n "$COUNT"
)

SELECTED_COUNT="${#SELECTED[@]}"
(( SELECTED_COUNT > 0 )) || fail "目录中没有 jpg/jpeg/png 图片：$SOURCE_ABS"
(( SELECTED_COUNT == COUNT )) \
  || fail "可用图片只有 $SELECTED_COUNT 张，少于请求的 $COUNT 张；请减小 COUNT"

STAGE_DIR="$(mktemp -d "$ROOT/.calib-stage.XXXXXX")"
MANIFEST_TMP="$(mktemp "$ROOT/.dataset-stage.XXXXXX")"
cleanup() {
  if [[ -n "${STAGE_DIR:-}" && -e "$STAGE_DIR" ]]; then
    rm -rf -- "$STAGE_DIR"
  fi
  if [[ -n "${MANIFEST_TMP:-}" && -e "$MANIFEST_TMP" ]]; then
    rm -f -- "$MANIFEST_TMP"
  fi
}
trap cleanup EXIT

CALIB_REL="$(realpath -m --relative-to="$ROOT" "$CALIB_ABS")"
index=0
for source in "${SELECTED[@]}"; do
  index=$((index + 1))
  extension="${source##*.}"
  extension="${extension,,}"
  printf -v filename '%04d.%s' "$index" "$extension"
  cp -- "$source" "$STAGE_DIR/$filename"
  printf '%s/%s\n' "$CALIB_REL" "$filename" >> "$MANIFEST_TMP"
done

mkdir -p -- "$(dirname "$CALIB_ABS")" "$(dirname "$DATASET_ABS")"
if [[ -d "$CALIB_ABS" ]]; then
  rmdir -- "$CALIB_ABS"
fi
mv -- "$STAGE_DIR" "$CALIB_ABS"
STAGE_DIR=""
mv -- "$MANIFEST_TMP" "$DATASET_ABS"
MANIFEST_TMP=""
trap - EXIT

echo "[✓] 校准集已生成"
echo "    来源目录：$SOURCE_ABS"
echo "    图片数量：$SELECTED_COUNT"
echo "    校准目录：$CALIB_REL"
echo "    校准清单：$(realpath --relative-to="$ROOT" "$DATASET_ABS")"
echo
echo "下一步："
echo "  ONNX=model/best.onnx DTYPE=i8 DATASET=$(realpath --relative-to="$ROOT" "$DATASET_ABS") OUTPUT=model/best-int8.rknn bash convert.sh"
