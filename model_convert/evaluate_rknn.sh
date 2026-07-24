#!/usr/bin/env bash
# 在同一 YOLO 验证集上比较非量化/INT8 RKNN 的逐类 AP50 与 mAP50-95。
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-rknn-convert-152:latest}"
MODE="${MODE:-auto}"                                  # auto/rebuild/rknn
ONNX="${ONNX:-model/best.onnx}"
CALIBRATION="${CALIBRATION:-dataset.txt}"
FP_MODEL="${FP_MODEL:-model/best-fp.rknn}"
INT8_MODEL="${INT8_MODEL:-model/best-int8.rknn}"
PLATFORM="${PLATFORM:-rk3588}"
DATA_YAML="${DATA_YAML:-../model_train/_yolov5_data.yaml}"
DATASET_ROOT="${DATASET_ROOT:-../model_train/datasets/construction-ppe}"
ANCHORS="${ANCHORS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
IMGSZ="${IMGSZ:-640}"
PREPROCESS="${PREPROCESS:-stretch}"
CONF="${CONF:-0.001}"
NMS="${NMS:-0.65}"
MAX_DET="${MAX_DET:-300}"
CLASS_IDS="${CLASS_IDS:-}"
LIMIT="${LIMIT:-0}"
TARGET="${TARGET:-}"
DEVICE_ID="${DEVICE_ID:-}"
REBUILD_IMAGE="${REBUILD_IMAGE:-0}"

# 模型已经复制到板端工程时，也允许直接复用 project1/model 下的文件。
if [ ! -f "$FP_MODEL" ] && [ -f "../../project1/model/best-fp.rknn" ]; then
  FP_MODEL="../../project1/model/best-fp.rknn"
fi
if [ ! -f "$INT8_MODEL" ] && [ -f "../../project1/model/best-int8.rknn" ]; then
  INT8_MODEL="../../project1/model/best-int8.rknn"
fi

if [ -z "$ANCHORS" ]; then
  for candidate in \
    "model/RK_anchors.txt" \
    "../model_train/yolov5/RK_anchors.txt" \
    "../model_train/RK_anchors.txt" \
    "../../project1/model/RK_anchors.txt"; do
    if [ -f "$candidate" ]; then
      ANCHORS="$candidate"
      break
    fi
  done
fi

[ -f "$DATA_YAML" ] || { echo "[!] 找不到 data.yaml: $DATA_YAML"; exit 1; }
[ -d "$DATASET_ROOT" ] || { echo "[!] 找不到数据集根目录: $DATASET_ROOT"; exit 1; }
[ -n "$ANCHORS" ] && [ -f "$ANCHORS" ] || {
  echo "[!] 找不到 RK_anchors.txt；请设置 ANCHORS=/实际路径/RK_anchors.txt"
  exit 1
}

case "$MODE" in
  auto)
    if [ -n "$TARGET" ]; then
      MODE="rknn"
    elif [ -f "$ONNX" ] && [ -f "$CALIBRATION" ]; then
      MODE="rebuild"
    else
      echo "[!] Toolkit 1.5.2 不能在 x86 模拟器直接运行已导出的 .rknn。"
      echo "    离线评测请保留 ONNX=$ONNX 和 CALIBRATION=$CALIBRATION；"
      echo "    或连接板端后使用 MODE=rknn TARGET=rk3588。"
      exit 1
    fi
    ;;
  rebuild)
    [ -f "$ONNX" ] || { echo "[!] 找不到 ONNX: $ONNX"; exit 1; }
    [ -f "$CALIBRATION" ] || { echo "[!] 找不到校准清单: $CALIBRATION"; exit 1; }
    ;;
  rknn)
    [ -n "$TARGET" ] || {
      echo "[!] MODE=rknn 必须设置 TARGET=rk3588 并连接实际板端"
      exit 1
    }
    [ -f "$FP_MODEL" ] || { echo "[!] 找不到 FP 模型: $FP_MODEL"; exit 1; }
    [ -f "$INT8_MODEL" ] || { echo "[!] 找不到 INT8 模型: $INT8_MODEL"; exit 1; }
    ;;
  *)
    echo "[!] MODE 只能是 auto、rebuild 或 rknn"
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR" wheels

if [ "$REBUILD_IMAGE" = "1" ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[*] 构建转换/评测镜像 $IMAGE ..."
  BUILD_ARGS=()
  if [ -n "${RKNN_WHL_URL:-}" ]; then
    BUILD_ARGS+=(--build-arg "RKNN_WHL_URL=$RKNN_WHL_URL")
  fi
  docker build "${BUILD_ARGS[@]}" -t "$IMAGE" .
fi

DATA_YAML_ABS="$(realpath "$DATA_YAML")"
DATASET_ROOT_ABS="$(realpath "$DATASET_ROOT")"
ANCHORS_ABS="$(realpath "$ANCHORS")"
OUTPUT_DIR_ABS="$(realpath "$OUTPUT_DIR")"

ARGS=(
  --data /inputs/data.yaml
  --dataset-root /dataset
  --anchors /inputs/RK_anchors.txt
  --platform "$PLATFORM"
  --imgsz "$IMGSZ"
  --preprocess "$PREPROCESS"
  --conf "$CONF"
  --nms "$NMS"
  --max-det "$MAX_DET"
  --limit "$LIMIT"
  --output-dir /output
)
MOUNTS=(
  -v "$(pwd):/work"
  -v "$DATA_YAML_ABS:/inputs/data.yaml:ro"
  -v "$DATASET_ROOT_ABS:/dataset:ro"
  -v "$ANCHORS_ABS:/inputs/RK_anchors.txt:ro"
  -v "$OUTPUT_DIR_ABS:/output"
)

if [ "$MODE" = "rebuild" ]; then
  ONNX_ABS="$(realpath "$ONNX")"
  # dataset.txt 内默认是相对 /work 的 calib/...，因此 model_convert 整体挂载到 /work。
  case "$(realpath "$CALIBRATION")" in
    "$(pwd)"/*) ;;
    *)
      echo "[!] rebuild 模式的校准清单必须在 model_convert 内，确保其中相对图片路径在 /work 可见"
      exit 1
      ;;
  esac
  CALIBRATION_REL="$(realpath --relative-to="$(pwd)" "$CALIBRATION")"
  ARGS+=(--onnx /inputs/best.onnx --calibration "/work/$CALIBRATION_REL")
  MOUNTS+=(-v "$ONNX_ABS:/inputs/best.onnx:ro")
  echo "[*] mode=rebuild（ONNX 同进程分别 build FP/INT8，再用模拟器评测）"
  echo "[*] onnx=$ONNX calibration=$CALIBRATION"
else
  FP_MODEL_ABS="$(realpath "$FP_MODEL")"
  INT8_MODEL_ABS="$(realpath "$INT8_MODEL")"
  ARGS+=(
    --fp-model /inputs/best-fp.rknn
    --int8-model /inputs/best-int8.rknn
    --target "$TARGET"
  )
  [ -n "$DEVICE_ID" ] && ARGS+=(--device-id "$DEVICE_ID")
  MOUNTS+=(
    -v "$FP_MODEL_ABS:/inputs/best-fp.rknn:ro"
    -v "$INT8_MODEL_ABS:/inputs/best-int8.rknn:ro"
  )
  echo "[*] mode=rknn（在连接的 $TARGET 板端运行现成 RKNN）"
  echo "[*] FP=$FP_MODEL INT8=$INT8_MODEL device_id=${DEVICE_ID:-自动}"
fi

[ -n "$CLASS_IDS" ] && ARGS+=(--class-ids "$CLASS_IDS")

echo "[*] val=$DATASET_ROOT (由 $DATA_YAML 的 val 指定)"
echo "[*] anchors=$ANCHORS preprocess=$PREPROCESS class_ids=${CLASS_IDS:-全部}"

docker run --rm \
  "${MOUNTS[@]}" \
  -w /work \
  "$IMAGE" \
  python3 evaluate_rknn.py "${ARGS[@]}"

echo "[✓] 结果已写入: $OUTPUT_DIR/comparison.csv"
