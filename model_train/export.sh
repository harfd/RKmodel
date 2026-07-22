#!/usr/bin/env bash
# 把训练得到的 best.pt 导出为 ONNX（标准导出）
# 用法：
#   bash export.sh                                     # 导出 runs/exp/weights/best.pt
#   WEIGHTS=runs/myexp/weights/best.pt bash export.sh
#
# 注意：这里导出的是"标准 ONNX"，可在 PC 上用 onnxruntime 验证模型是否正常。
#       要上 RK3588，需另外用 airockchip 的 ultralytics fork 导出"RKNN 友好 ONNX"，
#       再用 rknn-toolkit2 转成 .rknn —— 那是单独的转换容器，见 README「下一步」。
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-yolo-train:latest}"
WEIGHTS="${WEIGHTS:-runs/exp/weights/best.pt}"   # 相对 model_train/ 的路径
IMGSZ="${IMGSZ:-640}"
OPSET="${OPSET:-12}"

if [ ! -f "$WEIGHTS" ]; then
  echo "[!] 找不到权重文件：$WEIGHTS"
  echo "    先跑 train.sh，或用 WEIGHTS=<路径> 指定。"
  exit 1
fi

docker run --rm -it \
  -v "$(pwd)":/workspace -w /workspace \
  "$IMAGE" bash -lc "
    yolo export model='$WEIGHTS' format=onnx opset=$OPSET imgsz=$IMGSZ
  "

ONNX="${WEIGHTS%.pt}.onnx"
echo "[✓] 已导出标准 ONNX： model_train/$ONNX"
echo "    ↳ 上板前请改用 airockchip fork 重导 + rknn-toolkit2 转 .rknn（见 README 第 6 节）"
