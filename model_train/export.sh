#!/usr/bin/env bash
# 把训练得到的经典 YOLOv5 best.pt 导出为"RKNN 友好 ONNX"(airockchip/yolov5 的 --rknpu)。
# 会同时生成 RK_anchors.txt —— 板端后处理要用到这些 anchor（务必和 postprocess.cpp 对上）。
# 用法：
#   bash export.sh
#   WEIGHTS=runs/ppe/weights/best.pt bash export.sh
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-yolo-train:latest}"
WEIGHTS="${WEIGHTS:-runs/ppe/weights/best.pt}"   # 相对本目录

[ -d yolov5 ] || { echo "[!] 缺 yolov5/ 目录"; exit 1; }
[ -f "$WEIGHTS" ] || { echo "[!] 找不到权重： $WEIGHTS（先训练，或用 WEIGHTS=... 指定）"; exit 1; }

docker run --rm -it -v "$(pwd)":/workspace -w /workspace "$IMAGE" bash -lc "
  cd /workspace/yolov5
  python export.py --rknpu --weight /workspace/$WEIGHTS
"

ONNX="${WEIGHTS%.pt}.onnx"
echo "[✓] 已导出 RKNN 友好 ONNX： model_train/$ONNX"
echo "    同时生成 RK_anchors.txt（anchor 值）—— 板端 postprocess.cpp 的 anchor 要与之一致"
echo "    下一步： 拷到 ../model_convert/model/ 用 1.5.2 容器做 i8 量化"
