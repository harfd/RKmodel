#!/usr/bin/env bash
# 训练"经典 anchor 版 YOLOv5"(airockchip/yolov5)——产出能复用项目现有 YOLOv5 后处理的模型。
# 注意：不是 ultralytics 包里的 yolov5nu(那是 anchor-free 的 YOLOv5u，和现有后处理不匹配)。
#
# 前置：
#   1) 本目录下已 clone: git clone https://github.com/airockchip/yolov5.git
#   2) 已跑 bash prepare_dataset.sh 生成 _yolov5_data.yaml
# 用法：
#   bash train.sh
#   CFG=yolov5n.yaml WEIGHTS=yolov5n.pt EPOCHS=200 NAME=ppe bash train.sh
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-yolo-train:latest}"
DATA_YAML="${DATA_YAML:-_yolov5_data.yaml}"   # prepare_dataset.sh 生成的
CFG="${CFG:-yolov5s.yaml}"                    # 模型结构 yolov5n/s/m.yaml（多路实时选 n 或 s）
WEIGHTS="${WEIGHTS:-yolov5s.pt}"              # 预训练权重（首次自动从 github 下，需联网/代理）
EPOCHS="${EPOCHS:-150}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
NAME="${NAME:-ppe}"
USE_GPU="${USE_GPU:-1}"

GPU_FLAG=""; [ "$USE_GPU" = "1" ] && GPU_FLAG="--gpus all"

[ -d yolov5 ] || { echo "[!] 缺 yolov5/ 目录。先执行： git clone https://github.com/airockchip/yolov5.git"; exit 1; }
[ -f "$DATA_YAML" ] || { echo "[!] 缺 $DATA_YAML。先执行： bash prepare_dataset.sh"; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" .
mkdir -p runs

echo "[*] data=$DATA_YAML cfg=$CFG weights=$WEIGHTS epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH name=$NAME gpu=$USE_GPU"

docker run --rm -it $GPU_FLAG --ipc=host \
  -v "$(pwd)":/workspace -w /workspace \
  "$IMAGE" bash -lc "
    cd /workspace/yolov5
    # 补 yolov5 需要但基础镜像可能缺的轻量依赖(不动已装的 CUDA torch)
    pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm seaborn thop gitpython psutil pyyaml requests matplotlib pandas 2>/dev/null || true
    python train.py \
      --data /workspace/$DATA_YAML --cfg $CFG --weights $WEIGHTS \
      --img $IMGSZ --epochs $EPOCHS --batch-size $BATCH \
      --project /workspace/runs --name $NAME --exist-ok
  "

echo "[✓] 训练完成： runs/$NAME/weights/best.pt"
echo "    下一步导出： WEIGHTS=runs/$NAME/weights/best.pt bash export.sh"
