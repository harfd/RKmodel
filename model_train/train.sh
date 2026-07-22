#!/usr/bin/env bash
# 启动 YOLOv8 训练任务（容器内运行，产物持久化到宿主机 model_train/ 目录）
# 用法示例：
#   bash train.sh                                   # 用默认参数(非洲野生动物数据集)训练
#   DATA=construction-ppe.yaml EPOCHS=150 bash train.sh
#   DATA=coco8.yaml EPOCHS=5 bash train.sh          # 冒烟测试链路
#   USE_GPU=0 bash train.sh                         # 无 NVIDIA GPU 时用 CPU(很慢)
set -euo pipefail
cd "$(dirname "$0")"

# ---------- 可配置参数(用环境变量覆盖) ----------
IMAGE="${IMAGE:-yolo-train:latest}"
DATA="${DATA:-african-wildlife.yaml}"   # 数据集 yaml：内置的会自动下载，也可指向自己的
MODEL="${MODEL:-yolov8n.pt}"            # 基础权重：yolov8n/s/m/l/x.pt（首次自动下载）
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"                    # 显存小可设小，或设 -1 让其自动
NAME="${NAME:-exp}"                     # 输出子目录：runs/<NAME>
USE_GPU="${USE_GPU:-1}"                 # 1=用 GPU，0=CPU
# -----------------------------------------------

GPU_FLAG=""
[ "$USE_GPU" = "1" ] && GPU_FLAG="--gpus all"

# 镜像不存在则先构建
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[*] 镜像 $IMAGE 不存在，开始构建（首次较慢，镜像约数 GB）..."
  docker build -t "$IMAGE" .
fi

mkdir -p datasets runs

echo "[*] 训练参数： data=$DATA model=$MODEL epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH gpu=$USE_GPU name=$NAME"

# --ipc=host / --shm-size：PyTorch 多进程 dataloader 需要更大的共享内存，否则会 worker 被杀
docker run --rm -it $GPU_FLAG --ipc=host \
  -v "$(pwd)":/workspace -w /workspace \
  "$IMAGE" bash -lc "
    yolo settings datasets_dir=/workspace/datasets >/dev/null 2>&1 || true
    yolo detect train \
      data='$DATA' model='$MODEL' \
      epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH \
      project=/workspace/runs name='$NAME'
  "

echo "[✓] 训练完成。最优权重： model_train/runs/$NAME/weights/best.pt"
echo "    下一步导出 ONNX： WEIGHTS=runs/$NAME/weights/best.pt bash export.sh"
