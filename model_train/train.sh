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
mkdir -p runs _patch

# --weights 路径解析：容器内会 cd 到 /workspace/yolov5，故本地已存在的权重(如 runs/ppe/weights/last.pt)
# 要加 /workspace/ 前缀；裸名(如 yolov5s.pt)保持原样，让 yolov5 自动下载。
if [ -f "$WEIGHTS" ]; then WEIGHTS_ARG="/workspace/$WEIGHTS"; else WEIGHTS_ARG="$WEIGHTS"; fi

# sitecustomize 补丁：让 torch.load 默认 weights_only=False（PyTorch>=2.6 加载旧 yolov5 权重需要）
cat > _patch/sitecustomize.py <<'PYEOF'
try:
    import torch, functools
    _orig = torch.load
    @functools.wraps(_orig)
    def _load(*a, **k):
        k.setdefault('weights_only', False)
        return _orig(*a, **k)
    torch.load = _load
except Exception:
    pass
try:
    # 补回 NumPy 2.0 移除的名字（旧 yolov5 会用到），一次性覆盖，避免逐个报错
    import numpy as np
    for _o, _n in (('trapz','trapezoid'),('in1d','isin'),('row_stack','vstack'),
                   ('product','prod'),('cumproduct','cumprod'),('sometrue','any'),
                   ('alltrue','all'),('round_','round'),
                   ('float_','float64'),('complex_','complex128'),('unicode_','str_'),
                   ('string_','bytes_'),('int0','intp'),('uint0','uintp'),
                   ('longfloat','longdouble'),('singlecomplex','complex64'),('cfloat','complex128')):
        if not hasattr(np, _o) and hasattr(np, _n):
            setattr(np, _o, getattr(np, _n))
    for _o, _v in (('NaN',np.nan),('NAN',np.nan),('Inf',np.inf),('Infinity',np.inf),
                   ('infty',np.inf),('PINF',np.inf),('NINF',-np.inf)):
        if not hasattr(np, _o):
            setattr(np, _o, _v)
except Exception:
    pass
PYEOF

echo "[*] data=$DATA_YAML cfg=$CFG weights=$WEIGHTS epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH name=$NAME gpu=$USE_GPU"

docker run --rm -it $GPU_FLAG --ipc=host \
  -v "$(pwd)":/workspace -w /workspace \
  "$IMAGE" bash -lc "
    cd /workspace/yolov5
    # 挂载目录属主 UID 与容器不同，git 会报 dubious ownership，放行一下
    git config --global --add safe.directory '*'
    # 一次性装齐 yolov5 依赖，但排除 torch/torchvision(避免动掉 CUDA torch；yolov5 的 pin 都是 >= 不会降级)
    grep -viE '^[[:space:]]*(torch|torchvision)([[:space:]]|>|=|<|\$)' requirements.txt > /tmp/req.txt 2>/dev/null || cp requirements.txt /tmp/req.txt
    pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/req.txt 2>/dev/null || true
    export PYTHONPATH=/workspace/_patch:\${PYTHONPATH:-}
    python train.py \
      --data /workspace/$DATA_YAML --cfg $CFG --weights $WEIGHTS_ARG \
      --img $IMGSZ --epochs $EPOCHS --batch-size $BATCH \
      --project /workspace/runs --name $NAME --exist-ok
  "

echo "[✓] 训练完成： runs/$NAME/weights/best.pt"
echo "    下一步导出： WEIGHTS=runs/$NAME/weights/best.pt bash export.sh"
