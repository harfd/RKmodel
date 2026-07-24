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

mkdir -p _patch
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

docker run --rm -it -v "$(pwd)":/workspace -w /workspace "$IMAGE" bash -lc "
  cd /workspace/yolov5
  git config --global --add safe.directory '*'
  export PYTHONPATH=/workspace/_patch:\${PYTHONPATH:-}
  grep -viE '^[[:space:]]*(torch|torchvision)([[:space:]]|>|=|<|\$)' requirements.txt > /tmp/req.txt 2>/dev/null || cp requirements.txt /tmp/req.txt
  pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/req.txt onnx onnxscript 2>/dev/null || true
  python export.py --rknpu --weight /workspace/$WEIGHTS
"

ONNX="${WEIGHTS%.pt}.onnx"
if [ -f "$ONNX" ]; then
  echo "[✓] 已导出 RKNN 友好 ONNX： model_train/$ONNX"
  echo "    同时生成 RK_anchors.txt（anchor 值）—— 板端 postprocess.cpp 的 anchor 要与之一致"
  echo "    下一步： 拷到 ../model_convert/model/ 用 1.5.2 容器做 i8 量化"
else
  echo "[✗] 导出失败：未生成 $ONNX（见上方 export failure 报错）。补依赖后重试。"
  exit 1
fi
