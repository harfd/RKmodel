#!/usr/bin/env bash
# 启动 ONNX -> RKNN 转换（容器内运行 convert.py，产物落到本目录）
# 用法示例：
#   bash convert.sh                                        # 默认转 model/best.onnx 为 i8 量化 rknn
#   ONNX=model/yolov8n.onnx DTYPE=fp bash convert.sh       # 不量化，先验证转换流程
#   RKNN_TOOLKIT_VERSION=2.3.0 bash convert.sh             # 指定与板端匹配的版本重建镜像
set -euo pipefail
cd "$(dirname "$0")"

# ---------- 可配置参数 ----------
IMAGE="${IMAGE:-rknn-convert-152:latest}"               # 1.5.2 版镜像(和板端 1.5.2 运行时对齐)
ONNX="${ONNX:-model/best.onnx}"                          # 输入 ONNX（相对本目录）
PLATFORM="${PLATFORM:-rk3588}"
DTYPE="${DTYPE:-i8}"                                     # i8/u8=量化, fp=不量化
DATASET="${DATASET:-dataset.txt}"                        # 量化校准图清单
OUTPUT="${OUTPUT:-}"                                     # 输出路径，空则同名 .rknn
# --------------------------------

# 镜像不存在则构建
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[*] 构建镜像 $IMAGE (rknn-toolkit2 1.5.2，需先放好 wheels/，见 README)..."
  docker build -t "$IMAGE" .
fi

[ -f "$ONNX" ] || { echo "[!] 找不到 ONNX: $ONNX（先在训练容器导出，见 README）"; exit 1; }
if [ "$DTYPE" != "fp" ] && [ ! -f "$DATASET" ]; then
  echo "[!] 量化需要校准清单 $DATASET（每行一张图路径）。或用 DTYPE=fp 跳过量化。"
  exit 1
fi

OUT_ARG=""
[ -n "$OUTPUT" ] && OUT_ARG="-o $OUTPUT"

echo "[*] onnx=$ONNX platform=$PLATFORM dtype=$DTYPE dataset=$DATASET"
docker run --rm -it -v "$(pwd)":/work -w /work "$IMAGE" \
  python3 convert.py "$ONNX" "$PLATFORM" "$DTYPE" -d "$DATASET" $OUT_ARG

echo "[✓] 完成。把生成的 .rknn 用 scp 拷到 RK3588 项目的 model/ 目录即可。"
