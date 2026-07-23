# ONNX -> RKNN 转换镜像（rknn-toolkit2，x86_64 专用）
# 与训练镜像分开：rknn-toolkit2 对 numpy/onnx/protobuf 版本有硬约束，和训练依赖会打架
# 说明见同目录 README.md
FROM ubuntu:22.04

# ⚠️ 该版本必须与 RK3588 板上的 librknnrt.so 版本一致，否则生成的 .rknn 在板上加载失败
ARG RKNN_TOOLKIT_VERSION=2.3.2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip git \
        libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 \
        libsm6 libgl1-mesa-glx libprotobuf-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# rknn-toolkit2 2.3.2 要求 onnx>=1.16.1，但默认会拉到 onnx 1.18+（删了 onnx.mapping 会报错）。
# 固定到 1.16.1：既满足其依赖，又仍保留 onnx.mapping。
RUN pip3 install --no-cache-dir "rknn-toolkit2==${RKNN_TOOLKIT_VERSION}" "onnx==1.16.1"

WORKDIR /work
CMD ["bash"]
