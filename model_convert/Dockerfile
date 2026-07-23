# ONNX -> RKNN 转换镜像（rknn-toolkit2 1.5.2，匹配板端 1.5.2 运行时 / 0.8.2 驱动）
# ⚠️ 1.5.2 不在 PyPI，需从 rockchip-linux/rknn-toolkit2 的 v1.5.2 拿 wheel + requirements，
#    放到本目录 wheels/ 下再 build。获取方法见 README「准备 1.5.2 wheel」。
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip \
        libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 \
        libsm6 libgl1-mesa-glx libprotobuf-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# wheels/ 里应包含：rknn_toolkit2-1.5.2*-cp310-cp310-linux_x86_64.whl 和 requirements_cp310-1.5.2.txt
COPY wheels/ /wheels/
# 先按官方 requirements 装依赖(numpy/onnx/protobuf 等都在里面被固定住)，再装 wheel 本体
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /wheels/requirements_cp310-1.5.2.txt
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple /wheels/rknn_toolkit2-1.5.2*.whl

WORKDIR /work
CMD ["bash"]
