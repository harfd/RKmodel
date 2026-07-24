# ONNX -> RKNN 转换镜像（rknn-toolkit2 1.5.2，匹配板端 1.5.2 运行时 / 0.8.2 驱动）
# 1.5.2 不在 PyPI：构建时从 RKNN_WHL_URL 下载 wheel(cp310)；或把 whl 放到 wheels/ 用本地的。
# github 连不上就换镜像地址： docker build --build-arg RKNN_WHL_URL=<镜像URL> -t ... .
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ARG RKNN_WHL_URL=https://raw.githubusercontent.com/rockchip-linux/rknn-toolkit2/v1.5.2/packages/rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl

# 换成清华 apt 源（官方 archive.ubuntu.com 在国内慢/易 502）
RUN sed -i 's@http://archive.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g; s@http://security.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip curl ca-certificates \
        libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 \
        libsm6 libgl1-mesa-glx libprotobuf-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# 1.5.2 的 packages/ 没有单独 requirements 文件，这里固定住兼容的关键依赖
# （numpy<1.24 保留旧别名；onnx==1.14.1 保留 onnx.mapping；protobuf<4）
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
      "numpy<1.24" "onnx==1.14.1" "protobuf<4" \
      onnxruntime scipy ruamel.yaml psutil requests fast-histogram

# 取 wheel：本地 wheels/ 里有就用本地，否则从 RKNN_WHL_URL 下载
COPY wheels/ /wheels/
RUN set -e; \
    if ls /wheels/rknn_toolkit2-1.5.2*.whl >/dev/null 2>&1; then \
        echo '[*] 使用本地 wheels/ 里的 whl'; \
    else \
        echo "[*] 本地无 whl，下载: $RKNN_WHL_URL"; \
        curl -fL "$RKNN_WHL_URL" -o /wheels/rknn_toolkit2-1.5.2-cp310.whl \
          || { echo '!! 下载失败：github 连不上。改用镜像地址 --build-arg RKNN_WHL_URL=<镜像> 或手动把 whl 放进 wheels/(见 README)'; exit 1; }; \
    fi; \
    pip3 install --no-cache-dir /wheels/rknn_toolkit2-1.5.2*.whl

WORKDIR /work
CMD ["bash"]
