# YOLOv8 训练镜像
# 基于 Ultralytics 官方 GPU 镜像（已内置 CUDA + PyTorch + ultralytics + yolo CLI）
# 说明见同目录 README.md
FROM ultralytics/ultralytics:latest

# 导出 ONNX 所需组件 + 经典 YOLOv5(airockchip/yolov5) 训练需要的额外依赖
# 基础镜像已含 torch/opencv 等；这里补 yolov5 需要但基础镜像可能缺的(不动 torch)
RUN pip install --no-cache-dir onnx onnxslim onnxruntime onnxscript \
        tqdm seaborn thop gitpython psutil pyyaml requests matplotlib pandas

# 训练工作区：数据集 / 基础权重 / 训练输出 都落在挂载进来的 /workspace，持久化到宿主机
WORKDIR /workspace
# 把 ultralytics 的配置(含 datasets_dir 设置)也放进挂载目录，避免每次容器重建丢配置
ENV YOLO_CONFIG_DIR=/workspace/.ultralytics

# 默认进 bash；实际训练/导出由 train.sh / export.sh 通过 docker run 传入命令
CMD ["bash"]
