# model_train —— YOLOv8 训练容器

「训练 → 上板」链路的**第一环**：训练 YOLOv8 并导出 ONNX。第二环（ONNX→RKNN）见 [`../model_convert/`](../model_convert/)，整体总览见 [`../README.md`](../README.md)。

基于 Ultralytics 官方镜像（内置 CUDA + PyTorch + ultralytics），产物持久化到本目录。

---

## 前置条件

| 项 | 要求 |
|---|---|
| 运行位置 | **WSL2 Ubuntu 终端**，且把 RKmodel 放 WSL2 内部（别在 `/mnt/d/...` 下跑，跨盘 IO 极慢） |
| Docker | Docker Desktop + WSL2 backend |
| GPU（推荐） | Windows 装好 NVIDIA 驱动才能 `--gpus all`；无独显用 `USE_GPU=0` 走 CPU（很慢，仅试链路） |

验证 GPU（可选）：`docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

---

## 目录结构

```
model_train/
├── Dockerfile      # 训练镜像
├── train.sh        # 启动训练
├── export.sh       # 导出标准 ONNX
├── README.md       # 本文件
├── .gitignore
├── datasets/       # (运行时) 数据集自动下载到这里
└── runs/           # (运行时) 训练输出：权重/曲线/结果图
```

---

## 快速开始

```bash
cd RKmodel/model_train

# ① 冒烟测试：8 张图跑 5 轮，几十秒，验证链路
DATA=coco8.yaml EPOCHS=5 bash train.sh

# ② 第一次完整训练：非洲野生动物(4 类, ~1500 图)
bash train.sh

# ③ 导出标准 ONNX（仅 PC 验证用）
bash export.sh
```
首次运行自动 `docker build`（镜像约数 GB）。最优权重：`runs/exp/weights/best.pt`。

---

## 参数（环境变量覆盖）

| 变量 | 默认 | 含义 |
|---|---|---|
| `DATA` | `african-wildlife.yaml` | 数据集 yaml，内置的自动下载，也可指自己的 |
| `MODEL` | `yolov8n.pt` | 基础权重 n/s/m/l/x，多路实时选 **n 或 s** |
| `EPOCHS` | `100` | 训练轮数 |
| `IMGSZ` | `640` | 输入尺寸，**保持 640**（和板端一致） |
| `BATCH` | `16` | 批大小，显存小改小或设 `-1` 自动 |
| `NAME` | `exp` | 输出子目录 `runs/<NAME>` |
| `USE_GPU` | `1` | `1`=GPU，`0`=CPU |

示例：
```bash
DATA=construction-ppe.yaml MODEL=yolov8s.pt EPOCHS=150 BATCH=-1 NAME=ppe bash train.sh
WEIGHTS=runs/ppe/weights/best.pt bash export.sh
```

---

## 推荐数据集（都自动下载，零准备）

| yaml | 规模/类别 | 用途 |
|---|---|---|
| `coco8.yaml` | 8 图 / 80 类 | 冒烟测试链路 |
| `african-wildlife.yaml` | ~1500 图 / 4 类 | **第一次完整训练**，易收敛、结果直观 |
| `construction-ppe.yaml` | ~1400 图 / 11 类(含 helmet/no_helmet/Person/vest) | **贴业务**：一个模型判断"戴没戴帽"，可替换现项目三模型+融合 |
| `VOC.yaml` | ~2 万图 / 20 类 | 中等规模基准 |

自选领域可去 Roboflow Universe 导出 YOLO 格式，放进 `datasets/` 并写自己的 `data.yaml`。

---

## 导出说明（重要）

- `export.sh` 导出的是**标准 ONNX**，只用于在 PC 上验证模型是否正常，**不能直接上板**。
- 上 RK3588 要用 **airockchip 的 ultralytics fork** 导"RKNN 友好 ONNX"，再交给 [`../model_convert/`](../model_convert/) 转 `.rknn`。fork 导出命令与后续步骤见 [`../README.md`](../README.md) 的「阶段②」。

---

## 常见问题

| 现象 | 处理 |
|---|---|
| `bash: train.sh: bad interpreter` / `\r` | CRLF 行尾：`sed -i 's/\r$//' *.sh` |
| `DataLoader worker ... killed` | 共享内存不足，脚本已加 `--ipc=host`；仍不行就把 `BATCH` 调小 |
| `could not select device driver ... gpu` | 未装 NVIDIA 驱动/容器工具；先跑 nvidia-smi 验证，或 `USE_GPU=0` |
| 训练极慢、卡在读图 | 目录在 `/mnt/d/...`，移到 WSL2 内部再跑 |
| OOM / 显存不足 | `BATCH` 调小或设 `-1`，或用 `yolov8n.pt` |
| 看训练结果 | `runs/<NAME>/` 里的 `results.png`、`weights/best.pt`、`val_batch*.jpg` |

---

## 一句话流程

```text
cd model_train → bash train.sh (→ runs/exp/weights/best.pt) → bash export.sh (→ best.onnx)
下一步：airockchip fork 重导 ONNX → ../model_convert 转 .rknn → 上板
```
