# RKmodel —— YOLOv8 训练 → RKNN 转换 → 上板 全流程

用 Docker 把「训练自己的 YOLOv8 模型」到「生成 RK3588 能跑的 `.rknn`」这条链路脚本化，分两个互相隔离的容器：

| 阶段 | 目录 | 干什么 | 依赖 |
|---|---|---|---|
| ① 训练 | [`model_train/`](model_train/) | 训练 YOLOv8 + 导出 ONNX | 新版 PyTorch/CUDA（ultralytics 官方镜像） |
| ② 转换 | [`model_convert/`](model_convert/) | ONNX → RKNN（int8 量化） | rknn-toolkit2（旧 numpy/onnx） |

> 为什么分两个容器：训练依赖和 rknn-toolkit2 的依赖版本互相打架，隔离开最省心。

## 目录结构

```
RKmodel/
├── README.md                # 本文件(总览)
├── model_train/             # 阶段① 训练 + 导出 ONNX
│   ├── Dockerfile           #   训练镜像(基于 ultralytics 官方镜像)
│   ├── train.sh             #   启动训练
│   ├── export.sh            #   导出标准 ONNX
│   ├── .gitignore
│   ├── datasets/            #   (运行时) 数据集自动下载到这里
│   └── runs/                #   (运行时) 训练输出：权重/曲线/结果图
└── model_convert/           # 阶段② ONNX → RKNN
    ├── Dockerfile           #   rknn-toolkit2 镜像
    ├── convert.py           #   转换脚本
    ├── convert.sh           #   启动转换
    ├── dataset.txt          #   量化校准图清单(模板)
    ├── README.md            #   转换阶段详细说明
    └── .gitignore
```

## 端到端流程

```
[阶段①] model_train                         [阶段②] model_convert            [板端]
 数据集 --train.sh--> best.pt                  best.onnx --convert.sh-->        RK3588
                       │                        (rknn-toolkit2 量化)            │
                       ├─ export.sh ─> 标准ONNX(仅PC验证)                        │
                       └─ airockchip fork ─> RKNN友好ONNX ──拷入 model_convert──┘
                                                            └─> best.rknn --scp--> 换后处理跑起来
```

---

## 前置条件（两个阶段通用）

| 项 | 要求 |
|---|---|
| WSL2 | 必须能正常启动（`HypervisorPresent` 为 True）。起不来先修虚拟化（`bcdedit /set hypervisorlaunchtype auto` + 启用"虚拟机平台"+重启） |
| Docker Desktop | 开启 WSL2 backend、启用 Ubuntu 集成 |
| 运行位置 | 在 **WSL2 的 Ubuntu 终端**里跑，**且把 RKmodel 放到 WSL2 内部**（如 `~/RKmodel`），别在 `/mnt/d/...` 下跑——跨盘 IO 极慢 |
| GPU（阶段①推荐） | Windows 装好 NVIDIA 驱动才能 `--gpus all`；无独显用 `USE_GPU=0` 走 CPU（很慢，仅试链路）。阶段②纯 CPU，无需 GPU |

验证 GPU 直通（可选）：
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## 阶段① 训练（`model_train/`）

```bash
cd RKmodel/model_train

# 冒烟测试：8 张图跑 5 轮，几十秒，只验证链路通不通
DATA=coco8.yaml EPOCHS=5 bash train.sh

# 第一次完整训练：非洲野生动物(4 类, ~1500 图)，GPU 上几十分钟
bash train.sh

# 训练完导出标准 ONNX（仅用于 PC 上验证模型，不直接上板）
bash export.sh
```
首次运行自动 `docker build`（镜像约数 GB）。最优权重在 `model_train/runs/exp/weights/best.pt`。

### 参数（环境变量覆盖 train.sh）

| 变量 | 默认 | 含义 |
|---|---|---|
| `DATA` | `african-wildlife.yaml` | 数据集 yaml，内置的自动下载，也可指自己的 |
| `MODEL` | `yolov8n.pt` | 基础权重 n/s/m/l/x，多路实时选 **n 或 s** |
| `EPOCHS` | `100` | 训练轮数 |
| `IMGSZ` | `640` | 输入尺寸，**保持 640**（和板端项目一致） |
| `BATCH` | `16` | 批大小，显存小改小或设 `-1` 自动 |
| `NAME` | `exp` | 输出子目录 `runs/<NAME>` |
| `USE_GPU` | `1` | `1`=GPU，`0`=CPU |

### 推荐数据集（都自动下载）

| yaml | 规模/类别 | 用途 |
|---|---|---|
| `coco8.yaml` | 8 图 / 80 类 | 冒烟测试链路 |
| `african-wildlife.yaml` | ~1500 图 / 4 类 | **第一次完整训练**，易收敛 |
| `construction-ppe.yaml` | ~1400 图 / 11 类(含 helmet/no_helmet/Person/vest) | **贴业务**：一个模型判断"戴没戴帽"，可替换现项目三模型+融合 |
| `VOC.yaml` | ~2 万图 / 20 类 | 中等规模基准 |

---

## 阶段② 转换（`model_convert/`）

细节见 [`model_convert/README.md`](model_convert/README.md)，要点：

```bash
cd RKmodel/model_convert
mkdir -p model calib

# 1) 放入"RKNN 友好 ONNX"(用 airockchip fork 导，见下方) 到 model/
# 2) 先不量化试通链路
ONNX=model/best.onnx DTYPE=fp bash convert.sh
# 3) 放校准图 + 填 dataset.txt 后，正式 int8 量化
ONNX=model/best.onnx DTYPE=i8 bash convert.sh
```
产物 `model/best.rknn`。

**⚠️ 输入 ONNX 必须用 airockchip fork 导**（标准 ONNX 上板量化不好且后处理不匹配）。在训练镜像里临时装 fork 导一次：
```bash
cd RKmodel/model_train
docker run --rm -it -v "$(pwd)":/workspace -w /workspace yolo-train:latest bash -lc "
  pip install -q 'git+https://github.com/airockchip/ultralytics_yolov8.git' &&
  yolo export model=runs/exp/weights/best.pt format=onnx imgsz=640
"
# 得到的 onnx 拷到 ../model_convert/model/
```

**⚠️ rknn-toolkit2 版本要与板上 `librknnrt.so` 一致**：`RKNN_TOOLKIT_VERSION=2.3.2 bash convert.sh`。

---

## 完整跑一遍（从零到 .rknn）

```bash
# 阶段①：训练 + 导出 RKNN 友好 ONNX
cd RKmodel/model_train
DATA=construction-ppe.yaml MODEL=yolov8n.pt EPOCHS=150 NAME=ppe bash train.sh
docker run --rm -it -v "$(pwd)":/workspace -w /workspace yolo-train:latest bash -lc "
  pip install -q 'git+https://github.com/airockchip/ultralytics_yolov8.git' &&
  yolo export model=runs/ppe/weights/best.pt format=onnx imgsz=640"

# 阶段②：转 RKNN
cd ../model_convert && mkdir -p model calib
cp ../model_train/runs/ppe/weights/best.onnx model/
# (往 calib/ 放校准图并填好 dataset.txt)
RKNN_TOOLKIT_VERSION=<与板端一致> DTYPE=i8 bash convert.sh

# 上板
scp model/best.rknn cat@<板子IP>:~/.../project1/.../model/RK3588/
```

上板后别忘了：**换 YOLOv8 后处理**（现项目是 YOLOv5 anchor 版，不通用，从 `rknn_model_zoo/examples/yolov8` 搬 C++ 后处理）、调 `rknn_lite` 的 `class_num`、更新标签文件。

---

## 跨阶段要牢记的坑

1. **WSL2 必须正常** + **RKmodel 放 WSL 内部**（不放 `/mnt/d`），否则训练极慢。
2. **CRLF 行尾**：脚本报 `bad interpreter` 就 `sed -i 's/\r$//' */*.sh`。
3. **版本对齐**：rknn-toolkit2 版本 == 板上 `librknnrt.so` 版本。
4. **导出要用 fork**：上板的 ONNX 用 airockchip fork，不是标准导出。
5. **后处理要换**：YOLOv8 anchor-free ≠ 现项目 YOLOv5 anchor。
6. **许可证**：Ultralytics YOLOv8 是 AGPL-3.0，商用需注意授权。

## 一句话流程

```text
model_train:  数据集 → train.sh → best.pt →(airockchip fork)→ RKNN友好 ONNX
model_convert: ONNX →(对齐版本 + 量化)→ best.rknn
板端:          scp → 换 YOLOv8 后处理 → 跑起来
```
