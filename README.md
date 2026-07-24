# RKmodel —— 经典 YOLOv5 训练 → RKNN(1.5.2) 转换 → 上板 全流程

用 Docker 把「训练自己的 YOLOv5 模型」到「生成 RK3588 能跑的 `.rknn`」脚本化，分两个隔离的容器。

> **为什么是经典 YOLOv5 + 1.5.2（不是 YOLOv8 + 2.3.2）？**
> - 板端运行时是 **librknnrt 1.5.2**、NPU 驱动 **0.8.2**（不动板子，保证原有 person/helmet/callplay 模型继续能用）→ PC 转换工具最高只能 **1.5.2**。
> - 项目现有后处理是 **经典 anchor 版 YOLOv5**（`postprocess.cpp` 里有 anchor 表）→ 训经典 YOLOv5 **能直接复用这套 C++ 后处理**，几乎不用改板端代码。
> - ⚠️ 用的是 **airockchip/yolov5**（经典 anchor 版），**不是** ultralytics 包里的 `yolov5nu`（那是 anchor-free 的 YOLOv5u，和现有后处理不匹配）。

| 阶段 | 目录 | 干什么 | 工具 |
|---|---|---|---|
| ① 训练+导出 | `model_train/` | 经典 YOLOv5 训练 → `--rknpu` 导 RKNN 友好 ONNX | airockchip/yolov5 |
| ② 转换 | `model_convert/` | ONNX → RKNN（i8 量化） | rknn-toolkit2 **1.5.2** |

## 目录结构

```
RKmodel/
├── README.md
├── model_train/
│   ├── Dockerfile              # 训练镜像(ultralytics 基础镜像 + yolov5 依赖)
│   ├── prepare_dataset.sh      # 下数据集 + 生成 yolov5 格式 yaml
│   ├── train.sh                # 训练经典 YOLOv5
│   ├── export.sh               # 导出 RKNN 友好 ONNX(--rknpu)
│   ├── yolov5/                 # (需自行 git clone airockchip/yolov5)
│   ├── datasets/  runs/        # (运行时生成)
├── model_convert/
│   ├── Dockerfile              # rknn-toolkit2 1.5.2(从本地 wheel 装)
│   ├── prepare_calibration.sh  # 随机选取校准图 + 生成 dataset.txt
│   ├── wheels/                 # 可选：本地放置 1.5.2 的 cp310 wheel
│   ├── convert.py  convert.sh  dataset.txt  model/  calib/
```

## 端到端流程

```
[① model_train]                                  [② model_convert]        [板端 RK3588]
 clone airockchip/yolov5
 prepare_dataset.sh ─► 数据集 + _yolov5_data.yaml
 train.sh ─► best.pt
 export.sh(--rknpu) ─► best.onnx + RK_anchors.txt ──► prepare_calibration.sh + convert.sh ─► best-int8.rknn ─► scp
                                                                                    │
                                            复用现有 YOLOv5 后处理，只改 class_num/anchors/标签
```

---

## 前置条件

- Docker（你这台是 Docker Desktop + WSL 集成，加速/代理在 Docker Desktop 的 Settings 里配）。
- 训练阶段建议有 NVIDIA GPU（`--gpus all`），无 GPU 用 `USE_GPU=0`（慢）。
- 联网：训练下预训练权重/数据集、build 拉镜像，都走你在 Docker Desktop 配好的镜像加速/代理。

---

## 阶段① 训练 + 导出（`model_train/`）

```bash
cd RKmodel/model_train

# 0) 拿经典 anchor 版 YOLOv5(RK 优化 fork)  —— 只需一次
git clone https://github.com/airockchip/yolov5.git

# 1) 下数据集 + 生成 yolov5 格式 yaml(默认 construction-ppe)
bash prepare_dataset.sh
#    生成 _yolov5_data.yaml，打印出来核对一下 names 顺序(=类别索引)

# 2) 训练(经典 YOLOv5，多路实时选 yolov5n/s)
CFG=yolov5s.yaml WEIGHTS=yolov5s.pt EPOCHS=150 NAME=ppe bash train.sh
#    结果：runs/ppe/weights/best.pt

# 3) 导出 RKNN 友好 ONNX(--rknpu，会生成 RK_anchors.txt)
WEIGHTS=runs/ppe/weights/best.pt bash export.sh
#    结果：runs/ppe/weights/best.onnx + yolov5/RK_anchors.txt
```

参数（train.sh 环境变量）：`CFG`(yolov5n/s/m.yaml)、`WEIGHTS`、`EPOCHS`、`IMGSZ`(保持 640)、`BATCH`、`NAME`、`USE_GPU`。

> **想要更好的 i8 精度** → 用 ReLU 激活训练（你项目原来的 `*_relu` 模型就是这么来的）。airockchip/yolov5 支持把激活换成 ReLU（见其 `README_rkopt.md`），按它的说明改模型 cfg 的 activation 即可。默认 SiLU 也能转能跑，只是量化掉点略多。

---

## 阶段② 转换（`model_convert/`，rknn-toolkit2 1.5.2）

### 先准备 1.5.2 的 wheel（一次）
Dockerfile 默认会下载官方 cp310 wheel。如果网络不稳定，也可以预先放到本地：
```bash
cd RKmodel/model_convert && mkdir -p wheels
wget -O wheels/rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl \
  https://raw.githubusercontent.com/rockchip-linux/rknn-toolkit2/v1.5.2/packages/rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl
```
> Ubuntu 22.04 容器使用 Python 3.10，因此必须保留完整的 **cp310-cp310-linux_x86_64** wheel 文件名。

### 转换（i8 量化 + 校准图）
```bash
cd ~/RKmodel/model_convert
mkdir -p model
cp ~/RKmodel/model_train/runs/ppe/weights/best.onnx model/

# 先做非量化模型，验证 ONNX -> RKNN 链路
ONNX=model/best.onnx DTYPE=fp OUTPUT=model/best-fp.rknn bash convert.sh

# 从 construction-ppe/images/train 随机选 200 张并生成 dataset.txt
bash prepare_calibration.sh

# 正式量化，显式指定输出名，避免覆盖 FP 模型
ONNX=model/best.onnx \
DTYPE=i8 \
DATASET=dataset.txt \
OUTPUT=model/best-int8.rknn \
bash convert.sh
```

`prepare_calibration.sh` 默认参数：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SOURCE_DIR` | `../model_train/datasets/construction-ppe/images/train` | 原始图片目录 |
| `COUNT` | `200` | 随机选取数量 |
| `CALIB_DIR` | `calib/construction-ppe-<COUNT>` | 复制后的校准图目录 |
| `DATASET` | `dataset.txt` | 生成的相对路径清单 |

自定义示例：

```bash
SOURCE_DIR=../model_train/datasets/construction-ppe/images/val \
COUNT=300 \
CALIB_DIR=calib/construction-ppe-val-300 \
DATASET=dataset_val_300.txt \
bash prepare_calibration.sh
```

校准图片不需要标签。随机抽取后应人工检查，确保覆盖所有实际类别、远近目标、遮挡、白天/夜间、逆光和少量无目标背景；不要使用大量连续重复帧。目标目录非空或清单已有真实路径时，脚本会停止，避免混入上一版校准集。

---

## 阶段③ 上板（复用现有后处理，改动很小）

```bash
scp model/best-int8.rknn cat@<板子IP>:~/.../project1/.../model/RK3588/
```
板端只需（**不用重写后处理**，因为是经典 YOLOv5）：
1. 换模型文件路径 / 文件名；
2. **改 `class_num`** 为你的类别数（construction-ppe 是 11，或你精简后的数量）；
3. **核对 anchor**：把 `postprocess.cpp` 里的 anchor 表和导出生成的 `RK_anchors.txt` 对上（若你没改 anchor、用默认，一般就是一致的）；
4. 更新标签文件（类别名，顺序=`_yolov5_data.yaml` 里的 names）；
5. 若用单个多类模型替代原三模型，可相应精简 `rknn_infer`/融合逻辑。

---

## 跨阶段要点

1. **版本铁律**：PC 转换 = **1.5.2**，板端运行时 = **1.5.2**，两边必须一致。
2. **经典 YOLOv5 ≠ YOLOv5u**：一定用 `airockchip/yolov5`，别用 ultralytics 包的 `yolov5nu`。
3. **anchor 要对上**：导出的 `RK_anchors.txt` vs 板端 `postprocess.cpp` 的 anchor。
4. **老模型安全**：全程不动板子的运行时/驱动，person/helmet/callplay 照常工作。
5. **脚本行尾**：报 `bad interpreter` 就 `sed -i 's/\r$//' */*.sh`。

## 一句话流程

```text
clone airockchip/yolov5 → prepare_dataset.sh → train.sh → export.sh(--rknpu)
  → prepare_calibration.sh → model_convert(1.5.2, i8) → best-int8.rknn → scp
  → 改 class_num/anchors/标签 → 跑起来
```

---

## 量化前后逐类别 mAP 验证

在同一套带标签的验证集上依次运行 FP RKNN 和 INT8 RKNN：

```bash
cd ~/RKmodel/model_convert

# Toolkit 1.5.2 不能在 x86 模拟器直接 load_rknn；
# 默认从原 ONNX + 同一校准清单分别重建 FP/INT8 后评测。
MODE=rebuild LIMIT=20 bash evaluate_rknn.sh

# 全量验证集，生成正式逐类 AP50 / mAP50-95 对比
MODE=rebuild bash evaluate_rknn.sh
```

如果当前业务只关心 construction-ppe 原始类别 ID `0,1,2,6`：

```bash
CLASS_IDS=0,1,2,6 bash evaluate_rknn.sh
```

输出位于 `model_convert/eval_results/comparison.csv`。默认采用与当前板端一致的
640×640 直接拉伸；只有板端也使用等比例补边时，才设置
`PREPROCESS=letterbox`。校准集不用于报告 mAP，正式结果必须来自完整验证集或
独立测试集。若要直接验证两个最终 `.rknn` 文件，需连接 RK3588 后使用
`MODE=rknn TARGET=rk3588`。详细参数和结果解释见
`model_convert/README.md` 第 12 节。
