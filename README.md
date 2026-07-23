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
│   ├── wheels/                 # (需自行放 1.5.2 的 whl + requirements)
│   ├── convert.py  convert.sh  dataset.txt  model/  calib/
```

## 端到端流程

```
[① model_train]                                  [② model_convert]        [板端 RK3588]
 clone airockchip/yolov5
 prepare_dataset.sh ─► 数据集 + _yolov5_data.yaml
 train.sh ─► best.pt
 export.sh(--rknpu) ─► best.onnx + RK_anchors.txt ──► convert.sh(i8, 1.5.2) ─► best.rknn ──► scp
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
1.5.2 不在 PyPI，从仓库拿：
```bash
cd RKmodel/model_convert && mkdir -p wheels
# 挂代理/镜像 clone，checkout v1.5.2，把 wheel 和 requirements 拷进 wheels/
git clone https://github.com/rockchip-linux/rknn-toolkit2.git /tmp/rk152
cd /tmp/rk152 && git checkout tags/v1.5.2 -b v1.5.2
cp packages/rknn_toolkit2-1.5.2*cp310*.whl  packages/requirements_cp310-1.5.2.txt  <RKmodel路径>/model_convert/wheels/
```
> Ubuntu22.04 容器是 python3.10 → 用 **cp310** 的 whl。若 1.5.2 只提供 cp38，把 Dockerfile 基础镜像换成 `ubuntu:20.04` 并改用 cp38 的 whl+requirements。

### 转换（i8 量化 + 校准图）
```bash
cd RKmodel/model_convert && mkdir -p model calib
cp ~/RKmodel/model_train/runs/ppe/weights/best.onnx model/

# 从训练集挑 ~200 张当校准图
cp $(ls ~/RKmodel/model_train/datasets/construction-ppe/images/train/*.jpg | shuf | head -200) calib/
ls calib/*.jpg > dataset.txt

ONNX=model/best.onnx DTYPE=i8 bash convert.sh    # 首次自动 build 1.5.2 镜像
#   产物：model/best.rknn
```
（先 `DTYPE=fp` 跑一遍验证转换本身没问题，再 `i8` 量化。）

---

## 阶段③ 上板（复用现有后处理，改动很小）

```bash
scp model/best.rknn cat@<板子IP>:~/.../project1/.../model/RK3588/
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
  → model_convert(1.5.2, i8) → best.rknn → scp → 改 class_num/anchors/标签 → 跑起来
```
