# model_convert —— ONNX → RKNN 1.5.2 转换与量化

本目录使用 x86_64 Docker 容器把经典 YOLOv5 的 RKNN 友好 ONNX 转成 RK3588 可加载的 `.rknn`，支持先生成 FP 模型验证链路，再使用代表性校准集生成 INT8 量化模型。

版本固定关系：

```text
airockchip/yolov5 导出 ONNX opset 12
    ↓
rknn-toolkit2 1.5.2 转换
    ↓
RK3588 板端 librknnrt 1.5.2 / NPU 驱动 0.8.2
```

不要随意升级 PC 侧 Toolkit。板端运行时保持 1.5.2 时，PC 转换工具也必须保持 1.5.2。

## 1. 目录结构

```text
RKmodel/model_convert/
├── Dockerfile                  # rknn-toolkit2 1.5.2 转换镜像
├── convert.py                  # load_onnx → build → export_rknn
├── convert.sh                  # 构建镜像并启动转换
├── prepare_calibration.sh      # 随机选择校准图并生成清单
├── dataset.txt                 # 默认校准清单；初始为说明模板
├── wheels/                     # 可选：本地 cp310 wheel
├── model/                      # 输入 ONNX、输出 RKNN
└── calib/                      # 脚本生成的校准图片
```

`model/`、`calib/`、`wheels/` 和模型产物不应提交到 Git。

## 2. 前置条件

- 在 x86_64 WSL2 Ubuntu 中执行；rknn-toolkit2 转换 wheel 不是 ARM wheel。
- Docker Desktop 已启用 WSL2 集成。
- 输入模型由 `model_train/export.sh` 导出，必须是 RKNN 友好的经典 YOLOv5 ONNX。
- ONNX 主域 opset 必须不大于 12。`model_train/export.sh` 已强制 `dynamo=False`、`--opset 12` 并在导出后校验。
- 板端 `librknnrt.so` 为 1.5.2。

## 3. 准备 RKNN Toolkit wheel

Dockerfile 默认从 Rockchip 仓库下载：

```text
rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl
```

如果构建时网络不稳定，可以手动放入 `wheels/`：

```bash
cd ~/RKmodel/model_convert
mkdir -p wheels

wget -O wheels/rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl \
  https://raw.githubusercontent.com/rockchip-linux/rknn-toolkit2/v1.5.2/packages/rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl
```

必须保留完整 wheel 文件名；不能缩短成 `rknn_toolkit2-1.5.2-cp310.whl`。

## 4. 放入 ONNX

以训练任务 `ppe2` 为例：

```bash
cd ~/RKmodel/model_convert
mkdir -p model

cp ../model_train/runs/ppe2/weights/best.onnx model/best.onnx
```

如需检查 opset：

```bash
docker run --rm \
  -v "$(pwd)":/work \
  -w /work \
  rknn-convert-152:latest \
  python3 -c 'import onnx; m=onnx.load("model/best.onnx"); print([(x.domain or "ai.onnx", x.version) for x in m.opset_import])'
```

主域应显示 opset 12。

## 5. 先生成 FP 模型

第一次转换先关闭量化，用于确认 ONNX、算子和 Toolkit 链路正常：

```bash
cd ~/RKmodel/model_convert

ONNX=model/best.onnx \
DTYPE=fp \
OUTPUT=model/best-fp.rknn \
bash convert.sh
```

成功后得到：

```text
model/best-fp.rknn
```

## 6. 选择量化校准集

当前 construction-ppe 图片目录为：

```text
../model_train/datasets/construction-ppe/images/train
../model_train/datasets/construction-ppe/images/val
```

默认从训练集随机选取 200 张：

```bash
cd ~/RKmodel/model_convert
bash prepare_calibration.sh
```

默认生成：

```text
calib/construction-ppe-200/0001.jpg
calib/construction-ppe-200/0002.jpg
...
dataset.txt
```

`dataset.txt` 使用相对于 `model_convert` 的路径，例如：

```text
calib/construction-ppe-200/0001.jpg
calib/construction-ppe-200/0002.jpg
```

这些相对路径在转换容器中对应 `/work/calib/...`。不要把 `/home/<用户>/...` 绝对路径写入清单，因为 `convert.sh` 只把当前 `model_convert` 目录挂载为 `/work`。

### 脚本参数

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SOURCE_DIR` | `../model_train/datasets/construction-ppe/images/train` | 校准图来源 |
| `COUNT` | `200` | 随机选择数量，必须为正整数 |
| `CALIB_DIR` | `calib/construction-ppe-<COUNT>` | 复制后的图片目录，必须在本目录内 |
| `DATASET` | `dataset.txt` | 输出清单，必须在本目录内 |

从验证集选择 300 张并使用独立文件名：

```bash
SOURCE_DIR=../model_train/datasets/construction-ppe/images/val \
COUNT=300 \
CALIB_DIR=calib/construction-ppe-val-300 \
DATASET=dataset_val_300.txt \
bash prepare_calibration.sh
```

为了防止旧图片混入新校准集：

- `CALIB_DIR` 已经非空时，脚本会停止。
- `DATASET` 已经包含真实路径时，脚本会停止。
- 需要重新抽样时，请换一个 `CALIB_DIR` 和 `DATASET` 名称。

### 校准集选取原则

- 建议先使用 150～300 张真实部署场景图片。
- 覆盖全部类别、远近目标、大小目标、遮挡、不同背景和摄像头角度。
- 覆盖白天、夜间、逆光、阴影以及实际会出现的曝光范围。
- 可以包含少量无目标背景，但比例应接近真实部署情况。
- 不需要标签文件；Toolkit 只读取图片统计张量分布。
- 避免大量连续视频相邻帧、重复图、全黑图、损坏图和无关场景。
- 校准集不要代替独立测试集；量化后仍需使用独立验证集比较精度。
- 校准图片的预处理应尽量与板端输入一致。如果板端先做 640×640 letterbox，应确认校准阶段和板端的缩放、颜色顺序、均值和归一化一致。

## 7. 生成 INT8 量化模型

使用默认 `dataset.txt`：

```bash
cd ~/RKmodel/model_convert

ONNX=model/best.onnx \
DTYPE=i8 \
DATASET=dataset.txt \
OUTPUT=model/best-int8.rknn \
bash convert.sh
```

使用自定义清单：

```bash
ONNX=model/best.onnx \
DTYPE=i8 \
DATASET=dataset_val_300.txt \
OUTPUT=model/best-int8-val300.rknn \
bash convert.sh
```

成功日志应包含：

```text
load_onnx
build: quant=True dataset=...
export_rknn: model/best-int8.rknn
```

当前 `convert.py` 中的 `i8` 和 `u8` 都只是打开 `do_quantization=True`，没有分别设置不同的 `quantized_dtype`；现阶段统一使用 `DTYPE=i8`。

## 8. 参数说明

| 变量 | 默认值 | 说明 |
|---|---|---|
| `IMAGE` | `rknn-convert-152:latest` | 转换镜像名 |
| `ONNX` | `model/best.onnx` | 输入 ONNX，相对本目录 |
| `PLATFORM` | `rk3588` | 目标平台 |
| `DTYPE` | `i8` | `fp` 为非量化；`i8`/`u8` 打开量化 |
| `DATASET` | `dataset.txt` | 量化校准清单 |
| `OUTPUT` | 空 | 输出路径；为空时覆盖 ONNX 同名 `.rknn` |
| `RKNN_WHL_URL` | Dockerfile 内官方地址 | 自动下载 wheel 的地址 |

建议始终显式设置 `OUTPUT`，分别保留 FP 和 INT8 模型。

## 9. 量化后验证

将两个模型都部署到 RK3588，使用同一批独立测试图片或视频比较：

- 各类别检出数量、漏检和误检；
- 置信度变化；
- 小目标、远距离目标和遮挡场景；
- Precision、Recall、mAP（有标注数据时）；
- 推理耗时和内存占用。

如果 INT8 精度下降明显：

1. 删除重复、模糊或无关校准图；
2. 增加漏检场景、暗光、小目标和稀有类别图片；
3. 将校准集从 200 张增加到 300～500 张；
4. 检查板端预处理与 `convert.py` 中 `mean_values=[[0,0,0]]`、`std_values=[[255,255,255]]` 是否一致；
5. 对比 FP 模型，确认问题确实来自量化而不是 ONNX 或后处理。

## 10. 上板

```bash
scp model/best-int8.rknn cat@<板子IP>:~/.../project1/.../model/RK3588/
```

板端还需要核对：

- 模型路径；
- `class_num`；
- 标签顺序；
- `RK_anchors.txt` 与 `postprocess.cpp` 中 anchors；
- 输出张量量化参数的反量化处理。

## 11. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `not a valid wheel filename` | wheel 被改成了不完整文件名，恢复完整 cp310/ABI/平台标签 |
| `Unsupported onnx opset 18, need <= 12` | 使用 `model_train/export.sh` 重新导出，必须 opset 12 |
| `dataset.txt` 为空或找不到图片 | 运行 `prepare_calibration.sh`，并确保清单使用 `/work` 下可见的相对路径 |
| `CALIB_DIR 已非空` | 换一个新的目标目录，防止混入上一版图片 |
| FP 成功但量化 `build` 失败 | 检查每个清单路径、图片格式和损坏文件 |
| INT8 精度下降明显 | 改善校准集代表性并检查预处理一致性 |
| 板端 `rknn_init` 失败 | PC Toolkit 和板端 `librknnrt.so` 版本不一致 |

## 12. 比较量化前后逐类别 mAP

`accuracy_analysis` 分析的是 RKNN 各层张量误差，不是目标检测的 mAP。逐类别
mAP 必须让两个模型在**同一套带 YOLO 标签的验证集**上完整推理，再用相同的
预处理、anchors、置信度阈值和 NMS 统计。

本目录已提供：

```text
evaluate_rknn.py   # RKNN 推理、YOLOv5 后处理、逐类 AP/mAP 统计
evaluate_rknn.sh   # 复用转换镜像、挂载模型/验证集并生成对比结果
```

### 为什么不能直接在 x86 模拟器加载两个 `.rknn`

rknn-toolkit2 1.5.2 的 `load_rknn()` 不支持模拟器推理，直接调用会报：

```text
RKNN model that loaded by 'load_rknn' not support inference on the simulator
```

因此脚本提供两种可靠模式：

1. `MODE=rebuild`（默认自动选择）：读取原 ONNX，在同一 Python 进程中分别
   `build(do_quantization=False/True)`，随即用 Toolkit 模拟器评测。INT8 必须使用
   正式转换时的同一份校准清单。这是纯 x86/WSL 下最方便的量化损失对比。
2. `MODE=rknn`：连接实际 RK3588，直接运行现成的 `best-fp.rknn` 和
   `best-int8.rknn`。这是验证两个最终文件的严格方式。

离线模式默认文件：

```text
model/best.onnx
dataset.txt
../model_train/_yolov5_data.yaml
../model_train/datasets/construction-ppe
```

先用少量图片检查推理链路：

```bash
cd ~/RKmodel/model_convert
MODE=rebuild LIMIT=20 bash evaluate_rknn.sh
```

`LIMIT` 非零时不能作为正式 mAP。确认输出形状、类别数和 anchors 均无报错后，
使用全部验证集：

```bash
MODE=rebuild bash evaluate_rknn.sh
```

如需直接验证两个最终 RKNN 文件，把 RK3588 以 Toolkit 支持的 USB/ADB 方式连接
到转换主机后运行：

```bash
MODE=rknn \
TARGET=rk3588 \
DEVICE_ID=<多设备时填写，单设备可省略> \
FP_MODEL=model/best-fp.rknn \
INT8_MODEL=model/best-int8.rknn \
bash evaluate_rknn.sh
```

如果模型只放在同级工作区的 `project1/model/`，脚本会自动尝试其中的
`best-fp.rknn`、`best-int8.rknn` 和 `RK_anchors.txt`。

当前板端通过 RGA 直接缩放到 640×640，所以默认使用 `PREPROCESS=stretch`。
如果部署代码改成等比例补黑边，则评测也必须改为：

```bash
PREPROCESS=letterbox bash evaluate_rknn.sh
```

若模型仍是 construction-ppe 的 11 类输出，但项目目前只显示原始类别
`0,1,2,6`（helmet、gloves、vest、Person），可只汇总这四类：

```bash
CLASS_IDS=0,1,2,6 bash evaluate_rknn.sh
```

这只是选择统计类别，不会把 11 类模型误当成 4 类解码。类别 ID 必须以训练生成的
`_yolov5_data.yaml` 为准，不能用只有四行的显示标签文件替代完整 names。

其他常用覆盖参数：

```bash
MODE=rebuild \
ONNX=model/best.onnx \
CALIBRATION=dataset.txt \
ANCHORS=/path/to/RK_anchors.txt \
DATA_YAML=../model_train/_yolov5_data.yaml \
DATASET_ROOT=../model_train/datasets/construction-ppe \
OUTPUT_DIR=eval_results \
bash evaluate_rknn.sh
```

结果写入：

```text
eval_results/fp_metrics.csv
eval_results/int8_metrics.csv
eval_results/comparison.csv
eval_results/metrics.json
```

`comparison.csv` 中 `delta = INT8 - FP`。重点看每个类别的 `AP50`、
`mAP50-95` 及其下降量；`targets=0` 的类别没有验证标注，结果留空且不参与平均。
默认 `CONF=0.001`、`NMS=0.65` 是为 mAP 保留低分候选，不能照搬线上显示阈值。

校准集只用于统计 INT8 张量分布，正式精度不能在校准集上报告；应使用训练过程中
未参与梯度更新的完整 `images/val + labels/val`，更严格时另留独立 test 集。
`MODE=rebuild` 虽然需要校准集来重新 build INT8，但计算 AP 的图片仍由
`data.yaml` 的 `val` 指定，两者用途不会混在一起。
