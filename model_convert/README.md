# model_convert —— ONNX → RKNN 转换容器

把 YOLOv8 的 ONNX 转成 RK3588 能跑的 `.rknn`（含 int8 量化），用 **rknn-toolkit2** 独立容器完成。
这是「训练 → 上板」链路的**第二环**；第一环（训练 + 导出 ONNX）在训练容器 `model_train`（见其 README）。

```
best.pt --(model_train + airockchip fork)--> RKNN友好ONNX --(本容器 rknn-toolkit2)--> best.rknn --scp--> RK3588
```

> 为什么单独一个容器：rknn-toolkit2 对 numpy/onnx/protobuf 版本要求很旧，和训练用的新 PyTorch 环境装一起必冲突，所以隔离开。

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 运行环境 | **x86_64** 的 WSL2 Ubuntu（rknn-toolkit2 只支持 x86，不能在 ARM/板子上跑） |
| Docker | Docker Desktop + WSL2 backend（**这一步不需要 GPU**，纯 CPU 转换） |
| 版本匹配 | rknn-toolkit2 版本**必须与板上 `librknnrt.so` 一致**，见第 4 节 |

---

## 2. 输入的 ONNX 从哪来（关键）

**不要直接用 `model_train` 里 `export.sh` 导出的"标准 ONNX"上板**——那个后处理结构和 RKNN 不匹配。要用 **airockchip 的 ultralytics fork** 导出"RKNN 友好 ONNX"。

在**训练容器**里临时装上 fork 导一次即可（在 `model_train/` 目录）：

```bash
# 进 model_train 的训练镜像，装 airockchip fork 后导出
docker run --rm -it -v "$(pwd)":/workspace -w /workspace yolo-train:latest bash -lc "
  pip install -q 'git+https://github.com/airockchip/ultralytics_yolov8.git' &&
  yolo export model=runs/exp/weights/best.pt format=onnx imgsz=640
"
```
> fork 的导出会把检测头改成 RKNN 友好结构（去掉不好量化的子图）。具体命令以
> [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) 的 `examples/yolov8/README` 为准（不同版本略有差异）。

把得到的 `best.onnx` 拷进本目录的 `model/` 下，即可转换。
也可以先用官方 `rknn_model_zoo` 里预导出的 `yolov8n.onnx` 试通链路。

---

## 3. 目录结构

```
RKmodel/model_convert/
├── Dockerfile      # rknn-toolkit2 转换镜像
├── convert.py      # 转换脚本(load_onnx -> build+量化 -> export_rknn)
├── convert.sh      # 启动转换
├── dataset.txt     # 量化校准图清单(模板，需自己填)
├── README.md       # 本文件
├── .gitignore
├── model/          # (自建) 放输入 onnx / 输出 rknn
└── calib/          # (自建) 放量化校准图片
```

---

## 4. 先对齐版本（否则白转）

在 **RK3588 板子上**查运行时版本：
```bash
strings /usr/lib/librknnrt.so | grep -i "librknnrt version"   # 或看你项目里的 .so
```
拿到版本号（如 `2.3.2`）后，转换时用同一版本构建镜像：
```bash
RKNN_TOOLKIT_VERSION=2.3.2 bash convert.sh
```
版本不一致时，生成的 `.rknn` 在板上 `rknn_init` 会失败。

---

## 5. 使用步骤

```bash
# 在 x86 WSL2 Ubuntu 终端，进入本目录
cd ~/RKmodel/model_convert        # 或你放置的位置
mkdir -p model calib

# ① 放入 RKNN 友好 ONNX
cp /path/to/best.onnx model/

# ② 先不量化试通链路（快，验证转换本身没问题）
ONNX=model/best.onnx DTYPE=fp bash convert.sh

# ③ 正式量化：先往 calib/ 放 100~300 张校准图，编辑 dataset.txt 列出它们
#    然后：
ONNX=model/best.onnx DTYPE=i8 bash convert.sh
```
产物在 `model/best.rknn`。首次运行会自动 `docker build`。

---

## 6. 参数说明（环境变量覆盖）

| 变量 | 默认 | 含义 |
|---|---|---|
| `ONNX` | `model/best.onnx` | 输入 ONNX（相对本目录） |
| `PLATFORM` | `rk3588` | 目标平台 |
| `DTYPE` | `i8` | `i8`/`u8`=量化，`fp`=不量化 |
| `DATASET` | `dataset.txt` | 量化校准图清单 |
| `OUTPUT` | 空 | 输出路径，空则与输入同名 `.rknn` |
| `RKNN_TOOLKIT_VERSION` | `2.3.2` | rknn-toolkit2 版本，**与板端对齐** |

---

## 7. 上板部署

```bash
# 把 rknn 拷到板子项目的 model 目录
scp model/best.rknn cat@<板子IP>:~/.../project1/.../model/RK3588/
```
然后在 RK3588 上：
- 换掉模型路径（或文件名）；
- ⚠️ **后处理也要换**：现项目是 YOLOv5 anchor 版后处理，YOLOv8 是 anchor-free，
  直接从 `rknn_model_zoo/examples/yolov8` 的 C/C++ 后处理搬过来替换 `post_process`；
- 调 `rknn_lite` 的 `class_num` 为你的类别数，更新标签文件。

---

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| 板上 `rknn_init` 失败 / 版本报错 | rknn-toolkit2 版本与 `librknnrt.so` 不一致，见第 4 节重建镜像 |
| `build failed` 且用了量化 | `dataset.txt` 为空或路径错；先用 `DTYPE=fp` 排除是不是量化问题 |
| 量化后精度掉很多 | 校准图不够/不具代表性，多放些真实场景图；或先 `fp` 验证是模型本身 OK |
| `bash: convert.sh: bad interpreter` | CRLF 行尾：`sed -i 's/\r$//' *.sh` |
| 拉不动 rknn-toolkit2 | pip 源问题，或换 `RKNN_TOOLKIT_VERSION`；也可从 Rockchip 官方 whl 装 |
| 转换报不支持的算子 | ONNX 不是 airockchip fork 导的（用了标准导出），见第 2 节重导 |

---

## 9. 一句话流程

```text
(model_train) best.pt --airockchip fork--> best.onnx
   ↓ 拷进 model/
(本容器) 对齐版本 → DTYPE=fp 试通 → 放校准图+dataset.txt → DTYPE=i8 量化 → best.rknn
   ↓ scp 到板子 + 换 YOLOv8 后处理
RK3588 跑起来
```
