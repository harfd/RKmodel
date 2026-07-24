#!/usr/bin/env python3
"""在同一 YOLO 验证集上比较 FP 与 INT8 RKNN 的逐类 mAP。"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import yaml

    def load_yaml(stream):
        return yaml.safe_load(stream)

except ImportError:
    # rknn-toolkit2 1.5.2 的既有镜像默认安装 ruamel.yaml。
    from ruamel.yaml import YAML

    def load_yaml(stream):
        return YAML(typ="safe").load(stream)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def parse_args():
    parser = argparse.ArgumentParser(
        description="比较两个 YOLOv5 RKNN 模型的逐类 AP50 和 mAP50-95"
    )
    parser.add_argument("--fp-model", default="", help="非量化 RKNN（连接板端评测模式）")
    parser.add_argument("--int8-model", default="", help="INT8 RKNN（连接板端评测模式）")
    parser.add_argument(
        "--onnx",
        default="",
        help="离线模式输入 ONNX；在同一进程分别 build FP/INT8 后使用模拟器评测",
    )
    parser.add_argument(
        "--calibration",
        default="",
        help="离线 INT8 build 使用的校准清单，必须与正式转换使用同一份",
    )
    parser.add_argument("--platform", default="rk3588", help="离线 build 的目标平台")
    parser.add_argument("--data", required=True, help="YOLO data.yaml，用于读取 val 和 names")
    parser.add_argument(
        "--dataset-root",
        default="",
        help="数据集根目录覆盖值；推荐显式指定，以避开 data.yaml 中的容器绝对路径",
    )
    parser.add_argument("--anchors", required=True, help="RK_anchors.txt，包含 18 个数")
    parser.add_argument("--imgsz", type=int, default=640, help="模型方形输入尺寸")
    parser.add_argument(
        "--preprocess",
        choices=("stretch", "letterbox"),
        default="stretch",
        help="stretch 与当前板端 RGA 直接缩放一致；letterbox 为等比例黑边",
    )
    parser.add_argument("--conf", type=float, default=0.001, help="mAP 统计前最低置信度")
    parser.add_argument("--nms", type=float, default=0.65, help="逐类别 NMS IoU 阈值")
    parser.add_argument("--max-det", type=int, default=300, help="每张图片最大检测数")
    parser.add_argument(
        "--class-ids",
        default="",
        help="只在结果中显示指定原始类别 ID，例如 0,1,2,6；模型仍按完整类别数解码",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评测前 N 张；0=全部。非零只适合冒烟测试，不能作为正式 mAP",
    )
    parser.add_argument("--output-dir", default="eval_results", help="CSV/JSON 输出目录")
    parser.add_argument(
        "--target",
        default="",
        help="评测现成 RKNN 文件时必须填 rk3588，并连接实际板端",
    )
    parser.add_argument("--device-id", default="", help="连接多块板时指定 device_id")
    args = parser.parse_args()

    for name in ("data", "anchors"):
        path = Path(getattr(args, name))
        if not path.is_file():
            parser.error(f"找不到文件: {path}")
    if args.onnx:
        if not Path(args.onnx).is_file():
            parser.error(f"找不到 ONNX: {args.onnx}")
        if not args.calibration or not Path(args.calibration).is_file():
            parser.error(f"离线 INT8 评测需要校准清单: {args.calibration or '<未设置>'}")
    else:
        for name in ("fp_model", "int8_model"):
            path = Path(getattr(args, name))
            if not path.is_file():
                parser.error(f"找不到文件: {path}")
        if not args.target:
            parser.error(
                "rknn-toolkit2 1.5.2 不能在模拟器中直接运行 load_rknn 的模型。"
                "请提供 --onnx/--calibration 做离线重建评测，"
                "或设置 --target rk3588 连接实际板端。"
            )
    if args.dataset_root and not Path(args.dataset_root).is_dir():
        parser.error(f"找不到数据集根目录: {args.dataset_root}")
    if args.imgsz <= 0 or args.max_det <= 0 or args.limit < 0:
        parser.error("imgsz/max-det 必须为正数，limit 必须大于等于 0")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 < args.nms <= 1.0:
        parser.error("conf 必须在 [0,1]，nms 必须在 (0,1]")
    return args


def load_dataset_config(data_path):
    data_path = Path(data_path).resolve()
    with data_path.open("r", encoding="utf-8") as stream:
        config = load_yaml(stream) or {}

    names_value = config.get("names")
    if isinstance(names_value, dict):
        try:
            ordered = sorted(names_value.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError):
            raise ValueError("data.yaml 的 names 字典键必须是类别 ID")
        names = [str(value) for _, value in ordered]
    elif isinstance(names_value, list):
        names = [str(value) for value in names_value]
    else:
        nc = int(config.get("nc", 0))
        if nc <= 0:
            raise ValueError("data.yaml 必须提供 names 或正整数 nc")
        names = [f"class_{index}" for index in range(nc)]

    nc = int(config.get("nc", len(names)))
    if nc != len(names):
        raise ValueError(f"data.yaml 中 nc={nc}，但 names 有 {len(names)} 项")
    if "val" not in config:
        raise ValueError("data.yaml 缺少 val")
    return data_path, config, names


def resolve_dataset_root(data_path, config, override):
    if override:
        return Path(override).resolve()

    configured = Path(str(config.get("path", ".")))
    if not configured.is_absolute():
        configured = (data_path.parent / configured).resolve()
    if not configured.is_dir():
        raise FileNotFoundError(
            f"data.yaml 的 path 在当前容器不可见: {configured}\n"
            "请用 --dataset-root 显式指定实际数据集目录。"
        )
    return configured


def expand_image_source(source, dataset_root):
    source = Path(str(source))
    if not source.is_absolute():
        source = dataset_root / source
    source = source.resolve()

    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if source.is_file() and source.suffix.lower() == ".txt":
        images = []
        with source.open("r", encoding="utf-8-sig") as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                image = Path(line)
                if not image.is_absolute():
                    candidate = (source.parent / image).resolve()
                    image = candidate if candidate.exists() else (dataset_root / image).resolve()
                images.append(image)
        return images
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    raise FileNotFoundError(f"验证集路径不存在或不受支持: {source}")


def collect_validation_images(config, dataset_root):
    val_sources = config["val"]
    if not isinstance(val_sources, (list, tuple)):
        val_sources = [val_sources]
    images = []
    for source in val_sources:
        images.extend(expand_image_source(source, dataset_root))
    images = sorted(dict.fromkeys(path.resolve() for path in images))
    if not images:
        raise RuntimeError("验证集中没有找到图片")
    missing = [path for path in images if not path.is_file()]
    if missing:
        sample = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"验证清单中有 {len(missing)} 张图片不存在，例如:\n{sample}")
    return images


def label_path_for_image(image_path):
    parts = list(image_path.resolve().parts)
    image_indexes = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if not image_indexes:
        raise ValueError(f"图片路径中没有 images 目录，无法推导 labels 路径: {image_path}")
    index = image_indexes[-1]
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_ground_truth(image_path, width, height, class_count):
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        # YOLO 允许负样本没有标签文件。
        return np.empty((0, 5), dtype=np.float32)

    targets = []
    with label_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            fields = raw_line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(
                    f"{label_path}:{line_number} 不是 YOLO 检测标签（应为 class cx cy w h）"
                )
            class_id = int(float(fields[0]))
            if class_id < 0 or class_id >= class_count:
                raise ValueError(
                    f"{label_path}:{line_number} 类别 {class_id} 超出 [0,{class_count - 1}]"
                )
            cx, cy, box_width, box_height = map(float, fields[1:])
            x1 = (cx - box_width / 2.0) * width
            y1 = (cy - box_height / 2.0) * height
            x2 = (cx + box_width / 2.0) * width
            y2 = (cy + box_height / 2.0) * height
            targets.append((class_id, x1, y1, x2, y2))
    return np.asarray(targets, dtype=np.float32).reshape(-1, 5)


def load_anchors(path):
    values = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for token in stream.read().replace(",", " ").split():
            values.append(float(token))
    if len(values) != 18:
        raise ValueError(f"{path} 应包含 18 个 anchor 数值，实际为 {len(values)} 个")
    return np.asarray(values, dtype=np.float32).reshape(3, 3, 2)


def preprocess_image(image, image_size, method):
    height, width = image.shape[:2]
    if method == "stretch":
        prepared = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        transform = ("stretch", width / image_size, height / image_size, 0.0, 0.0)
    else:
        scale = min(image_size / width, image_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        left = (image_size - resized_width) // 2
        top = (image_size - resized_height) // 2
        prepared = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        prepared[top : top + resized_height, left : left + resized_width] = resized
        transform = ("letterbox", 1.0 / scale, 1.0 / scale, float(left), float(top))
    return cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB), transform


def restore_boxes(boxes, transform, original_width, original_height):
    if boxes.size == 0:
        return boxes
    _, scale_x, scale_y, pad_x, pad_y = transform
    restored = boxes.copy()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - pad_x) * scale_x
    restored[:, [1, 3]] = (restored[:, [1, 3]] - pad_y) * scale_y
    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, original_width)
    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, original_height)
    return restored


def normalize_output_head(output, class_count):
    channels = 3 * (5 + class_count)
    array = np.asarray(output)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim == 3:
        if array.shape[0] == channels:
            return array.reshape(3, 5 + class_count, array.shape[1], array.shape[2])
        if array.shape[-1] == channels:
            array = array.transpose(2, 0, 1)
            return array.reshape(3, 5 + class_count, array.shape[1], array.shape[2])
    elif array.ndim == 4:
        if array.shape[:2] == (3, 5 + class_count):
            return array
        if array.shape[0] == 3 and array.shape[-1] == 5 + class_count:
            return array.transpose(0, 3, 1, 2)

    raise ValueError(
        f"无法识别输出形状 {tuple(np.asarray(output).shape)}；"
        f"当前 data.yaml 类别数为 {class_count}，期望每个检测头 {channels} 通道"
    )


def box_iou_one_to_many(box, boxes):
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-9)


def nms(boxes, scores, threshold):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = box_iou_one_to_many(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= threshold]
    return keep


def decode_outputs(outputs, anchors, image_size, class_count, conf_threshold, nms_threshold, max_det):
    if len(outputs) != 3:
        raise ValueError(f"经典 YOLOv5 应有 3 个输出，实际为 {len(outputs)} 个")

    heads = [normalize_output_head(output, class_count) for output in outputs]
    heads.sort(key=lambda head: head.shape[-1], reverse=True)
    all_boxes = []
    all_scores = []
    all_classes = []

    for head_index, head in enumerate(heads):
        grid_height, grid_width = head.shape[-2:]
        if image_size % grid_width != 0 or grid_height != grid_width:
            raise ValueError(
                f"输出网格 {grid_height}x{grid_width} 与输入 {image_size} 不匹配"
            )
        stride = image_size / grid_width
        grid_x, grid_y = np.meshgrid(
            np.arange(grid_width, dtype=np.float32),
            np.arange(grid_height, dtype=np.float32),
        )
        grid = np.stack((grid_x, grid_y), axis=0)[None, ...]

        box_xy = (head[:, 0:2] * 2.0 - 0.5 + grid) * stride
        anchor = anchors[head_index, :, :, None, None]
        box_wh = np.square(head[:, 2:4] * 2.0) * anchor
        xywh = np.concatenate((box_xy, box_wh), axis=1)
        xywh = xywh.transpose(0, 2, 3, 1).reshape(-1, 4)

        objectness = head[:, 4].reshape(-1)
        class_probabilities = head[:, 5:].transpose(0, 2, 3, 1).reshape(-1, class_count)
        class_ids = class_probabilities.argmax(axis=1)
        scores = objectness * class_probabilities[np.arange(class_probabilities.shape[0]), class_ids]
        selected = scores >= conf_threshold
        if not np.any(selected):
            continue

        xywh = xywh[selected]
        boxes = np.empty_like(xywh)
        boxes[:, 0:2] = xywh[:, 0:2] - xywh[:, 2:4] / 2.0
        boxes[:, 2:4] = xywh[:, 0:2] + xywh[:, 2:4] / 2.0
        all_boxes.append(boxes)
        all_scores.append(scores[selected])
        all_classes.append(class_ids[selected])

    if not all_boxes:
        return np.empty((0, 6), dtype=np.float32)

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    class_ids = np.concatenate(all_classes)
    kept = []
    for class_id in np.unique(class_ids):
        indexes = np.flatnonzero(class_ids == class_id)
        kept.extend(indexes[nms(boxes[indexes], scores[indexes], nms_threshold)])
    kept = np.asarray(kept, dtype=np.int64)
    kept = kept[np.argsort(scores[kept])[::-1]][:max_det]
    return np.column_stack((boxes[kept], scores[kept], class_ids[kept])).astype(np.float32)


def parse_class_ids(value, class_count):
    if not value.strip():
        return list(range(class_count))
    try:
        class_ids = sorted(set(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError:
        raise ValueError("--class-ids 必须是逗号分隔的整数")
    invalid = [class_id for class_id in class_ids if not 0 <= class_id < class_count]
    if invalid:
        raise ValueError(f"--class-ids 超出范围 [0,{class_count - 1}]: {invalid}")
    if not class_ids:
        raise ValueError("--class-ids 没有有效类别")
    return class_ids


def run_model(quantized, args, images, anchors, class_count):
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError("当前环境没有 rknn-toolkit2；请通过 evaluate_rknn.sh 运行") from exc

    rknn = RKNN(verbose=False)
    model_name = "INT8" if quantized else "FP"
    if args.onnx:
        print(f"\n[*] {model_name}: 从 ONNX build 后用 Toolkit 模拟器评测")
        result = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=args.platform,
        )
        if result != 0:
            rknn.release()
            raise RuntimeError(f"{model_name} rknn.config 失败")
        if rknn.load_onnx(model=args.onnx) != 0:
            rknn.release()
            raise RuntimeError(f"load_onnx 失败: {args.onnx}")
        dataset = args.calibration if quantized else None
        if rknn.build(do_quantization=quantized, dataset=dataset) != 0:
            rknn.release()
            raise RuntimeError(
                f"{model_name} build 失败"
                + (f"，校准清单: {dataset}" if quantized else "")
            )
    else:
        model_path = args.int8_model if quantized else args.fp_model
        print(f"\n[*] {model_name}: 加载现成模型并在目标板运行: {model_path}")
        if rknn.load_rknn(str(model_path)) != 0:
            rknn.release()
            raise RuntimeError(f"load_rknn 失败: {model_path}")

    runtime_kwargs = {}
    if args.target:
        runtime_kwargs["target"] = args.target
    if args.device_id:
        runtime_kwargs["device_id"] = args.device_id
    if rknn.init_runtime(**runtime_kwargs) != 0:
        rknn.release()
        if args.onnx:
            raise RuntimeError(f"{model_name} 模拟器 init_runtime 失败")
        raise RuntimeError(
            f"{model_name} init_runtime 失败；请检查板端 USB/ADB 连接、TARGET 和 DEVICE_ID"
        )

    ground_truth = {}
    predictions = {}
    try:
        total = len(images)
        for image_index, image_path in enumerate(images):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"OpenCV 无法读取图片: {image_path}")
            height, width = image.shape[:2]
            ground_truth[image_index] = load_ground_truth(
                image_path, width, height, class_count
            )
            input_image, transform = preprocess_image(image, args.imgsz, args.preprocess)
            outputs = rknn.inference(inputs=[input_image])
            if outputs is None:
                raise RuntimeError(f"RKNN inference 返回空结果: {image_path}")
            detections = decode_outputs(
                outputs,
                anchors,
                args.imgsz,
                class_count,
                args.conf,
                args.nms,
                args.max_det,
            )
            if detections.size:
                detections[:, :4] = restore_boxes(
                    detections[:, :4], transform, width, height
                )
            predictions[image_index] = detections
            if (image_index + 1) % 50 == 0 or image_index + 1 == total:
                print(f"    {image_index + 1}/{total}")
    finally:
        rknn.release()
    return ground_truth, predictions


def interpolated_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def calculate_class_metrics(ground_truth, predictions, class_id):
    gt_by_image = {}
    target_count = 0
    prediction_rows = []

    for image_id, targets in ground_truth.items():
        selected = targets[targets[:, 0].astype(np.int64) == class_id, 1:5]
        gt_by_image[image_id] = selected
        target_count += len(selected)
    for image_id, detections in predictions.items():
        if detections.size == 0:
            continue
        selected = detections[detections[:, 5].astype(np.int64) == class_id]
        for detection in selected:
            prediction_rows.append((float(detection[4]), image_id, detection[:4]))
    prediction_rows.sort(key=lambda row: row[0], reverse=True)

    if target_count == 0:
        return {
            "targets": 0,
            "predictions": len(prediction_rows),
            "ap50": None,
            "map50_95": None,
            "aps": [None] * len(IOU_THRESHOLDS),
        }
    if not prediction_rows:
        aps = [0.0] * len(IOU_THRESHOLDS)
        return {
            "targets": target_count,
            "predictions": 0,
            "ap50": 0.0,
            "map50_95": 0.0,
            "aps": aps,
        }

    aps = []
    for threshold in IOU_THRESHOLDS:
        matched = {
            image_id: np.zeros(len(boxes), dtype=bool)
            for image_id, boxes in gt_by_image.items()
        }
        true_positives = np.zeros(len(prediction_rows), dtype=np.float32)
        false_positives = np.zeros(len(prediction_rows), dtype=np.float32)

        for prediction_index, (_, image_id, box) in enumerate(prediction_rows):
            candidates = gt_by_image.get(image_id, np.empty((0, 4), dtype=np.float32))
            if candidates.size == 0:
                false_positives[prediction_index] = 1.0
                continue
            ious = box_iou_one_to_many(box, candidates)
            ious[matched[image_id]] = -1.0
            best = int(np.argmax(ious))
            if ious[best] >= threshold:
                true_positives[prediction_index] = 1.0
                matched[image_id][best] = True
            else:
                false_positives[prediction_index] = 1.0

        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)
        recall = cumulative_tp / max(target_count, 1)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        aps.append(interpolated_ap(recall, precision))

    return {
        "targets": target_count,
        "predictions": len(prediction_rows),
        "ap50": aps[0],
        "map50_95": float(np.mean(aps)),
        "aps": aps,
    }


def evaluate_metrics(ground_truth, predictions, names, selected_class_ids):
    rows = []
    for class_id in selected_class_ids:
        metrics = calculate_class_metrics(ground_truth, predictions, class_id)
        rows.append({"class_id": class_id, "class_name": names[class_id], **metrics})

    valid = [row for row in rows if row["ap50"] is not None]
    summary = {
        "map50": float(np.mean([row["ap50"] for row in valid])) if valid else None,
        "map50_95": (
            float(np.mean([row["map50_95"] for row in valid])) if valid else None
        ),
        "evaluated_classes": len(valid),
    }
    return rows, summary


def format_metric(value):
    return "-" if value is None or not math.isfinite(value) else f"{value:.4f}"


def print_metrics(title, rows, summary):
    print(f"\n===== {title} =====")
    print(f"{'ID':>3}  {'class':<24} {'targets':>8} {'preds':>8} {'AP50':>8} {'mAP50-95':>10}")
    for row in rows:
        print(
            f"{row['class_id']:>3}  {row['class_name']:<24.24} "
            f"{row['targets']:>8} {row['predictions']:>8} "
            f"{format_metric(row['ap50']):>8} {format_metric(row['map50_95']):>10}"
        )
    print(
        f"{'ALL':>3}  {'mean':<24} {'':>8} {'':>8} "
        f"{format_metric(summary['map50']):>8} {format_metric(summary['map50_95']):>10}"
    )


def csv_value(value):
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.8f}"


def write_outputs(output_dir, fp_rows, fp_summary, int8_rows, int8_summary, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, rows, summary in (
        ("fp", fp_rows, fp_summary),
        ("int8", int8_rows, int8_summary),
    ):
        with (output_dir / f"{name}_metrics.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["class_id", "class_name", "targets", "predictions", "AP50", "mAP50-95"])
            for row in rows:
                writer.writerow(
                    [
                        row["class_id"],
                        row["class_name"],
                        row["targets"],
                        row["predictions"],
                        csv_value(row["ap50"]),
                        csv_value(row["map50_95"]),
                    ]
                )
            writer.writerow(["ALL", "mean", "", "", csv_value(summary["map50"]), csv_value(summary["map50_95"])])

    with (output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "class_id",
                "class_name",
                "targets",
                "fp_AP50",
                "int8_AP50",
                "delta_AP50",
                "fp_mAP50-95",
                "int8_mAP50-95",
                "delta_mAP50-95",
            ]
        )
        for fp_row, int8_row in zip(fp_rows, int8_rows):
            ap50_delta = (
                int8_row["ap50"] - fp_row["ap50"]
                if fp_row["ap50"] is not None and int8_row["ap50"] is not None
                else None
            )
            map_delta = (
                int8_row["map50_95"] - fp_row["map50_95"]
                if fp_row["map50_95"] is not None and int8_row["map50_95"] is not None
                else None
            )
            writer.writerow(
                [
                    fp_row["class_id"],
                    fp_row["class_name"],
                    fp_row["targets"],
                    csv_value(fp_row["ap50"]),
                    csv_value(int8_row["ap50"]),
                    csv_value(ap50_delta),
                    csv_value(fp_row["map50_95"]),
                    csv_value(int8_row["map50_95"]),
                    csv_value(map_delta),
                ]
            )
        writer.writerow(
            [
                "ALL",
                "mean",
                "",
                csv_value(fp_summary["map50"]),
                csv_value(int8_summary["map50"]),
                csv_value(
                    int8_summary["map50"] - fp_summary["map50"]
                    if fp_summary["map50"] is not None and int8_summary["map50"] is not None
                    else None
                ),
                csv_value(fp_summary["map50_95"]),
                csv_value(int8_summary["map50_95"]),
                csv_value(
                    int8_summary["map50_95"] - fp_summary["map50_95"]
                    if fp_summary["map50_95"] is not None
                    and int8_summary["map50_95"] is not None
                    else None
                ),
            ]
        )

    json_data = {
        "metadata": metadata,
        "fp": {"classes": fp_rows, "summary": fp_summary},
        "int8": {"classes": int8_rows, "summary": int8_summary},
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(json_data, stream, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    data_path, config, names = load_dataset_config(args.data)
    dataset_root = resolve_dataset_root(data_path, config, args.dataset_root)
    images = collect_validation_images(config, dataset_root)
    if args.limit:
        images = images[: args.limit]
        print(f"[!] LIMIT={args.limit}：本次只是冒烟测试，结果不是完整验证集 mAP")
    selected_class_ids = parse_class_ids(args.class_ids, len(names))
    anchors = load_anchors(args.anchors)

    print(f"[*] 验证图片: {len(images)}")
    print(f"[*] 类别总数: {len(names)}；统计类别: {selected_class_ids}")
    print(f"[*] 预处理: {args.preprocess}；conf={args.conf}；NMS={args.nms}")
    if args.onnx:
        print(
            "[*] 模式: ONNX 离线重建；FP/INT8 将使用与 convert.py 相同配置分别 build。"
        )
    else:
        print(f"[*] 模式: 现成 RKNN 上板评测；target={args.target}")
    print("[*] 同一验证集将顺序评测 FP 和 INT8，耗时约为单模型的两倍。")

    fp_ground_truth, fp_predictions = run_model(
        False, args, images, anchors, len(names)
    )
    int8_ground_truth, int8_predictions = run_model(
        True, args, images, anchors, len(names)
    )

    fp_rows, fp_summary = evaluate_metrics(
        fp_ground_truth, fp_predictions, names, selected_class_ids
    )
    int8_rows, int8_summary = evaluate_metrics(
        int8_ground_truth, int8_predictions, names, selected_class_ids
    )
    print_metrics("FP RKNN", fp_rows, fp_summary)
    print_metrics("INT8 RKNN", int8_rows, int8_summary)

    metadata = {
        "mode": "onnx_rebuild_simulator" if args.onnx else "rknn_target",
        "onnx": str(Path(args.onnx)) if args.onnx else None,
        "calibration": str(Path(args.calibration)) if args.calibration else None,
        "fp_model": str(Path(args.fp_model)) if args.fp_model else None,
        "int8_model": str(Path(args.int8_model)) if args.int8_model else None,
        "data": str(data_path),
        "dataset_root": str(dataset_root),
        "anchors": str(Path(args.anchors)),
        "image_count": len(images),
        "image_size": args.imgsz,
        "preprocess": args.preprocess,
        "confidence_threshold": args.conf,
        "nms_threshold": args.nms,
        "iou_thresholds": [round(float(value), 2) for value in IOU_THRESHOLDS],
        "selected_class_ids": selected_class_ids,
        "limit": args.limit,
    }
    write_outputs(
        args.output_dir,
        fp_rows,
        fp_summary,
        int8_rows,
        int8_summary,
        metadata,
    )
    print(f"\n[✓] 对比结果: {Path(args.output_dir) / 'comparison.csv'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        sys.exit(f"[!] {error}")
