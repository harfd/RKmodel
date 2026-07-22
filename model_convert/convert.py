#!/usr/bin/env python3
# ONNX -> RKNN 转换（在 rknn-toolkit2 容器内运行）
# 用法：
#   python3 convert.py <onnx路径> [平台=rk3588] [dtype=i8|u8|fp] [-o 输出.rknn] [-d dataset.txt]
# 例：
#   python3 convert.py model/best.onnx rk3588 i8 -d dataset.txt
import os
import sys
import argparse
from rknn.api import RKNN


def main():
    ap = argparse.ArgumentParser(description="ONNX -> RKNN")
    ap.add_argument("onnx", help="输入 ONNX 路径")
    ap.add_argument("platform", nargs="?", default="rk3588",
                    help="目标平台，默认 rk3588")
    ap.add_argument("dtype", nargs="?", default="i8", choices=["i8", "u8", "fp"],
                    help="i8/u8=量化(默认 i8)，fp=不量化")
    ap.add_argument("-o", "--output", default=None, help="输出 .rknn 路径")
    ap.add_argument("-d", "--dataset", default="dataset.txt",
                    help="量化校准图清单(每行一张图路径)")
    args = ap.parse_args()

    if not os.path.isfile(args.onnx):
        sys.exit(f"[!] 找不到 ONNX: {args.onnx}")

    do_quant = args.dtype in ("i8", "u8")
    if do_quant and not os.path.isfile(args.dataset):
        sys.exit(f"[!] 量化需要校准清单: {args.dataset}（或用 dtype=fp 跳过量化）")

    out = args.output or (os.path.splitext(args.onnx)[0] + ".rknn")

    rknn = RKNN(verbose=True)
    # mean=0/std=255：让 RKNN 内部对输入做 /255 归一化，板端可直接喂 0-255 的 uint8
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform=args.platform)

    print(f"[*] load_onnx: {args.onnx}")
    if rknn.load_onnx(model=args.onnx) != 0:
        sys.exit("[!] load_onnx 失败")

    print(f"[*] build: quant={do_quant} dataset={args.dataset if do_quant else '-'}")
    if rknn.build(do_quantization=do_quant,
                  dataset=(args.dataset if do_quant else None)) != 0:
        sys.exit("[!] build 失败")

    print(f"[*] export_rknn: {out}")
    if rknn.export_rknn(out) != 0:
        sys.exit("[!] export_rknn 失败")

    rknn.release()
    print(f"[✓] 完成 -> {out}")


if __name__ == "__main__":
    main()
