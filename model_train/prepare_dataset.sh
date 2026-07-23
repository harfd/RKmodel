#!/usr/bin/env bash
# 下载 ultralytics 内置数据集，并生成"经典 yolov5"训练用的 data.yaml。
# 经典 yolov5(airockchip/yolov5) 的 yaml 需要 nc + names(列表)，和 ultralytics 的格式略不同，这里自动转换。
# 用法：
#   bash prepare_dataset.sh                         # 默认 construction-ppe
#   DATA=african-wildlife.yaml bash prepare_dataset.sh
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-yolo-train:latest}"
DATA="${DATA:-construction-ppe.yaml}"    # ultralytics 内置数据集名(会自动下载)
OUT="${OUT:-_yolov5_data.yaml}"          # 生成的 yolov5 格式 yaml

docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" .
mkdir -p datasets

docker run --rm -it -v "$(pwd)":/workspace -w /workspace "$IMAGE" bash -lc "
  yolo settings datasets_dir=/workspace/datasets >/dev/null 2>&1 || true
  python - <<'PY'
import os, yaml
from ultralytics.data.utils import check_det_dataset
d = check_det_dataset('${DATA}')                 # 不存在则自动下载
root = str(d['path'])
def rel(p):
    p = p[0] if isinstance(p,(list,tuple)) else p
    p = str(p)
    return os.path.relpath(p, root) if os.path.isabs(p) else p
names = d['names']
names_list = names if isinstance(names, list) else [names[k] for k in sorted(names)]
out = {'path': root,
       'train': rel(d.get('train','images/train')),
       'val':   rel(d.get('val','images/val')),
       'nc': len(names_list),
       'names': names_list}
yaml.safe_dump(out, open('/workspace/${OUT}','w'), allow_unicode=True, sort_keys=False)
print('===== 生成 ${OUT} =====')
print(open('/workspace/${OUT}').read())
PY
"
echo "[✓] 数据集就绪，yolov5 训练用的 yaml: ${OUT}（里面 names 顺序=类别索引，务必核对一眼）"
