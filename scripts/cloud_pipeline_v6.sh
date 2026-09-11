#!/bin/bash
# v6（2026-09-10）：在 v5 基础上加"标签质量守门"重训 mix3。
# 背景：供应商 7 月树冠在大疆 0723/0811 影像上约 40% 的切片局部错位 >1 m（供应商正射西北与东南边缘几何畸变，
# 位移场修不平），mix2 把这些切片当成"树在别处/没有树"学，导致西侧 1 ha 几乎全漏检。
# 做法：coregister 写入 reg_dist（配准后到最近模型预测的距离），dataset 按切片剔除达标比 <0.4 的切片（分布双峰，谷在 0.4～0.5）。
# 前提：/root/work 下 treeseg/ prepv06 prepv07 prep0723 prep0811 prep0811_reg labels_v06/v07 labels_0723_v2/0811_v2.gpkg field_0811.npz yolo11m-seg.pt
W=/root/work; PY=$W/py; cd $W/treeseg
echo "[$(date +%T)] 数据集 mix3（0723/0811 走质量守门）"
rm -rf $W/dataset_mix3
for pair in "prepv06 labels_v06 v06 -" "prepv07 labels_v07 v07 -" "prep0723 labels_0723_v2 d0723 reg_dist" "prep0811 labels_0811_v2 d0811 reg_dist"; do
  set -- $pair
  QA=""; [ "$4" != "-" ] && QA="--qa-field $4 --qa-thr 1.0 --qa-min-frac 0.4"
  $PY -m treeseg.dataset --raster $W/$1/composite.tif --labels $W/$2.gpkg --out $W/dataset_mix3 --tile 1024 --overlap 128 --block 3 --val-frac 0.2 --prefix $3 --seed 1 $QA | tail -2 | head -1
done
sed -i "s#^path: .*#path: $W/dataset_mix3#" $W/dataset_mix3/data.yaml
echo "train $(ls $W/dataset_mix3/images/train | wc -l) 张，val $(ls $W/dataset_mix3/images/val | wc -l) 张：$(ls $W/dataset_mix3/images/train | cut -d_ -f1 | sort | uniq -c | tr "\n" " ")"
echo "[$(date +%T)] 训练（batch 4）"
rm -rf $W/runs/mix3_m
$PY -m treeseg.train --data $W/dataset_mix3/data.yaml --model $W/yolo11m-seg.pt --epochs 150 --imgsz 1024 --batch 4 --device 0 --workers 8 --mask-ratio 4 --patience 40 --project $W/runs --name mix3_m > $W/train_mix3.log 2>&1 || { echo "训练失败"; exit 1; }
sort -t, -k9 -g $W/runs/mix3_m/results.csv | tail -1 | awk -F, '{print "最佳轮 "$1" mAP50(B)="$9" mAP50(M)="$13}'
BEST=$W/runs/mix3_m/weights/best.pt
echo "[$(date +%T)] 0811 推理（conf 0.10，NMS 0.3，与 mix2 选定变体一致）"
$PY -m treeseg.predict --raster $W/prep0811/composite.tif --weights $BEST --out $W/pred_mix3_0811 --device 0 --ms-dir $W/prep0811 --conf 0.10 --nms-iou 0.3 2>&1 | grep -E "合并后"
for ref in labels_0811 labels_0811_v2; do
  echo "=== 0811 vs $ref"; $PY -m treeseg.evaluate --pred $W/pred_mix3_0811/crowns.gpkg --ref $W/$ref.gpkg --auto-shift --max-dist 1.5 2>&1 | grep -E "匹配|冠幅"
done
echo "[$(date +%T)] 其他期次（conf 0.15）对照"
for pair in "v07 v07" "0723 0723"; do
  set -- $pair
  $PY -m treeseg.predict --raster $W/prep$1/composite.tif --weights $BEST --out $W/pred_mix3_$1 --conf 0.15 --device 0 2>&1 | grep -E "合并后" | sed "s/^/  $1: /"
  echo "=== $1 vs labels_$2"; $PY -m treeseg.evaluate --pred $W/pred_mix3_$1/crowns.gpkg --ref $W/labels_$2.gpkg --auto-shift --max-dist 1.5 2>&1 | grep -E "匹配|冠幅"
done
echo "=== 西侧漏检区（1 ha，x 631247-631348 y 2628440-2628543）各版本检出数"
$PY - <<'EOF' 2>&1 | grep -v Warning
import numpy as np
from shapely.geometry import box
from treeseg import geo
rb = box(631247, 2628440, 631348, 2628543)
for name, p in [("供应商7月标签", "/root/work/labels_v07.gpkg"), ("mix2", "/root/work/pred_mix2_0811_c10n3/crowns.gpkg"), ("mix3", "/root/work/pred_mix3_0811/crowns.gpkg")]:
    print(f"  {name}: {sum(1 for g, _ in geo.read_vector(p, 4544) if rb.contains(g.centroid))} 株")
EOF
echo "[$(date +%T)] 配准到供应商框架并打包"
$PY -m treeseg.coregister --moving $W/pred_mix3_0811/crowns.gpkg --field-in $W/field_0811.npz --out-vector $W/pred_mix3_0811/crowns_reg.gpkg 2>&1 | grep -v Warning
echo "=== 配准后预测 vs 供应商 7 月树冠（不做平移估计，直接比）"
$PY -m treeseg.evaluate --pred $W/pred_mix3_0811/crowns_reg.gpkg --ref $W/labels_v07.gpkg --max-dist 1.5 2>&1 | grep -E "匹配|冠幅"
rm -rf $W/export_0811_mix3 && $PY -m treeseg.export_platform --crowns $W/pred_mix3_0811/crowns_reg.gpkg --tif $W/prep0811_reg/composite.tif --out $W/export_0811_mix3 --name nahua_0811_mix3 --date 2026-08-11 2>&1 | grep -v Warning
echo "[$(date +%T)] ALL_DONE"
