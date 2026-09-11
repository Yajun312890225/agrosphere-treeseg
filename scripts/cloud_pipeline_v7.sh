#!/bin/bash
# v7（2026-09-11）：mix4 = mix3 四期（v06 v07 0723 0811）+ 大疆 0524（5 月物候，此前无样本 F1 仅 0.46）。
# 0524 标签：供应商 6 月树冠按位移场套到 0524 影像（参考点 = SAM 冠形标签 ∪ 三套模型预测，逐切片守门 106/107 通过，
# 说明 0524 正射与供应商框架只差整体平移；0723/0811 仍有约 40% 切片对不上，照旧守门）。
# 训练后三期大疆影像各出一版预测 → 配准到供应商框架（0811 复用 field_0811.npz 保证底图不变）→ 跨期 ID 台账 → 三个导入 zip。
# 前提（scripts/cloud_upload.sh 会放齐）：/root/work 下 treeseg/ prepv06 prepv07 prep0723 prep0811 prep0524
#   labels_v06/v07/0723_v2/0811_v2/0524_v2/0322/sam_v2.gpkg field_0811.npz yolo11m-seg.pt
W=/root/work; PY=$W/py; N=mix4; cd $W/treeseg
echo "[$(date +%T)] 数据集 $N（大疆三期走质量守门）"
rm -rf $W/dataset_$N
for pair in "prepv06 labels_v06 v06 -" "prepv07 labels_v07 v07 -" "prep0723 labels_0723_v2 d0723 reg_dist" "prep0811 labels_0811_v2 d0811 reg_dist" "prep0524 labels_0524_v2 d0524 reg_dist"; do
  set -- $pair
  QA=""; [ "$4" != "-" ] && QA="--qa-field $4 --qa-thr 1.0 --qa-min-frac 0.4"
  $PY -m treeseg.dataset --raster $W/$1/composite.tif --labels $W/$2.gpkg --out $W/dataset_$N --tile 1024 --overlap 128 --block 3 --val-frac 0.2 --prefix $3 --seed 1 $QA | tail -2 | head -1
done
sed -i "s#^path: .*#path: $W/dataset_$N#" $W/dataset_$N/data.yaml
echo "train $(ls $W/dataset_$N/images/train | wc -l) 张，val $(ls $W/dataset_$N/images/val | wc -l) 张：$(ls $W/dataset_$N/images/train | cut -d_ -f1 | sort | uniq -c | tr "\n" " ")"
echo "[$(date +%T)] 训练（batch 4，约 90 分钟）"
rm -rf $W/runs/${N}_m
$PY -m treeseg.train --data $W/dataset_$N/data.yaml --model $W/yolo11m-seg.pt --epochs 150 --imgsz 1024 --batch 4 --device 0 --workers 8 --mask-ratio 4 --patience 40 --project $W/runs --name ${N}_m > $W/train_$N.log 2>&1 || { echo "训练失败，见 $W/train_$N.log"; exit 1; }
sort -t, -k9 -g $W/runs/${N}_m/results.csv | tail -1 | awk -F, '{print "最佳轮 "$1" mAP50(B)="$9" mAP50(M)="$13}'
BEST=$W/runs/${N}_m/weights/best.pt
echo "[$(date +%T)] 推理：大疆三期 conf 0.10 NMS 0.3；v07 对照 conf 0.15"
for per in 0524 0723 0811; do
  $PY -m treeseg.predict --raster $W/prep$per/composite.tif --weights $BEST --out $W/pred_${N}_$per --device 0 --ms-dir $W/prep$per --conf 0.10 --nms-iou 0.3 2>&1 | grep -E "合并后" | sed "s/^/  $per: /"
done
$PY -m treeseg.predict --raster $W/prepv07/composite.tif --weights $BEST --out $W/pred_${N}_v07 --device 0 --conf 0.15 2>&1 | grep -E "合并后" | sed "s/^/  v07: /"
echo "[$(date +%T)] 评估（--auto-shift，1.5 m；大疆期全图指标受错位标签拖低，重点看召回与西侧区）"
for pair in "0524 labels_0524_v2" "0524 labels_sam_v2" "0524 labels_0322" "0723 labels_0723_v2" "0811 labels_0811_v2" "v07 labels_v07"; do
  set -- $pair
  echo "=== $1 vs $2"; $PY -m treeseg.evaluate --pred $W/pred_${N}_$1/crowns.gpkg --ref $W/$2.gpkg --auto-shift --max-dist 1.5 2>&1 | grep -E "匹配|冠幅"
done
echo "=== 西侧漏检区（1 ha，x 631247-631348 y 2628440-2628543）检出数（供应商 7 月标签 354 株，mix3 于 0811 为 369）"
$PY - <<PYEOF 2>&1 | grep -v Warning
from shapely.geometry import box
from treeseg import geo
rb = box(631247, 2628440, 631348, 2628543)
for per in ["0524", "0723", "0811"]:
    print(f"  $N {per}: {sum(1 for g, _ in geo.read_vector(f'/root/work/pred_${N}_{per}/crowns.gpkg', 4544) if rb.contains(g.centroid))} 株")
PYEOF
echo "[$(date +%T)] 配准到供应商框架 + 底图重采样（0811 复用 field_0811.npz；0723/0524 以供应商 7 月树冠为 fixed 估计新场）"
for per in 0524 0723 0811; do
  P=$W/pred_${N}_$per
  if [ -f $W/field_$per.npz ]; then FIELD="--field-in $W/field_$per.npz"; else FIELD="--fixed $W/labels_v07.gpkg --field-out $W/field_$per.npz"; fi
  RAS=""; [ -f $W/prep${per}_reg/composite.tif ] || RAS="--rasters $W/prep$per/composite.tif --out-dir $W/prep${per}_reg --threads 8"
  echo "=== $per"; $PY -m treeseg.coregister --moving $P/crowns.gpkg $FIELD --out-vector $P/crowns_reg.gpkg $RAS 2>&1 | grep -v Warning
  $PY -m treeseg.evaluate --pred $P/crowns_reg.gpkg --ref $W/labels_v07.gpkg --max-dist 1.5 2>&1 | grep -E "匹配" | sed "s/^/  配准后 vs 供应商7月（不平移）: /"
done
echo "[$(date +%T)] 跨期 ID 台账 0524 → 0723 → 0811（已配准，不再估平移）"
rm -f $W/registry_$N.gpkg
for pair in "0524 2026-05-24" "0723 2026-07-22" "0811 2026-08-11"; do
  set -- $pair
  $PY -m treeseg.registry --registry $W/registry_$N.gpkg --new $W/pred_${N}_$1/crowns_reg.gpkg --date $2 --out $W/registry_$N.gpkg --no-auto-shift 2>&1 | grep -v Warning | sed "s/^/  $1: /"
done
echo "[$(date +%T)] 打包三期（ID 来自台账，跨期一致）"
for pair in "0524 2026-05-24" "0723 2026-07-22" "0811 2026-08-11"; do
  set -- $pair
  rm -rf $W/export_$1_$N && $PY -m treeseg.export_platform --crowns $W/pred_${N}_$1/crowns_with_ids.gpkg --tif $W/prep$1_reg/composite.tif --out $W/export_$1_$N --name nahua_$1_$N --date $2 2>&1 | grep -v Warning
done
echo "[$(date +%T)] ALL_DONE"
