#!/bin/bash
# v5（2026-09-10）：v06 + v07 + 大疆 0723 + 大疆 0811（配准标签）四期混合重训 -> 0811 推理变体评估（conf / TTA）
# 前提：/root/work 下 treeseg/ prepv06 prepv07 prep0723 prep0811 labels_v06/v07/0723/0811.gpkg yolo11m-seg.pt，/root/work/py
W=/root/work; PY=$W/py; cd $W/treeseg
echo "[$(date +%T)] 数据集 mix2"
rm -rf $W/dataset_mix2
for pair in "prepv06 labels_v06 v06" "prepv07 labels_v07 v07" "prep0723 labels_0723 d0723" "prep0811 labels_0811 d0811"; do
  set -- $pair
  $PY -m treeseg.dataset --raster $W/$1/composite.tif --labels $W/$2.gpkg --out $W/dataset_mix2 --tile 1024 --overlap 128 --block 3 --val-frac 0.2 --prefix $3 --seed 1 | tail -1
done
sed -i "s#^path: .*#path: $W/dataset_mix2#" $W/dataset_mix2/data.yaml
echo "train $(ls $W/dataset_mix2/images/train | wc -l) 张，val $(ls $W/dataset_mix2/images/val | wc -l) 张：$(ls $W/dataset_mix2/images/train | cut -d_ -f1 | sort | uniq -c | tr "\n" " ")"
echo "[$(date +%T)] 训练（batch 4）"
rm -rf $W/runs/mix2_m
$PY -m treeseg.train --data $W/dataset_mix2/data.yaml --model $W/yolo11m-seg.pt --epochs 150 --imgsz 1024 --batch 4 --device 0 --workers 8 --mask-ratio 4 --patience 40 --project $W/runs --name mix2_m > $W/train_mix2.log 2>&1 || { echo "训练失败"; exit 1; }
sort -t, -k9 -g $W/runs/mix2_m/results.csv | tail -1 | awk -F, '{print "最佳轮 "$1" mAP50(B)="$9" mAP50(M)="$13}'
BEST=$W/runs/mix2_m/weights/best.pt
echo "[$(date +%T)] 0811 推理变体评估（参考：配准后的 7 月树冠 labels_0811）"
for v in "c15 --conf 0.15" "c10 --conf 0.10" "c15tta --conf 0.15 --tta" "c10tta --conf 0.10 --tta" "c10n3 --conf 0.10 --nms-iou 0.3"; do
  set -- $v; tag=$1; shift
  $PY -m treeseg.predict --raster $W/prep0811/composite.tif --weights $BEST --out $W/pred_mix2_0811_$tag --device 0 --ms-dir $W/prep0811 "$@" 2>&1 | grep -E "合并后" | sed "s/^/  $tag: /"
  echo "=== 变体 $tag"; $PY -m treeseg.evaluate --pred $W/pred_mix2_0811_$tag/crowns.gpkg --ref $W/labels_0811.gpkg --auto-shift --max-dist 1.5 2>&1 | grep -E "匹配|冠幅"
done
echo "[$(date +%T)] 其他期次（conf 0.15）对照"
for pair in "v07 v07" "0723 0723"; do
  set -- $pair
  $PY -m treeseg.predict --raster $W/prep$1/composite.tif --weights $BEST --out $W/pred_mix2_$1 --conf 0.15 --device 0 2>&1 | grep -E "合并后" | sed "s/^/  $1: /"
  echo "=== $1 vs labels_$2"; $PY -m treeseg.evaluate --pred $W/pred_mix2_$1/crowns.gpkg --ref $W/labels_$2.gpkg --auto-shift --max-dist 1.5 2>&1 | grep -E "匹配|冠幅"
done
echo "[$(date +%T)] ALL_DONE"
