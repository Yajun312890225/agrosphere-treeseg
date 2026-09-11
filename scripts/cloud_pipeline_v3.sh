#!/bin/bash
# 云训练机流水线 v3（2026-09-10）：
#   A. 现有模型（SAM 标签版 best.pt）预测四期影像，与供应商 6/7 月树冠比对
#   B. 用供应商真实树冠多边形（6 月 + 7 月）重训 yolo11m-seg
#   C. 新模型再预测四期并比对
# 前提：/root/work 下已有 treeseg/ prep0524 prep0811 prep_v06 prep_v07 labels_v06/v07.gpkg best.pt yolo11m-seg.pt，/root/work/py 包装脚本
set -e
W=/root/work
PY=$W/py
cd $W/treeseg
CONF=0.15

predict_all() {  # $1=权重 $2=输出前缀
    for p in 0524 0811 v06 v07; do
        ms=""; [ -f $W/prep$p/ms_NIR.tif ] && ms="--ms-dir $W/prep$p"
        $PY -m treeseg.predict --raster $W/prep$p/composite.tif --weights $1 --out $W/$2_$p --conf $CONF --device 0 $ms 2>&1 | grep -vE "已推理|Warning"
    done
}
evaluate_all() {  # $1=输出前缀
    for pair in "0524 v06" "v06 v06" "v07 v07" "0811 v07"; do
        set -- $pair
        echo "=== 预测 $1 vs 供应商 $2 树冠"
        $PY -m treeseg.evaluate --pred $W/${PFX}_$1/crowns.gpkg --ref $W/labels_$2.gpkg --auto-shift --max-dist 1.5 2>&1 | grep -vE "Warning"
    done
}

echo "[$(date +%T)] A. 现有模型预测四期"
predict_all $W/best.pt pred_sam
PFX=pred_sam evaluate_all

echo "[$(date +%T)] B. 供应商多边形标签 -> 数据集（6 月 + 7 月，同一空间块划分，避免跨期泄漏）"
rm -rf $W/dataset_vendor
$PY -m treeseg.dataset --raster $W/prep_v06/composite.tif --labels $W/labels_v06.gpkg --out $W/dataset_vendor --tile 1024 --overlap 128 --block 3 --val-frac 0.2 --prefix v06 --seed 1 | tail -1
$PY -m treeseg.dataset --raster $W/prep_v07/composite.tif --labels $W/labels_v07.gpkg --out $W/dataset_vendor --tile 1024 --overlap 128 --block 3 --val-frac 0.2 --prefix v07 --seed 1 | tail -1
sed -i "s#^path: .*#path: $W/dataset_vendor#" $W/dataset_vendor/data.yaml
echo "train $(ls $W/dataset_vendor/images/train | wc -l) 张，val $(ls $W/dataset_vendor/images/val | wc -l) 张"

echo "[$(date +%T)] B. 训练"
rm -rf $W/runs/vendor_m
$PY -m treeseg.train --data $W/dataset_vendor/data.yaml --model $W/yolo11m-seg.pt --epochs 150 --imgsz 1024 --batch 8 --device 0 --workers 8 --mask-ratio 4 --patience 40 --project $W/runs --name vendor_m > $W/train_vendor.log 2>&1
sort -t, -k9 -g $W/runs/vendor_m/results.csv | tail -1 | awk -F, '{print "最佳轮 "$1" mAP50(B)="$9" mAP50(M)="$13}'

echo "[$(date +%T)] C. 新模型预测四期"
predict_all $W/runs/vendor_m/weights/best.pt pred_vendor
PFX=pred_vendor evaluate_all
echo "[$(date +%T)] PIPELINE_DONE"
