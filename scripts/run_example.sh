#!/bin/zsh
# 端到端示例：那花宏哥 0524 多光谱飞行 + 供应商 0322 树点标签
# 按实际路径修改下面几个变量后逐段执行。
set -e
PROJ=$(cd "$(dirname "$0")/.." && pwd)
TERRA=/Users/xunyou/Downloads/map                       # DJI Terra 成果目录
WORK=$HOME/treeseg_work/nahua0524                      # 工作目录（影像、数据集、预测都放这里）
VENDOR_PTS="/Users/xunyou/Desktop/宏哥/宏哥树点-ID-病虫害-产量.shp"
VENDOR_CSV="/Users/xunyou/Desktop/宏哥/宏哥八角-ID-病虫害-估产.csv"
TE="631042 2627881 631789 2628956"                      # 裁剪范围（EPSG:4544），留空则全幅
cd "$PROJ"

# 1. 影像准备：composite.tif(5cm RGB) + ms_*.tif/ndvi.tif/dsm.tif(20cm)
python -m treeseg.prepare --terra-dir "$TERRA" --out "$WORK/prep" --gsd 0.05 --ms-gsd 0.2 --te ${=TE} --threads 4

# 2. 标签：供应商树点 + 冠幅面积 → 圆 → NDVI 精修。--shift 用 evaluate --auto-shift 或人工量取
python -m treeseg.labels --raster "$WORK/prep/composite.tif" --source "$VENDOR_PTS" --type points \
  --id-field ID --area-csv "$VENDOR_CSV" --shift 1.14 1.39 --refine-ndvi "$WORK/prep/ndvi.tif" --out "$WORK/labels_0322.gpkg"
# 若有供应商树冠多边形（如 单株树冠估产.shp），优先用它：
# python -m treeseg.labels --raster "$WORK/prep/composite.tif" --source 单株树冠估产.shp --type polygons --id-field tree_id --out "$WORK/labels_crowns.gpkg"

# 3. 切片数据集（1024px ≈ 51m，重叠 128px）
python -m treeseg.dataset --raster "$WORK/prep/composite.tif" --labels "$WORK/labels_0322.gpkg" --out "$WORK/dataset" --tile 1024 --overlap 128 --prefix n0524

# 4. 训练（GPU 机器上执行）
python -m treeseg.train --data "$WORK/dataset/data.yaml" --model yolo11m-seg.pt --epochs 100 --imgsz 1024 --batch 8 --project "$WORK/runs" --name m_rgb

# 5. 全幅预测 + 逐株指数
python -m treeseg.predict --raster "$WORK/prep/composite.tif" --weights "$WORK/runs/m_rgb/weights/best.pt" --out "$WORK/pred" --ms-dir "$WORK/prep"

# 6. 与供应商树点对比
python -m treeseg.evaluate --pred "$WORK/pred/crowns.gpkg" --ref "$VENDOR_PTS" --auto-shift --max-dist 1.5

# 7. 跨期 ID 台账（首期初始化，后续每期重复执行）
python -m treeseg.registry --registry "$HOME/treeseg_work/registry_nahua.gpkg" --new "$WORK/pred/crowns.gpkg" --date 2026-05-24 --out "$HOME/treeseg_work/registry_nahua.gpkg"

# 8. 打包成 AgroSphere「AI 单树监测」导入格式（zip 内：底图 tif + 树点/树冠/病虫害多边形 shp + csv）
python -m treeseg.export_platform --crowns "$WORK/pred/crowns_with_ids.gpkg" --tif "$WORK/prep/composite.tif" \
  --out "$WORK/export" --name nahua_20260524 --date 2026-05-24
# 然后在管理后台 AI 单树监测 → 导入批次 上传 "$WORK/export/nahua_20260524.zip"（或服务器本地路径模式 local_archive_path）
