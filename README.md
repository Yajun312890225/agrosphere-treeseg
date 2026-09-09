# agrosphere-treeseg：八角单株树冠提取（深度学习）

用无人机正射影像（大疆智图成果）训练实例分割模型，输出逐株树冠多边形、树点与冠幅面积，并支持跨期单株 ID 台账。标签来自供应商已交付的单株结果（树点 + 冠幅面积，或树冠多边形）。病虫害与产量不在本项目范围，逐株光谱指数（NDVI/NDRE/GNDVI）会随预测结果一并输出，供后续建模。

## 流程

```
DJI Terra 成果 ──prepare──> composite.tif(5cm) + ms_*.tif/ndvi.tif/dsm.tif(20cm)
供应商结果    ──labels───> labels.gpkg（树冠多边形）
两者          ──dataset──> YOLO seg 数据集（1024px 切片 + data.yaml）
              ──train────> best.pt
              ──predict──> crowns.gpkg / trees.csv（含逐株指数）
              ──evaluate─> 与供应商树点的召回/精度/欠分割统计
              ──registry─> 跨期 ID 台账
```

完整命令见 `scripts/run_example.sh`，每个模块 `python -m treeseg.<模块> -h` 可看参数。

## 环境安装

推荐 conda（GDAL 用 pip 装容易失败）：

```bash
conda create -n treeseg python=3.11 -y
conda activate treeseg
conda install -c conda-forge gdal=3.9 -y
# 按 https://pytorch.org 选与显卡 CUDA 版本匹配的命令，例如：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

验证：`python -c "from osgeo import gdal; import ultralytics, torch; print(torch.cuda.is_available())"` 应输出 `True`。

注意：不要用 `--system-site-packages` 复用 Homebrew/系统的 numpy、scipy。它们链接的 libomp 会与 torch 自带的 libomp 冲突，表现为导入 ultralytics 后段错误（退出码 139）。用干净的 conda 环境或 venv 全部由 pip 安装即可避免。

## 数据要求

| 项 | 要求 |
|---|---|
| 影像 | 大疆智图目录：`result.tif`（RGB），可选 `result_Red/Green/RedEdge/NIR.tif`、`dsm.tif` |
| 标签 | 供应商树冠多边形 shp（最佳）；或树点 shp/csv + 冠幅面积（自动生成圆形伪标签，可用 NDVI 精修） |
| 坐标 | 默认统一到 EPSG:4544（CGCS2000 3 度带 105E），其它地区用 `--epsg` 改 |
| 对齐 | 标签与影像若来自不同架次会有整体偏移（那花 0322 与 0524 差 1.1/1.4 m），用 `evaluate --auto-shift` 估计后填到 `labels --shift` |

多期数据合并训练：对每一期分别跑 prepare + labels + dataset（`--prefix` 不同，`--out` 同一个目录）即可。

## 硬件要求

| 阶段 | 最低 | 推荐 | 说明 |
|---|---|---|---|
| prepare（gdalwarp） | 16 GB 内存，50 GB 磁盘/期 | 32 GB，NVMe SSD | 7 GB RGB 正射重采样约 5～10 分钟 |
| train | NVIDIA 8 GB 显存（yolo11s-seg，batch 4） | 16～24 GB 显存（yolo11m/l-seg，batch 8～16），32 GB 内存 | 约 500 张 1024px 切片、100 轮：RTX 4090 约 1 小时，RTX 3060 12G 约 4 小时，RTX 5060 Ti 16G 约 2 小时 |
| predict | 任意 GPU 或 CPU | GPU | 0.37 km² 约 500 切片，GPU 1～2 分钟，CPU 15～30 分钟 |

你们做大疆智图的那台机器（i5-14600KF、RTX 5060 Ti 16 GB、32 GB 内存）可以直接训练。Mac（Apple Silicon）能用 `--device mps` 训练 n/s 级别模型，速度约为 RTX 4090 的 1/10，只适合调试。CPU 不建议训练。

## 训练建议

1. 先用 `yolo11m-seg.pt`、`imgsz 1024`、`epochs 100` 跑一版，看 val 的 mask mAP50 与 `evaluate` 的召回/精度。
2. 标签质量决定上限：树点+面积生成的圆形伪标签只能学到"大概位置和大小"；拿到供应商的树冠多边形后重新生成标签再训一版，通常召回和冠幅一致性会明显提高。
3. 郁闭林分（树冠相连）欠分割最常见。`evaluate` 里"含 2 点以上树冠"数就是欠分割量；可尝试 `--bands cir`（近红外假彩色）或更高 `imgsz`。
4. 跨期比较前，飞行务必开 RTK 固定解并放反射率标定板，否则 ID 匹配靠整体平移估计，局部误差仍可达 1 m 以上。

## 输出说明

`predict` 输出 `crowns.gpkg`（可直接在 QGIS 打开）与 `trees.csv`：`id, area_m2, conf, cx, cy, lon, lat, ndvi_mean, ndre_mean, gndvi_mean`。`registry` 输出的 `crowns_with_ids.gpkg` 带跨期一致的 `tree_id`。

## 对接 AgroSphere 平台

`export_platform` 把结果打成平台「AI 单树监测」导入格式的 zip，平台侧不需要改代码：

```bash
python -m treeseg.export_platform --crowns pred/crowns_with_ids.gpkg --tif prep/composite.tif --out export --name nahua_20260524 --date 2026-05-24
```

zip 内容与平台 `internal/logic/plot/monitor_analysis_service.go` 的约定对应：

| 文件 | 说明 |
|---|---|
| `<name>_basemap.tif` | 本批次底图，平台会自动切片；用 prepare 的 composite.tif（5 cm）或大疆 result.tif 都可以 |
| `<name>_tree_point.shp` | 树点：`ID, Lon, Lat, Pest(=0), tree_area, hash, conf, ndvi, ndre` |
| `<name>_tree_crown.shp` | 树冠多边形：`ID, tree_area, hash`；已按 `--simplify`（默认 0.05 m）简化，平台单株几何上限 16 KB |
| `<name>_pest_polygon.shp` | 病虫害多边形，本项目无检测结果时为空图层（平台要求文件必须存在） |
| `<name>.csv` | `PT_ID, PT_hash, Pest, tree_area, ndvi_mean, ndre_mean`；无 Yield 列，平台会告警并按无产量处理 |

导入时平台按 `detected_at` 排时间线，所以 `--date` 填飞行日期。ID 用 `registry` 的 `tree_id` 时跨批次一致；直接用 predict 的 crowns.gpkg 则每批次从 T000001 重新编号。

## 已知限制

- 目前只有一类（tree），不区分八角与其它树种；训练区域外的杂木也会被检出，评估时用 `--footprint` 限定范围。
- 圆形伪标签的冠幅面积来自供应商估计值，训练出的模型冠幅偏差会继承该误差。
- `registry` 的匹配只用质心距离，树冠大幅变化或密植错位时可能串号，需要人工抽查。
