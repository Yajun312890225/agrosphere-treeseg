# CLAUDE.md

本文件为 Claude Code 在本仓库工作时的指引。

## 项目定位

八角单株树冠提取的深度学习项目，服务于 AgroSphere 平台（`/Users/xunyou/Desktop/AgroSphere`，go-zero 后端）的 AI 单树监测链路。用无人机正射影像（大疆智图成果）训练 Ultralytics YOLO-seg 实例分割模型，输出逐株树冠多边形、树点、冠幅面积、逐株光谱指数，并维护跨期单株 ID 台账。**病虫害与产量不在本项目范围**（见下文"已验证结论"）。

## 常用命令

```bash
# 环境（不要用 --system-site-packages 复用 Homebrew 的 numpy/scipy，会和 torch 的 libomp 冲突段错误）
conda create -n treeseg python=3.11 && conda activate treeseg
conda install -c conda-forge gdal
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # 按显卡选
pip install -r requirements.txt

# 全流程（路径见 scripts/run_example.sh）
python -m treeseg.prepare  --terra-dir <DJI Terra 目录> --out <work>/prep --gsd 0.05 --ms-gsd 0.2 --te XMIN YMIN XMAX YMAX
python -m treeseg.labels   --raster <work>/prep/composite.tif --source <供应商 shp/csv> --type points|polygons ... --out <work>/labels.gpkg
python -m treeseg.dataset  --raster <work>/prep/composite.tif --labels <work>/labels.gpkg --out <work>/dataset --prefix <期次>
python -m treeseg.train    --data <work>/dataset/data.yaml --model yolo11m-seg.pt --epochs 100 --imgsz 1024 --batch 8
python -m treeseg.predict  --raster <work>/prep/composite.tif --weights best.pt --out <work>/pred --ms-dir <work>/prep
python -m treeseg.evaluate --pred <work>/pred/crowns.gpkg --ref <供应商树点> --auto-shift
python -m treeseg.registry --registry registry.gpkg --new <work>/pred/crowns.gpkg --date YYYY-MM-DD --out registry.gpkg
```

本机（Mac，16 GB）可用环境：`~/nahua_work/venv_dl/bin/python`（干净 venv，含 GDAL 3.12 绑定 + ultralytics）。跑任何重计算都加 `nice -n 19` 和 `OMP_NUM_THREADS=2`，用户明确要求不要拖慢机器；不要整幅加载 5cm 影像到内存。

## 代码结构

```
treeseg/geo.py       坐标/矢量读写/平移估计(互相关+ICP)/一对一匹配；vector_epsg 等要保持 GDAL 数据源引用
treeseg/prepare.py   DJI Terra 目录 → composite.tif(5cm RGB 或 cir) + ms_*.tif/ndvi.tif/dsm.tif(20cm)，全部 gdalwarp 子进程
treeseg/labels.py    供应商结果 → labels.gpkg；points 类型按 tree_area 生成圆并可用 NDVI 精修
treeseg/dataset.py   切片 + YOLO seg 标签，按空间块划分 train/val
treeseg/train.py     ultralytics 训练封装（翻转/旋转增强，无 mixup）
treeseg/predict.py   滑窗推理 → 核心区过滤 → 跨切片 IoU NMS → gpkg/csv + 逐株 NDVI/NDRE/GNDVI
treeseg/evaluate.py  与参考树点/树冠比对：召回、精度、欠分割（含 2 点以上树冠数）、冠幅相关
treeseg/registry.py  跨期 ID 台账（平移估计 + 质心 1.5m 内匹配继承 ID）
treeseg/export_platform.py  打包成 AgroSphere AI 单树监测导入 zip（底图 tif + 树点/树冠/病虫害多边形 shp + csv）
```

约定：注释、日志、错误信息用简体中文；矢量统一 EPSG:4544（CGCS2000 3 度带 105E）；GDAL Python 对象注意生命周期（`ogr.Open(p).GetLayer()` 这种链式写法会崩）。

## 数据事实（截至 2026-09-09）

- `/Users/xunyou/Downloads/map`：DJI Terra 成果，Mavic 3M 于 **2026-05-24** 飞（1855 张，航高 37.6 m，GSD 1.3 cm，RTK 全 SINGLE，无反射率标定）。含 RGB、四个单波段多光谱、DSM、index_map、XYZ 瓦片。
- `/Users/xunyou/Desktop/宏哥`：供应商 **2026-03-22** 结果：6591 株树点（ID/Lon/Lat）、`宏哥八角-ID-病虫害-估产.csv`（Pest 0/1 共 442 株、Yield、tree_area）、病虫害矩形 351 / 多边形 451。树冠多边形 shp 主文件缺失。相对 0524 影像整体偏移 **dx=1.14 dy=1.39 m**。
- 供应商 6、7 月结果（`单株树冠估产.shp`，真实冠形，EPSG:4544，tree_id 跨月一致）已被用户删除；若再拿到，是最好的训练标签。
- `/Users/xunyou/Downloads/那花红哥八角0524.zip`（43 GB）、`那花周哥0811.zip`（46 GB）：原始 DJI 数据包。
- 本机工作目录 `~/nahua_work`：prep0524（合成影像）、labels_0322.gpkg、dataset（122 train / 13 val，8693 实例）、runs/smoke（1 轮冒烟权重，无实用价值）。

## 已验证结论（不要重复做）

- 传统方法（NDVI 掩膜 + DSM/NIR 树顶 + 分水岭）与供应商树点一对一匹配只有 48%～73%，郁闭林分欠分割严重，所以转深度学习。
- 0524 多光谱上，供应商 0322 病虫害标记无任何光谱差异（12 个特征 AUC 0.50～0.54，逻辑回归 CV AUC 0.59；病害多边形内 NDVI 反而更高），产量与特征 R² 0.07。**从影像复现供应商病虫害/产量不可行**，必须有同期地面调查数据。
- 大疆智图自带语义分割（segment.tif）不支持八角，整幅判为 other，无用。

## 下一步

1. 在 GPU 机器上正式训练（用户的智图 PC 有 RTX 5060 Ti 16G；或阿里云 gn7i 按量/抢占式，训练完释放；或 AutoDL 4090 约 ¥2/h）。
2. 拿到供应商树冠多边形后重新生成标签，多期合并训练。
3. ~~对接平台导入格式~~ 已完成：`export_platform` 输出已用 AgroSphere 导入代码（locateMonitorArchiveFiles → validate → readMonitorShapefile → buildMonitorTreeImports）验证通过，平台无需改造。
