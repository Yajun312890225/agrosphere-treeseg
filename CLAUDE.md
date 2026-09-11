# CLAUDE.md

本文件为 Claude Code 在本仓库工作时的指引。

## 项目定位

八角单株树冠提取的深度学习项目，服务于 AgroSphere 平台（`/Users/xunyou/Desktop/AgroSphere`，go-zero 后端）的 AI 单树监测链路。用无人机正射影像（大疆智图成果）训练 Ultralytics YOLO-seg 实例分割模型，输出逐株树冠多边形、树点、冠幅面积、逐株光谱指数，并维护跨期单株 ID 台账。**病虫害与产量不在本项目范围**（见下文"已验证结论"）。

训练全流程见项目技能 `.claude/skills/treeseg-train/SKILL.md`（/treeseg-train）。

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
# 只有树点时必须再过一遍 SAM2（圆形伪标签 YOLO 学不动，见"已验证结论"）；sam2.1_b.pt 从 ultralytics assets 下载
python -m treeseg.labels_sam --raster <work>/prep/composite.tif --labels <work>/labels.gpkg --sam sam2.1_b.pt --ndvi <work>/prep/ndvi.tif --out <work>/labels_sam.gpkg
python -m treeseg.dataset  --raster <work>/prep/composite.tif --labels <work>/labels_sam.gpkg --out <work>/dataset --prefix <期次> --block 3
python -m treeseg.train    --data <work>/dataset/data.yaml --model yolo11m-seg.pt --epochs 100 --imgsz 1024 --batch 8
python -m treeseg.predict  --raster <work>/prep/composite.tif --weights best.pt --out <work>/pred --ms-dir <work>/prep
python -m treeseg.evaluate --pred <work>/pred/crowns.gpkg --ref <供应商树点> --auto-shift
python -m treeseg.registry --registry registry.gpkg --new <work>/pred/crowns.gpkg --date YYYY-MM-DD --out registry.gpkg
python -m treeseg.verify_export --zip <export>/<name>.zip --out <work>/verify --ref <供应商树冠 gpkg> --detail XMIN YMIN XMAX YMAX   # 本地代替测试环境验证
```

本机（Mac，16 GB）可用环境：`~/nahua_work/venv_dl/bin/python`（干净 venv，含 GDAL 3.12 绑定 + ultralytics）。跑任何重计算都加 `nice -n 19` 和 `OMP_NUM_THREADS=2`，用户明确要求不要拖慢机器；不要整幅加载 5cm 影像到内存。

## 代码结构

```
treeseg/geo.py       坐标/矢量读写/平移估计(互相关+ICP)/一对一匹配；vector_epsg 等要保持 GDAL 数据源引用
treeseg/prepare.py   DJI Terra 目录 → composite.tif(5cm RGB 或 cir) + ms_*.tif/ndvi.tif/dsm.tif(20cm)，全部 gdalwarp 子进程
treeseg/labels.py    供应商结果 → labels.gpkg；points 类型按 tree_area 生成圆并可用 NDVI 精修
treeseg/labels_sam.py 以 labels.gpkg 质心为点提示跑 SAM2，多掩膜按 src_area 选最接近者 + NDVI 过滤，得到真实冠形标签（失败退回原标签）
treeseg/register_labels.py 把一期树冠标签按局部平移场配准到另一期影像（全局互相关 + 以模型预测为同名点的局部中位残差），用于大疆影像套供应商标签
treeseg/coregister.py 把一期影像及成果配准到参考坐标框架（--out-vector 会写 reg_dist 字段供 dataset.py 做标签质量守门；build_field 现为逐节点局部互相关，与旧的"全局平移+局部中位残差"效果相当，无标签区域的节点值不可信）：以两期树冠质心为同名点估计 25 m 格网位移场（可 --field-out/--field-in 复用），矢量按场平移，栅格以节点为 GCP 走 gdalwarp -tps。有供应商成果的期次以供应商树冠为 fixed，之后每期以上一期已配准预测为 fixed，保证跨期位置稳定。导出平台前必须做这一步（大疆 RTK 单点解偏 1～4 m 且不均匀）
treeseg/dataset.py   切片 + YOLO seg 标签，按空间块划分 train/val（--block 3 让验证块打散，6 会整块落在特殊区域）；--qa-field reg_dist 按切片剔除错位标签（错位标签当负样本比少一批样本伤得多）
treeseg/train.py     ultralytics 训练封装（翻转/旋转增强，无 mixup；nbs=batch 不做梯度累积，mask_ratio 默认 4）
treeseg/predict.py   滑窗推理 → 核心区过滤 → 跨切片 IoU NMS → gpkg/csv + 逐株 NDVI/NDRE/GNDVI（--tta 测试时增强）
treeseg/export_platform.py 导出前先 buffer 平滑再用 0.02 m 简化；不要用 0.05 m 简化 5 cm 像元轮廓，会砍成八边形
treeseg/evaluate.py  与参考树点/树冠比对：召回、精度、欠分割（含 2 点以上树冠数）、冠幅相关
treeseg/registry.py  跨期 ID 台账（平移估计 + 质心 1.5m 内匹配继承 ID）
treeseg/export_platform.py  打包成 AgroSphere AI 单树监测导入 zip（底图 tif + 树点/树冠/病虫害多边形 shp + csv）
treeseg/verify_export.py 移植平台导入解析逻辑（unzip → 定位 → 校验 → ogr2ogr → buildMonitorTreeImports），本地解析 zip 出 report.json + overview.jpg/.tif 大图 + --detail 放大窗 + --ref 比对，结果与测试库口径一致，不用上传测试环境
```

约定：注释、日志、错误信息用简体中文；矢量统一 EPSG:4544（CGCS2000 3 度带 105E）；GDAL Python 对象注意生命周期（`ogr.Open(p).GetLayer()` 这种链式写法会崩）。

## 数据事实（截至 2026-09-09）

- `/Users/xunyou/Downloads/map`：DJI Terra 成果，Mavic 3M 于 **2026-05-24** 飞（1855 张，航高 37.6 m，GSD 1.3 cm，RTK 全 SINGLE，无反射率标定）。含 RGB、四个单波段多光谱、DSM、index_map、XYZ 瓦片。
- `/Users/xunyou/Desktop/宏哥`：供应商 **2026-03-22** 结果：6591 株树点（ID/Lon/Lat）、`宏哥八角-ID-病虫害-估产.csv`（Pest 0/1 共 442 株、Yield、tree_area）、病虫害矩形 351 / 多边形 451。树冠多边形 shp 主文件缺失。相对 0524 影像整体偏移 **dx=1.14 dy=1.39 m**。
- 项目根目录下三个 zip（2026-09-10 已辨认，zip 内为 deflate，大 tif 需解压后用）：
  - `那花红哥八角0524.zip`（43 GB）= DJI Terra 成果 map/，与 `/Users/xunyou/Downloads/map` 完全相同（result.tif 6838006414 B），已处理为 prep0524。
  - `那花周哥0811.zip`（46 GB）= 同一块地 2026-08-11 的 DJI Terra 成果（EPSG:4326，1.1 cm，含 4 个多光谱 + DSM），已处理为 `~/nahua_work/prep0811`。
  - `那花6、7月_病虫害与估产合并版.zip`（14.5 GB）= 供应商成果：2026-06/2026-07 各有 RGB tif（1.2 cm，EPSG:4544）、多光谱 4 波段 tif（2 cm）、`单株病虫害估产.shp`（**真实树冠多边形**，5477/6116 株，字段 ID/Pest/Yield/Longitude/Latitude，两期 5358 个 ID 一致，Pest=1 共 223/275）。小文件解在 `~/nahua_work/vendor_0607/`，RGB 已处理为 prep_v06/prep_v07，多边形已转为 labels_v06.gpkg/labels_v07.gpkg（最好的训练标签）。
- `那花0723-DOM/`（20 GB，已解压的 DJI Terra 目录）：2026-07-21/22 航拍（目录名是处理日期），同一块地，含 4 波段多光谱 + DSM，是供应商 7 月结果的原料；已处理为 `~/nahua_work/prep0723`。
- 各期合成影像统一用 `--te 631042 2627881 631789 2628956`、5 cm、EPSG:4544，网格完全对齐。供应商 6 月树冠相对 0524 影像偏移 dx=0.88 dy=1.50 m。
- 本机工作目录 `~/nahua_work`：prep0524/prep0811/prep_v06/prep_v07、labels_0322/labels_sam_v2/labels_v06/labels_v07.gpkg、dataset（圆标签，勿用）、runs/sam_m2（SAM 标签第一版模型 best.pt）、pred_sam_m2(_c0.15)。

## 已验证结论（不要重复做）

- 传统方法（NDVI 掩膜 + DSM/NIR 树顶 + 分水岭）与供应商树点一对一匹配只有 48%～73%，郁闭林分欠分割严重，所以转深度学习。
- 0524 多光谱上，供应商 0322 病虫害标记无任何光谱差异（12 个特征 AUC 0.50～0.54，逻辑回归 CV AUC 0.59；病害多边形内 NDVI 反而更高），产量与特征 R² 0.07。**从影像复现供应商病虫害/产量不可行**，必须有同期地面调查数据。
- 大疆智图自带语义分割（segment.tif）不支持八角，整幅判为 other，无用。
- **圆形伪标签（树点 + tree_area）YOLO 学不动**：yolo11m-seg 训练集 mAP50 仅 0.11、验证集 0.005，换 ultralytics 版本/超参/纯检测都一样。改用 labels_sam.py 的 SAM2 冠形标签后训练集 0.39、验证集 0.27（2026-09-09，A10）。
- ultralytics 小数据集两个坑：默认 nbs=64 使 batch 8 每轮只更新 2 次；mask_ratio=2 在 24G 显存上 OOM 且无收益。均已在 train.py 处理。

## 下一步

1. ~~SAM 标签第一版~~ 已被供应商标签版取代：`~/nahua_work/runs/vendor_m/best.pt`（yolo11m-seg，v06+v07 合并 142 训/10 验，**batch 4**（batch 8 会在掩膜损失处 OOM），验证集 mAP50 0.81，2026-09-10）。与供应商树冠一对一匹配（conf 0.15，1.5 m）：供应商自家影像 v06/v07 上 F1 0.81（召回 78%～82%，精度 80%～85%，质心中位差 0.27 m）；大疆影像 0723/0811 上 F1 0.65（召回 59%～63%，精度 68%～73%），0524 上 F1 0.48。完整报告 `~/nahua_work/compare_v3_report.log`，各期预测 `~/nahua_work/pred_vendor_<期次>/`。
2. **当前最佳模型 `~/nahua_work/runs/mix3_m/best.pt`**（四期混合 + 标签质量守门，300 训/24 验，2026-09-10 深夜，`scripts/cloud_pipeline_v6.sh`）。**关键发现**：供应商 7 月树冠在大疆 0723/0811 影像上约 40% 的切片局部错位 >1 m（供应商正射西北、东南边缘几何畸变，位移场修不平），mix2 把这些切片学成"没有树"，西侧 1 ha（354 株）只检出 31 株。处理：`coregister.py --out-vector` 写 `reg_dist`（到最近模型预测的距离），`dataset.py --qa-field reg_dist --qa-thr 1.0 --qa-min-frac 0.4` 整片剔除（逐切片达标比双峰，谷在 0.4～0.5；0723/0811 各剔 47/45 片）。mix3 在该区检出 369 株；**只在标签对得上的 76 个切片内比**（错位切片里正确检出会被记成误检，全图精度指标失真）：mix_m F1 0.864、mix2 0.882、mix3 0.878，而错位切片内 mix2 只检 542 株、mix3 1614 株（标签 1957）。全图：0811 召回 68%、精度 78%（受错位标签拖低）；v07 F1 0.855；0723 0.742。标签 labels_0723_v2/labels_0811_v2.gpkg（含 reg_dist），预测 `~/nahua_work/pred_mix3_0811/`（crowns_reg.gpkg 为配准版）。测试环境批次 17 已替换为 mix3（5892 株，source_layer treeseg-mix3-reg）：相对供应商批次 16，标签对得上区域的 4224 株中位距离 0.28 m、1 m 内 84%；其余 1668 株（供应商正射西北/东南畸变区）中位 2.75 m，这是供应商成果自身的几何问题，25 m 位移场修不平，两套框架在那里注定对不上，需用户决定以哪方为准。上一版 mix2 记录如下：
   **mix2 `~/nahua_work/runs/mix2_m/best.pt`**（v06 + v07 + 0723 + 0811 四期混合，378 训/39 验，2026-09-10 晚）：0811 上召回 65%、精度 86%、F1 0.74（conf 0.10）；v07 F1 0.84；0723 F1 0.76。测试环境批次 17 已用它的配准结果替换：相对供应商批次 16 的树点中位距离 2.08→0.31 m，1 m 内占比 16%→75%。ultralytics 分割模型不支持 TTA（augment 静默无效）。上一版 mix_m 记录如下：
   **mix_m `~/nahua_work/runs/mix_m/best.pt`**（v06 + v07 + 大疆 0723 三期混合，266 训/24 验，batch 4，150 轮，验证集 mAP50 0.73）。用 `treeseg/register_labels.py` 把供应商 7 月树冠按局部平移场配准到大疆影像（以模型预测为同名点，残差中位 0.67→0.30 m）得到 labels_0723/labels_0811.gpkg。与配准后的供应商树冠比对：供应商影像 v06/v07 F1 0.84/0.85（精度 87%～90%），大疆 0723 F1 0.75（召回 69%、精度 81%），0811 F1 0.71（标签比影像早 3 周），0524 F1 0.46（5 月物候无训练样本）。各期预测 `~/nahua_work/pred_mix_<期次>/`，报告 `compare_v4_mix_report.log`。
3. 大疆影像上差的主要原因是配准而非检测：大疆智图成果（RTK SINGLE）相对供应商成果整体偏 0723 dx=3.15 dy=4.14 m、0811 1.21/3.36 m、0524 0.93/1.46 m，全局平移后质心中位差仍 0.67 m（供应商影像上 0.27 m），说明偏移不均匀。下一步：a) 把 0723 影像 + 按平移量套上的 v07 标签加进训练集，让模型直接学大疆影像的色彩和纹理；b) 无人机后续航飞用 RTK 固定解或布控制点；c) 供应商标签只做训练，验收应以同期地面抽查为准。
3. 云机脚本：`scripts/cloud_pipeline_v6.sh`（守门数据集 → 训练 → 各期预测评估 → 配准 → 打包，当前版本）；v5 为 mix2，v3 为更早的三段式。导入测试环境：zip 传到 99 的 `/opt/agrosphere-test/data/import/`，admin 登录后 POST `/api/v1/plot-analysis/import` 带 `local_archive_path=/app/data/import/<zip>` 与 `analysis_id=17` 原地替换。
3. ~~对接平台导入格式~~ 已完成：`export_platform` 输出已用 AgroSphere 导入代码（locateMonitorArchiveFiles → validate → readMonitorShapefile → buildMonitorTreeImports）验证通过，平台无需改造。
