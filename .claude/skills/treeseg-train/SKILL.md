---
name: treeseg-train
description: 那花八角单株树冠 YOLO-seg 模型的完整训练流程：新一期影像预处理、供应商标签配准与切片级质量守门、云机（A10）训练、各期预测评估、配准到统一坐标框架、打包平台导入 zip，并用 verify_export 在本地核对结果（不上传测试环境）。当用户要"重训模型 / 加一期数据训练 / 处理新一期影像出结果 / 验证导出包"时使用。
---

# treeseg-train

项目 `/Users/xunyou/Desktop/agrosphere-treeseg`，数据在 `~/nahua_work`，训练在阿里云 A10（`ssh 101.37.237.221`，见 CLAUDE.md 与记忆）。本机只做矢量计算与出图，重计算一律 `nice -n 19` + `OMP_NUM_THREADS=2`。

## 0. 检查

```bash
ssh 101.37.237.221 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; ls /root/work; df -h /root | tail -1'
```
机器不在或 `/root/work` 空：按 `/root/work/install.sh`（或记忆里的安装步骤）重装，约 11 分钟。**所有 python 用包装脚本 `/root/work/py`**。

## 1. 新一期影像预处理（本机）

```bash
PY=~/nahua_work/venv_dl/bin/python; RUN="OMP_NUM_THREADS=2 nice -n 19 $PY"
# 大疆智图目录（含 result.tif + result_*.tif 多光谱 + dsm.tif）
$RUN -m treeseg.prepare --terra-dir <DJI目录> --out ~/nahua_work/prep<期次> --gsd 0.05 --ms-gsd 0.2 --te 631042 2627881 631789 2628956
# 供应商单张正射
$RUN -m treeseg.prepare --rgb <正射.tif> --out ~/nahua_work/prep_v<期次> --gsd 0.05 --te 631042 2627881 631789 2628956
```
所有期次同一 `--te`、5 cm、EPSG:4544，网格对齐是后面一切叠加的前提。

## 2. 标签

- 供应商树冠多边形：`treeseg.labels --type polygons` → `labels_v<期>.gpkg`。
- 只有树点：`treeseg.labels --type points` 再过 `treeseg.labels_sam`（圆形伪标签学不动）。
- **把供应商标签套到大疆影像上必须配准 + 写 reg_dist**（用当前最佳模型在该期影像上的预测做同名点）：
```bash
$RUN -m treeseg.predict --raster ~/nahua_work/prep<期>/composite.tif --weights <当前best.pt> --out ~/nahua_work/pred_tmp_<期> --conf 0.15   # 云机上跑更快
$RUN -m treeseg.coregister --moving ~/nahua_work/labels_v07.gpkg --fixed ~/nahua_work/pred_tmp_<期>/crowns.gpkg \
    --out-vector ~/nahua_work/labels_<期>_v2.gpkg --field-out ~/nahua_work/field_v07_to_<期>.npz
```
供应商正射边缘有 2～5 m 非平滑畸变，约 40% 切片对不上，这些切片**必须**在 dataset 阶段剔除，否则模型把那里学成背景。

## 3. 上传云机

```bash
H=101.37.237.221
rsync -az --exclude '__pycache__' --exclude '*.pt' --exclude '*.zip' --exclude '那花*' ~/Desktop/agrosphere-treeseg/treeseg/ $H:/root/work/treeseg/treeseg/
rsync -az ~/nahua_work/labels_*.gpkg $H:/root/work/
rsync -az ~/nahua_work/prep<期>/ $H:/root/work/prep<期>/        # 每期约 0.3～0.6 GB
rsync -az ~/Desktop/agrosphere-treeseg/scripts/cloud_pipeline_v6.sh $H:/root/work/
```

## 4. 训练 + 预测 + 打包（云机，一条脚本）

以 `scripts/cloud_pipeline_v6.sh` 为模板复制一份改期次列表、模型名（mix<N>_m）与守门参数，然后：
```bash
ssh $H 'setsid nohup /root/work/cloud_pipeline_v6.sh > /root/work/pipeline_v6.log 2>&1 < /dev/null & disown'
ssh $H 'cut -d, -f1,9,13 /root/work/runs/mix3_m/results.csv | tail -1; grep -E "^\[|===|匹配" /root/work/pipeline_v6.log | tail'   # 进度
```
脚本做的事：dataset（v06/v07 直接用，大疆期 `--qa-field reg_dist --qa-thr 1.0 --qa-min-frac 0.4`，`--block 3 --seed 1`）→ train（yolo11m-seg，150 轮，imgsz 1024，**batch 4**，mask_ratio 4，patience 40）→ predict 目标期（conf 0.10，NMS 0.3）→ evaluate 各期 → 漏检区计数 → coregister `--field-in field_<期>.npz` 到供应商框架 → export_platform（底图用 `prep<期>_reg/composite.tif`）。300 张约 60 分钟，378 张约 80 分钟。

新一期第一次做配准底图时（只需一次）：
```bash
ssh $H 'cd /root/work/treeseg; W=/root/work; setsid nohup $W/py -m treeseg.coregister --moving $W/pred_<期>/crowns.gpkg --fixed <参考树冠.gpkg> \
  --rasters $W/prep<期>/composite.tif $W/prep<期>/ms_Red.tif $W/prep<期>/ms_Green.tif $W/prep<期>/ms_RedEdge.tif $W/prep<期>/ms_NIR.tif $W/prep<期>/ndvi.tif \
  --out-dir $W/prep<期>_reg --field-out $W/field_<期>.npz --threads 6 > $W/coreg_<期>.log 2>&1 < /dev/null & disown'
```
参考框架：有供应商成果的期次用供应商树冠；否则用上一期**已配准**的预测。

## 5. 拉回与本地验证（不上传测试环境）

```bash
mkdir -p ~/nahua_work/runs/mix<N>_m ~/nahua_work/pred_mix<N>_<期>
rsync -az $H:/root/work/runs/mix<N>_m/weights/best.pt $H:/root/work/runs/mix<N>_m/results.csv ~/nahua_work/runs/mix<N>_m/
rsync -az $H:/root/work/pred_mix<N>_<期>/ ~/nahua_work/pred_mix<N>_<期>/
rsync -az --partial --inplace $H:/root/work/export_<期>_mix<N>/*.zip ~/nahua_work/export_<期>/   # 约 0.5 MB/s，386 MB 十几分钟；勿并发两个 rsync
$RUN -m treeseg.verify_export --zip ~/nahua_work/export_<期>/<name>.zip --out ~/nahua_work/verify_<期> \
    --ref ~/nahua_work/labels_v07.gpkg --detail 631247 2628440 631348 2628543
```
verify_export 按平台导入代码同口径解析（株数、>16KB 丢几何、告警、2 GB 解码预算），输出 `report.json`、`trees.csv`、`overview.jpg/.tif`（可进 QGIS）、`detail_k.jpg`，并给出与参考的距离分布和一对一匹配。看图重点：西侧 `631247 2628440 631348 2628543` 这 1 ha（历史漏检区）、密植区 `631301 2628131 631477 2628210`。

## 6. 判断好坏

- 只看"标签对得上的切片"内的召回/精度（错位切片里正确检出会被记成误检）；全图指标受供应商标签错位拖低。
- 平台位置一致性：对得上区域中位距离应 ≤0.3 m、1 m 内 ≥80%；畸变区 2～3 m 是供应商成果问题。
- 结果好则更新 CLAUDE.md「下一步」里的当前最佳模型与数字。

## 已知结论（不要重做）

圆形伪标签不可学；TTA 无效；batch 8 OOM；0524（5 月）无训练样本 F1 只有 0.46；病虫害/产量无法从影像复现。
