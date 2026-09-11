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
机器不在或 `/root/work` 空（用户重建实例后磁盘全空、可能换 IP）：`scripts/cloud_upload.sh <ip>` 一键完成清 host key → 后台装环境（`scripts/cloud_install.sh`，约 11 分钟）→ 上传代码、标签、位移场、五期合成影像（约 2.1 GB）。只改了代码时 `scripts/cloud_upload.sh <ip> only-code`。**所有 python 用包装脚本 `/root/work/py`**。

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
- **该期没有可靠预测时**（模型召回不到一半，如 0524）：只用预测做 `--fixed` 会把漏检记成错位、守门分布是平的。参考点要用 SAM 冠形标签（`labels_sam` 以已配准树点做提示在该期影像上生成）∪ 各模型预测，质心 1 m 内去重后再做 fixed（0524 的做法：`fixed_0524_ref.gpkg`，守门 106/107 通过）。
- 造好标签后先在本机看逐切片达标比直方图（对齐好的期次是单峰或双峰谷在 0.4～0.5）再决定阈值，不要盲跑。

## 3. 上传云机

```bash
scripts/cloud_upload.sh <ip>             # 全量：环境 + 代码 + 标签 + 五期影像（新增期次先把 prep 目录与标签加进脚本清单）
scripts/cloud_upload.sh <ip> only-code   # 只同步代码与 cloud_pipeline_v*.sh
```
本机 prep_v06/prep_v07 在云机上叫 prepv06/prepv07（脚本已映射）。

## 4. 训练 + 预测 + 打包（云机，一条脚本）

以 `scripts/cloud_pipeline_v7.sh`（当前版本，mix4）为模板复制一份改期次列表、模型名（mix<N>_m）与守门参数，然后：
```bash
ssh $H 'setsid nohup /root/work/cloud_pipeline_v7.sh > /root/work/pipeline_v7.log 2>&1 < /dev/null & disown'
ssh $H 'cut -d, -f1,9,13 /root/work/runs/mix4_m/results.csv | tail -1; grep -E "^\[|===|匹配|株" /root/work/pipeline_v7.log | tail'   # 进度
```
脚本做的事：dataset（v06/v07 直接用，大疆期 `--qa-field reg_dist --qa-thr 1.0 --qa-min-frac 0.4`，`--block 3 --seed 1`）→ train（yolo11m-seg，150 轮，imgsz 1024，**batch 4**，mask_ratio 4，patience 40）→ predict 大疆三期（conf 0.10，NMS 0.3）+ v07 对照 → evaluate 各期 → 漏检区计数 → coregister 到供应商框架（0811 复用 `field_0811.npz` 保证底图不变，其他期以 labels_v07 为 fixed 估计并保存 `field_<期>.npz`，顺带把 composite 重采样成 `prep<期>_reg`）→ registry 台账 0524→0723→0811（`--no-auto-shift`）→ 三期各打一个 zip（ID 来自台账）。300 张约 60 分钟，400 张约 90 分钟；整条流水线约 2.5～3 小时，结束行 ALL_DONE。

新一期第一次做配准底图时（只需一次）：
```bash
ssh $H 'cd /root/work/treeseg; W=/root/work; setsid nohup $W/py -m treeseg.coregister --moving $W/pred_<期>/crowns.gpkg --fixed <参考树冠.gpkg> \
  --rasters $W/prep<期>/composite.tif $W/prep<期>/ms_Red.tif $W/prep<期>/ms_Green.tif $W/prep<期>/ms_RedEdge.tif $W/prep<期>/ms_NIR.tif $W/prep<期>/ndvi.tif \
  --out-dir $W/prep<期>_reg --field-out $W/field_<期>.npz --threads 6 > $W/coreg_<期>.log 2>&1 < /dev/null & disown'
```
参考框架：有供应商成果的期次用供应商树冠；否则用上一期**已配准**的预测。

## 5. 验证与拉回（不上传测试环境）

云机→本机只有 0.5～1.3 MB/s，所以**先在云机跑 verify_export，只拉 20 MB 的图和报告**，zip 后台慢拉：
```bash
# 云机：两版都出——带供应商紫线（--ref）和只画我们的（不带 --ref，用户主要看这版）
ssh $H 'cd /root/work/treeseg && setsid nohup bash -c "for per in 0524 0723 0811; do /root/work/py -m treeseg.verify_export --zip /root/work/export_\${per}_mix<N>/nahua_\${per}_mix<N>.zip --out /root/work/verify_\${per}_mix<N> --ref /root/work/labels_v07.gpkg --detail 631247 2628440 631348 2628543 --detail 631301 2628131 631477 2628210 --threads 8; done; echo VERIFY_DONE" > /root/work/verify.log 2>&1 < /dev/null & disown'
for p in 0524 0723 0811; do rsync -az --exclude extract $H:/root/work/verify_${p}_mix<N>/ ~/nahua_work/verify_${p}_mix<N>/; done
# 本机出"只有我们"的版本（zip 拉回后，几秒一期）：
$RUN -m treeseg.verify_export --zip ~/nahua_work/export_<期>_mix<N>/nahua_<期>_mix<N>.zip --out ~/nahua_work/verify_<期>_mix<N>_own --detail 631247 2628440 631348 2628543 --detail 631301 2628131 631477 2628210
```
verify_export 按平台导入代码同口径解析（株数、>16KB 丢几何、告警、2 GB 解码预算），输出 `report.json`、`trees.csv`、`overview.jpg/.tif`（可进 QGIS）、`detail_k.jpg`。看图重点：西侧 `631247 2628440 631348 2628543` 这 1 ha（历史漏检区）、密植区 `631301 2628131 631477 2628210`。

**释放云机前拉回**（后台 `nohup … & disown`，同一文件勿并发两个 rsync）：
```bash
rsync -az $H:/root/work/runs/mix<N>_m/weights/best.pt $H:/root/work/runs/mix<N>_m/results.csv ~/nahua_work/runs/mix<N>_m/
for p in 0524 0723 0811; do rsync -az $H:/root/work/pred_mix<N>_$p/ ~/nahua_work/pred_mix<N>_$p/; done
rsync -az $H:/root/work/registry_mix<N>.gpkg $H:/root/work/field_*.npz $H:/root/work/*.log ~/nahua_work/
for p in 0524 0723 0811; do rsync -az --partial --inplace $H:/root/work/export_${p}_mix<N>/*.zip ~/nahua_work/export_${p}_mix<N>/; done   # 400 MB 5～13 分钟/个
python -c "import zipfile,sys; [print(p, zipfile.ZipFile(p).testzip() is None) for p in sys.argv[1:]]" ~/nahua_work/export_*_mix<N>/*.zip
```
数据集与 `prep*_reg` 可重新生成，不用拉。拉完校验后告诉用户"可以回收"。

## 6. 判断好坏

- 只看"标签对得上的切片"内的召回/精度（错位切片里正确检出会被记成误检）；全图指标受供应商标签错位拖低。
- 平台位置一致性：对得上区域中位距离应 ≤0.3 m、1 m 内 ≥80%；畸变区 2～3 m 是供应商成果问题。
- 召回低先看是影像还是模型：在训练切片上叠标签（黄）与预测（青）出诊断图（做法见 pipeline 后的 diag 脚本思路：读 dataset 切片 + labels txt + crowns.gpkg）。0524 杂草区 159 株标签只检 27、干净区 160 检 172，就是影像问题，加标签没用。
- 结果好则更新 CLAUDE.md「下一步」里的当前最佳模型与数字。

## 7. 收尾

更新 CLAUDE.md（模型演进、结果位置）、`~/.claude/projects/.../memory/`、Obsidian「那花八角单株提取-项目总览」等笔记（/obs），提交 git（大文件已 .gitignore）。

## 已知结论（不要重做）

圆形伪标签不可学；TTA 无效；batch 8 OOM；病虫害/产量无法从影像复现。
0524（5 月）：无样本时 F1 0.46，mix4 加了 100 张五月切片仍只有召回 42%——杂草未割、RGB 上无冠缘，是影像问题，不要再靠加标签解决（要割草后飞或加 DSM 通道）。
供应商 6 月与 7 月树冠同 ID 几何**完全相同**（7 月只是复制 6 月再改属性、加了 758 株），所以 v06 换 v07 不会带来新的位置信息。
0524 标签（`labels_0524_v2.gpkg`）的配准参考点用的是 SAM 冠形标签 ∪ 三套预测（模型在 0524 上召回不到一半，只用预测做参考会把漏检当错位）。
