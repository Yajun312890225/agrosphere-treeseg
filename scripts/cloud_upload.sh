#!/bin/bash
# 新开/重建训练机后一键准备：清 host key → 后台装环境 → 上传代码、标签、位移场、各期合成影像、流水线脚本。
# 用法：scripts/cloud_upload.sh <ip> [only-code]
#   only-code：只同步代码与脚本（机器没重建、数据还在时用）
# 上传总量约 2 GB（prep 5 期 ≈ 1.98 GB + 标签 0.17 GB）；环境安装与上传并行，各约 11～15 分钟。
set -e
H=${1:?用法: cloud_upload.sh <ip> [only-code]}
S=$(cd "$(dirname "$0")/.." && pwd)
D=$HOME/nahua_work
R="rsync -az --partial --inplace"
ssh-keygen -R "$H" >/dev/null 2>&1 || true
SSH="ssh -o StrictHostKeyChecking=accept-new $H"
$SSH 'mkdir -p /root/work/treeseg'
if [ "$2" != "only-code" ] && ! $SSH 'test -x /root/work/py'; then
  scp -q "$S/scripts/cloud_install.sh" "$H:/root/work/install.sh"
  $SSH 'chmod +x /root/work/install.sh; setsid nohup /root/work/install.sh > /root/work/install.log 2>&1 < /dev/null & disown'
  echo "[$(date +%T)] 环境安装已在云机后台启动（tail /root/work/install.log 看进度，INSTALL_DONE 为止）"
fi
echo "[$(date +%T)] 代码与脚本"
$R --exclude '__pycache__' --exclude '*.pt' --exclude 'weights' "$S/treeseg/" "$H:/root/work/treeseg/treeseg/"
$R "$S/scripts/"cloud_pipeline_v*.sh "$H:/root/work/"
$SSH 'chmod +x /root/work/cloud_pipeline_v*.sh'
[ "$2" = "only-code" ] && { echo "[$(date +%T)] 代码同步完成"; exit 0; }
echo "[$(date +%T)] 标签与位移场"
$R "$D"/labels_v06.gpkg "$D"/labels_v07.gpkg "$D"/labels_0723_v2.gpkg "$D"/labels_0811_v2.gpkg "$D"/labels_0524_v2.gpkg "$D"/labels_0322.gpkg "$D"/labels_sam_v2.gpkg "$H:/root/work/"
$R "$D"/field_0811.npz "$D"/field_v07_to_0723.npz "$D"/field_v07_to_0811.npz "$D"/field_v06_to_0524.npz "$H:/root/work/"
for p in "prep_v06 prepv06" "prep_v07 prepv07" "prep0723 prep0723" "prep0811 prep0811" "prep0524 prep0524"; do
  set -- $p
  echo "[$(date +%T)] $1 -> /root/work/$2 ($(du -sh "$D/$1" | cut -f1))"
  $R "$D/$1/" "$H:/root/work/$2/"
done
echo "[$(date +%T)] 上传完成；确认环境：ssh $H 'tail -3 /root/work/install.log'"
