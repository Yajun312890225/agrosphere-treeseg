#!/bin/bash
# 训练机（阿里云 gn7i / A10，Ubuntu 22.04，全空系统盘）环境安装，约 11 分钟。
# miniconda(清华) + GDAL(conda-forge 清华) + torch/ultralytics(阿里内网 pypi) + 权重(ghfast 镜像)。
# 用法（本机）：scripts/cloud_upload.sh <ip> 会自动把本脚本传上去并后台执行；日志 /root/work/install.log，结束行 INSTALL_DONE。
set -e
cd /root
E=/root/miniconda3/envs/treeseg
echo "[$(date +%T)] 1/5 miniconda"
wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /root/miniconda3 >/dev/null
/root/miniconda3/bin/conda create -y -q -n treeseg python=3.11 gdal -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ --override-channels >/dev/null 2>&1
echo "[$(date +%T)] 2/5 torch"
M="-i http://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com"
$E/bin/pip install -q $M torch torchvision 2>&1 | grep -iE "error" || true
echo "[$(date +%T)] 3/5 ultralytics 8.3.253 + 依赖"
$E/bin/pip install -q $M "numpy>=1.26" "scipy>=1.11" "scikit-image>=0.22" "shapely>=2.0" pyyaml tqdm "ultralytics==8.3.253" 2>&1 | grep -iE "error" || true
echo "[$(date +%T)] 4/5 包装脚本 + 权重"
cat > /root/work/py <<"EOP"
#!/bin/bash
# treeseg 环境 python 包装：设置 PROJ/GDAL 数据路径、显存分配策略，并把 env 的 bin（gdalwarp/ogr2ogr）加进 PATH
E=/root/miniconda3/envs/treeseg
export PROJ_DATA=$E/share/proj GDAL_DATA=$E/share/gdal PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=$E/bin:$PATH
exec $E/bin/python "$@"
EOP
chmod +x /root/work/py
mkdir -p /root/work/treeseg/weights /root/.config/Ultralytics
G=https://ghfast.top/https://github.com/ultralytics/assets/releases/download
curl -sL -o /root/work/yolo11m-seg.pt $G/v8.3.0/yolo11m-seg.pt
curl -sL -o /root/work/treeseg/weights/yolo11n.pt $G/v8.3.0/yolo11n.pt
curl -sL -o /root/work/treeseg/sam2.1_b.pt $G/v8.3.0/sam2.1_b.pt
ls -la /root/work/yolo11m-seg.pt /root/work/treeseg/weights/yolo11n.pt /root/work/treeseg/sam2.1_b.pt
echo "[$(date +%T)] 5/5 验证"
/root/work/py -c "
import torch, ultralytics
from osgeo import gdal, osr
s=osr.SpatialReference(); s.ImportFromEPSG(4544)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('ultralytics', ultralytics.__version__, 'GDAL', gdal.__version__, 'EPSG:4544', s.GetName())"
echo "[$(date +%T)] INSTALL_DONE"
