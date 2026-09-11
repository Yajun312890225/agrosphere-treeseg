"""把大疆智图（DJI Terra）成果目录转换为训练/推理用的合成影像。

输入目录需包含 result.tif（RGB 正射），可选 result_Red/Green/RedEdge/NIR.tif（多光谱）与 dsm.tif。
输出（均为目标 EPSG 投影坐标）：
  composite.tif  3 波段 Byte，训练/推理输入（rgb 或 cir=NIR/Red/Green 假彩色）
  ms_<band>.tif  多光谱各波段 Float32（ms-gsd 分辨率，供逐株指数统计）
  ndvi.tif       Float32 NDVI（ms-gsd 分辨率，供标签精修与指数统计）
  dsm.tif        Float32（ms-gsd 分辨率）
"""
import argparse
import os
import subprocess
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
MS_BANDS = ["Red", "Green", "RedEdge", "NIR"]


def warp(src, dst, epsg, gsd, te, threads, extra):
    cmd = ["gdalwarp", "-q", "-overwrite", "-t_srs", f"EPSG:{epsg}", "-tr", str(gsd), str(gsd), "-tap",
           "-r", "average", "-wo", f"NUM_THREADS={threads}", "-wm", "256",
           "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER"]
    if te:
        cmd += ["-te", *map(str, te)]
    cmd += extra + [src, dst]
    print("  $", " ".join(cmd[:6]), "...", os.path.basename(src), "->", os.path.basename(dst), flush=True)
    subprocess.run(cmd, check=True)


def stretch_to_byte(arr, valid, lo=2, hi=98):
    v = arr[valid]
    if v.size == 0:
        return np.zeros(arr.shape, np.uint8)
    a, b = np.percentile(v, [lo, hi])
    out = np.clip((arr - a) / max(b - a, 1e-6) * 254 + 1, 1, 255)
    out[~valid] = 0
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terra-dir", default=None, help="DJI Terra 成果目录（含 result.tif 等）")
    ap.add_argument("--rgb", default=None, help="直接指定 RGB 正射 tif（如供应商 RGB_6月.tif），不走 Terra 目录布局；与 --terra-dir 二选一或同时给")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--epsg", type=int, default=4544, help="目标投影 EPSG，默认 4544（CGCS2000 3度带 105E）")
    ap.add_argument("--gsd", type=float, default=0.05, help="composite 分辨率(m)，默认 0.05")
    ap.add_argument("--ms-gsd", type=float, default=0.2, help="多光谱/NDVI/DSM 分辨率(m)，默认 0.2")
    ap.add_argument("--bands", choices=["rgb", "cir"], default="rgb", help="composite 波段组合")
    ap.add_argument("--te", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), help="裁剪范围（目标投影坐标）")
    ap.add_argument("--threads", type=int, default=2, help="gdalwarp 线程数，默认 2 以免拖慢机器")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.terra_dir and not a.rgb:
        raise SystemExit("需要 --terra-dir 或 --rgb")
    td = a.terra_dir or ""
    rgb_src = a.rgb or os.path.join(td, "result.tif")
    has_ms = bool(td) and all(os.path.exists(os.path.join(td, f"result_{b}.tif")) for b in MS_BANDS)
    has_dsm = bool(td) and os.path.exists(os.path.join(td, "dsm.tif"))
    print(f"RGB: {rgb_src}；多光谱波段: {'有' if has_ms else '无'}；DSM: {'有' if has_dsm else '无'}")

    # 1. 多光谱 + NDVI + DSM（低分辨率，用于统计）
    if has_ms:
        for b in MS_BANDS:
            warp(os.path.join(td, f"result_{b}.tif"), os.path.join(a.out, f"ms_{b}.tif"), a.epsg, a.ms_gsd, a.te, a.threads,
                 ["-ot", "Float32", "-dstnodata", "nan"])
        r = gdal.Open(os.path.join(a.out, "ms_Red.tif"))
        n = gdal.Open(os.path.join(a.out, "ms_NIR.tif"))
        R = r.GetRasterBand(1).ReadAsArray().astype(np.float32)
        N = n.GetRasterBand(1).ReadAsArray().astype(np.float32)
        ndvi = (N - R) / (N + R + 1e-6)
        ndvi[~(np.isfinite(R) & np.isfinite(N) & (N > 0))] = np.nan
        o = gdal.GetDriverByName("GTiff").Create(os.path.join(a.out, "ndvi.tif"), r.RasterXSize, r.RasterYSize, 1, gdal.GDT_Float32,
                                                 ["COMPRESS=DEFLATE", "TILED=YES"])
        o.SetGeoTransform(r.GetGeoTransform())
        o.SetProjection(r.GetProjection())
        o.GetRasterBand(1).SetNoDataValue(float("nan"))
        o.GetRasterBand(1).WriteArray(ndvi)
        o.FlushCache()
        del R, N, ndvi, o
    if has_dsm:
        warp(os.path.join(td, "dsm.tif"), os.path.join(a.out, "dsm.tif"), a.epsg, a.ms_gsd, a.te, a.threads,
             ["-ot", "Float32", "-srcnodata", "-9999", "-dstnodata", "-9999"])

    # 2. composite
    comp = os.path.join(a.out, "composite.tif")
    if a.bands == "rgb":
        warp(rgb_src, comp, a.epsg, a.gsd, a.te, a.threads,
             ["-ot", "Byte", "-b", "1", "-b", "2", "-b", "3", "-srcnodata", "0", "-dstnodata", "0"])
    else:
        if not has_ms:
            raise SystemExit("bands=cir 需要多光谱波段")
        tmp = {}
        for b in ["NIR", "Red", "Green"]:
            tmp[b] = os.path.join(a.out, f"_tmp_{b}.tif")
            warp(os.path.join(td, f"result_{b}.tif"), tmp[b], a.epsg, a.gsd, a.te, a.threads, ["-ot", "Float32", "-dstnodata", "nan"])
        ref = gdal.Open(tmp["NIR"])
        o = gdal.GetDriverByName("GTiff").Create(comp, ref.RasterXSize, ref.RasterYSize, 3, gdal.GDT_Byte,
                                                 ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"])
        o.SetGeoTransform(ref.GetGeoTransform())
        o.SetProjection(ref.GetProjection())
        # 分块拉伸：先在抽样上定拉伸阈值，再逐块写出，避免整幅读入内存
        for i, b in enumerate(["NIR", "Red", "Green"]):
            ds = gdal.Open(tmp[b])
            band = ds.GetRasterBand(1)
            sample = band.ReadAsArray(buf_xsize=min(2000, ds.RasterXSize), buf_ysize=min(2000, ds.RasterYSize)).astype(np.float32)
            lo, hi = np.nanpercentile(sample[np.isfinite(sample) & (sample > 0)], [2, 98])
            ob = o.GetRasterBand(i + 1)
            ob.SetNoDataValue(0)
            bs = 2048
            for y in range(0, ds.RasterYSize, bs):
                h = min(bs, ds.RasterYSize - y)
                arr = band.ReadAsArray(0, y, ds.RasterXSize, h).astype(np.float32)
                valid = np.isfinite(arr) & (arr > 0)
                out = np.clip((arr - lo) / max(hi - lo, 1e-6) * 254 + 1, 1, 255)
                out[~valid] = 0
                ob.WriteArray(out.astype(np.uint8), 0, y)
            ds = None
            os.remove(tmp[b])
        o.FlushCache()
    print("完成：", comp)


if __name__ == "__main__":
    main()
