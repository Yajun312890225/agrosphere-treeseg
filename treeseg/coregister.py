"""把一期影像及其成果配准到参考坐标框架（非均匀位移场）。

背景：大疆智图成果（RTK 单点解）相对供应商正射整体偏 1～4 m 且各处不一致，平台上多期叠加时同一株树
对不上。本脚本以"两期树冠质心"为同名点（树不会动）：全局平移粗配 → 一对一匹配 → 规则格网上取局部中位残差
→ 平滑成位移场；然后用该场 (1) 平移矢量成果，(2) 以格网节点为 GCP 用 gdalwarp -tps 重采样栅格。

参考框架的选择：有供应商成果的期次以供应商树冠为 --fixed；之后的期次以上一期"已配准"的预测树冠为 --fixed，
这样所有期次都落在同一框架里，跨期位置才稳定。

示例：
  python -m treeseg.coregister --moving pred_0811/crowns.gpkg --fixed labels_v07.gpkg \
      --rasters prep0811/composite.tif prep0811/ms_Red.tif ... --out-dir prep0811_reg --out-vector pred_0811/crowns_reg.gpkg
"""
import argparse
import os
import subprocess
import numpy as np
from osgeo import gdal
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from shapely.affinity import translate
from . import geo

gdal.UseExceptions()


def build_field(moving_xy, fixed_xy, max_dist, radius, min_pts, grid, bounds, search=6.0):
    """返回 (xs, ys, DX, DY, info)：格网节点上的总位移，把 moving 框架映射到 fixed 框架。

    每个节点在 radius 邻域内独立做"密度互相关 + ICP"（geo.estimate_shift），而不是在全局平移后取匹配残差：
    后者要求局部偏差已在 max_dist 内，遇到局部偏差 2 m 以上的区域会因匹配对不足退回全局平移，标签留在错位状态
    （2026-09-10 那花 0811 西侧 1 ha 就是这样被训成背景的）。互相关不依赖初始匹配，可吃下 search 范围内的局部偏差。
    """
    dx, dy = geo.estimate_shift(moving_xy, fixed_xy)
    mtree, ftree = cKDTree(moving_xy), cKDTree(fixed_xy)
    x0, y0, x1, y1 = bounds
    xs = np.arange(x0 - grid, x1 + 2 * grid, grid)
    ys = np.arange(y0 - grid, y1 + 2 * grid, grid)
    DX = np.full((len(ys), len(xs)), dx)
    DY = np.full_like(DX, dy)
    W = np.zeros_like(DX)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            for r in (radius, radius * 1.6):
                mi = mtree.query_ball_point([x, y], r)
                fi = ftree.query_ball_point([x, y], r + search)
                if len(mi) >= min_pts and len(fi) >= min_pts:
                    # 以全局平移为初值，在 ±search 内搜局部平移
                    ldx, ldy = geo.estimate_shift(moving_xy[mi] + [dx, dy], fixed_xy[fi], search=search, res=0.25, icp_dist=1.2)
                    if np.hypot(ldx, ldy) <= search:
                        DX[j, i], DY[j, i] = dx + ldx, dy + ldy
                        W[j, i] = len(mi)
                    break
    # 无数据节点用邻近有效节点填充，再轻微平滑抑制单节点噪声
    if (W > 0).any() and (W == 0).any():
        vj, vi = np.where(W > 0)
        nj, ni = np.where(W == 0)
        _, k = cKDTree(np.c_[vj, vi]).query(np.c_[nj, ni])
        DX[nj, ni] = DX[vj[k], vi[k]]
        DY[nj, ni] = DY[vj[k], vi[k]]
    DX = gaussian_filter(DX, 0.8)
    DY = gaussian_filter(DY, 0.8)
    moved = moving_xy + np.c_[[sample_field(xs, ys, DX, DY, x, y) for x, y in moving_xy]]
    _, _, d = geo.match_one_to_one(moved, fixed_xy, max_dist)
    local = np.hypot(DX - dx, DY - dy)
    info = dict(global_dx=dx, global_dy=dy, n_match=len(d), resid_med=float(np.median(d)) if len(d) else float("nan"),
                local_med=float(np.median(local[W > 0])) if (W > 0).any() else 0.0, local_max=float(local.max()),
                n_nodes=int((W > 0).sum()), n_total=int(W.size))
    return xs, ys, DX, DY, info


def sample_field(xs, ys, DX, DY, x, y):
    """双线性插值取点 (x, y) 处的位移。"""
    fx = np.clip((x - xs[0]) / (xs[1] - xs[0]), 0, len(xs) - 1.001)
    fy = np.clip((y - ys[0]) / (ys[1] - ys[0]), 0, len(ys) - 1.001)
    i, j = int(fx), int(fy)
    tx, ty = fx - i, fy - j
    def bil(A):
        return (A[j, i] * (1 - tx) * (1 - ty) + A[j, i + 1] * tx * (1 - ty) + A[j + 1, i] * (1 - tx) * ty + A[j + 1, i + 1] * tx * ty)
    return float(bil(DX)), float(bil(DY))


def warp_raster(src, dst, xs, ys, DX, DY, threads):
    """以格网节点为 GCP（源像素 → 目标世界坐标），gdalwarp -tps 重采样到与源相同的分辨率与投影。"""
    ds = gdal.Open(src)
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    gcps = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            col, row = geo.world_to_pixel(gt, x, y)
            if -0.1 * W <= col <= 1.1 * W and -0.1 * H <= row <= 1.1 * H:
                gcps += ["-gcp", f"{col:.2f}", f"{row:.2f}", f"{x + DX[j, i]:.3f}", f"{y + DY[j, i]:.3f}"]
    epsg = geo.raster_epsg(src)
    vrt = dst + ".gcp.vrt"
    subprocess.run(["gdal_translate", "-q", "-of", "VRT", "-a_srs", f"EPSG:{epsg}", *gcps, src, vrt], check=True)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    cmd = ["gdalwarp", "-q", "-overwrite", "-tps", "-t_srs", f"EPSG:{epsg}", "-tr", str(abs(gt[1])), str(abs(gt[5])), "-tap",
           "-r", "bilinear", "-wo", f"NUM_THREADS={threads}", "-wm", "512", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
           "-co", "BIGTIFF=IF_SAFER"]
    if nodata is not None:
        cmd += ["-srcnodata", str(nodata), "-dstnodata", str(nodata)]
    subprocess.run(cmd + [vrt, dst], check=True)
    os.remove(vrt)
    ds = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moving", required=True, help="待配准期次的树冠/树点 gpkg（模型预测）")
    ap.add_argument("--fixed", default=None, help="参考框架的树冠/树点 gpkg（供应商成果或上一期已配准的预测）；给 --field-in 时可省")
    ap.add_argument("--rasters", nargs="*", default=[], help="需要一并重采样的栅格（composite/ms/ndvi/dsm）")
    ap.add_argument("--out-dir", default=None, help="重采样栅格输出目录（文件名不变）")
    ap.add_argument("--out-vector", default=None, help="平移后的 moving 矢量输出路径")
    ap.add_argument("--field-out", default=None, help="把位移场存成 npz 便于复用/检查")
    ap.add_argument("--field-in", default=None, help="直接加载已保存的位移场 npz（跳过估计），保证底图与后续预测用同一个场")
    ap.add_argument("--max-dist", type=float, default=1.5)
    ap.add_argument("--radius", type=float, default=60.0, help="每个节点做局部互相关的邻域半径(m)")
    ap.add_argument("--search", type=float, default=6.0, help="局部平移相对全局平移的搜索范围(m)")
    ap.add_argument("--min-pts", type=int, default=30, help="邻域内两侧点数下限，不足则半径放大 1.6 倍，再不足用邻近节点值")
    ap.add_argument("--grid", type=float, default=25.0, help="位移场格网间距(m)")
    ap.add_argument("--dist-field", default="reg_dist", help="写入 --out-vector 的字段：配准后到最近 fixed 点的距离(m)，供 dataset 做标签质量守门")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    epsg = geo.vector_epsg(a.moving)
    mov = geo.read_vector(a.moving, epsg)
    mxy = np.array([[g.centroid.x, g.centroid.y] for g, _ in mov])
    if a.field_in:
        f = np.load(a.field_in)
        xs, ys, DX, DY = f["xs"], f["ys"], f["DX"], f["DY"]
        print(f"加载位移场 {a.field_in}：{len(xs)}x{len(ys)} 节点")
    else:
        fix = geo.read_vector(a.fixed, epsg)
        fxy = np.array([[g.centroid.x, g.centroid.y] for g, _ in fix])
        bounds = (min(mxy[:, 0].min(), fxy[:, 0].min()), min(mxy[:, 1].min(), fxy[:, 1].min()),
                  max(mxy[:, 0].max(), fxy[:, 0].max()), max(mxy[:, 1].max(), fxy[:, 1].max()))
        if a.rasters:
            gt, _, W, H = geo.raster_info(a.rasters[0])
            bounds = (min(bounds[0], gt[0]), min(bounds[1], gt[3] + H * gt[5]), max(bounds[2], gt[0] + W * gt[1]), max(bounds[3], gt[3]))
        xs, ys, DX, DY, info = build_field(mxy, fxy, a.max_dist, a.radius, a.min_pts, a.grid, bounds, a.search)
        print(f"moving {len(mxy)} 株，fixed {len(fxy)} 株；全局平移 dx={info['global_dx']:.2f} dy={info['global_dy']:.2f} m；"
              f"局部互相关有效节点 {info['n_nodes']}/{info['n_total']}，局部修正中位 {info['local_med']:.2f} m，最大 {info['local_max']:.2f} m")
        # 自检：位移场作用于 moving 后与 fixed 重新匹配
        moved = np.array([sample_field(xs, ys, DX, DY, x, y) for x, y in mxy]) + mxy
        _, _, d2 = geo.match_one_to_one(moved, fxy, a.max_dist)
        print(f"配准后：匹配 {len(d2)} 对，残差中位 {np.median(d2):.2f} m，90% {np.percentile(d2, 90):.2f} m")
        # 仅全局平移的对照，便于判断局部场带来的收益
        _, _, d1 = geo.match_one_to_one(mxy + [info['global_dx'], info['global_dy']], fxy, a.max_dist)
        print(f"（对照：仅全局平移 匹配 {len(d1)} 对，残差中位 {np.median(d1):.2f} m）")

    if a.field_out:
        np.savez(a.field_out, xs=xs, ys=ys, DX=DX, DY=DY)
    if a.out_vector:
        geoms, attrs = [], []
        moved = np.array([sample_field(xs, ys, DX, DY, x, y) for x, y in mxy]) + mxy
        dist = np.full(len(mxy), np.nan)
        if a.fixed:
            fxy_all = np.array([[g.centroid.x, g.centroid.y] for g, _ in geo.read_vector(a.fixed, epsg)])
            dist, _ = cKDTree(fxy_all).query(moved)
        for k, ((g, at), (x, y)) in enumerate(zip(mov, mxy)):
            geoms.append(translate(g, moved[k, 0] - x, moved[k, 1] - y))
            d = dict(at)
            d[a.dist_field] = float(dist[k])
            attrs.append(d)
        geo.write_gpkg(a.out_vector, geoms, attrs, epsg)
        print("矢量 ->", a.out_vector)
    if a.rasters:
        os.makedirs(a.out_dir, exist_ok=True)
        for r in a.rasters:
            dst = os.path.join(a.out_dir, os.path.basename(r))
            print("  重采样", os.path.basename(r), flush=True)
            warp_raster(r, dst, xs, ys, DX, DY, a.threads)
        print("栅格 ->", a.out_dir)


if __name__ == "__main__":
    main()
