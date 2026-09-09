"""把 composite.tif + labels.gpkg 切成 YOLO 实例分割数据集。

- 按 tile/overlap 滑窗切片，写 JPEG 图片与 YOLO seg 标签（class 0 + 归一化多边形顶点）
- 训练/验证按空间块（block×block 个切片为一块）划分，避免相邻切片泄漏
- 无标签切片按 neg-frac 比例保留少量作为负样本
"""
import argparse
import os
import random
import numpy as np
from osgeo import gdal
from shapely.geometry import box
from shapely.strtree import STRtree
from . import geo

gdal.UseExceptions()


def save_jpeg(path, arr):
    """arr: (3,h,w) uint8 → JPEG。"""
    drv = gdal.GetDriverByName("MEM")
    m = drv.Create("", arr.shape[2], arr.shape[1], 3, gdal.GDT_Byte)
    for i in range(3):
        m.GetRasterBand(i + 1).WriteArray(np.ascontiguousarray(arr[i]))
    gdal.GetDriverByName("JPEG").CreateCopy(path, m, options=["QUALITY=95"])
    aux = path + ".aux.xml"
    if os.path.exists(aux):
        os.remove(aux)


def poly_to_yolo(poly, x0, y0, tile, gt):
    """世界坐标多边形 → 切片内归一化顶点列表。"""
    ext = np.array(poly.exterior.coords)
    cols = (ext[:, 0] - gt[0]) / gt[1] - x0
    rows = (ext[:, 1] - gt[3]) / gt[5] - y0
    cols = np.clip(cols / tile, 0, 1)
    rows = np.clip(rows / tile, 0, 1)
    return np.c_[cols, rows].ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", required=True)
    ap.add_argument("--labels", required=True, help="labels.gpkg（可多个，逗号分隔，会合并）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--block", type=int, default=6, help="空间块边长（切片数）")
    ap.add_argument("--neg-frac", type=float, default=0.05)
    ap.add_argument("--min-cover", type=float, default=0.3, help="被切片边界截断的树冠，保留面积比例下限")
    ap.add_argument("--min-area", type=float, default=1.0, help="最小树冠面积 m²")
    ap.add_argument("--max-nodata", type=float, default=0.5, help="切片无数据比例上限")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="t", help="文件名前缀（多期数据合并训练时区分）")
    a = ap.parse_args()
    random.seed(a.seed)

    ds = gdal.Open(a.raster)
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    epsg = geo.raster_epsg(a.raster)
    polys = []
    for lp in a.labels.split(","):
        polys += [geo.largest_polygon(g.buffer(0)) for g, _ in geo.read_vector(lp, epsg) if g.area >= a.min_area]
    tree = STRtree(polys)
    print(f"影像 {W}x{H} px，标签 {len(polys)} 个，切片 {a.tile} 重叠 {a.overlap}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    step = a.tile - a.overlap
    xs = list(range(0, max(W - a.tile, 0) + 1, step))
    ys = list(range(0, max(H - a.tile, 0) + 1, step))
    if xs[-1] + a.tile < W:
        xs.append(W - a.tile)
    if ys[-1] + a.tile < H:
        ys.append(H - a.tile)
    # 空间块划分
    blocks = sorted({(ix // a.block, iy // a.block) for ix in range(len(xs)) for iy in range(len(ys))})
    random.shuffle(blocks)
    val_blocks = set(blocks[:max(1, int(len(blocks) * a.val_frac))])
    stats = {"train": 0, "val": 0, "neg": 0, "skip": 0, "inst": 0}
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            split = "val" if (ix // a.block, iy // a.block) in val_blocks else "train"
            arr = ds.ReadAsArray(x0, y0, a.tile, a.tile)[:3]
            nodata = (arr.max(axis=0) == 0).mean()
            if nodata > a.max_nodata:
                stats["skip"] += 1
                continue
            wx0, wy0 = geo.pixel_to_world(gt, x0, y0 + a.tile)
            wx1, wy1 = geo.pixel_to_world(gt, x0 + a.tile, y0)
            tb = box(wx0, wy0, wx1, wy1)
            lines = []
            for idx in tree.query(tb):
                p = polys[idx]
                c = p.intersection(tb)
                if c.is_empty:
                    continue
                c = geo.largest_polygon(c)
                if c.area < a.min_cover * p.area or c.area < a.min_area:
                    continue
                v = poly_to_yolo(c.simplify(abs(gt[1])), x0, y0, a.tile, gt)
                if len(v) < 6:
                    continue
                lines.append("0 " + " ".join(f"{t:.5f}" for t in v))
            if not lines:
                if random.random() > a.neg_frac:
                    stats["skip"] += 1
                    continue
                stats["neg"] += 1
            name = f"{a.prefix}_{x0}_{y0}"
            save_jpeg(os.path.join(a.out, "images", split, name + ".jpg"), arr)
            with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            stats[split] += 1
            stats["inst"] += len(lines)
    with open(os.path.join(a.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: tree\n")
    print(f"完成：train {stats['train']} 张，val {stats['val']} 张（含负样本 {stats['neg']}），实例 {stats['inst']}，跳过 {stats['skip']}")
    print("data.yaml ->", os.path.join(a.out, "data.yaml"))


if __name__ == "__main__":
    main()
