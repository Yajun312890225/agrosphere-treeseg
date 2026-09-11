"""用训练好的模型对整幅 composite.tif 滑窗推理，输出树冠多边形与树点表。

输出：
  <out>/crowns.gpkg   树冠多边形（id, area_m2, conf, cx, cy, ndvi_mean, ndre_mean ...）
  <out>/trees.csv     同上属性的表格（含经纬度）
"""
import argparse
import csv
import os
import numpy as np
from osgeo import gdal, osr
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree
from . import geo

gdal.UseExceptions()


def tile_windows(W, H, tile, overlap):
    step = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, step))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    return xs, ys


def zonal_means(polys, ms_dir, epsg):
    """把多边形栅格化到多光谱网格，统计每株 NDVI/NDRE/GNDVI 均值。"""
    from osgeo import ogr
    paths = {b: os.path.join(ms_dir, f"ms_{b}.tif") for b in ["Red", "Green", "RedEdge", "NIR"]}
    if not all(os.path.exists(p) for p in paths.values()):
        return {}
    ref = gdal.Open(paths["NIR"])
    gt = ref.GetGeoTransform()
    mem = gdal.GetDriverByName("MEM").Create("", ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Int32)
    mem.SetGeoTransform(gt)
    mem.SetProjection(ref.GetProjection())
    vds = ogr.GetDriverByName("Memory").CreateDataSource("v")
    lyr = vds.CreateLayer("p", geo.srs_from_epsg(epsg), ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    # 大冠先烧、小冠后烧，避免小树冠被重叠的大树冠完全覆盖
    for i in sorted(range(len(polys)), key=lambda k: -polys[k].area):
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(ogr.CreateGeometryFromWkb(polys[i].wkb))
        f.SetField("id", i + 1)
        lyr.CreateFeature(f)
    gdal.RasterizeLayer(mem, [1], lyr, options=["ATTRIBUTE=id"])
    lab = mem.ReadAsArray().ravel()
    n = len(polys) + 1
    R, G, RE, N = [gdal.Open(paths[b]).ReadAsArray().astype(np.float32).ravel() for b in ["Red", "Green", "RedEdge", "NIR"]]
    eps = 1e-6
    out = {}
    for name, arr in [("ndvi", (N - R) / (N + R + eps)), ("ndre", (N - RE) / (N + RE + eps)), ("gndvi", (N - G) / (N + G + eps))]:
        ok = np.isfinite(arr)
        cnt = np.bincount(lab, weights=ok.astype(np.float64), minlength=n)  # 只统计有效像元
        s = np.bincount(lab, weights=np.where(ok, arr, 0), minlength=n)
        out[name + "_mean"] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)[1:]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", required=True, help="composite.tif")
    ap.add_argument("--weights", required=True, help="best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256, help="重叠像素，需大于最大冠幅直径")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.4, help="跨切片合并时的 IoU 阈值")
    ap.add_argument("--max-det", type=int, default=600)
    ap.add_argument("--min-area", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ms-dir", default=None, help="prepare 输出目录，含 ms_*.tif 时附带逐株指数")
    ap.add_argument("--limit", type=int, default=0, help="只推理前 N 个切片（调试）")
    ap.add_argument("--tta", action="store_true",
                    help="测试时增强。注意 ultralytics 8.3 的分割模型不支持 augment=True（静默回退为单尺度，2026-09-10 实测结果与不开完全一致），仅保留接口")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    ds = gdal.Open(a.raster)
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    epsg = geo.raster_epsg(a.raster)
    xs, ys = tile_windows(W, H, a.tile, a.overlap)
    core = a.overlap // 2
    polys, confs = [], []
    n_tiles = 0
    for y0 in ys:
        for x0 in xs:
            if a.limit and n_tiles >= a.limit:
                break
            arr = ds.ReadAsArray(x0, y0, a.tile, a.tile)[:3]
            if (arr.max(axis=0) == 0).mean() > 0.95:
                continue
            n_tiles += 1
            img = np.ascontiguousarray(arr.transpose(1, 2, 0)[:, :, ::-1])  # RGB -> BGR（ultralytics 约定）
            kw = dict(imgsz=a.tile, conf=a.conf, iou=a.iou, max_det=a.max_det, retina_masks=True, verbose=False, augment=a.tta)
            if a.device:
                kw["device"] = a.device
            res = model.predict(img, **kw)[0]
            if res.masks is None:
                continue
            # 只保留质心落在切片核心区（去掉半个重叠带）的实例，避免边界重复
            cx0 = core if x0 > 0 else 0
            cy0 = core if y0 > 0 else 0
            cx1 = a.tile - core if x0 + a.tile < W else a.tile
            cy1 = a.tile - core if y0 + a.tile < H else a.tile
            for seg, c in zip(res.masks.xy, res.boxes.conf.cpu().numpy()):
                if len(seg) < 3:
                    continue
                p = Polygon(seg).buffer(0)
                if p.is_empty:
                    continue
                p = geo.largest_polygon(p)
                mx, my = p.centroid.x, p.centroid.y
                if not (cx0 <= mx < cx1 and cy0 <= my < cy1):
                    continue
                ext = np.array(p.exterior.coords)
                wx, wy = geo.pixel_to_world(gt, ext[:, 0] + x0, ext[:, 1] + y0)
                wp = Polygon(np.c_[wx, wy]).buffer(0)
                if wp.area < a.min_area:
                    continue
                polys.append(geo.largest_polygon(wp))
                confs.append(float(c))
        print(f"  已推理 {n_tiles} 切片，累计实例 {len(polys)}", flush=True)
    # 跨切片 NMS
    order = np.argsort(confs)[::-1]
    tree = STRtree(polys)
    keep = []
    suppressed = np.zeros(len(polys), bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in tree.query(polys[i]):
            if j == i or suppressed[j]:
                continue
            inter = polys[i].intersection(polys[j]).area
            if inter <= 0:
                continue
            union = polys[i].area + polys[j].area - inter
            if inter / union > a.nms_iou or inter / polys[j].area > 0.7:
                suppressed[j] = True
    final = [polys[i] for i in keep]
    fconf = [confs[i] for i in keep]
    print(f"合并后树冠 {len(final)} 株（合并前 {len(polys)}）")
    os.makedirs(a.out, exist_ok=True)
    extra = zonal_means(final, a.ms_dir, epsg) if a.ms_dir else {}
    tr = osr.CoordinateTransformation(geo.srs_from_epsg(epsg), geo.srs_from_epsg(4326))
    attrs = []
    for i, (p, c) in enumerate(zip(final, fconf)):
        lon, lat, _ = tr.TransformPoint(p.centroid.x, p.centroid.y)
        d = {"id": i + 1, "area_m2": float(p.area), "conf": c, "cx": p.centroid.x, "cy": p.centroid.y, "lon": lon, "lat": lat}
        for k, v in extra.items():
            d[k] = float(v[i])
        attrs.append(d)
    geo.write_gpkg(os.path.join(a.out, "crowns.gpkg"), final, attrs, epsg)
    with open(os.path.join(a.out, "trees.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(attrs[0].keys()) if attrs else ["id"])
        w.writeheader()
        w.writerows(attrs)
    print("输出:", os.path.join(a.out, "crowns.gpkg"), os.path.join(a.out, "trees.csv"))


if __name__ == "__main__":
    main()
