"""把供应商成果转换为训练标签（树冠多边形 GeoPackage，目标投影与 composite 一致）。

支持三类来源：
  --type polygons  树冠多边形 shp/gpkg（如“单株树冠估产.shp”），直接使用
  --type points    树点 shp/gpkg/csv + 冠幅面积 → 圆形伪标签，可用 NDVI 精修成实际冠形
  --type boxes     矩形框 shp（如病虫害矩形），直接当多边形
"""
import argparse
import math
import numpy as np
from osgeo import gdal
from scipy import ndimage as ndi
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from skimage import measure
from . import geo


def refine_with_ndvi(circle: Polygon, ndvi_ds, gt, thr: float, min_keep=0.3):
    """在圆内取 NDVI>thr 且与圆心连通的像元，转成多边形；覆盖太少则退回圆。"""
    minx, miny, maxx, maxy = circle.bounds
    c0, r0 = geo.world_to_pixel(gt, minx, maxy)
    c1, r1 = geo.world_to_pixel(gt, maxx, miny)
    c0, r0 = int(math.floor(c0)), int(math.floor(r0))
    c1, r1 = int(math.ceil(c1)), int(math.ceil(r1))
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, ndvi_ds.RasterXSize), min(r1, ndvi_ds.RasterYSize)
    if c1 - c0 < 3 or r1 - r0 < 3:
        return circle
    arr = ndvi_ds.GetRasterBand(1).ReadAsArray(c0, r0, c1 - c0, r1 - r0)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    wx, wy = geo.pixel_to_world(gt, xx + 0.5, yy + 0.5)
    cx, cy = circle.centroid.x, circle.centroid.y
    rad = math.sqrt(circle.area / math.pi)
    inside = (wx - cx) ** 2 + (wy - cy) ** 2 <= rad ** 2
    mask = inside & np.isfinite(arr) & (arr > thr)
    if mask.sum() < min_keep * inside.sum():
        return circle
    lab, n = ndi.label(mask)
    ci, ri = int((cx - gt[0]) / gt[1]) - c0, int((cy - gt[3]) / gt[5]) - r0
    ci, ri = min(max(ci, 0), mask.shape[1] - 1), min(max(ri, 0), mask.shape[0] - 1)
    keep = lab[ri, ci]
    if keep == 0:  # 圆心不在植被上，取最大连通块
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = sizes.argmax()
    m = ndi.binary_fill_holes(lab == keep)
    pad = np.pad(m, 1)
    contours = measure.find_contours(pad.astype(np.uint8), 0.5)
    if not contours:
        return circle
    cont = max(contours, key=len)
    pts = [geo.pixel_to_world(gt, c0 + p[1] - 1 + 0.5, r0 + p[0] - 1 + 0.5) for p in cont]
    poly = Polygon(pts).buffer(0)
    poly = geo.largest_polygon(poly).simplify(abs(gt[1]) * 0.7)
    if not poly.is_valid or poly.area < min_keep * circle.area:
        return circle
    return poly


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", required=True, help="composite.tif，用于确定目标 EPSG")
    ap.add_argument("--source", required=True, help="供应商矢量文件（shp/gpkg/csv）")
    ap.add_argument("--type", choices=["polygons", "points", "boxes"], required=True)
    ap.add_argument("--id-field", default=None, help="ID 字段名（缺省用序号）")
    ap.add_argument("--area-field", default=None, help="points 类型：冠幅面积字段(m²)")
    ap.add_argument("--area-csv", default=None, help="points 类型：面积在另一张 CSV 时的路径")
    ap.add_argument("--area-csv-id", default="PT_ID")
    ap.add_argument("--area-csv-field", default="tree_area")
    ap.add_argument("--default-area", type=float, default=10.0, help="无面积信息时的默认冠幅(m²)")
    ap.add_argument("--csv-epsg", type=int, default=4326, help="csv 来源坐标系")
    ap.add_argument("--csv-x", default="Lon")
    ap.add_argument("--csv-y", default="Lat")
    ap.add_argument("--shift", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"), help="整体平移(m)，把标签叠到影像上")
    ap.add_argument("--refine-ndvi", default=None, help="ndvi.tif；points 类型用它把圆精修成冠形")
    ap.add_argument("--ndvi-thr", type=float, default=0.6)
    ap.add_argument("--out", required=True, help="输出 labels.gpkg")
    a = ap.parse_args()

    epsg = geo.raster_epsg(a.raster)
    if a.source.lower().endswith(".csv"):
        feats = geo.read_points_csv(a.source, a.csv_x, a.csv_y, a.csv_epsg, epsg)
    else:
        feats = geo.read_vector(a.source, epsg)
    print(f"读取 {len(feats)} 个要素，目标 EPSG:{epsg}，平移 {a.shift}")

    area_lookup = {}
    if a.area_csv:
        import csv
        for r in csv.DictReader(open(a.area_csv, encoding="utf-8-sig")):
            try:
                area_lookup[r[a.area_csv_id]] = float(r[a.area_csv_field])
            except (KeyError, ValueError):
                pass
    ndvi_ds = gt = None
    if a.refine_ndvi:
        ndvi_ds = gdal.Open(a.refine_ndvi)
        gt = ndvi_ds.GetGeoTransform()

    geoms, attrs = [], []
    dx, dy = a.shift
    n_refined = 0
    for i, (g, at) in enumerate(feats):
        tid = str(at.get(a.id_field, i + 1)) if a.id_field else str(i + 1)
        if a.type in ("polygons", "boxes"):
            poly = geo.largest_polygon(g.buffer(0))
            src_area = poly.area
        else:
            area = None
            if a.area_field and at.get(a.area_field) not in (None, ""):
                area = float(at[a.area_field])
            elif tid in area_lookup:
                area = area_lookup[tid]
            src_area = area if area and area > 0 else a.default_area
            c = Point(g.centroid.x + dx, g.centroid.y + dy)
            poly = c.buffer(math.sqrt(src_area / math.pi), resolution=16)
            if ndvi_ds is not None:
                p2 = refine_with_ndvi(poly, ndvi_ds, gt, a.ndvi_thr)
                n_refined += p2 is not poly
                poly = p2
            geoms.append(poly)
            attrs.append({"tree_id": tid, "src_area": float(src_area), "area_m2": float(poly.area)})
            continue
        from shapely.affinity import translate
        poly = translate(poly, dx, dy)
        geoms.append(poly)
        attrs.append({"tree_id": tid, "src_area": float(src_area), "area_m2": float(poly.area)})
    geo.write_gpkg(a.out, geoms, attrs, epsg)
    areas = np.array([g.area for g in geoms])
    print(f"写出 {len(geoms)} 个树冠标签 -> {a.out}；NDVI 精修 {n_refined} 个；面积中位 {np.median(areas):.1f} m²")


if __name__ == "__main__":
    main()
