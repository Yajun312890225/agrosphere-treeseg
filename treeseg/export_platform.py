"""把 predict/registry 的树冠结果打包成 AgroSphere「AI 单树监测」导入格式的 zip。

平台导入要求（见 AgroSphere internal/logic/plot/monitor_analysis_service.go）：
  - 一个 GeoTIFF（本批次底图，任意坐标系）
  - 树点 shp（文件名含 tree_point/树点；字段 ID, Lon, Lat, Pest, tree_area, hash）
  - 树冠 shp（文件名含 crown/树冠；字段 ID, tree_area）
  - 病虫害多边形 shp（文件名含 polygon/多边形；本项目无检测结果时输出空图层）
  - 可选 CSV（PT_ID, Pest, tree_area）
三个 shp 都必须带 .shp/.shx/.dbf/.prj。ID 优先用 tree_id（registry 输出），否则用 id。
"""
import argparse
import csv
import hashlib
import os
import shutil
import zipfile
import numpy as np
from osgeo import ogr, osr
from shapely.geometry import Point
from . import geo

ogr.UseExceptions()


def write_shp(path, geoms, attrs, fields, epsg, geom_type):
    """写 ESRI Shapefile（UTF-8 编码，字段类型由 fields 指定：'str'/'int'/'float'）。"""
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(path):
        drv.DeleteDataSource(path)
    ds = drv.CreateDataSource(path)
    srs = geo.srs_from_epsg(epsg)
    lyr = ds.CreateLayer(os.path.splitext(os.path.basename(path))[0], srs, geom_type, options=["ENCODING=UTF-8"])
    for name, typ in fields:
        fd = ogr.FieldDefn(name, {"str": ogr.OFTString, "int": ogr.OFTInteger, "float": ogr.OFTReal}[typ])
        if typ == "str":
            fd.SetWidth(80)
        if typ == "float":
            fd.SetWidth(24)
            fd.SetPrecision(10)
        lyr.CreateField(fd)
    defn = lyr.GetLayerDefn()
    for g, a in zip(geoms, attrs):
        f = ogr.Feature(defn)
        f.SetGeometry(ogr.CreateGeometryFromWkb(g.wkb))
        for name, _ in fields:
            v = a.get(name)
            if v is not None:
                f.SetField(name, v.item() if isinstance(v, np.generic) else v)
        lyr.CreateFeature(f)
    ds = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crowns", required=True, help="predict 的 crowns.gpkg 或 registry 的 crowns_with_ids.gpkg")
    ap.add_argument("--tif", default=None, help="本批次底图 GeoTIFF（prepare 的 composite.tif 或大疆 result.tif）；缺省不打包，导入会失败")
    ap.add_argument("--out", required=True, help="输出目录，zip 也写在这里")
    ap.add_argument("--name", default="treeseg_batch", help="文件名前缀与 zip 名")
    ap.add_argument("--date", default="", help="采集日期 YYYY-MM-DD，写进 hash 以保证跨批次唯一")
    ap.add_argument("--id-prefix", default="T", help="没有 tree_id 时用 id 生成的 ID 前缀")
    ap.add_argument("--simplify", type=float, default=0.05, help="树冠多边形简化容差(m)；平台单株几何上限 16KB，默认 0.05")
    ap.add_argument("--no-zip", action="store_true", help="只生成目录，不压缩")
    a = ap.parse_args()

    epsg = geo.vector_epsg(a.crowns)
    feats = geo.read_vector(a.crowns, epsg)
    tr = osr.CoordinateTransformation(geo.srs_from_epsg(epsg), geo.srs_from_epsg(4326))
    os.makedirs(a.out, exist_ok=True)

    pts, crowns, pattrs, cattrs, rows = [], [], [], [], []
    for g, at in feats:
        tid = at.get("tree_id") or f"{a.id_prefix}{int(at.get('id', 0)):06d}"
        c = g.centroid
        lon, lat, _ = tr.TransformPoint(c.x, c.y)
        area = float(at.get("area_m2") or g.area)
        h = hashlib.sha256(f"{a.name}|{a.date}|{tid}".encode()).hexdigest()
        pts.append(Point(c.x, c.y))
        gs = g.simplify(a.simplify, preserve_topology=True) if a.simplify > 0 else g
        crowns.append(gs if not gs.is_empty else g)
        pattrs.append({"ID": str(tid), "Lon": lon, "Lat": lat, "Pest": 0, "tree_area": area, "hash": h,
                       "conf": float(at.get("conf") or 0), "ndvi": at.get("ndvi_mean"), "ndre": at.get("ndre_mean")})
        cattrs.append({"ID": str(tid), "tree_area": area, "hash": h})
        rows.append({"PT_ID": str(tid), "PT_hash": h, "Pest": 0, "tree_area": f"{area:.4f}",
                     "ndvi_mean": "" if at.get("ndvi_mean") is None else f"{at['ndvi_mean']:.4f}",
                     "ndre_mean": "" if at.get("ndre_mean") is None else f"{at['ndre_mean']:.4f}"})

    base = os.path.join(a.out, a.name)
    write_shp(base + "_tree_point.shp", pts, pattrs,
              [("ID", "str"), ("Lon", "float"), ("Lat", "float"), ("Pest", "int"), ("tree_area", "float"), ("hash", "str"),
               ("conf", "float"), ("ndvi", "float"), ("ndre", "float")], epsg, ogr.wkbPoint)
    write_shp(base + "_tree_crown.shp", crowns, cattrs, [("ID", "str"), ("tree_area", "float"), ("hash", "str")], epsg, ogr.wkbPolygon)
    write_shp(base + "_pest_polygon.shp", [], [], [("ID", "str"), ("hash", "str")], epsg, ogr.wkbPolygon)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["PT_ID", "PT_hash", "Pest", "tree_area", "ndvi_mean", "ndre_mean"])
        w.writeheader()
        w.writerows(rows)
    tif_name = None
    if a.tif:
        tif_name = a.name + "_basemap.tif"
        dst = os.path.join(a.out, tif_name)
        if not os.path.exists(dst):
            os.link(a.tif, dst) if os.stat(a.tif).st_dev == os.stat(a.out).st_dev else shutil.copyfile(a.tif, dst)
    print(f"树点/树冠 {len(pts)} 株，病虫害多边形 0 个 -> {a.out}")
    if a.no_zip:
        return
    zpath = base + ".zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for fn in sorted(os.listdir(a.out)):
            if fn.startswith(a.name) and not fn.endswith(".zip"):
                # TIF 本身已压缩，用 STORED 省时间
                z.write(os.path.join(a.out, fn), fn, compress_type=zipfile.ZIP_STORED if fn.endswith(".tif") else zipfile.ZIP_DEFLATED)
    print("zip ->", zpath, f"{os.path.getsize(zpath)/1e6:.0f} MB")
    if not a.tif:
        print("警告：未指定 --tif，zip 缺少底图 GeoTIFF，平台导入会报“缺少原始 GeoTIFF”")


if __name__ == "__main__":
    main()
