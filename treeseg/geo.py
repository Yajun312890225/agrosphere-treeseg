"""坐标、矢量读写、平移估计与树点匹配的公共工具。"""
import csv
import math
import os
import numpy as np
from osgeo import gdal, ogr, osr
from scipy import ndimage as ndi, signal
from scipy.spatial import cKDTree
from shapely import wkb
from shapely.geometry import Polygon, MultiPolygon, Point

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()


def srs_from_epsg(epsg: int) -> osr.SpatialReference:
    s = osr.SpatialReference()
    s.ImportFromEPSG(epsg)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def raster_info(path: str):
    """返回 (geotransform, wkt, width, height)。"""
    ds = gdal.Open(path)
    return ds.GetGeoTransform(), ds.GetProjection(), ds.RasterXSize, ds.RasterYSize


def vector_epsg(path: str, layer_name: str | None = None) -> int:
    """读取矢量图层的 EPSG。必须持有数据源引用，链式 ogr.Open(p).GetLayer() 会因数据源被回收而崩溃。"""
    ds = ogr.Open(path)
    if ds is None:
        raise FileNotFoundError(path)
    layer = ds.GetLayerByName(layer_name) if layer_name else ds.GetLayer(0)
    s = layer.GetSpatialRef()
    if s is None:
        raise ValueError(f"矢量文件无坐标系：{path}")
    s = s.Clone()
    s.AutoIdentifyEPSG()
    code = s.GetAuthorityCode(None)
    if code is None:
        raise ValueError(f"无法识别矢量 EPSG：{path}")
    return int(code)


def raster_epsg(path: str) -> int:
    ds = gdal.Open(path)
    s = osr.SpatialReference(wkt=ds.GetProjection())
    s.AutoIdentifyEPSG()
    code = s.GetAuthorityCode(None)
    if code is None:
        raise ValueError(f"无法识别影像 EPSG：{path}")
    return int(code)


def world_to_pixel(gt, x, y):
    col = (x - gt[0]) / gt[1]
    row = (y - gt[3]) / gt[5]
    return col, row


def pixel_to_world(gt, col, row):
    return gt[0] + col * gt[1], gt[3] + row * gt[5]


def _transform_to(layer, epsg: int):
    src = layer.GetSpatialRef()
    dst = srs_from_epsg(epsg)
    if src is None:
        return None
    src = src.Clone()
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if src.IsSame(dst):
        return None
    return osr.CoordinateTransformation(src, dst)


def read_vector(path: str, epsg: int, layer_name: str | None = None):
    """读取矢量文件（shp/gpkg/geojson），转换到目标 EPSG，返回 [(geom(shapely), attrs(dict))]。"""
    ds = ogr.Open(path)
    if ds is None:
        raise FileNotFoundError(path)
    layer = ds.GetLayerByName(layer_name) if layer_name else ds.GetLayer(0)
    tr = _transform_to(layer, epsg)
    defn = layer.GetLayerDefn()
    fields = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
    out = []
    for f in layer:
        g = f.GetGeometryRef()
        if g is None:
            continue
        g = g.Clone()
        if tr is not None:
            g.Transform(tr)
        attrs = {k: f.GetField(k) for k in fields}
        out.append((wkb.loads(bytes(g.ExportToWkb())), attrs))
    return out


def read_points_csv(path: str, x_field="X", y_field="Y", epsg_in: int | None = None, epsg_out: int | None = None,
                    encoding="utf-8-sig"):
    """读取 CSV 点表；表头含乱码时按列序号回退。返回 [(Point, attrs)]。"""
    raw = open(path, "rb").read()
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    rows = list(csv.reader(text.splitlines()))
    header = rows[0]
    if x_field not in header or y_field not in header:
        raise ValueError(f"CSV 缺少 {x_field}/{y_field} 列，表头为 {header}")
    xi, yi = header.index(x_field), header.index(y_field)
    tr = None
    if epsg_in and epsg_out and epsg_in != epsg_out:
        tr = osr.CoordinateTransformation(srs_from_epsg(epsg_in), srs_from_epsg(epsg_out))
    out = []
    for r in rows[1:]:
        if len(r) <= max(xi, yi):
            continue
        try:
            x, y = float(r[xi]), float(r[yi])
        except ValueError:
            continue
        if tr is not None:
            x, y, _ = tr.TransformPoint(x, y)
        out.append((Point(x, y), {header[i]: r[i] for i in range(min(len(header), len(r)))}))
    return out


def write_gpkg(path: str, geoms, attrs_list, epsg: int, layer_name="crowns", geom_type=ogr.wkbPolygon):
    """写 GeoPackage。attrs_list 为与 geoms 等长的 dict 列表，字段类型按首条推断。"""
    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(path):
        drv.DeleteDataSource(path)
    ds = drv.CreateDataSource(path)
    layer = ds.CreateLayer(layer_name, srs_from_epsg(epsg), geom_type)
    if attrs_list:
        for k, v in attrs_list[0].items():
            t = ogr.OFTInteger64 if isinstance(v, (int, np.integer)) else ogr.OFTReal if isinstance(v, (float, np.floating)) else ogr.OFTString
            layer.CreateField(ogr.FieldDefn(k, t))
    defn = layer.GetLayerDefn()
    for g, a in zip(geoms, attrs_list):
        f = ogr.Feature(defn)
        f.SetGeometry(ogr.CreateGeometryFromWkb(g.wkb))
        for k, v in a.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            f.SetField(k, v.item() if isinstance(v, np.generic) else v)
        layer.CreateFeature(f)
    ds = None


def largest_polygon(geom):
    """MultiPolygon / GeometryCollection 取面积最大的多边形部分；不含多边形时返回空 Polygon。"""
    if isinstance(geom, Polygon):
        return geom
    polys = []
    for g in getattr(geom, "geoms", []):
        if isinstance(g, Polygon):
            polys.append(g)
        elif hasattr(g, "geoms"):  # 嵌套集合（如 GeometryCollection 内含 MultiPolygon）
            sub = largest_polygon(g)
            if not sub.is_empty:
                polys.append(sub)
    if not polys:
        return Polygon()
    return max(polys, key=lambda g: g.area)


def density_grid(xy: np.ndarray, x0, y0, w, h, res, sigma=2.0):
    g = np.zeros((h, w), np.float32)
    ix = ((xy[:, 0] - x0) / res).astype(int)
    iy = ((xy[:, 1] - y0) / res).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    np.add.at(g, (iy[ok], ix[ok]), 1)
    return ndi.gaussian_filter(g, sigma)


def estimate_shift(moving: np.ndarray, fixed: np.ndarray, search=25.0, res=0.5, icp_dist=3.0, icp_iter=5):
    """估计把 moving 点集叠到 fixed 点集上的整体平移 (dx, dy)：互相关粗配 + 最近邻中位数精修。"""
    allxy = np.vstack([moving, fixed])
    x0, y0 = allxy[:, 0].min() - search - 5, allxy[:, 1].min() - search - 5
    w = int((allxy[:, 0].max() + search + 5 - x0) / res) + 1
    h = int((allxy[:, 1].max() + search + 5 - y0) / res) + 1
    gm = density_grid(moving, x0, y0, w, h, res)
    gf = density_grid(fixed, x0, y0, w, h, res)
    cc = signal.fftconvolve(gf, gm[::-1, ::-1], mode="same")
    cy, cx = np.array(cc.shape) // 2
    win = int(search / res)
    sub = cc[cy - win:cy + win + 1, cx - win:cx + win + 1]
    py, px = np.unravel_index(sub.argmax(), sub.shape)
    dx, dy = (px - win) * res, (py - win) * res
    tree = cKDTree(fixed)
    for _ in range(icp_iter):
        d, j = tree.query(moving + [dx, dy], distance_upper_bound=icp_dist)
        ok = np.isfinite(d)
        if ok.sum() < 10:
            break
        dx += float(np.median(fixed[j[ok], 0] - (moving[ok, 0] + dx)))
        dy += float(np.median(fixed[j[ok], 1] - (moving[ok, 1] + dy)))
    return float(dx), float(dy)


def match_one_to_one(pred: np.ndarray, ref: np.ndarray, max_dist: float):
    """按距离升序贪心做一对一匹配。返回 (pred_idx, ref_idx, dist) 三个数组。"""
    if len(pred) == 0 or len(ref) == 0:
        return np.array([], int), np.array([], int), np.array([])
    tree = cKDTree(ref)
    k = min(5, len(ref))
    d, j = tree.query(pred, k=k, distance_upper_bound=max_dist)
    d = np.atleast_2d(d).reshape(len(pred), k)
    j = np.atleast_2d(j).reshape(len(pred), k)
    cand = [(d[i, m], i, j[i, m]) for i in range(len(pred)) for m in range(k) if np.isfinite(d[i, m])]
    cand.sort()
    used_p, used_r, out = set(), set(), []
    for dist, i, r in cand:
        if i in used_p or r in used_r:
            continue
        used_p.add(i)
        used_r.add(r)
        out.append((i, r, dist))
    if not out:
        return np.array([], int), np.array([], int), np.array([])
    a = np.array(out)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2]


def classical_tree_tops(surface: np.ndarray, veg: np.ndarray, gt, sigma=2.5, min_dist_px=5):
    """传统方法树顶种子（平滑后局部极大值），用于无模型时估计平移或做对照。返回世界坐标 (n,2)。"""
    from skimage.feature import peak_local_max
    s = ndi.gaussian_filter(np.where(np.isfinite(surface), surface, 0).astype(np.float32), sigma)
    peaks = peak_local_max(np.where(veg, s, -1e9), min_distance=min_dist_px, threshold_abs=-1e8,
                           labels=veg.astype(np.int32), exclude_border=False)
    xs, ys = pixel_to_world(gt, peaks[:, 1] + 0.5, peaks[:, 0] + 0.5)
    return np.c_[xs, ys]
