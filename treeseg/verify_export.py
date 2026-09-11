"""本地验证 export_platform 打出的 zip：按 AgroSphere 平台导入逻辑解析，并渲染一张带树冠/树点的大图。

移植自 AgroSphere internal/logic/plot/monitor_analysis_service.go（ImportMonitorBatch 一路：
unzipMonitorArchive → locateMonitorArchiveFiles → validateMonitorLocatedFiles → validateMonitorGeoTIF
→ readMonitorShapefile(ogr2ogr → GeoJSON EPSG:4326, COORDINATE_PRECISION=7) → readMonitorCSV
→ buildMonitorTreeImports → buildMonitorDetectionImports）以及 tile_service.ProcessTIFForAnalysis 的解码预算检查。
目的：不用上传测试环境就能知道平台会怎么解析这个包（株数、丢几何、告警），并直接看图核对结果。

输出（--out 目录）：
  report.json        与平台 MonitorImportReport 同口径的解析结果
  trees.csv          平台将落库的逐株记录（tree_id, lon, lat, pest, tree_area, crown_bytes）
  overview.jpg/.tif  降采样底图 + 树冠（青）+ 树点（黄，病虫害红）+ 参考（紫）；tif 带地理参考可进 QGIS
  detail_<k>.jpg     --detail 指定窗口的全分辨率放大图
示例：
  python -m treeseg.verify_export --zip export_0811/nahua_0811_mix3.zip --out verify_0811 \
      --ref ~/nahua_work/labels_v07.gpkg --detail 631247 2628440 631348 2628543
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import zipfile
import numpy as np
from osgeo import gdal, ogr, osr
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from . import geo

gdal.UseExceptions()

# 与平台 models.DefaultMonitorImportLimits / ProcessTIFForAnalysis 一致
MAX_FILES = 200
MAX_SINGLE_FILE_BYTES = 8 * 1024 ** 3
MAX_DECODE_BYTES = 2 * 1024 ** 3          # width*height*4 超过则平台拒绝切片
MAX_CROWN_GEOMETRY_BYTES = 16 * 1024      # 单株树冠 GeoJSON 超过则平台丢弃几何
ALLOWED_EXT = {".tif", ".tiff", ".shp", ".shx", ".dbf", ".prj", ".cpg", ".csv", ".xlsx", ".json", ".xml"}


def is_macos_metadata(name):
    for seg in name.replace("\\", "/").split("/"):
        if seg in ("__MACOSX", ".DS_Store") or seg.startswith("._"):
            return True
    return False


def allowed_ext(name):
    n = name.lower()
    return n.endswith(".shp.xml") or os.path.splitext(n)[1] in ALLOWED_EXT


def unzip_archive(zpath, extract_dir):
    """对应 unzipMonitorArchive：跳过 mac 元数据，拒绝符号链接/越界路径/非法扩展名/超大文件。"""
    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
        if len(infos) > MAX_FILES:
            raise ValueError(f"ZIP 文件数量超过限制 {MAX_FILES}")
        os.makedirs(extract_dir, exist_ok=True)
        n = 0
        for zi in infos:
            if zi.is_dir() or is_macos_metadata(zi.filename):
                continue
            if (zi.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"ZIP 中不允许符号链接: {zi.filename}")
            if zi.file_size > MAX_SINGLE_FILE_BYTES:
                raise ValueError(f"单文件超过限制: {zi.filename}")
            if not allowed_ext(zi.filename):
                raise ValueError(f"ZIP 包含不允许的文件类型: {zi.filename}")
            target = os.path.normpath(os.path.join(extract_dir, zi.filename))
            if not target.startswith(os.path.abspath(extract_dir)):
                raise ValueError(f"ZIP 路径越界: {zi.filename}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(zi) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
            n += 1
    return n


def normalize_name(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def locate_files(root):
    """对应 locateMonitorArchiveFiles：按文件名关键字识别 tif/csv/树点/树冠/病虫害多边形。"""
    found = {"tif": "", "csv": "", "xlsx": "", "tree_point": "", "tree_crown": "", "pest_polygon": ""}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d != "__MACOSX"]
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            if is_macos_metadata(os.path.relpath(p, root)):
                continue
            low = fn.lower()
            if low.endswith((".tif", ".tiff")):
                found["tif"] = found["tif"] or p
            elif low.endswith(".csv"):
                found["csv"] = found["csv"] or p
            elif low.endswith(".xlsx"):
                found["xlsx"] = found["xlsx"] or p
            elif low.endswith(".shp"):
                n = normalize_name(low)
                if "树点" in n or "tree_point" in n or "treepoint" in n:
                    found["tree_point"] = p
                elif "树冠" in n or "tree_crown" in n or "crown" in n:
                    found["tree_crown"] = p
                elif "多边形" in n or "polygon" in n or "poly" in n:
                    found["pest_polygon"] = p
    return found


def validate_files(found):
    """对应 validateMonitorLocatedFiles + validateMonitorShapefileComponents。"""
    if not found["tif"]:
        raise ValueError("缺少原始 GeoTIFF")
    for label, key in (("树点 shp", "tree_point"), ("树冠 shp", "tree_crown"), ("病虫害多边形 shp", "pest_polygon")):
        if not found[key]:
            raise ValueError(f"缺少{label}")
        base = os.path.splitext(found[key])[0]
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            if not os.path.exists(base + ext):
                raise ValueError(f"缺少 shp 组件: {os.path.basename(base + ext)}")


def validate_tif(path):
    """对应 validateMonitorGeoTIF（gdalinfo 文本检查）+ ProcessTIFForAnalysis 的解码预算。返回 (W, H, 波段数, 经纬度范围)。"""
    out = subprocess.run(["gdalinfo", path], capture_output=True, text=True, check=True).stdout
    if "Size is" not in out or "Pixel Size" not in out:
        raise ValueError("TIF 缺少尺寸或像元信息")
    if "Coordinate System is" not in out and "GEOGCRS" not in out and "PROJCRS" not in out:
        raise ValueError("TIF 缺少地理参考坐标系")
    ds = gdal.Open(path)
    W, H = ds.RasterXSize, ds.RasterYSize
    need = W * H * 4
    if need > MAX_DECODE_BYTES:
        raise ValueError(f"TIF 尺寸 {W}x{H} 解码需 {need/2**30:.1f}GB，超出平台预算 {MAX_DECODE_BYTES/2**30:.1f}GB，需先降采样")
    gt = ds.GetGeoTransform()
    src = osr.SpatialReference(wkt=ds.GetProjection())
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, geo.srs_from_epsg(4326))
    xs = [gt[0], gt[0] + W * gt[1]]
    ys = [gt[3], gt[3] + H * gt[5]]
    lons, lats = [], []
    for x in xs:
        for y in ys:
            lon, lat, _ = tr.TransformPoint(x, y)
            lons.append(lon)
            lats.append(lat)
    return W, H, ds.RasterCount, (min(lons), min(lats), max(lons), max(lats))


def read_shapefile(shp, workdir, layer):
    """对应 readMonitorShapefile：ogr2ogr 转 EPSG:4326 GeoJSON（坐标 7 位小数），返回 feature 列表。"""
    gj = os.path.join(workdir, layer + ".geojson")
    if os.path.exists(gj):
        os.remove(gj)
    subprocess.run(["ogr2ogr", "-f", "GeoJSON", "-t_srs", "EPSG:4326", "-lco", "COORDINATE_PRECISION=7", gj, shp],
                   capture_output=True, text=True, check=True)
    with open(gj, encoding="utf-8") as f:
        return json.load(f).get("features", [])


def prop_string(props, *keys):
    for k in keys:
        if k in props and props[k] is not None:
            s = str(props[k]).strip()
            if s and s != "<nil>":
                return s
    for pk, v in props.items():
        for k in keys:
            if pk.lower() == k.lower() and v is not None:
                s = str(v).strip()
                if s and s != "<nil>":
                    return s
    return ""


def prop_float(props, *keys):
    for k in keys:
        if k in props and props[k] is not None:
            v = props[k]
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return float(v)
            try:
                return float(str(v).strip())
            except ValueError:
                pass
    return None


def parse_bool(raw):
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "是"):
        return True
    if s in ("0", "false", "no", "n", "否", ""):
        return False
    raise ValueError(raw)


def prop_bool(props, default, *keys):
    for k in keys:
        if k in props and props[k] is not None:
            v = props[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return v != 0
            try:
                return parse_bool(v)
            except ValueError:
                pass
    return default


def read_csv_table(path, warnings):
    """对应 readMonitorCSV：只有 tree_id 必需，pest/yield/tree_area 可缺（缺列时用 shp 属性兜底）。"""
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        warnings.append("CSV 不是有效 UTF-8，若来源为 GBK 请先转为 UTF-8 后上传")
        text = raw.decode("latin-1")
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        raise ValueError("CSV 为空")
    headers = [h.strip().lower() for h in rows[0]]

    def find(*aliases):
        for a in aliases:
            if a.lower() in headers:
                return headers.index(a.lower())
        return -1
    idx_id = find("tree_id", "id", "pt_id", "treeid")
    idx_pest, idx_yield, idx_area = find("pest", "病虫害"), find("yield_value", "yield", "预测产量"), find("tree_area", "area", "树冠面积")
    if idx_id < 0:
        raise ValueError("CSV 缺少 tree_id 列，无法与单树关联")
    for name, i in (("pest", idx_pest), ("yield_value", idx_yield), ("tree_area", idx_area)):
        if i < 0:
            warnings.append(f"CSV 缺少 {name} 列，该字段改用 shp 属性兜底")
    table = {"has_pest": idx_pest >= 0, "has_yield": idx_yield >= 0, "has_area": idx_area >= 0, "rows": {}}
    skipped = 0

    def field(r, i):
        return r[i] if 0 <= i < len(r) else ""
    for n, r in enumerate(rows[1:], start=2):
        tid = field(r, idx_id).strip()
        if not tid:
            skipped += 1
            continue
        pest = False
        if table["has_pest"]:
            try:
                pest = parse_bool(field(r, idx_pest))
            except ValueError:
                skipped += 1
                warnings.append(f"CSV 第 {n} 行 pest 无效，已跳过")
                continue
        yv = av = None
        if table["has_yield"] and field(r, idx_yield).strip():
            try:
                yv = float(field(r, idx_yield))
            except ValueError:
                warnings.append(f"CSV 第 {n} 行 yield_value 无效，已置空")
        if table["has_area"] and field(r, idx_area).strip():
            try:
                av = float(field(r, idx_area))
            except ValueError:
                warnings.append(f"CSV 第 {n} 行 tree_area 无效，已置空")
        if tid in table["rows"]:
            warnings.append(f"CSV tree_id={tid} 重复，后出现的记录覆盖前一条")
        table["rows"][tid] = dict(pest=pest, yield_value=yv, tree_area=av)
    return table, skipped


def build_tree_imports(points, crowns, table, warnings):
    """对应 buildMonitorTreeImports：树点为主，树冠按 ID 挂接，超 16KB 的树冠几何丢弃。"""
    crown_by_id = {}
    for f in crowns:
        tid = prop_string(f.get("properties") or {}, "PT_ID", "ID", "tree_id")
        if tid:
            crown_by_id[tid] = f
    trees, skipped, dropped_geom, near_limit = [], 0, 0, 0
    for f in points:
        props = f.get("properties") or {}
        tid = prop_string(props, "ID", "tree_id", "PT_ID")
        if not tid:
            skipped += 1
            continue
        g = f.get("geometry") or {}
        lon = lat = 0.0
        if str(g.get("type", "")).lower() == "point" and len(g.get("coordinates") or []) >= 2:
            lon, lat = g["coordinates"][:2]
        else:
            lon, lat = prop_float(props, "Lon", "lon", "longitude") or 0, prop_float(props, "Lat", "lat", "latitude") or 0
        if lon == 0 or lat == 0:
            skipped += 1
            warnings.append(f"tree_id={tid} 缺少有效坐标，已跳过")
            continue
        pest = prop_bool(props, False, "病虫害", "Pest", "pest")
        yv = prop_float(props, "Yield", "yield_value", "预测产量")
        area = prop_float(props, "tree_area", "area")
        row = table["rows"].get(tid) if table else None
        if row is not None:
            if table["has_pest"]:
                pest = row["pest"]
            if table["has_yield"]:
                yv = row["yield_value"]
            if table["has_area"]:
                area = row["tree_area"]
        elif table is not None:
            warnings.append(f"tree_id={tid} 在 CSV 中缺少对应记录，已使用 shp 属性兜底")
        crown_json, crown_bytes = "", 0
        cf = crown_by_id.get(tid)
        if cf is not None:
            crown_json = json.dumps(cf["geometry"], separators=(",", ":"))
            crown_bytes = len(crown_json.encode())
            if crown_bytes > MAX_CROWN_GEOMETRY_BYTES:
                dropped_geom += 1
                crown_json = ""
            elif crown_bytes > MAX_CROWN_GEOMETRY_BYTES * 0.75:
                near_limit += 1
            cp = cf.get("properties") or {}
            if area is None:
                area = prop_float(cp, "tree_area", "area")
            if yv is None:
                yv = prop_float(cp, "Yield", "yield_value")
        trees.append(dict(tree_id=tid, tree_hash=prop_string(props, "hash", "PT_hash"), lon=lon, lat=lat, pest=pest,
                          yield_value=yv, tree_area=area, crown_json=crown_json, crown_bytes=crown_bytes))
    return trees, skipped, dropped_geom, near_limit


def build_detections(polys, trees, warnings):
    """对应 buildMonitorDetectionImports：病虫害多边形必须挂到已有 tree_id。"""
    valid = {t["tree_id"] for t in trees}
    detected = set()
    n = 0
    for f in polys:
        tid = prop_string(f.get("properties") or {}, "ID", "tree_id", "PT_ID")
        if not tid:
            warnings.append("病虫害多边形 shp 存在缺少 ID 的检测图形，已跳过")
            continue
        if tid not in valid:
            warnings.append(f"病虫害多边形 shp tree_id={tid} 无对应单树，已跳过")
            continue
        if not (f.get("geometry") or {}).get("coordinates"):
            warnings.append(f"病虫害多边形 shp tree_id={tid} 检测图形无效，已跳过")
            continue
        detected.add(tid)
        n += 1
    return n, detected


def dedupe(warnings, limit=30):
    seen, out = set(), []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:limit], len(out)


def to_raster_xy(lonlat, ds):
    """经纬度 → 影像坐标系。"""
    src = osr.SpatialReference(wkt=ds.GetProjection())
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(geo.srs_from_epsg(4326), src)
    pts = tr.TransformPoints([(float(a), float(b)) for a, b in lonlat])
    return np.array([[p[0], p[1]] for p in pts])


def render(base_ds, gsd, trees, crown_polys, ref_geoms, out_prefix, window=None, threads=2):
    """渲染底图 + 叠加。window=(xmin,ymin,xmax,ymax) 时按原分辨率裁窗，否则整幅降采样到 gsd。"""
    import cv2
    gt = base_ds.GetGeoTransform()
    if window is None:
        tmp = out_prefix + "_base.tif"
        gdal.Translate(tmp, base_ds, xRes=gsd, yRes=gsd, resampleAlg="average", bandList=list(range(1, min(3, base_ds.RasterCount) + 1)),
                       creationOptions=["COMPRESS=DEFLATE", "TILED=YES", f"NUM_THREADS={threads}"])
        ds = gdal.Open(tmp)
        arr = ds.ReadAsArray()
        ogt = ds.GetGeoTransform()
    else:
        c0, r0 = geo.world_to_pixel(gt, window[0], window[3])
        c1, r1 = geo.world_to_pixel(gt, window[2], window[1])
        c0, r0 = max(int(c0), 0), max(int(r0), 0)
        c1, r1 = min(int(c1), base_ds.RasterXSize), min(int(r1), base_ds.RasterYSize)
        arr = base_ds.ReadAsArray(c0, r0, c1 - c0, r1 - r0)[:min(3, base_ds.RasterCount)]
        ogt = (gt[0] + c0 * gt[1], gt[1], 0.0, gt[3] + r0 * gt[5], 0.0, gt[5])
        ds = None
    if arr.ndim == 2:
        arr = np.stack([arr] * 3)
    img = np.ascontiguousarray(arr[:3].transpose(1, 2, 0)[:, :, ::-1])  # BGR 供 OpenCV
    H, W = img.shape[:2]
    lw = 1 if window is None else 2

    def px(xy):
        c, r = geo.world_to_pixel(ogt, xy[:, 0], xy[:, 1])
        return np.c_[c, r].astype(np.int32)
    for g in ref_geoms:  # 参考：紫
        if g.geom_type == "Point":
            c, r = geo.world_to_pixel(ogt, g.x, g.y)
            cv2.circle(img, (int(c), int(r)), 3 if window is None else 6, (255, 0, 255), -1)
        elif g.geom_type == "Polygon":
            cv2.polylines(img, [px(np.array(g.exterior.coords))], True, (255, 0, 255), lw)
    for g in crown_polys:  # 树冠：青
        cv2.polylines(img, [px(np.array(g.exterior.coords))], True, (255, 255, 0), lw)
    for t in trees:  # 树点：黄，病虫害红
        c, r = geo.world_to_pixel(ogt, t["x"], t["y"])
        if 0 <= c < W and 0 <= r < H:
            cv2.circle(img, (int(c), int(r)), 2 if window is None else 5, (0, 0, 255) if t["pest"] else (0, 255, 255), -1)
    cv2.imwrite(out_prefix + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if window is None:
        drv = gdal.GetDriverByName("GTiff")
        o = drv.Create(out_prefix + ".tif", W, H, 3, gdal.GDT_Byte, options=["COMPRESS=JPEG", "JPEG_QUALITY=85", "TILED=YES", "PHOTOMETRIC=YCBCR"])
        o.SetGeoTransform(ogt)
        o.SetProjection(base_ds.GetProjection())
        for b in range(3):
            o.GetRasterBand(b + 1).WriteArray(img[:, :, 2 - b])
        o = None
        os.remove(tmp)
    return W, H


def compare_ref(txy, ref_xy, max_dist):
    """与参考树点/树冠质心比：最近距离分布（平台上两批次位置一致性口径）+ 一对一匹配召回/精度。"""
    d, _ = cKDTree(ref_xy).query(txy)
    lines = [f"每株到参考最近点距离：中位 {np.median(d):.2f} m；0.5 m 内 {100*(d<=0.5).mean():.0f}%，1 m 内 {100*(d<=1).mean():.0f}%，1.5 m 内 {100*(d<=1.5).mean():.0f}%"]
    qs = np.percentile(txy[:, 1], [0, 25, 50, 75, 100])
    lines.append("  南北四段中位距离：" + "，".join(f"{np.median(d[(txy[:,1]>=lo)&(txy[:,1]<=hi)]):.2f}" for lo, hi in zip(qs[:-1], qs[1:])) + " m（南→北）")
    fp = STRtree([Point(*p).buffer(6.0) for p in ref_xy])
    in_fp = np.array([len(fp.query(Point(*p))) > 0 for p in txy])
    pi, ri, dm = geo.match_one_to_one(txy, ref_xy, max_dist)
    rec, prec = len(ri) / len(ref_xy), len(pi) / max(in_fp.sum(), 1)
    lines.append(f"  {max_dist} m 内一对一匹配 {len(pi)} 株：召回 {rec:.1%}，范围内精度 {prec:.1%}，F1 {2*rec*prec/max(rec+prec,1e-9):.3f}；"
                 f"范围外预测 {(~in_fp).sum()} 株" + (f"，匹配质心中位差 {np.median(dm):.2f} m" if len(dm) else ""))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="export_platform 输出的 zip")
    ap.add_argument("--out", required=True, help="输出目录（解压也放这里的 extract/ 下，用完可删）")
    ap.add_argument("--ref", default=None, help="参考树点/树冠（gpkg/shp，任意坐标系），用于比对与叠加")
    ap.add_argument("--gsd", type=float, default=0.2, help="整幅渲染分辨率(m)，5 cm 底图用 0.2 约 3700x5400 px")
    ap.add_argument("--detail", type=float, nargs=4, action="append", metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="影像坐标系下的放大窗口，可多次指定，按原分辨率出图")
    ap.add_argument("--max-dist", type=float, default=1.5)
    ap.add_argument("--keep-extract", action="store_true", help="保留解压目录")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    extract = os.path.join(a.out, "extract")
    warnings = []
    report = {"import_status": "completed", "errors": []}

    def fail(code, msg):
        report["import_status"] = "failed"
        report["errors"].append({"code": code, "message": msg})
        print(f"平台会拒绝导入 [{code}]：{msg}")
        json.dump(report, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=2)
        raise SystemExit(1)

    try:
        n = unzip_archive(a.zip, extract)
    except (ValueError, zipfile.BadZipFile) as e:
        fail("unzip_failed", str(e))
    print(f"解压 {n} 个文件 -> {extract}")
    found = locate_files(extract)
    try:
        validate_files(found)
        W, H, bands, bbox = validate_tif(found["tif"])
    except (ValueError, subprocess.CalledProcessError) as e:
        fail("validate_failed", str(e))
    print(f"底图 {os.path.basename(found['tif'])}：{W}x{H}，{bands} 波段，解码 {W*H*4/2**30:.2f} GB（预算 2 GB），"
          f"经纬度范围 {bbox[0]:.6f},{bbox[1]:.6f} ~ {bbox[2]:.6f},{bbox[3]:.6f}")
    for k in ("tree_point", "tree_crown", "pest_polygon", "csv"):
        print(f"  {k}: {os.path.relpath(found[k], extract) if found[k] else '（无）'}")

    try:
        points = read_shapefile(found["tree_point"], a.out, "tree_point")
        crowns = read_shapefile(found["tree_crown"], a.out, "tree_crown")
        polys = read_shapefile(found["pest_polygon"], a.out, "pest_polygon")
    except subprocess.CalledProcessError as e:
        fail("ogr2ogr_failed", e.stderr)
    table, csv_skipped = None, 0
    if found["csv"]:
        try:
            table, csv_skipped = read_csv_table(found["csv"], warnings)
        except ValueError as e:
            fail("csv_invalid", str(e))
    trees, skipped, dropped_geom, near_limit = build_tree_imports(points, crowns, table, warnings)
    n_det, detected = build_detections(polys, trees, warnings)
    pest = sum(1 for t in trees if t["pest"])
    with_crown = sum(1 for t in trees if t["crown_json"])
    top, n_warn = dedupe(warnings)
    report.update(dict(total_trees=len(trees), pest_trees=pest, healthy_trees=len(trees) - pest, trees_with_crown=with_crown,
                       crown_geometry_dropped_over_16kb=dropped_geom, crown_geometry_near_limit=near_limit,
                       detected_geometry_tree_count=len(detected), pest_polygons=n_det,
                       missing_geometry_pest_trees=sum(1 for t in trees if t["pest"] and t["tree_id"] not in detected),
                       skipped_rows=skipped + csv_skipped, warning_count=n_warn, warnings=top,
                       point_features=len(points), crown_features=len(crowns), csv_rows=len(table["rows"]) if table else 0,
                       tif=dict(width=W, height=H, bands=bands, bbox_lonlat=bbox)))
    print(f"平台口径：total_trees {len(trees)}，pest {pest}，带树冠几何 {with_crown}（>16KB 丢弃 {dropped_geom}，接近上限 {near_limit}），"
          f"病虫害多边形 {n_det}，跳过 {skipped + csv_skipped}，告警 {n_warn} 条")
    for w in top[:10]:
        print("  告警：", w)
    if len(trees) == 0:
        fail("no_trees", "没有可导入的单树")

    # 坐标转到影像坐标系，输出逐株表
    base_ds = gdal.Open(found["tif"])
    txy = to_raster_xy([(t["lon"], t["lat"]) for t in trees], base_ds)
    for t, (x, y) in zip(trees, txy):
        t["x"], t["y"] = float(x), float(y)
    with open(os.path.join(a.out, "trees.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tree_id", "lon", "lat", "x", "y", "pest", "tree_area", "yield_value", "crown_bytes"])
        for t in trees:
            w.writerow([t["tree_id"], f"{t['lon']:.7f}", f"{t['lat']:.7f}", f"{t['x']:.3f}", f"{t['y']:.3f}", int(t["pest"]),
                        "" if t["tree_area"] is None else f"{t['tree_area']:.3f}", "" if t["yield_value"] is None else t["yield_value"], t["crown_bytes"]])
    areas = np.array([t["tree_area"] for t in trees if t["tree_area"] is not None])
    if len(areas):
        print(f"冠幅面积：中位 {np.median(areas):.1f} m²，10%～90% {np.percentile(areas,10):.1f}～{np.percentile(areas,90):.1f} m²")

    # 树冠几何（平台落库的那份，已经过 7 位小数量化）转到影像坐标系
    src = osr.SpatialReference(wkt=base_ds.GetProjection())
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(geo.srs_from_epsg(4326), src)
    crown_polys = []
    for t in trees:
        if not t["crown_json"]:
            continue
        g = ogr.CreateGeometryFromJson(t["crown_json"])
        g.Transform(tr)
        from shapely import wkb
        s = wkb.loads(bytes(g.ExportToWkb()))
        crown_polys.append(geo.largest_polygon(s) if s.geom_type == "MultiPolygon" else s)

    ref_geoms, ref_xy = [], None
    if a.ref:
        epsg = geo.raster_epsg(found["tif"])
        ref_geoms = [g for g, _ in geo.read_vector(a.ref, epsg)]
        ref_xy = np.array([[g.centroid.x, g.centroid.y] for g in ref_geoms])
        print(f"参考 {os.path.basename(a.ref)}：{len(ref_xy)} 株")
        for line in compare_ref(txy, ref_xy, a.max_dist):
            print(line)
        report["ref_compare"] = dict(ref=a.ref, n_ref=len(ref_xy))

    print(f"渲染整幅（{a.gsd} m）…")
    W2, H2 = render(base_ds, a.gsd, trees, crown_polys, ref_geoms, os.path.join(a.out, "overview"), threads=a.threads)
    print(f"  overview.jpg / overview.tif：{W2}x{H2} px")
    for k, win in enumerate(a.detail or []):
        w2, h2 = render(base_ds, a.gsd, trees, crown_polys, ref_geoms, os.path.join(a.out, f"detail_{k}"), window=tuple(win))
        n_in = int(((txy[:, 0] >= win[0]) & (txy[:, 0] <= win[2]) & (txy[:, 1] >= win[1]) & (txy[:, 1] <= win[3])).sum())
        n_ref = int(((ref_xy[:, 0] >= win[0]) & (ref_xy[:, 0] <= win[2]) & (ref_xy[:, 1] >= win[1]) & (ref_xy[:, 1] <= win[3])).sum()) if ref_xy is not None else None
        print(f"  detail_{k}.jpg：{w2}x{h2} px，窗口内树 {n_in} 株" + (f"，参考 {n_ref} 株" if n_ref is not None else ""))
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=2)
    if not a.keep_extract:
        shutil.rmtree(extract, ignore_errors=True)
    print("报告 ->", os.path.join(a.out, "report.json"))


if __name__ == "__main__":
    main()
