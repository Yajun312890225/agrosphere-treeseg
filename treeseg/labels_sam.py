"""用 SAM2 以树点为提示，在影像上生成真实冠形训练标签。

背景：供应商只给树点 + 冠幅面积，"圆形伪标签"在郁闭林分里是对连续冠层的任意切分，
YOLO 拟合不了（训练集 mAP50 都不到 0.11）。本脚本用 SAM2 的点提示在 composite.tif 上
直接分割每株树冠，得到与影像一致的冠形；SAM 结果不合理的退回原标签。

流程：
  1. 读 labels.gpkg（treeseg.labels 生成，字段 tree_id/src_area），取每个多边形质心为点提示
  2. 按 tile/overlap 滑窗，每个切片只处理质心落在核心区的树，避免边界重复
  3. SAM2 多掩膜输出（每点 3 个候选），过滤后按面积与 src_area 最接近的候选选取
  4. 掩膜 → 连通块 → 轮廓 → 世界坐标多边形；重叠过大的按得分去重
  5. 写 labels_sam.gpkg（tree_id, src_area, area_m2, source=sam|fallback, score）

示例：
  python -m treeseg.labels_sam --raster prep/composite.tif --labels labels_0322.gpkg --sam sam2.1_b.pt --out labels_sam.gpkg
"""
import argparse
import math
import os
import numpy as np
from osgeo import gdal
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from skimage import measure
from . import geo
from .predict import tile_windows

gdal.UseExceptions()


def mask_to_polygon(mask: np.ndarray, px: int, py: int, x0: int, y0: int, gt, simplify_m: float):
    """布尔掩膜 → 含提示点 (px,py) 的连通块 → 世界坐标多边形；失败返回 None。"""
    lab, n = ndi.label(mask)
    if n == 0:
        return None
    keep = lab[py, px]
    if keep == 0:
        # 提示点恰好在掩膜外（边界抖动），取 3 像素邻域内出现最多的标号
        win = lab[max(py - 3, 0):py + 4, max(px - 3, 0):px + 4]
        vals = win[win > 0]
        if vals.size == 0:
            return None
        keep = np.bincount(vals).argmax()
    m = ndi.binary_fill_holes(lab == keep)
    contours = measure.find_contours(np.pad(m, 1).astype(np.uint8), 0.5)
    if not contours:
        return None
    cont = max(contours, key=len)
    cols = x0 + cont[:, 1] - 1 + 0.5
    rows = y0 + cont[:, 0] - 1 + 0.5
    wx, wy = geo.pixel_to_world(gt, cols, rows)
    poly = Polygon(np.c_[wx, wy]).buffer(0)
    if poly.is_empty:
        return None
    poly = geo.largest_polygon(poly).simplify(simplify_m)
    if not poly.is_valid or poly.is_empty:
        return None
    return poly


def mean_ndvi(poly, ndvi_ds, ndvi_gt):
    """多边形内 NDVI 均值（按像元中心采样）；无有效像元返回 nan。"""
    from shapely import contains_xy
    minx, miny, maxx, maxy = poly.bounds
    c0, r0 = geo.world_to_pixel(ndvi_gt, minx, maxy)
    c1, r1 = geo.world_to_pixel(ndvi_gt, maxx, miny)
    c0, r0 = max(int(math.floor(c0)), 0), max(int(math.floor(r0)), 0)
    c1, r1 = min(int(math.ceil(c1)) + 1, ndvi_ds.RasterXSize), min(int(math.ceil(r1)) + 1, ndvi_ds.RasterYSize)
    if c1 - c0 < 1 or r1 - r0 < 1:
        return float("nan")
    arr = ndvi_ds.GetRasterBand(1).ReadAsArray(c0, r0, c1 - c0, r1 - r0).astype(np.float32)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    wx, wy = geo.pixel_to_world(ndvi_gt, xx + 0.5, yy + 0.5)
    inside = contains_xy(poly, wx.ravel(), wy.ravel()).reshape(arr.shape)
    vals = arr[inside & np.isfinite(arr)]
    if vals.size == 0:  # 多边形比像元还小，退化为取质心所在像元
        vals = arr[np.isfinite(arr)]
    return float(vals.mean()) if vals.size else float("nan")


def build_predictor(sam_path: str, device):
    from ultralytics.models.sam import SAM2Predictor
    ov = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=sam_path, verbose=False)
    if device:
        ov["device"] = device
    p = SAM2Predictor(overrides=ov)
    p.setup_model()
    return p


def sam_masks(pred, bgr: np.ndarray, pts_px: np.ndarray, chunk: int):
    """对一张切片跑 SAM2 点提示，返回 (N,3,H,W) 布尔掩膜与 (N,3) 得分。"""
    import torch
    import torch.nn.functional as F
    pred.setup_source(bgr)
    im = None
    for batch in pred.dataset:
        pred.batch = batch
        im = pred.preprocess(batch[1])
        pred.features = pred.get_im_features(im)
        break
    H, W = bgr.shape[:2]
    masks_all, scores_all = [], []
    for s in range(0, len(pts_px), chunk):
        pts = pts_px[s:s + chunk]
        with torch.inference_mode():
            out = pred.prompt_inference(im, points=pts, labels=np.ones(len(pts)), multimask_output=True)
        logits, scores = out[0].detach(), out[1].detach()
        n = len(pts)
        logits = logits.reshape(n, -1, *logits.shape[-2:]).float()
        up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False) > 0
        masks_all.append(up.cpu().numpy())
        scores_all.append(scores.reshape(n, -1).float().cpu().numpy())
    pred.features = None
    return np.concatenate(masks_all), np.concatenate(scores_all)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", required=True, help="composite.tif")
    ap.add_argument("--labels", required=True, help="treeseg.labels 生成的 labels.gpkg（用质心做点提示，原多边形做退路）")
    ap.add_argument("--sam", default="sam2.1_b.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=64, help="每批提示点数")
    ap.add_argument("--min-area", type=float, default=1.0, help="掩膜最小面积 m²")
    ap.add_argument("--min-ratio", type=float, default=0.25, help="掩膜面积 / src_area 下限")
    ap.add_argument("--max-ratio", type=float, default=3.0, help="掩膜面积 / src_area 上限")
    ap.add_argument("--iou-dup", type=float, default=0.6, help="两个结果 IoU 超过此值视为重复，保留得分高者")
    ap.add_argument("--simplify", type=float, default=0.1, help="多边形简化容差 m")
    ap.add_argument("--ndvi", default=None, help="ndvi.tif：掩膜内 NDVI 均值低于阈值的 SAM 结果作废；退回的原标签也低于阈值则整株丢弃（树点落在裸地/已砍伐）")
    ap.add_argument("--ndvi-thr", type=float, default=0.6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个切片（调试）")
    a = ap.parse_args()

    ds = gdal.Open(a.raster)
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    gsd = abs(gt[1])
    epsg = geo.raster_epsg(a.raster)
    feats = geo.read_vector(a.labels, epsg)
    cents = np.array([[g.centroid.x, g.centroid.y] for g, _ in feats])
    cols, rows = geo.world_to_pixel(gt, cents[:, 0], cents[:, 1])
    src_area = np.array([float(at.get("src_area") or g.area) for g, at in feats])
    print(f"影像 {W}x{H} px GSD {gsd} m，标签 {len(feats)} 个，切片 {a.tile} 重叠 {a.overlap}")

    ndvi_ds = ndvi_gt = None
    if a.ndvi:
        ndvi_ds = gdal.Open(a.ndvi)
        ndvi_gt = ndvi_ds.GetGeoTransform()
    pred = build_predictor(a.sam, a.device)
    xs, ys = tile_windows(W, H, a.tile, a.overlap)
    core = a.overlap // 2
    result = [None] * len(feats)   # (poly, area, source, score)
    n_tiles = 0
    for y0 in ys:
        for x0 in xs:
            if a.limit and n_tiles >= a.limit:
                break
            cx0 = x0 + (core if x0 > 0 else 0)
            cy0 = y0 + (core if y0 > 0 else 0)
            cx1 = x0 + (a.tile - core if x0 + a.tile < W else a.tile)
            cy1 = y0 + (a.tile - core if y0 + a.tile < H else a.tile)
            idx = np.where((cols >= cx0) & (cols < cx1) & (rows >= cy0) & (rows < cy1))[0]
            if len(idx) == 0:
                continue
            arr = ds.ReadAsArray(x0, y0, a.tile, a.tile)[:3]
            if (arr.max(axis=0) == 0).mean() > 0.95:
                continue
            n_tiles += 1
            bgr = np.ascontiguousarray(arr.transpose(1, 2, 0)[:, :, ::-1])
            pts = np.c_[cols[idx] - x0, rows[idx] - y0].astype(np.float32)
            masks, scores = sam_masks(pred, bgr, pts, a.chunk)
            n_ok = 0
            for k, i in enumerate(idx):
                px, py = int(min(max(pts[k, 0], 0), a.tile - 1)), int(min(max(pts[k, 1], 0), a.tile - 1))
                best = None
                for m in range(masks.shape[1]):
                    mk = masks[k, m]
                    area = float(mk.sum()) * gsd * gsd
                    if area < a.min_area or area < a.min_ratio * src_area[i] or area > a.max_ratio * src_area[i]:
                        continue
                    # 只看含提示点的连通块面积，避免大掩膜里挂着远处碎片
                    poly = mask_to_polygon(mk, px, py, x0, y0, gt, a.simplify)
                    if poly is None:
                        continue
                    if poly.area < a.min_area or poly.area < a.min_ratio * src_area[i] or poly.area > a.max_ratio * src_area[i]:
                        continue
                    if ndvi_ds is not None and not (mean_ndvi(poly, ndvi_ds, ndvi_gt) >= a.ndvi_thr):
                        continue  # 分到了裸地/枯枝
                    key = (abs(math.log(poly.area / src_area[i])), -float(scores[k, m]))
                    if best is None or key < best[0]:
                        best = (key, poly, float(scores[k, m]))
                if best is not None:
                    result[i] = (best[1], best[1].area, "sam", best[2])
                    n_ok += 1
            print(f"  切片 {n_tiles} ({x0},{y0})：提示 {len(idx)} 株，SAM 成功 {n_ok}", flush=True)
        else:
            continue
        break

    # 退路 + NDVI 丢弃 + 去重
    geoms, attrs = [], []
    dropped = np.zeros(len(feats), bool)
    n_lowndvi = 0
    for i, (g, at) in enumerate(feats):
        if result[i] is None:
            fb = geo.largest_polygon(g.buffer(0))
            if ndvi_ds is not None and not (mean_ndvi(fb, ndvi_ds, ndvi_gt) >= a.ndvi_thr):
                dropped[i] = True
                n_lowndvi += 1
            result[i] = (fb, fb.area, "fallback", 0.0)
    polys = [r[0] for r in result]
    tree = STRtree(polys)
    order = sorted(range(len(polys)), key=lambda i: -result[i][3])
    for i in order:
        if dropped[i]:
            continue
        for j in tree.query(polys[i]):
            if j == i or dropped[j]:
                continue
            inter = polys[i].intersection(polys[j]).area
            if inter <= 0:
                continue
            if inter / (polys[i].area + polys[j].area - inter) > a.iou_dup:
                dropped[j] = True
    for i, (g, at) in enumerate(feats):
        if dropped[i]:
            continue
        poly, area, source, score = result[i]
        geoms.append(poly)
        attrs.append({"tree_id": str(at.get("tree_id", i + 1)), "src_area": float(src_area[i]),
                      "area_m2": float(area), "source": source, "score": float(score)})
    geo.write_gpkg(a.out, geoms, attrs, epsg)
    n_sam = sum(1 for x in attrs if x["source"] == "sam")
    areas = np.array([x["area_m2"] for x in attrs if x["source"] == "sam"])
    print(f"写出 {len(geoms)} 个标签 -> {a.out}；SAM {n_sam}，退回原标签 {len(geoms) - n_sam}，"
          f"NDVI 过低丢弃 {n_lowndvi}，重复丢弃 {int(dropped.sum()) - n_lowndvi}；"
          + (f"SAM 面积中位 {np.median(areas):.1f} m²" if len(areas) else "无 SAM 结果"))


if __name__ == "__main__":
    main()
