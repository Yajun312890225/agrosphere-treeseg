"""把预测树冠与参考数据（供应商树点或树冠多边形）对比，输出株数、匹配率、冠幅一致性。"""
import argparse
import numpy as np
from osgeo import gdal
from shapely.geometry import Point
from shapely.strtree import STRtree
from scipy import ndimage as ndi
from . import geo


def load_xy(path, epsg, csv_x, csv_y, csv_epsg):
    if path.lower().endswith(".csv"):
        feats = geo.read_points_csv(path, csv_x, csv_y, csv_epsg, epsg)
    else:
        feats = geo.read_vector(path, epsg)
    geoms = [g for g, _ in feats]
    xy = np.array([[g.centroid.x, g.centroid.y] for g in geoms])
    areas = np.array([g.area if g.geom_type != "Point" else np.nan for g in geoms])
    return geoms, xy, areas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="predict 输出的 crowns.gpkg")
    ap.add_argument("--ref", required=True, help="参考树点/树冠（shp/gpkg/csv）")
    ap.add_argument("--epsg", type=int, default=None, help="缺省取 pred 图层坐标系")
    ap.add_argument("--csv-x", default="Lon")
    ap.add_argument("--csv-y", default="Lat")
    ap.add_argument("--csv-epsg", type=int, default=4326)
    ap.add_argument("--max-dist", type=float, default=1.5, help="质心匹配距离阈值(m)")
    ap.add_argument("--auto-shift", action="store_true", help="先估计参考数据相对预测的整体平移")
    ap.add_argument("--footprint", type=float, default=6.0, help="参考点缓冲(m)构成范围，范围外的预测不计入误检")
    a = ap.parse_args()
    if a.epsg is None:
        a.epsg = geo.vector_epsg(a.pred)
    pg, pxy, parea = load_xy(a.pred, a.epsg, a.csv_x, a.csv_y, a.csv_epsg)
    rg, rxy, rarea = load_xy(a.ref, a.epsg, a.csv_x, a.csv_y, a.csv_epsg)
    print(f"预测 {len(pxy)} 株，参考 {len(rxy)} 株")
    if a.auto_shift:
        dx, dy = geo.estimate_shift(rxy, pxy)
        print(f"参考→预测 平移估计 dx={dx:.2f} dy={dy:.2f} m（已应用）")
        rxy = rxy + [dx, dy]
        from shapely.affinity import translate
        rg = [translate(g, dx, dy) for g in rg]
    # 范围：参考点缓冲
    fp = STRtree([Point(*p).buffer(a.footprint) for p in rxy])
    in_fp = np.array([len(fp.query(Point(*p))) > 0 for p in pxy])
    pi, ri, d = geo.match_one_to_one(pxy, rxy, a.max_dist)
    recall = len(ri) / len(rxy)
    precision = len(pi) / max(in_fp.sum(), 1)
    print(f"质心 {a.max_dist} m 内一对一匹配 {len(pi)} 株：召回 {recall:.1%}，范围内精度 {precision:.1%}，"
          f"F1 {2*recall*precision/max(recall+precision,1e-9):.3f}；范围内预测 {in_fp.sum()}，范围外 {(~in_fp).sum()}")
    if len(d):
        print(f"匹配质心距离：中位 {np.median(d):.2f} m，90% {np.percentile(d,90):.2f} m")
    # 参考点是否落在预测树冠内
    if all(g.geom_type == "Polygon" for g in pg):
        tree = STRtree(pg)
        hit = np.full(len(rxy), -1)
        for k, p in enumerate(rxy):
            for j in tree.query(Point(*p)):
                if pg[j].contains(Point(*p)):
                    hit[k] = j
                    break
        u, c = np.unique(hit[hit >= 0], return_counts=True)
        print(f"参考点落在预测树冠内 {np.mean(hit>=0):.1%}；含 1 点树冠 {(c==1).sum()}，含 2 点 {(c==2).sum()}，含 ≥3 点 {(c>=3).sum()}"
              f"（2 点以上=欠分割）")
    ok = np.isfinite(rarea[ri]) if len(ri) else np.array([], bool)
    if ok.sum() > 10:
        print(f"冠幅面积：预测中位 {np.median(parea[pi][ok]):.1f} m² vs 参考 {np.median(rarea[ri][ok]):.1f} m²，相关 {np.corrcoef(parea[pi][ok], rarea[ri][ok])[0,1]:.2f}")
    else:
        print(f"预测冠幅中位 {np.nanmedian(parea):.1f} m²（参考无面积信息）")


if __name__ == "__main__":
    main()
