"""把一套树冠标签非均匀配准到另一期影像上（局部平移场）。

场景：供应商 7 月树冠是在供应商自家正射上勾的，大疆智图成果（RTK 单点解）相对它偏 1～4 m
且各处不一致，用单一平移量套标签仍有半米以上残差。本脚本用"模型在目标影像上的预测结果"作为
同名点：全局平移粗配 → 一对一匹配 → 每株标签取周边匹配对残差的中位数作为局部修正。

  python -m treeseg.register_labels --labels labels_v07.gpkg --pred pred_vendor_0723/crowns.gpkg --out labels_0723.gpkg
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree
from shapely.affinity import translate
from . import geo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="待配准的标签 gpkg（如供应商 7 月树冠）")
    ap.add_argument("--pred", required=True, help="模型在目标影像上的预测 crowns.gpkg，作为同名点")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dist", type=float, default=1.5, help="一对一匹配距离阈值(m)")
    ap.add_argument("--radius", type=float, default=40.0, help="局部残差统计半径(m)")
    ap.add_argument("--min-pts", type=int, default=15, help="半径内匹配对少于此数则半径加倍，再不足退回全局平移")
    a = ap.parse_args()

    epsg = geo.vector_epsg(a.pred)
    labels = geo.read_vector(a.labels, epsg)
    preds = geo.read_vector(a.pred, epsg)
    lxy = np.array([[g.centroid.x, g.centroid.y] for g, _ in labels])
    pxy = np.array([[g.centroid.x, g.centroid.y] for g, _ in preds])
    dx, dy = geo.estimate_shift(lxy, pxy)
    print(f"标签 {len(lxy)} 株，预测 {len(pxy)} 株；全局平移 dx={dx:.2f} dy={dy:.2f} m")
    shifted = lxy + [dx, dy]
    pi, li, d = geo.match_one_to_one(pxy, shifted, a.max_dist)
    resid = pxy[pi] - shifted[li]
    print(f"一对一匹配 {len(pi)} 对，全局平移后残差中位 {np.median(d):.2f} m，"
          f"残差向量中位 ({np.median(resid[:, 0]):+.2f}, {np.median(resid[:, 1]):+.2f})")

    tree = cKDTree(shifted[li])
    local = np.zeros_like(lxy)
    n_local = np.zeros(len(lxy), int)
    for i in range(len(lxy)):
        for r in (a.radius, a.radius * 2):
            idx = tree.query_ball_point(shifted[i], r)
            if len(idx) >= a.min_pts:
                local[i] = np.median(resid[idx], axis=0)
                n_local[i] = len(idx)
                break
    corrected = shifted + local
    # 自检：修正后重新匹配
    pi2, li2, d2 = geo.match_one_to_one(pxy, corrected, a.max_dist)
    print(f"局部修正：{(n_local > 0).sum()} 株有局部修正量（中位 {np.median(np.hypot(*local[n_local > 0].T)):.2f} m，"
          f"最大 {np.hypot(*local.T).max():.2f} m）；修正后匹配 {len(pi2)} 对，残差中位 {np.median(d2):.2f} m")

    geoms, attrs = [], []
    for i, (g, at) in enumerate(labels):
        tx, ty = dx + local[i, 0], dy + local[i, 1]
        geoms.append(translate(geo.largest_polygon(g.buffer(0)), tx, ty))
        attrs.append({"tree_id": str(at.get("tree_id", i + 1)), "src_area": float(at.get("src_area") or g.area),
                      "dx": float(tx), "dy": float(ty), "n_local": int(n_local[i])})
    geo.write_gpkg(a.out, geoms, attrs, epsg)
    print("写出", a.out)


if __name__ == "__main__":
    main()
