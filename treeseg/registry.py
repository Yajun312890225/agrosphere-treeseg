"""跨期单株 ID 台账：把新一期预测树冠与台账匹配，继承 ID，未匹配的发新 ID。

  python -m treeseg.registry --registry registry.gpkg --new preds/crowns.gpkg --date 2026-05-24 --out registry.gpkg
台账不存在时用新一期初始化。输出同时写 <new 目录>/crowns_with_ids.gpkg。
"""
import argparse
import os
import numpy as np
from shapely.affinity import translate
from . import geo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--date", required=True, help="本期采集日期 YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="更新后的台账（可与 --registry 相同）")
    ap.add_argument("--max-dist", type=float, default=1.5)
    ap.add_argument("--no-auto-shift", action="store_true", help="不估计整体平移（有 RTK 固定解时可关）")
    ap.add_argument("--id-prefix", default="T")
    a = ap.parse_args()
    epsg = geo.vector_epsg(a.new)
    new = geo.read_vector(a.new, epsg)
    nxy = np.array([[g.centroid.x, g.centroid.y] for g, _ in new])
    if not os.path.exists(a.registry):
        reg_geoms, reg_attrs = [], []
    else:
        reg = geo.read_vector(a.registry, epsg)
        reg_geoms = [g for g, _ in reg]
        reg_attrs = [at for _, at in reg]
    dx = dy = 0.0
    if reg_geoms and not a.no_auto_shift:
        rxy = np.array([[g.centroid.x, g.centroid.y] for g in reg_geoms])
        dx, dy = geo.estimate_shift(nxy, rxy)
        print(f"本期→台账 平移估计 dx={dx:.2f} dy={dy:.2f} m")
    nxy_s = nxy + [dx, dy]
    import re
    def _num(v):
        m = re.search(r"(\d+)$", str(v or ""))
        return int(m.group(1)) if m else 0
    next_num = 1 + max([_num(at.get("tree_id")) for at in reg_attrs] + [0])
    matched_reg = set()
    out_geoms, out_attrs = [], []
    if reg_geoms:
        rxy = np.array([[g.centroid.x, g.centroid.y] for g in reg_geoms])
        ni, ri, d = geo.match_one_to_one(nxy_s, rxy, a.max_dist)
        assign = {int(i): int(r) for i, r in zip(ni, ri)}
    else:
        assign = {}
    for i, (g, at) in enumerate(new):
        g2 = translate(g, dx, dy)
        if i in assign:
            r = assign[i]
            matched_reg.add(r)
            ra = dict(reg_attrs[r])
            ra.update({"last_seen": a.date, "seen_count": int(ra.get("seen_count", 0)) + 1, "missing_count": 0,
                       "area_m2": float(g2.area), "conf": float(at.get("conf", 0) or 0)})
            out_geoms.append(g2)
            out_attrs.append(ra)
            at["tree_id"] = ra["tree_id"]
        else:
            tid = f"{a.id_prefix}{next_num:06d}"
            next_num += 1
            out_geoms.append(g2)
            out_attrs.append({"tree_id": tid, "first_seen": a.date, "last_seen": a.date, "seen_count": 1, "missing_count": 0,
                              "area_m2": float(g2.area), "conf": float(at.get("conf", 0) or 0)})
            at["tree_id"] = tid
    # 台账里本期没见到的：保留并计数
    lost = 0
    for r, (g, ra) in enumerate(zip(reg_geoms, reg_attrs)):
        if r in matched_reg:
            continue
        ra = dict(ra)
        ra["missing_count"] = int(ra.get("missing_count", 0)) + 1
        out_geoms.append(g)
        out_attrs.append(ra)
        lost += 1
    keys = ["tree_id", "first_seen", "last_seen", "seen_count", "missing_count", "area_m2", "conf"]
    out_attrs = [{k: at.get(k) for k in keys} for at in out_attrs]
    geo.write_gpkg(a.out, out_geoms, out_attrs, epsg, layer_name="registry")
    new_out = os.path.join(os.path.dirname(a.new), "crowns_with_ids.gpkg")
    geo.write_gpkg(new_out, [translate(g, dx, dy) for g, _ in new], [dict(at) for _, at in new], epsg)
    print(f"本期 {len(new)} 株：继承 ID {len(assign)}，新增 {len(new)-len(assign)}；台账未见 {lost}；台账总数 {len(out_geoms)} -> {a.out}")
    print("本期带 ID 的树冠 ->", new_out)


if __name__ == "__main__":
    main()
