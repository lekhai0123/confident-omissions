"""E36: coverage of the targets by each source of external evidence at a matched number of regions (review M3).

v11 (ACDC adverse, four monitored segmenters). Sources of candidate regions, all stored in the same run:
  detector   - RT-DETR boxes (normal + gamma pass), ranked by detector score;
  second seg - connected regions that another segmenter labels as a thing where the monitored one predicts stuff
               (source 'xmodel'), ranked by size;
  depth      - Maag & Rottmann depth foreground regions where the monitored segmenter predicts no thing ('maag'),
               ranked by size.
For k regions per image x model (k = 1, 2, 5, 10, 20, all) we report the share of confident omissions into
background covered (>= 50% of their pixels) by one of the first k regions, the share of the regions that hit no
omitted object (excess regions), and coverage by class group and size. This measures P(region covers the target),
which bounds recall, separately from any scorer.

usage: python e36_source_coverage.py <v11 dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e_suite_v8 import ADVERSE, MODELS, STUFF  # noqa: E402

SRC = {"detector": (["rtdetr", "rtdetr_gamma"], "det_score"), "second segmenter": (["xmodel"], "p_area_frac"),
       "depth (Maag)": (["maag"], "p_area_frac")}
KS = (1, 2, 5, 10, 20, 1000)


def main():
    d, out = sys.argv[1:3]
    objs, props = [], []
    for tag in MODELS:
        o = pd.read_csv(os.path.join(d, f"objects_{tag}.csv"),
                        usecols=["stem", "condition", "instance_id", "class", "area", "omitted", "confident_omission", "predicted_as"])
        o["model"] = tag
        o = o[o.condition.isin(ADVERSE)]
        o["co_stuff"] = o.confident_omission & o.predicted_as.isin(STUFF)
        objs.append(o)
        p = pd.read_csv(os.path.join(d, f"proposals_{tag}.csv"),
                        usecols=["source", "det_src", "det_score", "p_area_frac", "stem", "condition", "gt_instance_id", "gt_cover"])
        p["model"] = tag
        p["det_src"] = p.det_src.fillna("xmodel")
        props.append(p[p.condition.isin(ADVERSE)])
    O = pd.concat(objs, ignore_index=True)
    P = pd.concat(props, ignore_index=True)
    om = O[O.omitted][["model", "stem", "condition", "instance_id"]].rename(columns={"instance_id": "gt_instance_id"})
    om["hits_omitted"] = True
    P = P.merge(om, on=["model", "stem", "condition", "gt_instance_id"], how="left")
    P["hits_omitted"] = P.hits_omitted.fillna(False).astype(bool) & (P.gt_cover >= 0.5)
    K = O[O.co_stuff]
    keys = list(zip(K.model, K.stem, K.condition, K.instance_id))
    units = O.groupby(["model", "stem", "condition"]).ngroups
    res = {"n_targets": len(keys), "units": int(units), "sources": {}}
    for name, (srcs, rank_col) in SRC.items():
        S = P[P.det_src.isin(srcs)].copy()
        S["rk"] = S.groupby(["model", "stem", "condition"])[rank_col].rank(ascending=False, method="first")
        r = {"regions_per_unit": round(len(S) / units, 2)}
        for k in KS:
            Sk = S[S.rk <= k]
            cov = set(zip(*[Sk[Sk.gt_cover >= 0.5][c] for c in ("model", "stem", "condition", "gt_instance_id")]))
            hit = np.array([kk in cov for kk in keys])
            r[f"k={k if k < 1000 else 'all'}"] = {"coverage": round(float(hit.mean()), 4),
                                                   "regions_per_unit": round(len(Sk) / units, 2),
                                                   "excess_share": round(float((~Sk.hits_omitted).mean()), 3) if len(Sk) else None}
        cov = set(zip(*[S[S.gt_cover >= 0.5][c] for c in ("model", "stem", "condition", "gt_instance_id")]))
        hit = np.array([kk in cov for kk in keys])
        r["coverage_by_group_all"] = {g: round(float(hit[K["class"].isin(cl).to_numpy()].mean()), 3)
                                      for g, cl in {"person/rider": ["person", "rider"], "car": ["car"],
                                                    "truck/bus/train": ["truck", "bus", "train"],
                                                    "two-wheeler": ["motorcycle", "bicycle"]}.items()}
        res["sources"][name] = r
        print(name, r)
    # union detector + others (incremental coverage at all regions)
    def covset(srcs):
        S = P[P.det_src.isin(srcs) & (P.gt_cover >= 0.5)]
        return set(zip(S.model, S.stem, S.condition, S.gt_instance_id))
    cd, cx, cm = covset(["rtdetr", "rtdetr_gamma"]), covset(["xmodel"]), covset(["maag"])
    res["incremental_over_detector"] = {
        "second segmenter": round(float(np.mean([k in (cx - cd) for k in keys])), 4),
        "depth (Maag)": round(float(np.mean([k in (cm - cd) for k in keys])), 4),
        "detector_only_coverage": round(float(np.mean([k in cd for k in keys])), 4)}
    print(res["incremental_over_detector"])
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
