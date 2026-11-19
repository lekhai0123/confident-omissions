"""E16: correlated failures between the monitored segmenter and each external witness (v11, ACDC).

Witnesses: detector (RT-DETR normal pass, score >= 0.3, covers >= 50% of the object), detector low threshold incl.
gamma pass (score >= 0.05), cross-model (another of the 4 segmenters gets >= 50% of the object right), depth fg/bg
(mean foreground probability >= 0.5; 500-frame local subset). For each monitored model and object category
(confident omission into background / other omission / well segmented): P(witness sees the object).
phi = correlation between "segmenter omits" and "witness misses" over all objects (0 = independent failures).

usage: python e16_correlated.py <v11 dir> <depth-fg csv> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
MODELS = ["segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"]
KEY = ["stem", "condition", "instance_id"]


def main():
    d, depth_csv, out = sys.argv[1:4]
    objs = {}
    for m in MODELS:
        o = pd.read_csv(os.path.join(d, f"objects_{m}.csv"),
                        usecols=KEY + ["omitted", "confident_omission", "predicted_as", "frac_true", "area"])
        o["co_stuff"] = o.confident_omission & o.predicted_as.isin(STUFF)
        objs[m] = o.set_index(KEY)
    det_hi, det_lo = set(), set()
    for m in MODELS[:1]:   # detector proposals are identical across monitored models
        for ch in pd.read_csv(os.path.join(d, f"proposals_{m}.csv"), chunksize=300_000,
                              usecols=["stem", "condition", "gt_instance_id", "gt_cover", "det_src", "det_score"]):
            c = ch[(ch.gt_cover >= 0.5) & ch.det_src.isin(["rtdetr", "rtdetr_gamma"])]
            det_lo |= set(zip(c.stem, c.condition, c.gt_instance_id))
            c2 = c[(c.det_src == "rtdetr") & (c.det_score >= 0.3)]
            det_hi |= set(zip(c2.stem, c2.condition, c2.gt_instance_id))
    dep = pd.read_csv(depth_csv).set_index(KEY).fg_mean
    good_all = pd.concat({m: objs[m].frac_true >= 0.5 for m in MODELS}, axis=1)
    report = {}
    for m in MODELS:
        o = objs[m].copy()
        others = [x for x in MODELS if x != m]
        o["w_det_hi"] = [k in det_hi for k in o.index]
        o["w_det_lo"] = [k in det_lo for k in o.index]
        o["w_xmodel"] = good_all.loc[o.index, others].any(axis=1).to_numpy()
        o["w_depth"] = dep.reindex(o.index).to_numpy() >= 0.5
        o["has_depth"] = dep.reindex(o.index).notna().to_numpy()
        o["cat"] = np.where(o.co_stuff, "CO-to-stuff", np.where(o.omitted, "other omitted", np.where(o.frac_true >= 0.5, "good", "partial")))
        o["sz"] = pd.cut(o.area, [0, 1024, 4096, 1e12], labels=["<1k px", "1k-4k", ">4k"])
        R = {}
        for w in ("w_det_hi", "w_det_lo", "w_xmodel", "w_depth"):
            sub = o[o.has_depth] if w == "w_depth" else o
            rates = {c: round(float(g[w].mean()), 3) for c, g in sub.groupby("cat") if len(g) >= 10}
            fails = sub.omitted.to_numpy().astype(float)
            misses = (~sub[w].to_numpy()).astype(float)
            phi = float(np.corrcoef(fails, misses)[0, 1]) if fails.std() and misses.std() else float("nan")
            by_size = {str(s): {c: round(float(g[w].mean()), 3) for c, g in gg.groupby("cat") if len(g) >= 10 and c in ("CO-to-stuff", "good")}
                       for s, gg in sub.groupby("sz", observed=True)}
            R[w] = {"P_sees": rates, "phi_omit_vs_miss": round(phi, 3), "n": int(len(sub)), "by_size": by_size,
                    "n_co_stuff": int((sub.cat == "CO-to-stuff").sum())}
        report[m] = R
        print(f"\n== {m}")
        for w, r in R.items():
            print(f"  {w:9s} P(sees): {r['P_sees']}  phi(omit, miss) {r['phi_omit_vs_miss']:+.3f}  (n CO {r['n_co_stuff']})")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
