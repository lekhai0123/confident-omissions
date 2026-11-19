"""E13: synthetic-vs-real rank agreement over 13 models (v9: 7 models + v12: 6 SegFormer variants).

v9 evaluated a 70% prefix of the image permutation used by v12, so every model is restricted to the common
(stem, condition) set before computing object-level quantities.
For each matched pair (real ACDC condition ~ synthetic corruption) Spearman rho across models of:
  - mIoU (from summary.json; per-image IoU counts are not stored, so this uses each run's own image set),
  - omitted share (object level, common images) - an accuracy measure computed on identical images,
  - confident-into-background share of omissions (common images).
p-values (exact permutation for n=13 via scipy) and a 95% bootstrap CI over images (stems resampled jointly
across models, so the CI reflects image sampling, not model sampling).

usage: python e13_rank.py <merged dir with objects_*.csv and summary.json> <v9 dir>
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
PAIRS = {"fog": ["fog-3", "fog-5"], "night": ["dark-3", "dark-5", "brightness-5"], "rain": ["motion_blur-3", "defocus_blur-3"],
         "snow": ["gaussian_noise-3", "shot_noise-3", "contrast-5"]}
N_BOOT = 1000


def main():
    d, v9 = sys.argv[1:3]
    summ = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    common = pd.read_csv(os.path.join(v9, "images_segformer-b0.csv"), usecols=["stem", "condition"]).drop_duplicates()
    conds = sorted(set(PAIRS) | {c for v in PAIRS.values() for c in v})
    agg = {}   # tag -> DataFrame indexed by (condition, stem): n_obj, n_om, n_co
    miou = {}
    for f in sorted(glob.glob(os.path.join(d, "objects_*.csv"))):
        tag = os.path.basename(f)[len("objects_"):-4]
        o = pd.read_csv(f, usecols=["stem", "condition", "omitted", "confident_omission", "predicted_as"])
        o = o[o.condition.isin(conds)].merge(common, on=["stem", "condition"])
        o["co_bg"] = o.confident_omission & o.predicted_as.isin(STUFF)
        agg[tag] = o.groupby(["condition", "stem"]).agg(n_obj=("omitted", "size"), n_om=("omitted", "sum"), n_co=("co_bg", "sum"))
        miou[tag] = {c: summ["models"][tag]["miou_by_condition"][c]["argmax"]["miou"] for c in conds}
    tags = list(agg)
    stems = {c: agg[tags[0]].loc[c].index.to_numpy() for c in conds}

    def stats(c, idx=None):
        om, cs = [], []
        for t in tags:
            g = agg[t].loc[c]
            if idx is not None:
                g = g.loc[idx]
            om.append(g.n_om.sum() / g.n_obj.sum())
            cs.append(g.n_co.sum() / max(1, g.n_om.sum()))
        return np.array(om), np.array(cs)

    rng = np.random.default_rng(0)
    report = {"models": tags, "n_models": len(tags), "pairs": {}}
    print(f"{len(tags)} models; images per condition: " + ", ".join(f"{c}={len(stems[c])}" for c in PAIRS))
    for real, syns in PAIRS.items():
        om_r, cs_r = stats(real)
        for s in syns:
            om_s, cs_s = stats(s)
            m_r = np.array([miou[t][real] for t in tags]); m_s = np.array([miou[t][s] for t in tags])
            rm, pm = spearmanr(m_r, m_s)
            ro, po = spearmanr(om_r, om_s)
            rc, pc = spearmanr(cs_r, cs_s)
            boot = []
            for _ in range(N_BOOT):
                ir = rng.choice(stems[real], len(stems[real])); is_ = rng.choice(stems[s], len(stems[s]))
                boot.append(spearmanr(stats(real, ir)[1], stats(s, is_)[1])[0])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            report["pairs"][f"{real}~{s}"] = {"rho_miou": round(rm, 3), "p_miou": round(pm, 4), "rho_omitted": round(ro, 3),
                                              "p_omitted": round(po, 4), "rho_conf_bg": round(rc, 3), "p_conf_bg": round(pc, 4),
                                              "rho_conf_bg_ci": [round(lo, 3), round(hi, 3)]}
            print(f"  {real:5s}~{s:16s} mIoU {rm:+.2f} (p {pm:.3f})  omitted {ro:+.2f} (p {po:.3f})  "
                  f"conf-bg {rc:+.2f} (p {pc:.3f}) CI [{lo:+.2f}, {hi:+.2f}]")
    P = report["pairs"].values()
    report["summary"] = {k: int(sum(v[f"rho_{k}"] > 0 and v[f"p_{k}"] < 0.05 for v in P)) for k in ("miou", "omitted", "conf_bg")}
    report["summary"]["conf_bg_ci_excludes_0_pos"] = int(sum(v["rho_conf_bg_ci"][0] > 0 for v in P))
    report["summary"]["conf_bg_ci_excludes_0_neg"] = int(sum(v["rho_conf_bg_ci"][1] < 0 for v in P))
    print("significant positive agreement (p<0.05) out of 10 pairs:", report["summary"])
    json.dump(report, open(os.path.join(d, "e13_rank.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
