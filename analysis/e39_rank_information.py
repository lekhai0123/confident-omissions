"""E39: does the confident-share ranking add information beyond mIoU and the omitted share? (review2, C1)

Same 15 configurations and common images as Table 2 / E32 / E35. For each real ACDC condition (fog, night, rain, snow)
and pooled over the four:
  - Spearman rho between the confident share and mIoU, and between the confident share and the omitted share
    (= one minus object-level recall of thing objects), across segmenters;
  - R^2 of a linear regression of the confident-share RANK on the ranks of mIoU and the omitted share (how much of the
    ranking is predictable from the two usual metrics), with a permutation p-value;
  - the same for 12 independent checkpoints.
Low correlations / low R^2 = the confident share ranks the segmenters differently from the usual metrics, so a
practitioner who selects by mIoU or object recall does not also select for few confident omissions.

Extended mode (E40 data, 19 configurations / 16 checkpoints): mIoU of every segmenter is pooled from per-image
confusion counts on the same common frames as the confident share (re-run of the 15 original segmenters, see E40),
and the objects of the four further checkpoints come from their own run.

usage: python e39_rank_information.py <15-model merged dir> <common dir> <out json> [<new acdc dir> <confusion acdc dir>]
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
REAL = ["fog", "night", "rain", "snow"]
VARIANTS = {"segformer-b0-512-1024", "segformer-b0-640-1280", "segformer-b0-768-768"}


def r2_rank(y, X):
    y = rankdata(y)
    Xr = np.column_stack([np.ones(len(y))] + [rankdata(x) for x in X])
    beta, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    res = y - Xr @ beta
    return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()


def conf_miou(df, common):
    """mIoU per real condition pooled from per-image confusion counts on the common frames"""
    df = df.copy()
    df["stem"] = df.stem.astype(str)
    df = df[df.condition.isin(REAL)].merge(common, on=["stem", "condition"])
    out = {}
    for c, g in df.groupby("condition"):
        tp, gt, pr = (g[[f"{k}_{i}" for i in range(19)]].sum().to_numpy() for k in ("tp", "gt", "pr"))
        iou = tp / np.maximum(gt + pr - tp, 1)
        out[c] = float(np.mean(iou[gt > 0]))
    return out


def main():
    d, common_dir, out = sys.argv[1:4]
    extended = len(sys.argv) >= 6
    summ = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    common = pd.read_csv(os.path.join(common_dir, "images_segformer-b0.csv"), usecols=["stem", "condition"]).drop_duplicates()
    common["stem"] = common.stem.astype(str)
    files = sorted(glob.glob(os.path.join(d, "objects_*.csv")))
    MI = {}
    if extended:
        new_dir, conf_dir = sys.argv[4:6]
        files += sorted(glob.glob(os.path.join(new_dir, "objects_*.csv")))
        for f in glob.glob(os.path.join(conf_dir, "confusion_*.csv")):
            MI[os.path.basename(f)[len("confusion_"):-4]] = conf_miou(pd.read_csv(f), common)
        for f in glob.glob(os.path.join(new_dir, "images_*.csv")):
            MI[os.path.basename(f)[len("images_"):-4]] = conf_miou(pd.read_csv(f), common)
    stats = {}
    for f in files:
        tag = os.path.basename(f)[len("objects_"):-4]
        o = pd.read_csv(f, usecols=["stem", "condition", "omitted", "confident_omission", "predicted_as"])
        o["stem"] = o.stem.astype(str)
        o = o[o.condition.isin(REAL)].merge(common, on=["stem", "condition"])
        o["co_bg"] = o.confident_omission & o.predicted_as.isin(STUFF)
        st = {}
        for c in REAL + ["all"]:
            q = o if c == "all" else o[o.condition == c]
            if extended:
                mi = np.mean([MI[tag][r] for r in REAL]) if c == "all" else MI[tag][c]
            else:
                mi = (np.mean([summ["models"][tag]["miou_by_condition"][r]["argmax"]["miou"] for r in REAL]) if c == "all"
                      else summ["models"][tag]["miou_by_condition"][c]["argmax"]["miou"])
            st[c] = {"miou": mi, "omitted": q.omitted.mean(), "cs": q.co_bg.sum() / max(1, q.omitted.sum()),
                     "cs_obj": q.co_bg.mean()}
        stats[tag] = st
    rng = np.random.default_rng(0)
    res = {}
    n_all, n_ind = ("19", "16") if extended else ("15", "12")
    print("segmenters:", len(stats))
    for set_name, tags in ((n_all, sorted(stats)), (n_ind, sorted(t for t in stats if t not in VARIANTS))):
        res[set_name] = {}
        for c in REAL + ["all"]:
            mi = np.array([stats[t][c]["miou"] for t in tags])
            om = np.array([stats[t][c]["omitted"] for t in tags])
            for meas in ("cs", "cs_obj"):
              cs = np.array([stats[t][c][meas] for t in tags])
              r2 = r2_rank(cs, [mi, om])
              perm = np.array([r2_rank(rng.permutation(cs), [mi, om]) for _ in range(5000)])
              res[set_name][f"{c}|{meas}"] = {"rho_cs_miou": round(float(spearmanr(cs, mi)[0]), 3), "p_cs_miou": round(float(spearmanr(cs, mi)[1]), 3),
                                  "rho_cs_omitted": round(float(spearmanr(cs, om)[0]), 3), "p_cs_omitted": round(float(spearmanr(cs, om)[1]), 3),
                                  "rho_miou_omitted": round(float(spearmanr(mi, om)[0]), 3),
                                  "R2_cs_rank_on_miou_omitted": round(float(r2), 3), "perm_p": round(float((perm >= r2).mean()), 4)}
              v = res[set_name][f"{c}|{meas}"]
              print(f"{set_name:2s} {c:5s} {meas:6s} rho(cs,mIoU) {v['rho_cs_miou']:+.2f} (p {v['p_cs_miou']:.2f})  rho(cs,omitted) {v['rho_cs_omitted']:+.2f} "
                    f"(p {v['p_cs_omitted']:.2f})  rho(mIoU,omitted) {v['rho_miou_omitted']:+.2f}  R2 {v['R2_cs_rank_on_miou_omitted']:.2f} (perm p {v['perm_p']:.3f})")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
