"""Table 1 (prevalence) from the same runs as the rest of the paper: Cityscapes clean = v10a (443 validation images
with objects; same images as Fig. 2 and the DeepLabV3+ run), ACDC = v11 (all adverse and reference images, the 470
targets of the verifier experiments), BDD100K = v13, WildDash2 = v15.

mIoU: Cityscapes clean (v10a summary), ACDC adverse = mean over fog / night / rain / snow (v11 summary), BDD100K
validation split (v13), WildDash2 = image-weighted mean over the two public splits (v15).
Omitted: share of thing objects omitted (pooled over the conditions). Confident share: confident omissions into
background per omitted object (pooled).

usage: python table1_prevalence.py <v10a> <v11> <v13> <v15> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
MODELS = ["segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"]
ADV = ["fog", "night", "rain", "snow"]


def obj(d, m, conds=None):
    o = pd.read_csv(os.path.join(d, f"objects_{m}.csv"), usecols=["condition", "omitted", "confident_omission", "predicted_as"])
    if conds is not None:
        o = o[o.condition.isin(conds)]
    cob = (o.confident_omission & o.predicted_as.isin(STUFF)).sum()
    return 100 * o.omitted.mean(), 100 * cob / max(1, o.omitted.sum()), int(cob)


def miou(d, m, conds, weights=None):
    s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))["models"][m]["miou_by_condition"]
    vals = [s[c]["argmax"]["miou"] for c in conds]
    w = weights or [1] * len(vals)
    return 100 * sum(v * x for v, x in zip(vals, w)) / sum(w)


def main():
    v10a, v11, v13, v15, out = sys.argv[1:6]
    rows = {}
    for m in MODELS:
        wd_n = pd.read_csv(os.path.join(v15, f"images_{m}.csv"), usecols=["condition"]).condition.value_counts()
        r = {"miou_cs": miou(v10a, m, ["clean"]), "miou_adv": miou(v11, m, ADV), "miou_bdd": miou(v13, m, ["bdd_val"]),
             "miou_wd": miou(v15, m, ["wd_train", "wd_val"], [int(wd_n["wd_train"]), int(wd_n["wd_val"])])}
        r["om_adv"], r["cs_adv"], r["n_adv"] = obj(v11, m, ADV)
        _, r["cs_ref"], _ = obj(v11, m, [c + "_ref" for c in ADV])
        _, r["cs_cs"], r["n_cs"] = obj(v10a, m, ["clean"])
        r["om_bdd"], r["cs_bdd"], r["n_bdd"] = obj(v13, m)
        r["om_wd"], r["cs_wd"], r["n_wd"] = obj(v15, m)
        rows[m] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}
        print(m, rows[m])
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
