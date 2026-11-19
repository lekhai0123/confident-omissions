"""E24: our corruption re-implementation vs the standard `imagecorruptions` package (v18), protocol of WRITING_KIT section 0.

(a) Agreement per model x condition between the two implementations, on the SAME Cityscapes validation images:
    mIoU and confident-into-background share of omissions (also the pooled confident share and omitted share).
    Ours: v10a (SegFormer-B0/B2/B5, Mask2Former-S) and v19a (DeepLabV3+ R101); library: v18.
(b) E13 repeated for the models that have both versions (4 here: the CNN has no ACDC run yet): Spearman rho across
    models between each real ACDC condition and its synthetic counterpart, once with our corruptions and once with
    the library's. With n = 4 no rho can reach p < 0.05 (exact minimum two-sided p = 1/12), so (b) is descriptive:
    the criterion is whether the pattern "mIoU agrees, confident share does not" is the same under both versions.
`dark` is not in the library, so the night~dark pairs exist only for our version (night~brightness-5 is kept).

usage: python e24_imagecorruptions.py <v10a dir> <v19a dir> <v18 dir> <e13 merged dir> <v9 dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
SYN = ["gaussian_noise-3", "shot_noise-3", "defocus_blur-3", "motion_blur-3", "fog-3", "fog-5", "brightness-5", "contrast-5"]
PAIRS = {"fog": ["fog-3", "fog-5"], "night": ["brightness-5"], "rain": ["motion_blur-3", "defocus_blur-3"],
         "snow": ["gaussian_noise-3", "shot_noise-3", "contrast-5"]}
MODELS4 = ["segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"]
COLS = ["stem", "condition", "omitted", "confident_omission", "predicted_as"]


def obj_stats(path, conds, stems=None):
    o = pd.read_csv(path, usecols=COLS)
    o = o[o.condition.isin(conds)]
    if stems is not None:
        o = o[o.stem.isin(stems)]
    o["co_bg"] = o.confident_omission & o.predicted_as.isin(STUFF)
    g = o.groupby("condition").agg(n=("omitted", "size"), om=("omitted", "sum"), co=("confident_omission", "sum"), cob=("co_bg", "sum"))
    return pd.DataFrame({"omitted": g.om / g.n, "conf_share": g.co / g.om.clip(lower=1), "co_bg_share": g.cob / g.om.clip(lower=1),
                         "n_co_bg": g.cob})


def miou(summary, tag, conds):
    return {c: summary["models"][tag]["miou_by_condition"][c]["argmax"]["miou"] for c in conds}


def main():
    v10a, v19a, v18, merged, v9, out = sys.argv[1:7]
    S10 = json.load(open(os.path.join(v10a, "summary.json"), encoding="utf-8"))
    S19 = json.load(open(os.path.join(v19a, "summary.json"), encoding="utf-8"))
    S18 = json.load(open(os.path.join(v18, "summary.json"), encoding="utf-8"))
    conds = ["clean"] + SYN
    lib_stems = set(pd.read_csv(os.path.join(v18, "objects_segformer-b0.csv"), usecols=["stem"]).stem)
    own_src = {t: (v10a, S10) for t in MODELS4} | {"deeplabv3plus-r101": (v19a, S19)}

    # (a) per model x condition agreement on identical images
    cells = []
    for t, (d, S) in own_src.items():
        own = obj_stats(os.path.join(d, f"objects_{t}.csv"), conds, lib_stems)
        lib = obj_stats(os.path.join(v18, f"objects_{t}.csv"), conds, lib_stems)
        mo, ml = miou(S, t, conds), miou(S18, t, conds)
        for c in conds:
            cells.append({"model": t, "condition": c, "miou_own": mo[c], "miou_lib": ml[c],
                          **{f"{k}_own": float(own.loc[c, k]) for k in own.columns}, **{f"{k}_lib": float(lib.loc[c, k]) for k in lib.columns}})
    C = pd.DataFrame(cells)
    Cs = C[C.condition != "clean"]
    agree = {}
    for k in ("miou", "omitted", "conf_share", "co_bg_share"):
        a, b = Cs[f"{k}_own"].to_numpy(), Cs[f"{k}_lib"].to_numpy()
        agree[k] = {"spearman": round(float(spearmanr(a, b)[0]), 3), "pearson": round(float(pearsonr(a, b)[0]), 3),
                    "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 4), "n_cells": len(a)}
    clean_check = {t: {"miou_own": round(100 * C[(C.model == t) & (C.condition == "clean")].miou_own.iloc[0], 2),
                       "miou_lib": round(100 * C[(C.model == t) & (C.condition == "clean")].miou_lib.iloc[0], 2)} for t in own_src}

    # (b) E13 with the 4 models that have real ACDC numbers and both synthetic versions
    Sm = json.load(open(os.path.join(merged, "summary.json"), encoding="utf-8"))
    common = pd.read_csv(os.path.join(v9, "images_segformer-b0.csv"), usecols=["stem", "condition"]).drop_duplicates()
    real = {}
    for t in MODELS4:
        o = pd.read_csv(os.path.join(merged, f"objects_{t}.csv"), usecols=COLS)
        o = o[o.condition.isin(PAIRS)].merge(common, on=["stem", "condition"])
        o["co_bg"] = o.confident_omission & o.predicted_as.isin(STUFF)
        g = o.groupby("condition").agg(n=("omitted", "size"), om=("omitted", "sum"), cob=("co_bg", "sum"))
        real[t] = {"omitted": g.om / g.n, "co_bg_share": g.cob / g.om.clip(lower=1), "miou": miou(Sm, t, list(PAIRS))}
    e13 = {}
    for r, syns in PAIRS.items():
        for s in syns:
            row = {}
            for ver in ("own", "lib"):
                sub = C[(C.condition == s) & C.model.isin(MODELS4)].set_index("model").loc[MODELS4]
                for k in ("miou", "omitted", "co_bg_share"):
                    rv = np.array([real[t][k][r] for t in MODELS4])
                    row[f"rho_{k}_{ver}"] = round(float(spearmanr(rv, sub[f"{k}_{ver}"].to_numpy())[0]), 3)
            e13[f"{r}~{s}"] = row
    summ = {f"{k}_{ver}_pos_pairs": int(sum(v[f"rho_{k}_{ver}"] > 0.5 for v in e13.values()))
            for k in ("miou", "omitted", "co_bg_share") for ver in ("own", "lib")}

    res = {"a_agreement_corrupted_cells": agree, "clean_check": clean_check, "cells": C.round(4).to_dict("records"),
           "b_e13_4models": e13, "b_summary_rho_gt_0.5": summ, "n_pairs": len(e13),
           "note": "own = v10a/v19a, lib = v18 (imagecorruptions), identical Cityscapes val images; n=4 in (b) is descriptive"}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("clean check:", clean_check)
    print("(a) agreement over", agree["miou"]["n_cells"], "corrupted model x condition cells:")
    for k, v in agree.items():
        print(f"   {k:12s} spearman {v['spearman']:+.3f} pearson {v['pearson']:+.3f} mean|diff| {v['mean_abs_diff']:.4f}")
    print(Cs[["model", "condition", "miou_own", "miou_lib", "co_bg_share_own", "co_bg_share_lib", "n_co_bg_own", "n_co_bg_lib"]].round(3).to_string(index=False))
    print("(b) E13, 4 models, rho own | lib:")
    for p, v in e13.items():
        print(f"   {p:22s} mIoU {v['rho_miou_own']:+.2f} | {v['rho_miou_lib']:+.2f}   omitted {v['rho_omitted_own']:+.2f} | {v['rho_omitted_lib']:+.2f}"
              f"   co-bg {v['rho_co_bg_share_own']:+.2f} | {v['rho_co_bg_share_lib']:+.2f}")
    print("pairs with rho > 0.5:", summ)


if __name__ == "__main__":
    main()
