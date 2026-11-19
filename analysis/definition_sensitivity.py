"""Sensitivity of contribution 2 to the definition of a confident omission into background.
Variants: omitted = frac_true <= {0.02, 0.05, 0.10}; confident = mean confidence over the object >= {0.8, 0.85, 0.9, 0.95}
(plus the original rule: >= 50% of pixels wrong with conf >= 0.9). Target: predicted_as is a stuff class.
Reports confident share of omissions per model x group and checks three qualitative claims."""
import os, sys, json
import pandas as pd, numpy as np
STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
MODELS = ["segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"]
def grp(c):
    if c == "clean": return "clean"
    if c in ("fog", "night", "rain", "snow"): return "adverse"
    if c.endswith("_ref"): return "ref"
    return "synthetic"
res = sys.argv[1]
rows = []
for m in MODELS:
    o = pd.concat([pd.read_csv(os.path.join(res, d, f"objects_{m}.csv"),
                               usecols=["condition", "frac_true", "mean_conf", "frac_conf_err", "predicted_as"])
                   for d in ("kaggle-v6", "kaggle-v7-acdc")])
    o["g"] = o.condition.map(grp)
    stuff = o.predicted_as.isin(STUFF)
    for om_t in (0.02, 0.05, 0.10):
        om = o.frac_true <= om_t
        for conf_name, conf in [("orig", o.frac_conf_err >= 0.5)] + [(f"mean>={t}", o.mean_conf >= t) for t in (0.8, 0.85, 0.9, 0.95)]:
            co = om & conf & stuff
            for g in ("clean", "synthetic", "ref", "adverse"):
                sel = (o.g == g).to_numpy()
                rows.append({"model": m, "omit_t": om_t, "conf": conf_name, "group": g,
                             "share": co[sel].sum() / max(1, om[sel].sum()), "n_co": int(co[sel].sum())})
R = pd.DataFrame(rows)
W = R.pivot_table(index=["omit_t", "conf", "model"], columns="group", values="share").reset_index()
checks = []
for (om_t, conf), g in W.groupby(["omit_t", "conf"]):
    g = g.set_index("model")
    strong = ["segformer-b5", "mask2former-swin-small"]
    c1 = all(abs(g.loc[m, "adverse"] - g.loc[m, "ref"]) <= max(0.35 * g.loc[m, "ref"], 0.01) for m in strong + ["segformer-b2"])
    c2 = all(g.loc[m, "synthetic"] > 2 * g.loc[m, "adverse"] for m in ["segformer-b0", "segformer-b2"])
    c3 = g.loc["mask2former-swin-small", "ref"] > 1.5 * max(g.loc[m, "ref"] for m in ["segformer-b2", "segformer-b5"])
    checks.append({"omit_t": om_t, "conf": conf, "strong_models_adverse≈ref": c1, "weak_models_synthetic>2x_real": c2,
                   "M2F_ref>1.5x_SegFormer": c3,
                   "b5_ref/adv": f"{100*g.loc['segformer-b5','ref']:.1f}/{100*g.loc['segformer-b5','adverse']:.1f}",
                   "b0_syn/adv": f"{100*g.loc['segformer-b0','synthetic']:.1f}/{100*g.loc['segformer-b0','adverse']:.1f}",
                   "m2f_ref": f"{100*g.loc['mask2former-swin-small','ref']:.1f}"})
C = pd.DataFrame(checks)
print(C.to_string(index=False))
print("\nall claims hold in", int((C.iloc[:, 2:5].all(axis=1)).sum()), "of", len(C), "variants")
R.to_csv(os.path.join(res, "paper", "definition_sensitivity.csv"), index=False)
C.to_csv(os.path.join(res, "paper", "definition_sensitivity_checks.csv"), index=False)
