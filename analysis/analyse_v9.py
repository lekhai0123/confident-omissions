"""v9 analysis: architecture (per-pixel vs mask classification) and unnormalised evidence (energy, RbA).

usage: python analyse_v9.py <v9 results dir>
Writes analysis_v9.json, v9_per_condition.csv, fig_v9_confident_share.png/.pdf into the results dir.

1. Confident share of omissions per model and group (Cityscapes clean, Cityscapes-C, ACDC ref, ACDC adverse),
   bootstrap CIs over images; robustness to the confidence threshold (mean confidence of omitted objects).
2. Evidence signals: recall of confident omissions at the 2% / 10% per-image budget for entropy, margin, energy,
   RbA, cross-model JS, Mahalanobis; flag rate on well-segmented objects; object-level AUROC CO vs good.
   "Evidence-aware" confident share: CO that are also missed by energy / RbA at the 10% budget.
3. Synthetic vs real across 8 models: Spearman rank agreement of mIoU and of the confident share for matched pairs.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_vs_real import ratio_ci  # noqa: E402

ACDC = ["fog", "night", "rain", "snow"]
ACDC_ALL = ACDC + [c + "_ref" for c in ACDC]
STUFF_NAMES = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light",
               "traffic sign"}
SIGNALS = ["ent", "margin", "energy", "rba", "xmodel_js", "maha_pred", "fused"]
PAIRS = {"fog": ["fog-3", "fog-5"], "night": ["dark-3", "dark-5", "brightness-5"], "rain": ["motion_blur-3", "defocus_blur-3"],
         "snow": ["gaussian_noise-3", "shot_noise-3", "contrast-5"]}


def group(cond: str) -> str:
    if cond in ACDC:
        return "ACDC adverse"
    if cond in ACDC_ALL:
        return "ACDC ref"
    return "Cityscapes clean" if cond == "clean" else "Cityscapes-C"


def head_type(tag: str) -> str:
    return "per-pixel" if tag.startswith("segformer") else "mask"


def share(g: pd.DataFrame, num: str, den: str | None) -> tuple[float, list]:
    n = g[num].to_numpy()
    d = g[den].to_numpy() if den else np.ones(len(g))
    est, ci = ratio_ci(n, d, g.stem.to_numpy(), n_boot=500)
    return round(est, 4), [round(ci[0], 4), round(ci[1], 4)]


def main():
    d = sys.argv[1]
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as fh:
        summ = json.load(fh)
    report = {"models": {}, "pairs": {}}
    per_rows = []
    for f in sorted(glob.glob(os.path.join(d, "objects_*.csv"))):
        tag = os.path.basename(f)[len("objects_"):-4]
        o = pd.read_csv(f)
        o["group"] = o.condition.map(group)
        o["co_energy_silent"] = o.confident_omission & (o.energy_hit10 < 0.5)
        o["co_rba_silent"] = o.confident_omission & (o.rba_hit10 < 0.5)
        o["co_to_stuff"] = o.confident_omission & o.predicted_as.isin(STUFF_NAMES)      # object vanishes into background
        o["co_thing2thing"] = o.confident_omission & ~o.predicted_as.isin(STUFF_NAMES)  # confused with another thing
        miou = {c: v["argmax"]["miou"] for c, v in summ["models"][tag]["miou_by_condition"].items()}
        R = {"head": head_type(tag), "groups": {}, "signals": {}, "threshold_robustness": {}}
        for grp, g in o.groupby("group"):
            r = {"n_objects": int(len(g)), "n_omitted": int(g.omitted.sum()), "n_co": int(g.confident_omission.sum())}
            for k, num, den in (("omitted_share", "omitted", None), ("confident_share", "confident_omission", "omitted"),
                                ("confident_share_to_stuff", "co_to_stuff", "omitted"), ("confident_share_thing2thing", "co_thing2thing", "omitted"),
                                ("energy_silent_share", "co_energy_silent", "omitted"), ("rba_silent_share", "co_rba_silent", "omitted")):
                r[k], r[k + "_ci"] = share(g, num, den)
            om = g[g.omitted]
            r["conf_ge"] = {t: round(float((om.mean_conf >= t).mean()), 3) for t in (0.7, 0.8, 0.9, 0.95)}
            R["groups"][grp] = r
        for c, g in o.groupby("condition"):
            per_rows.append({"model": tag, "head": R["head"], "condition": c, "group": group(c), "miou": miou.get(c, np.nan),
                             "omitted_share": g.omitted.mean(), "n_omitted": int(g.omitted.sum()),
                             "n_co": int(g.confident_omission.sum()),
                             "confident_share": g.confident_omission.sum() / max(1, g.omitted.sum()),
                             "confident_share_to_stuff": g.co_to_stuff.sum() / max(1, g.omitted.sum())})
        shifted = o[o.group != "Cityscapes clean"]
        co, good = shifted[shifted.co_to_stuff], shifted[shifted.frac_true >= 0.5]   # signals: omissions into background
        for s in SIGNALS:
            if f"{s}_hit10" not in o:
                continue
            y = np.r_[np.ones(len(co)), np.zeros(len(good))]
            x = np.r_[co[f"{s}_mean"].to_numpy(), good[f"{s}_mean"].to_numpy()]
            R["signals"][s] = {"recall_co_hit10": round(float((co[f"{s}_hit10"] >= 0.5).mean()), 3),
                               "recall_co_hit02": round(float((co[f"{s}_hit02"] >= 0.5).mean()), 3),
                               "flag_good_hit10": round(float((good[f"{s}_hit10"] >= 0.5).mean()), 3),
                               "auroc_co_vs_good": round(float(roc_auc_score(y, np.nan_to_num(x))), 3) if len(co) else None}
        report["models"][tag] = R
        print(f"\n== {tag} ({R['head']})")
        for grp in ("Cityscapes clean", "Cityscapes-C", "ACDC ref", "ACDC adverse"):
            if grp in R["groups"]:
                r = R["groups"][grp]
                print(f"  {grp:17s} conf-share {r['confident_share']:.3f} {r['confident_share_ci']} (to-stuff {r['confident_share_to_stuff']:.3f}, thing2thing {r['confident_share_thing2thing']:.3f})  energy-silent {r['energy_silent_share']:.3f}"
                      f"  rba-silent {r['rba_silent_share']:.3f}  omitted {r['omitted_share']:.3f}  n_co {r['n_co']}")
        print("  signals (recall@10%, flag good, AUROC):", {s: (v["recall_co_hit10"], v["flag_good_hit10"], v["auroc_co_vs_good"])
                                                       for s, v in R["signals"].items()})
    per = pd.DataFrame(per_rows)
    per.to_csv(os.path.join(d, "v9_per_condition.csv"), index=False)
    print("\n== synthetic vs real rank agreement across models (Spearman)")
    for real, syns in PAIRS.items():
        for s in syns:
            a = per[per.condition == s].set_index("model")
            b = per[per.condition == real].set_index("model")
            idx = a.index.intersection(b.index)
            if len(idx) < 4:
                continue
            rm = spearmanr(a.loc[idx, "miou"], b.loc[idx, "miou"])[0]
            rc = spearmanr(a.loc[idx, "confident_share_to_stuff"], b.loc[idx, "confident_share_to_stuff"])[0]
            report["pairs"][f"{real}~{s}"] = {"rho_miou": round(float(rm), 3), "rho_confident_share": round(float(rc), 3), "n_models": len(idx)}
            print(f"  {real:6s} ~ {s:16s} rho mIoU {rm:+.2f}   rho confident share {rc:+.2f}   (n={len(idx)})")
    with open(os.path.join(d, "analysis_v9.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    groups = ["Cityscapes clean", "Cityscapes-C", "ACDC ref", "ACDC adverse"]
    tags = list(report["models"])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    w = 0.2
    colors = ["#8a8a85", "#2a78d6", "#1baf7a", "#eb6834"]
    for j, grp in enumerate(groups):
        vals = [100 * report["models"][t]["groups"].get(grp, {}).get("confident_share", np.nan) for t in tags]
        cis = [report["models"][t]["groups"].get(grp, {}).get("confident_share_ci", [np.nan, np.nan]) for t in tags]
        err = np.array([[v - 100 * c[0], 100 * c[1] - v] for v, c in zip(vals, cis)]).T
        ax.bar(np.arange(len(tags)) + (j - 1.5) * w, vals, w * 0.9, yerr=err, color=colors[j], label=grp,
               error_kw={"elinewidth": 0.8, "ecolor": "#5c5c58"})
    ax.set_xticks(np.arange(len(tags)))
    ax.set_xticklabels([t.replace("segformer-", "SegF-").replace("mask2former-swin-", "M2F-").replace("maskformer-", "MF-")
                        for t in tags], fontsize=8)
    ax.set_ylabel("confident share of omissions (%)")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"fig_v9_confident_share.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
