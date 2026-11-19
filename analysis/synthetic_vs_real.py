"""Synthetic corruptions (v6, Cityscapes-C) vs real adverse conditions (v7, ACDC): are omissions confident?

usage: python synthetic_vs_real.py <v6 results dir> <v7 results dir> <out dir>

Per (dataset, condition, model): mIoU, omitted share of thing objects, confident-omission (CO) share, the
confident share of omissions (CO / omitted) and the "silent" share of omissions (entropy does not flag the
object at a 10% per-image budget). 95% CIs: bootstrap over source images. Pooled contrasts per model:
synthetic (all corrupted conditions) vs ACDC adverse vs ACDC normal-condition references.
Writes synthetic_vs_real.json / .csv and fig_confident_share.png / .pdf.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

MODELS = ["segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"]
LABEL = {"segformer-b0": "SegFormer-B0", "segformer-b2": "SegFormer-B2", "segformer-b5": "SegFormer-B5",
         "mask2former-swin-small": "Mask2Former-S"}
ADVERSE = ["fog", "night", "rain", "snow"]
GROUP_STYLE = {"Cityscapes-C (synthetic)": ("#2a78d6", "o"), "ACDC adverse (real)": ("#eb6834", "^"),
               "ACDC normal ref": ("#1baf7a", "s")}


def group_of(dataset: str, cond: str) -> str:
    if dataset == "cs":
        return "Cityscapes clean" if cond == "clean" else "Cityscapes-C (synthetic)"
    return "ACDC adverse (real)" if cond in ADVERSE else "ACDC normal ref"


def ratio_ci(num, den, groups, n_boot=1000, seed=0):
    """Bootstrap (over images) CI of sum(num)/sum(den)."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    sn = np.bincount(inv, weights=num.astype(float), minlength=len(ug))
    sd = np.bincount(inv, weights=den.astype(float), minlength=len(ug))
    vals = []
    for _ in range(n_boot):
        pk = rng.integers(0, len(ug), len(ug))
        d = sd[pk].sum()
        if d:
            vals.append(sn[pk].sum() / d)
    est = sn.sum() / sd.sum() if sd.sum() else float("nan")
    return float(est), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def stats(g: pd.DataFrame) -> dict:
    om = g.omitted.to_numpy()
    co = g.confident_omission.to_numpy()
    silent = om & (g.ent_hit10.to_numpy() < 0.5)
    ones = np.ones(len(g))
    r = {"n_objects": int(len(g)), "n_images": int(g.stem.nunique()), "n_omitted": int(om.sum()), "n_co": int(co.sum())}
    for name, num, den in (("omitted_share", om, ones), ("co_share", co, ones), ("confident_share_of_omissions", co, om),
                           ("silent_share_of_omissions", silent, om)):
        est, ci = ratio_ci(num, den, g.stem.to_numpy())
        r[name], r[name + "_ci"] = round(est, 4), [round(ci[0], 4), round(ci[1], 4)]
    return r


def load(d: str, dataset: str) -> tuple[pd.DataFrame, dict]:
    cols = ["stem", "condition", "omitted", "confident_omission", "ent_hit10", "frac_true"]
    frames = []
    for m in MODELS:
        f = os.path.join(d, f"objects_{m}.csv")
        o = pd.read_csv(f, usecols=cols)
        o["model"], o["dataset"] = m, dataset
        frames.append(o)
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as fh:
        summ = json.load(fh)
    miou = {m: {c: v["argmax"]["miou"] for c, v in summ["models"][m]["miou_by_condition"].items()} for m in MODELS}
    return pd.concat(frames, ignore_index=True), miou


def main():
    d6, d7, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    o6, miou6 = load(d6, "cs")
    o7, miou7 = load(d7, "acdc")
    obj = pd.concat([o6, o7], ignore_index=True)
    obj["group"] = [group_of(a, b) for a, b in zip(obj.dataset, obj.condition)]
    rows = []
    for (ds, m, c), g in obj.groupby(["dataset", "model", "condition"]):
        r = {"dataset": ds, "model": m, "condition": c, "group": group_of(ds, c),
             "miou": round((miou6 if ds == "cs" else miou7)[m].get(c, np.nan), 4), **stats(g)}
        rows.append(r)
    per = pd.DataFrame(rows)
    pooled = {}
    for (m, grp), g in obj.groupby(["model", "group"]):
        pooled.setdefault(m, {})[grp] = stats(g)
    report = {"per_condition": rows, "pooled": pooled}
    per.drop(columns=[c for c in per.columns if c.endswith("_ci")]).to_csv(os.path.join(out, "synthetic_vs_real.csv"), index=False)
    with open(os.path.join(out, "synthetic_vs_real.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    print("pooled: confident share of omissions [CI] | silent share of omissions | omitted share")
    for m in MODELS:
        for grp in ["Cityscapes clean", "Cityscapes-C (synthetic)", "ACDC normal ref", "ACDC adverse (real)"]:
            r = pooled[m][grp]
            print(f"  {m:24s} {grp:26s} conf {r['confident_share_of_omissions']:.3f} {r['confident_share_of_omissions_ci']} "
                  f"silent {r['silent_share_of_omissions']:.3f}  omitted {r['omitted_share']:.3f} (n_om {r['n_omitted']}, n_co {r['n_co']})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b29", "xtick.color": "#5c5c58",
                         "ytick.color": "#5c5c58"})
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for ax, m in zip(axes, MODELS):
        sub = per[(per.model == m) & (per.group != "Cityscapes clean")]
        for grp, (col, mk) in GROUP_STYLE.items():
            s = sub[sub.group == grp]
            ax.scatter(100 * s.omitted_share, 100 * s.confident_share_of_omissions, s=38, c=col, marker=mk,
                       edgecolors="white", linewidths=1.2, label=grp, zorder=3)
        clean = per[(per.model == m) & (per.group == "Cityscapes clean")]
        ax.scatter(100 * clean.omitted_share, 100 * clean.confident_share_of_omissions, s=60, facecolors="none",
                   edgecolors="#2b2b29", marker="o", linewidths=1.2, label="Cityscapes clean", zorder=4)
        for _, r in sub[sub.group == "ACDC adverse (real)"].iterrows():
            ax.annotate(r.condition, (100 * r.omitted_share, 100 * r.confident_share_of_omissions), fontsize=7,
                        color="#5c5c58", xytext=(4, 2), textcoords="offset points")
        ax.set_title(LABEL[m], fontsize=9, color="#2b2b29")
        ax.set_xlabel("omitted thing objects (%)")
        ax.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
    axes[0].set_ylabel("confident share of omissions (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_confident_share.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
