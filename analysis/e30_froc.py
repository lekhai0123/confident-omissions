"""E30: free-response curves (recall of confident omissions into background vs. strict false alarms per image) for
the main methods on ACDC (leave-one-condition-out), BDD100K and WildDash2 (trained on ACDC), exactly the E18 / E23
protocol (shared proposals, best-of-four variant for raw scores, matched false alarms on the test set).

usage: python e30_froc.py <v11> <v13> <v15> <out stem>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e18_comparison as E  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, feature_sets, load  # noqa: E402

GRID = np.round(np.arange(0.05, 2.0001, 0.05), 2)
SHOW = {  # label in the figure -> key in the score dict; colours follow the validated categorical order
    "Ours": ("Ours", "#2a78d6", "-"),
    "MetaSeg-style meta-classifier": ("MetaSeg-style", "#eb6834", "--"),
    "Rule": ("Rule", "#1baf7a", "--"),
    "Depth-based fg/bg score": ("Maag & Rottmann fg/bg on shared proposals", "#eda100", ":"),
    "Ensemble disagreement": ("Ensemble disagreement (cross-model JS)", "#6250d6", ":"),
    "RbA score": ("RbA score (Nayal et al. 2023)", "#e87ba4", ":"),
}


def raw(T, keys):
    m = {}
    for name, stem in E.RAW.items():
        b = E.best_raw(T, stem, keys)
        if b:
            m[name] = b[1]
    m["Rule"] = T.rule.to_numpy()
    return m


def main():
    v11, v13, v15, out = sys.argv[1:5]
    P, O = load(v11)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta_unc = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    Dd = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"])].copy()
    T = Dd[Dd.condition.isin(ADVERSE)]
    kdf = O[O.condition.isin(ADVERSE) & O.co_stuff]
    keys = list(zip(kdf.model, kdf.stem, kdf.instance_id))
    m = raw(T, keys)
    m["MetaSeg-style"] = E.loco(Dd, meta_unc)[T.index].to_numpy()
    m["Ours"] = E.loco(Dd, cheap)[T.index].to_numpy()
    data = {"ACDC": (T, keys, m)}
    m_ours, m_meta = E.gbm(Dd, cheap), E.gbm(Dd, meta_unc)
    for name, d in (("BDD100K", v13), ("WildDash2", v15)):
        PB, OB = load(d)
        TB = PB[PB.det_src.isin(["rtdetr", "rtdetr_gamma"])].copy()
        kb = OB[OB.co_stuff]
        kk = list(zip(kb.model, kb.stem, kb.instance_id))
        mb = raw(TB, kk)
        mb["MetaSeg-style"] = m_meta.predict_proba(TB[meta_unc].fillna(-1).to_numpy(np.float32))[:, 1]
        mb["Ours"] = m_ours.predict_proba(TB[cheap].fillna(-1).to_numpy(np.float32))[:, 1]
        data[name] = (TB, kk, mb)

    curves = {}
    for ds, (T, keys, m) in data.items():
        curves[ds] = {}
        for lab, (key, _, _) in SHOW.items():
            if key in m:
                curves[ds][lab] = [round(float(E.found_at(T, m[key], keys, fa).mean()), 4) for fa in GRID]
        print(ds, {k: (v[9], v[19], v[39]) for k, v in curves[ds].items()})   # 0.5 / 1.0 / 2.0
    with open(out + ".json", "w") as fh:
        json.dump({"grid": GRID.tolist(), "curves": curves}, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6,
                         "xtick.color": "#55554f", "ytick.color": "#55554f"})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, (ds, cs) in zip(axes, curves.items()):
        for lab, (_, col, ls) in SHOW.items():
            if lab in cs:
                ax.plot([0] + GRID.tolist(), [0] + [100 * v for v in cs[lab]], color=col, ls=ls,
                        lw=2.2 if lab == "Ours" else 1.4, label=lab)
        ax.set_title(ds, fontsize=8)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 85)
        ax.set_xlabel("strict false alarms per image")
        ax.grid(color="#e6e5e0", linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("recall (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, frameon=False, fontsize=6.5, handlelength=2.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)


if __name__ == "__main__":
    main()
