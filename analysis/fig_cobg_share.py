"""Figure: share of omissions that are confident omissions into background, synthetic vs real (paper Fig. 2).

Per (segmenter, condition): x = omitted share of thing objects (%), y = confident-into-background share of the
omitted objects (%). Sources: v10a (Cityscapes clean + 16 synthetic corruptions, 443 images), v11 (ACDC adverse
conditions and their reference frames), v13 (BDD100K), v15 (WildDash2 random subset). Marker shape and colour both
encode the source (colour is never the only cue).

usage: python fig_cobg_share.py <v10a> <v11> <v13> <v15> <out stem>
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
MODELS = ["segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"]
LABEL = {"segformer-b0": "SegFormer-B0", "segformer-b2": "SegFormer-B2", "segformer-b5": "SegFormer-B5",
         "mask2former-swin-small": "Mask2Former-S"}
# validated categorical slots (dataviz reference palette, light mode); shapes give a second cue
STYLE = {"Cityscapes-C (synthetic)": ("#2a78d6", "o", 22), "ACDC adverse (real)": ("#eb6834", "^", 46),
         "ACDC reference frames": ("#1baf7a", "s", 34), "BDD100K (real)": ("#6250d6", "D", 40),
         "WildDash2 (real)": ("#eda100", "P", 60), "Cityscapes clean": ("none", "o", 70)}
ADV = ["fog", "night", "rain", "snow"]


def rates(path):
    o = pd.read_csv(path, usecols=["condition", "omitted", "confident_omission", "predicted_as"])
    o["cob"] = o.confident_omission & o.predicted_as.isin(STUFF)
    g = o.groupby("condition").agg(n=("omitted", "size"), om=("omitted", "sum"), cob=("cob", "sum"))
    return pd.DataFrame({"x": 100 * g.om / g.n, "y": 100 * g.cob / g.om.clip(lower=1)})


def group(src, cond):
    if src == "v10a":
        return "Cityscapes clean" if cond == "clean" else "Cityscapes-C (synthetic)"
    if src == "v11":
        return "ACDC adverse (real)" if cond in ADV else ("ACDC reference frames" if cond.endswith("_ref") else None)
    return "BDD100K (real)" if src == "v13" else "WildDash2 (real)"


def main():
    dirs = dict(zip(("v10a", "v11", "v13", "v15"), sys.argv[1:5]))
    out = sys.argv[5]
    rows = []
    for m in MODELS:
        for src, d in dirs.items():
            r = rates(os.path.join(d, f"objects_{m}.csv"))
            if src in ("v13", "v15"):    # one point per dataset (pool its splits)
                o = pd.read_csv(os.path.join(d, f"objects_{m}.csv"), usecols=["omitted", "confident_omission", "predicted_as"])
                cob = (o.confident_omission & o.predicted_as.isin(STUFF)).sum()
                r = pd.DataFrame({"x": [100 * o.omitted.mean()], "y": [100 * cob / max(1, o.omitted.sum())]}, index=["all"])
            for cond, v in r.iterrows():
                g = group(src, cond)
                if g:
                    rows.append({"model": m, "group": g, "condition": cond, "x": v.x, "y": v.y})
    D = pd.DataFrame(rows)
    D.to_csv(out + ".csv", index=False)

    # drawn at the manuscript text width (390 pt = 5.4 in) so that LaTeX does not shrink the fonts
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6,
                         "xtick.color": "#55554f", "ytick.color": "#55554f", "xtick.labelsize": 8, "ytick.labelsize": 8})
    YMAX = 20.0      # points above are shown at the top edge with their value (all are synthetic noise corruptions)
    fig, axes2 = plt.subplots(2, 2, figsize=(5.4, 4.6), sharey=True)
    axes = axes2.ravel()
    for ax, m in zip(axes, MODELS):
        d = D[D.model == m]
        for g, (c, mk, s) in STYLE.items():
            e = d[(d.group == g) & (d.y <= YMAX)]
            if g == "Cityscapes clean":
                ax.scatter(e.x, e.y, s=s, facecolors="none", edgecolors="#1f1f1c", linewidths=1.2, zorder=4, label=g)
            else:
                ax.scatter(e.x, e.y, s=s, c=c, marker=mk, edgecolors="#fcfcfb", linewidths=0.8, zorder=3, label=g)
        over = d[d.y > YMAX]
        assert set(over.group) <= {"Cityscapes-C (synthetic)"}, over
        for j, (_, r) in enumerate(over.sort_values("x").iterrows()):
            ax.scatter([r.x], [YMAX + 0.4], marker=r"$\uparrow$", s=60, c=STYLE["Cityscapes-C (synthetic)"][0],
                       clip_on=False, zorder=5)
            dx, ha = ((-3, "right") if j % 2 == 0 else (3, "left")) if len(over) > 1 else (0, "center")
            ax.annotate(f"{r.y:.0f}%", (r.x, YMAX + 0.4), xytext=(dx, 3), textcoords="offset points", ha=ha,
                        fontsize=7.5, color="#55554f", annotation_clip=False)
        ax.set_ylim(-1, YMAX + 1.5)
        for _, r in d[(d.group == "ACDC adverse (real)") & (d.condition == "night")].iterrows():
            ax.annotate(r.condition, (r.x, r.y), xytext=(4, 2), textcoords="offset points", fontsize=7.5, color="#55554f")
        ax.set_title(LABEL[m], fontsize=9)
        ax.grid(color="#e6e5e0", linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes2[1]:
        ax.set_xlabel("omitted objects (%)")
    for ax in axes2[:, 0]:
        ax.set_ylabel("confident omissions into\nbackground (% of omitted)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=1.2)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    print(D.groupby(["model", "group"]).y.agg(["min", "median", "max", "size"]).round(1).to_string())


if __name__ == "__main__":
    main()
