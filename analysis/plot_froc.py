"""Re-plot the FROC figure (Fig. 4) from results/paper/fig_froc.json at the manuscript text width (5.4 in), so that
LaTeX does not shrink the fonts. The curves themselves are computed by e30_froc.py.

usage: python plot_froc.py <fig_froc.json> <out stem>
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e30_froc import SHOW  # noqa: E402


SHORT = {"MetaSeg-style meta-classifier": "MetaSeg-style", "Depth-based fg/bg score": "Depth-based fg/bg",
         "Ensemble disagreement": "Ensemble (4 models)"}


def main():
    src, out = sys.argv[1:3]
    r = json.load(open(src, encoding="utf-8"))
    grid = r["grid"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "xtick.color": "#55554f", "ytick.color": "#55554f"})
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.75), sharey=True)
    for ax, (ds, cs) in zip(axes, r["curves"].items()):
        for lab, (_, col, ls) in SHOW.items():
            if lab in cs:
                ax.plot([0] + grid, [0] + [100 * v for v in cs[lab]], color=col, ls=ls, lw=2.0 if lab == "Ours" else 1.3,
                        label=SHORT.get(lab, lab))
        ax.set_title(ds, fontsize=9)
        ax.set_xlim(0, 2)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
        ax.set_ylim(0, 85)
        ax.grid(color="#e6e5e0", linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1].set_xlabel("strict false alarms per image")
    axes[0].set_ylabel("recall (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=8, handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.83), w_pad=0.6)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)


if __name__ == "__main__":
    main()
