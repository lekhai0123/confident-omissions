"""Graphical abstract for the CVIU submission (>= 1328 px wide, landscape, readable at 13 cm).

Three panels on one ACDC night example (Mask2Former-S, the first success case of Fig. 3): input with the omitted
cyclist outlined, the segmenter's confident prediction (cyclist erased into background), and the verifier's flag;
plus a one-line result strip. Flags come from the leave-one-condition-out verifier with a threshold calibrated on the
training conditions (no leakage), as in e9_figure.py.

usage: python graphical_abstract.py <v11 dir> <acdc root> <out stem>
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebook"))
from e9_figure import NAMES, PALETTE  # noqa: E402
from e_suite_v8 import feature_sets, fit_scores, load, thresholds  # noqa: E402
import pilot  # noqa: E402

STEM, MODEL, HELD = "GP010376_frame_000049", "mask2former-swin-small", "night"
STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}


def main():
    v11, acdc_root, out = sys.argv[1:4]
    P, O = load(v11)
    cols = [c for c in feature_sets(P)["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    D = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"])]
    trm = ~D.condition.isin([HELD, HELD + "_ref"])
    tem = (D.condition == HELD) & (D.stem == STEM) & (D.model == MODEL)
    s_tr, s_te = fit_scores(D[trm], D[tem], cols, "pos_co_stuff")
    thr = thresholds(s_tr, D[trm])[1.0]
    F = D[tem][s_te > thr]
    ids = O[(O.stem == STEM) & (O.model == MODEL) & O.co_stuff].instance_id.tolist()

    it = {i["stem"]: i for i in pilot.list_acdc(acdc_root, ("train", "val"), False)}[STEM]
    x, inst, _ = pilot.load_acdc(it, 1.0)
    img = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    seg = pilot.Segmenter(NAMES[MODEL], "cuda")
    with torch.no_grad():
        logits = seg(x.cuda(), want_feat=False)[0][0]
    pred = logits.argmax(0).cpu().numpy()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    m = np.isin(inst, ids)
    ys, xs = np.where(m)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    w = max(4.0 * (xs.max() - xs.min()), 900)
    h = w * 0.62
    H, W = inst.shape
    x0, y0 = int(np.clip(cx - w / 2, 0, W - w)), int(np.clip(cy - h / 2, 0, H - h))
    x1, y1 = int(x0 + w), int(y0 + h)

    plt.rcParams.update({"font.size": 11})
    fig = plt.figure(figsize=(13.3, 4.5), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.13], hspace=0.02, wspace=0.04, left=0.01, right=0.99, top=0.86, bottom=0.01)
    titles = ["1  Night scene: a cyclist", "2  Segmenter: cyclist erased into road,\n    confidence $\\geq$ 0.9, low uncertainty",
              "3  Detector + verifier: omission flagged"]
    for k in range(3):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(PALETTE[np.clip(pred, 0, 18)] if k == 1 else img)
        if k in (0, 1):
            ax.contour(m, levels=[0.5], colors=["#ffd400"], linewidths=2.5)
        if k == 2:
            for _, b in F.iterrows():
                ax.add_patch(Rectangle((2 * b.box_x0, 2 * b.box_y0), 2 * (b.box_x1 - b.box_x0), 2 * (b.box_y1 - b.box_y0),
                                       fill=False, ec="#1baf7a", lw=3.5))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.axis("off")
        ax.set_title(titles[k], fontsize=12, loc="left")
    axt = fig.add_subplot(gs[1, :])
    axt.axis("off")
    axt.text(0.5, 0.5, "Recall at one false alarm per image:  58.9% ACDC  |  53.4% BDD100K  |  69.5% WildDash2"
             "     (OOD scores: $\\leq$ 10%)", ha="center", va="center", fontsize=13,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f2f2ef", ec="#8a8a85"))
    fig.suptitle("Confident omissions: road users that semantic segmenters erase without warning", fontsize=15, x=0.01, ha="left")
    fig.savefig(out + ".png", dpi=150)
    fig.savefig(out + ".pdf")
    print("flags:", len(F), "targets:", len(ids), "size px:", fig.get_size_inches() * 150)


if __name__ == "__main__":
    main()
