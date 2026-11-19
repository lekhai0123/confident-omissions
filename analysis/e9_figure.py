"""E9: qualitative successes and failure cases of the cheap verifier on ACDC (v11 features, Kaggle-consistent).
Scores for a condition come from a verifier trained on the other conditions (no leakage); threshold calibrated at 1 FA/img.
Left: image + ground-truth confident omissions into background (yellow) + flagged proposals (green = covers an omitted
object, red = false alarm). Right: the monitored model's prediction (recomputed locally; independent of dev statistics).

usage: python e9_figure.py <v11 dir> <acdc root> <out dir>
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebook"))
from e_suite_v8 import ADVERSE, feature_sets, fit_scores, load, thresholds  # noqa: E402
import pilot  # noqa: E402

PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                    [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], np.uint8)
NAMES = {"segformer-b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
         "mask2former-swin-small": "facebook/mask2former-swin-small-cityscapes-semantic"}


def main():
    v11, acdc_root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    P, O = load(v11)
    cols = [c for c in feature_sets(P)["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    D = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"])].copy()
    D["s"], D["flag"] = np.nan, False
    for held in ADVERSE:
        trm = ~D.condition.isin([held, held + "_ref"])
        tem = D.condition == held
        s_tr, s_te = fit_scores(D[trm], D[tem], cols, "pos_co_stuff")
        thr = thresholds(s_tr, D[trm])[1.0]
        D.loc[tem, "s"] = s_te
        D.loc[tem, "flag"] = s_te > thr
    F = D[D.flag]
    co = O[O.condition.isin(ADVERSE) & O.co_stuff & O.model.isin(NAMES)]
    found = set(zip(F[F.pos_co_stuff].model, F[F.pos_co_stuff].stem, F[F.pos_co_stuff].gt_instance_id))
    co = co.assign(found=[k in found for k in zip(co.model, co.stem, co.instance_id)])
    fa = F[~F.pos].groupby(["model", "stem"]).size().rename("n_fa")
    per = co.groupby(["model", "stem", "condition"]).agg(n=("found", "size"), n_found=("found", "sum"), area=("area", "max")).reset_index()
    per = per.join(fa, on=["model", "stem"]).fillna({"n_fa": 0})
    succ = per[(per.n_found >= 1) & (per.n_fa <= 1)].sort_values("area", ascending=False).groupby("condition").head(1)
    miss = per[(per.n_found == 0)].sort_values("area", ascending=False).head(2)
    fal = per[per.n_fa >= 3].sort_values("n_fa", ascending=False).head(1)
    sel = pd.concat([succ.assign(kind="success"), miss.assign(kind="missed"), fal.assign(kind="false alarms")])
    print(sel.to_string(index=False))
    items = {it["stem"]: it for it in pilot.list_acdc(acdc_root, ("train", "val"), False)}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    segs = {}
    fig, axes = plt.subplots(len(sel), 2, figsize=(10, 2.85 * len(sel)))
    for row, (_, r) in zip(axes, sel.iterrows()):
        it = items[r.stem]
        x, inst, _ = pilot.load_acdc(it, 1.0)
        img = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if r.model not in segs:
            segs[r.model] = pilot.Segmenter(NAMES[r.model], "cuda")
        with torch.no_grad():
            pred = segs[r.model](x.cuda(), want_feat=False)[0][0].argmax(0).cpu().numpy()
        row[0].imshow(img)
        for iid in co[(co.model == r.model) & (co.stem == r.stem)].instance_id:
            row[0].contour(inst == iid, levels=[0.5], colors=["#ffd400"], linewidths=1.6)
        for _, b in F[(F.model == r.model) & (F.stem == r.stem)].iterrows():
            row[0].add_patch(Rectangle((2 * b.box_x0, 2 * b.box_y0), 2 * (b.box_x1 - b.box_x0), 2 * (b.box_y1 - b.box_y0),
                                       fill=False, ec="#1baf7a" if b.pos else "#ff3b30", lw=1.6))
        row[0].set_title(f"{r.kind}: {r.condition}, {r.model}  (found {int(r.n_found)}/{int(r.n)}, false alarms {int(r.n_fa)})", fontsize=8)
        row[1].imshow(PALETTE[np.clip(pred, 0, 18)])
        for iid in co[(co.model == r.model) & (co.stem == r.stem)].instance_id:
            row[1].contour(inst == iid, levels=[0.5], colors=["#ffd400"], linewidths=1.6)
        row[1].set_title("prediction (yellow: object the model omitted)", fontsize=8)
        for a in row:
            a.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_e9_qualitative.{ext}"), dpi=130, bbox_inches="tight")
    sel.to_csv(os.path.join(out, "e9_selected.csv"), index=False)
    print("saved")


if __name__ == "__main__":
    main()
