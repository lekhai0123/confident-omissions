"""Teaser figure (Fig. 1 of the introduction): confident omissions are invisible to the segmenter's own uncertainty.

Two ACDC examples (the fog and night successes of Fig. 3): input with the ground-truth outline of the omitted object,
the segmenter's prediction, the calibrated confidence (max softmax with the segmenter's fitted temperature) and the
normalised entropy, cropped around the object. Panel titles give the mean value inside the object and over the image.

usage: python fig_teaser.py <v11 dir> <acdc root> <out stem>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebook"))
from e9_figure import NAMES as _NAMES, PALETTE  # noqa: E402

NAMES = dict(_NAMES, **{f"segformer-{b}": f"nvidia/segformer-{b}-finetuned-cityscapes-1024-1024" for b in ("b0", "b2", "b5")})
import pilot  # noqa: E402

STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
# Selection rule (no hand-picking): among the confident omissions into background on ACDC with >= 20,000 px, the
# targets whose entropy percentile is closest to the median over all 470 targets, one per condition and segmenter.
EXAMPLES = [("GOPR0351_frame_000706", "segformer-b0", "night", "person at night, SegFormer-B0"),
            ("GP020400_frame_000666", "segformer-b5", "rain", "truck in rain, SegFormer-B5")]


def main():
    v11, acdc_root, out = sys.argv[1:4]
    import pandas as pd
    summ = json.load(open(os.path.join(v11, "summary.json"), encoding="utf-8"))
    items = {i["stem"]: i for i in pilot.list_acdc(acdc_root, ("train", "val"), False)}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})  # drawn at text width (5.4 in): no LaTeX scaling
    fig, axes = plt.subplots(len(EXAMPLES), 4, figsize=(5.4, 0.95 * len(EXAMPLES) + 0.5))
    segs = {}
    for row, (stem, model, cond, label) in zip(axes, EXAMPLES):
        o = pd.read_csv(os.path.join(v11, f"objects_{model}.csv"), usecols=["stem", "condition", "instance_id", "confident_omission", "predicted_as"])
        ids = o[(o.stem == stem) & (o.condition == cond) & o.confident_omission & o.predicted_as.isin(STUFF)].instance_id.tolist()
        x, inst, _ = pilot.load_acdc(items[stem], 1.0)
        if model not in segs:
            segs[model] = pilot.Segmenter(NAMES[model], "cuda")
        with torch.no_grad():
            logits = segs[model](x.cuda(), want_feat=False)[0][0].float()
        T = summ["models"][model]["temperature"]
        p = torch.softmax(logits / T, 0)
        conf = p.max(0).values.cpu().numpy()
        ent = (-(p * p.clamp_min(1e-12).log()).sum(0) / np.log(p.shape[0])).cpu().numpy()
        pred = p.argmax(0).cpu().numpy()
        img = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        m = np.isin(inst, ids)
        ys, xs = np.where(m)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        w = max(2.6 * (xs.max() - xs.min()), 3.0 * (ys.max() - ys.min()) * 16 / 9, 700)
        h = w * 9 / 16
        H, W = m.shape
        x0, y0 = int(np.clip(cx - w / 2, 0, max(0, W - w))), int(np.clip(cy - h / 2, 0, max(0, H - h)))
        crop = (slice(y0, int(min(H, y0 + h))), slice(x0, int(min(W, x0 + w))))
        top10 = ent >= np.quantile(ent, 0.9)          # the pixel budget of the entropy baseline (most uncertain 10%)
        frac = 100 * top10[m].mean()
        frac_conf = 100 * (conf[m] >= 0.9).mean()
        print(stem, model, f"conf>=0.9 in object {frac_conf:.0f}%", f"object pixels in top-10% entropy {frac:.1f}%")
        pred_as = o[(o.stem == stem) & (o.condition == cond) & o.instance_id.isin(ids)].predicted_as.iloc[0]
        panels = [
            (img[crop], None, label.replace(", ", ",\n")),
            (PALETTE[np.clip(pred, 0, 18)][crop], None, f"prediction:\nobject $\\rightarrow$ {pred_as}"),
            (conf[crop], (0.0, 1.0), f"confidence $\\geq$ 0.9:\n{frac_conf:.0f}% of object"),
            (ent[crop], (0.0, 1.0), f"in top-10% entropy:\n{frac:.0f}% of object"),
        ]
        for k, (ax, (arr, rng, title)) in enumerate(zip(row, panels)):
            if rng is None:
                ax.imshow(arr)
            else:
                im = ax.imshow(arr, cmap="cividis", vmin=rng[0], vmax=rng[1])
            ax.contour(m[crop], levels=[0.5], colors=["#ffd400" if k < 2 else "#ffffff"], linewidths=1.6)
            ax.set_title(title, fontsize=7.5)
            ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.91, top=0.84, bottom=0.02, wspace=0.05, hspace=0.62)
    cax = fig.add_axes([0.935, 0.08, 0.012, 0.74])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    print("saved")


if __name__ == "__main__":
    main()
