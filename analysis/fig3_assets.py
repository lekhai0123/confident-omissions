"""Real-image asset pack for the manual redraw of Fig. 3 (method overview), and as reference input for image AI tools.

One real ACDC case used throughout (the night success of Fig. 5 and the graphical abstract): frame
GP010376_frame_000049, monitored segmenter Mask2Former-S, omitted object = bicycle predicted as road.
Writes individual PNGs (full frames and the same crop), an annotated asset sheet, and facts.json with the measured
values (detector class / score, verifier score and threshold, confidence and entropy statistics). The verifier score
comes from the leave-one-condition-out verifier (trained without night images) with the threshold calibrated on the
training conditions at one false alarm per image, exactly as in e9_figure.py.

usage: python fig3_assets.py <v11 dir> <acdc root> <out dir>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebook"))
from e9_figure import NAMES, PALETTE  # noqa: E402
from e_suite_v8 import feature_sets, fit_scores, load, thresholds  # noqa: E402
import pilot  # noqa: E402

STEM, MODEL, HELD = "GP010376_frame_000049", "mask2former-swin-small", "night"
STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
CLASS = {24: "person", 25: "rider", 26: "car", 27: "truck", 28: "bus", 31: "train", 32: "motorcycle", 33: "bicycle"}


def save(arr, path):
    Image.fromarray(arr).save(path)


def main():
    v11, acdc_root, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    P, O = load(v11)
    cols = [c for c in feature_sets(P)["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    D = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"])]
    trm = ~D.condition.isin([HELD, HELD + "_ref"])
    tem = (D.condition == HELD) & (D.stem == STEM) & (D.model == MODEL)
    s_tr, s_te = fit_scores(D[trm], D[tem], cols, "pos_co_stuff")
    thr = float(thresholds(s_tr, D[trm])[1.0])
    Tf = D[tem].assign(verifier_score=s_te)
    tgt = O[(O.stem == STEM) & (O.model == MODEL) & (O.condition == HELD) & O.co_stuff]
    ids = tgt.instance_id.tolist()
    flagged = Tf[Tf.verifier_score > thr]
    hit = flagged[flagged.gt_instance_id.isin(ids) & (flagged.gt_cover >= 0.5)].sort_values("verifier_score", ascending=False)
    bi = hit.iloc[0]

    summ = json.load(open(os.path.join(v11, "summary.json"), encoding="utf-8"))
    T = summ["models"][MODEL]["temperature"]
    it = {i["stem"]: i for i in pilot.list_acdc(acdc_root, ("train", "val"), False)}[STEM]
    x, inst, _ = pilot.load_acdc(it, 1.0)
    seg = pilot.Segmenter(NAMES[MODEL], "cuda")
    with torch.no_grad():
        logits = seg(x.cuda(), want_feat=False)[0][0].float()
    p = torch.softmax(logits / T, 0)
    conf = p.max(0).values.cpu().numpy()
    ent = (-(p * p.clamp_min(1e-12).log()).sum(0) / np.log(p.shape[0])).cpu().numpy()
    pred = p.argmax(0).cpu().numpy()
    img = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gamma = (np.power(x[0].permute(1, 2, 0).numpy(), 0.5) * 255).astype(np.uint8)
    m = np.isin(inst, ids)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    ent_rgb = (cm.cividis(np.clip(ent, 0, 1))[..., :3] * 255).astype(np.uint8)
    pred_rgb = PALETTE[np.clip(pred, 0, 18)]

    # crop around the omitted object (same crop for every crop asset)
    ys, xs = np.where(m)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    w = max(2.2 * (xs.max() - xs.min()), 2.2 * (ys.max() - ys.min()) * 4 / 3, 420)
    h = w * 3 / 4
    H, W = m.shape
    x0, y0 = int(np.clip(cx - w / 2, 0, W - w)), int(np.clip(cy - h / 2, 0, H - h))
    x1, y1 = int(min(W, x0 + w)), int(min(H, y0 + h))
    crop = (slice(y0, y1), slice(x0, x1))

    def outline(arr, mask, colour):
        a = arr.copy()
        from scipy.ndimage import binary_dilation
        edge = binary_dilation(mask, iterations=3) & ~mask
        a[edge] = colour
        return a

    def boxes(arr, rows, highlight=None):
        a = Image.fromarray(arr.copy())
        from PIL import ImageDraw
        d = ImageDraw.Draw(a)
        for _, r in rows.iterrows():
            is_bi = highlight is not None and r.name == highlight
            d.rectangle([2 * r.box_x0, 2 * r.box_y0, 2 * r.box_x1, 2 * r.box_y1],
                        outline=(27, 175, 122) if is_bi else (235, 104, 52), width=6 if is_bi else 3)
        return np.asarray(a)

    files = {
        "01_input_full.png": img,
        "02_input_gamma_full.png": gamma,
        "03_prediction_full.png": pred_rgb,
        "04_entropy_full.png": ent_rgb,
        "05_detector_boxes_full.png": boxes(gamma, Tf, highlight=bi.name),
        "10_crop_input.png": outline(img, m, (255, 212, 0))[crop],
        "11_crop_prediction_road.png": outline(pred_rgb, m, (255, 212, 0))[crop],
        "12_crop_entropy.png": outline(ent_rgb, m, (255, 255, 255))[crop],
        "13_crop_flagged_box.png": boxes(img, Tf.loc[[bi.name]], highlight=bi.name)[crop],
    }
    # ---- extra assets for the feature block: tight crop around the detector box b_i (full resolution) ----
    bx0, by0, bx1, by1 = 2 * int(bi.box_x0), 2 * int(bi.box_y0), 2 * int(bi.box_x1), 2 * int(bi.box_y1)
    mg = int(0.35 * max(bx1 - bx0, by1 - by0))
    tc = (slice(max(0, by0 - mg), min(H, by1 + mg)), slice(max(0, bx0 - mg), min(W, bx1 + mg)))
    box_mask = np.zeros_like(m)
    box_mask[by0:by1, bx0:bx1] = True
    stuff = pred <= 10                       # trainIds 0-10 = background (stuff) classes
    from PIL import ImageDraw

    def with_box(arr):
        a = Image.fromarray(arr.copy())
        ImageDraw.Draw(a).rectangle([bx0, by0, bx1, by1], outline=(27, 175, 122), width=7)
        return np.asarray(a)

    dis = (img.astype(np.float32) * 0.35).astype(np.uint8)          # darkened input
    sel = box_mask & stuff
    dis[sel] = (0.45 * img[sel] + 0.55 * np.array([235, 104, 52])).astype(np.uint8)   # orange: box pixels the segmenter calls background
    files.update({
        "14_disagreement_box_on_prediction.png": with_box(pred_rgb)[tc],
        "15_disagreement_map.png": with_box(dis)[tc],
        "16_entropy_in_box.png": with_box(ent_rgb)[tc],
    })
    box_stats = {
        "share_of_box_pixels_predicted_background": round(float(stuff[box_mask].mean()), 3),
        "share_of_box_pixels_predicted_road": round(float((pred[box_mask] == 0).mean()), 3),
        "mean_entropy_in_box": round(float(ent[box_mask].mean()), 3),
        "share_of_box_pixels_in_image_top10pct_entropy": round(float((ent >= np.quantile(ent, 0.9))[box_mask].mean()), 3),
        "mean_confidence_in_box": round(float(conf[box_mask].mean()), 3),
    }
    for name, arr in files.items():
        save(arr, os.path.join(out, name))
    road = np.array(PALETTE[0], dtype=np.uint8)
    save(np.tile(road, (60, 160, 1)), os.path.join(out, "20_legend_road_colour.png"))

    top10 = ent >= np.quantile(ent, 0.9)
    facts = {
        "frame": STEM, "dataset": "ACDC (anonymised release), night", "segmenter": "Mask2Former-S (Cityscapes)",
        "omitted_object": {"gt_class": CLASS.get(int(ids[0]) // 1000, "?"), "predicted_as": tgt.predicted_as.iloc[0],
                           "area_px": int(m.sum())},
        "confidence": {"share_of_object_pixels_conf_ge_0.9": round(float((conf[m] >= 0.9).mean()), 3),
                       "mean_conf_in_object": round(float(conf[m].mean()), 3)},
        "entropy": {"share_of_object_pixels_in_image_top10pct": round(float(top10[m].mean()), 3),
                    "mean_entropy_in_object": round(float(ent[m].mean()), 3), "mean_entropy_image": round(float(ent.mean()), 3)},
        "detector": {"proposals_in_frame": int(len(Tf)), "b_i_class": bi.det_label, "b_i_detector_score": round(float(bi.det_score), 3),
                     "b_i_pass": "gamma" if bi.det_src == "rtdetr_gamma" else "normal"},
        "inside_box_b_i": box_stats,
        "verifier": {"b_i_verifier_score": round(float(bi.verifier_score), 3), "threshold_1FA_training_calibration": round(thr, 3),
                     "flags_in_frame": int(len(flagged)), "flags_on_target": int(len(hit))},
        "note": "Verifier trained leave-one-condition-out (no night images); threshold calibrated on the training "
                "conditions at one false alarm per image. Boxes: detector proposals of both passes (orange), b_i green.",
    }
    json.dump(facts, open(os.path.join(out, "facts.json"), "w"), indent=1)
    print(json.dumps(facts, indent=1))

    # annotated asset sheet (reference input for image AI tools and for the redraw)
    panels = [("1  Input (night)", files["01_input_full.png"]), ("2  Gamma-corrected input", files["02_input_gamma_full.png"]),
              ("3  Segmentation (full frame)", files["03_prediction_full.png"]),
              ("4  Candidate boxes, both passes (b_i green)", files["05_detector_boxes_full.png"]),
              ("5  Input crop", files["10_crop_input.png"]),
              (f"6  Predicted as {facts['omitted_object']['predicted_as']} ({100 * facts['confidence']['share_of_object_pixels_conf_ge_0.9']:.0f}% px conf >= 0.9)", files["11_crop_prediction_road.png"]),
              (f"7  Low entropy ({100 * facts['entropy']['share_of_object_pixels_in_image_top10pct']:.0f}% px in image top-10%)", files["12_crop_entropy.png"]),
              (f"8  Flagged: detector {bi.det_label} {bi.det_score:.2f}, verifier {bi.verifier_score:.2f}", files["13_crop_flagged_box.png"])]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, (t, a) in zip(axes.ravel(), panels):
        ax.imshow(a)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle("Fig. 3 asset sheet: one real ACDC case (Mask2Former-S, night, bicycle erased into road)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "00_asset_sheet.png"), dpi=150)


if __name__ == "__main__":
    main()
