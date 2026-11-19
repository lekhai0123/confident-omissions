"""Offline analysis of the v4 feasibility run (per model): sanity, cheap baselines, ML rule, model transfer.

usage: python analyse_v4.py <results dir> [--boot 1000]
Writes <results dir>/analysis_v4.json and prints a compact report.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

THINGS = {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
SCORES = {  # name -> (column, sign): higher score = more suspicious
    "mean_entropy": ("mean_entropy", 1), "max_entropy": ("max_entropy", 1), "p95_entropy": ("p95_entropy", 1),
    "spatial_max_window_entropy": ("max_window_entropy", 1), "frac_low_conf": ("frac_low_conf", 1),
    "one_minus_mean_margin": ("one_minus_mean_margin", 1),
    "class_shift_js": ("class_shift_js", 1), "pred_frac_building": ("pred_frac_building", 1),
    "quality_blur": ("q_sharpness", -1), "quality_low_contrast": ("q_contrast", -1),
}
FAMILY = {"uncertainty": ["mean_entropy", "max_entropy", "p95_entropy", "spatial_max_window_entropy",
                          "frac_low_conf", "one_minus_mean_margin"],
          "class_shift": ["class_shift_js", "pred_frac_building"],
          "image_quality": ["quality_blur", "quality_low_contrast"]}


def recall_at_fpr(y, s, fpr):
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0 or y.sum() == 0:
        return None
    thr = neg[min(int(np.floor(fpr * len(neg))), len(neg) - 1)]
    return float((s[y == 1] > thr).mean())


def detection(y, s, groups, n_boot, rng):
    if y.sum() == 0 or y.sum() == len(y):
        return None
    out = {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
           "positive_rate": float(y.mean()), "recall_at_fpr05": recall_at_fpr(y, s, 0.05),
           "recall_at_fpr10": recall_at_fpr(y, s, 0.10)}
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.flatnonzero(inv == g) for g in range(len(ug))]
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_g[g] for g in rng.integers(0, len(ug), len(ug))])
        yy = y[pick]
        if 0 < yy.sum() < len(yy):
            vals.append(roc_auc_score(yy, s[pick]))
    if vals:
        out["auroc_ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


def analyse_model(obj: pd.DataFrame, img: pd.DataFrame, summ: dict, n_boot: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obj = obj.copy()
    obj["true_miss"] = obj.confident_omission & ~obj.predicted_as.isin(THINGS)
    conds = sorted(img.condition.unique(), key=lambda c: (c != "clean", c))
    res: dict = {"sanity": {}, "events": {}, "detection_within_condition": {}, "detection_pooled_corrupted": {},
                 "ml_rule": {}}
    # P0 sanity: mIoU and collapse rate per condition
    for c in conds:
        gi = img[img.condition == c]
        m = summ.get("miou_by_condition", {}).get(c, {})
        res["sanity"][c] = {"miou": m.get("argmax", {}).get("miou"), "miou_ml_rule": m.get("ml", {}).get("miou"),
                            "share_images_building_ge_60pct": float((gi.pred_frac_building >= 0.6).mean()),
                            "mean_pred_frac_building": float(gi.pred_frac_building.mean())}
    # events
    nn = obj[~obj.condition.str.startswith("gaussian_noise")]
    for name, d in (("all", obj), ("no_noise", nn)):
        co, tm = d[d.confident_omission], d[d.true_miss]
        res["events"][name] = {"confident_omissions": int(len(co)),
                               "unique_objects": int(co[["stem", "instance_id"]].drop_duplicates().shape[0]),
                               "true_misses_unique_objects": int(tm[["stem", "instance_id"]].drop_duplicates().shape[0]),
                               "source_images": int(co.stem.nunique())}
    res["events"]["by_condition"] = {c: int(obj[(obj.condition == c) & obj.confident_omission].shape[0]) for c in conds}
    # image labels: has a confident omission (and has a true miss)
    lab = obj.groupby(["stem", "condition"]).true_miss.any().rename("has_true_miss").reset_index()
    img = img.merge(lab, on=["stem", "condition"], how="left").fillna({"has_true_miss": False})
    for c in conds:
        gi = img[img.condition == c]
        y = gi.has_confident_omission.astype(int).to_numpy()
        entry = {}
        for sname, (col, sign) in SCORES.items():
            r = detection(y, sign * gi[col].to_numpy(dtype=float), gi.stem.to_numpy(), n_boot // 4, rng)
            if r:
                entry[sname] = r
        res["detection_within_condition"][c] = entry
    # pooled over corrupted conditions, CI grouped by source image (the circularity-relevant view)
    gc = img[img.condition != "clean"]
    for label in ("has_confident_omission", "has_true_miss"):
        y = gc[label].astype(int).to_numpy()
        res["detection_pooled_corrupted"][label] = {
            s: detection(y, sign * gc[col].to_numpy(dtype=float), gc.stem.to_numpy(), n_boot, rng)
            for s, (col, sign) in SCORES.items()}
    # ML rule (MetaFusion): does lowering the decision threshold for rare classes recover them?
    for name, d in (("all", obj), ("no_noise", nn)):
        co = d[d.confident_omission]
        if len(co):
            res["ml_rule"][name] = {"recovered_share": float((~co.omitted_ml).mean()),
                                    "recovered_majority_share": float((co.frac_true_ml >= 0.5).mean()),
                                    "median_mean_p_true": float(co.mean_p_true.median()),
                                    "share_true_class_never_top2": float((co.frac_true_in_top2 < 0.05).mean()),
                                    "omitted_objects_argmax": int(d.omitted.sum()), "omitted_objects_ml": int(d.omitted_ml.sum())}
    return res


def best_by_family(det: dict) -> dict:
    out = {}
    for fam, names in FAMILY.items():
        best = max(((n, det[n]) for n in names if det.get(n)), key=lambda kv: kv[1]["auroc"], default=None)
        if best:
            out[fam] = {"score": best[0], **{k: best[1][k] for k in ("auroc", "auprc", "recall_at_fpr10") if k in best[1]},
                        "auroc_ci95": best[1].get("auroc_ci95")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    summary = json.load(open(os.path.join(a.results, "summary.json")))
    report = {}
    for f in sorted(glob.glob(os.path.join(a.results, "objects_*.csv"))):
        tag = os.path.basename(f)[len("objects_"):-4]
        obj = pd.read_csv(f)
        img = pd.read_csv(os.path.join(a.results, f"images_{tag}.csv"))
        r = analyse_model(obj, img, summary["models"].get(tag, {}), a.boot)
        report[tag] = r
        print(f"\n===== {tag}")
        print("mIoU / ML-rule mIoU / share collapsed:")
        for c, s in r["sanity"].items():
            print(f"  {c:18s} {s['miou'] if s['miou'] is None else round(s['miou'], 3)} / "
                  f"{s['miou_ml_rule'] if s['miou_ml_rule'] is None else round(s['miou_ml_rule'], 3)} / "
                  f"{s['share_images_building_ge_60pct']:.2f}")
        print("events:", json.dumps(r["events"]["no_noise"]), "| by condition:", r["events"]["by_condition"])
        print("ML rule:", json.dumps(r["ml_rule"]))
        for label, det in r["detection_pooled_corrupted"].items():
            print(f"pooled corrupted, {label}: best per family {json.dumps(best_by_family(det))}")
    with open(os.path.join(a.results, "analysis_v4.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
