"""E20: our method under the original MetaFusion protocol (Chan et al. 2020, arXiv 1912.07420), for a table that
places our numbers next to the numbers reported in that paper.

Their protocol: Cityscapes val, class 'human' (person + rider), segment level. FN = ground-truth human not found by
the prediction; FP = predicted human segment without overlap with a ground-truth human. They report, for an adjusted
decision rule d, Delta = (FP(d) - FP(Bayes)) / (FN(Bayes) - FN(d)): extra false positives paid per removed false
negative (smaller is better), at several FN reductions.

Ours as an add-on to the Bayes (argmax) prediction: flagged person proposals become additional human segments.
  - removed FN: an omitted ground-truth human (<= 5% correct pixels; theirs: IoU = 0) overlapped by a flag
    (IoU > 0, their criterion; strict variant: the flag covers >= 50% of it),
  - added FP: a flag that overlaps no ground-truth human; flags are de-duplicated by box NMS (IoU 0.5) so that the
    normal and gamma detector passes do not double count,
  - flags overlapping an already segmented human are neither (they agree with the base prediction).
The verifier is trained on ACDC (v11), never on Cityscapes images, with the generic target (any omitted object),
since MetaFusion targets all missed humans. Data: v10a clean Cityscapes val (450 images, the 50 dev images excluded).
Also reported: the plain ML rule on the same models (FN only; its FP are not stored, the paper's ML numbers are cited).

usage: python e20_metafusion_protocol.py <v10a dir> <v11 dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e_suite_v8 import MODELS, feature_sets, load  # noqa: E402

HUMAN = {24, 25}
# reported in Chan et al. 2020, Table 2 (DeepLabV3+ on Cityscapes val, class human)
REPORTED = {
    "MobileNetV2 (mIoU 0.684)": {"FN_bayes": 839, "rows": [("ML rule", 476, 11.074), ("MetaFusion best", 611, 1.676),
                                                        ("MetaFusion @ML", 553, 2.590)]},
    "Xception65 (mIoU 0.753)": {"FN_bayes": 679, "rows": [("ML rule", 421, 8.640), ("MetaFusion best", 549, 2.531),
                                                       ("MetaFusion @ML", 508, 2.912)]},
}


def box_iou(a, b):
    x0, y0 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x1, y1 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def nms_keep(df, thr=0.5):
    keep = []
    for _, g in df.groupby("unit"):
        g = g.sort_values("s", ascending=False)
        boxes = g[["box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy(float)
        alive = np.ones(len(g), bool)
        for i in range(len(g)):
            if not alive[i]:
                continue
            keep.append(g.index[i])
            alive[i + 1:] &= box_iou(boxes[i], boxes[i + 1:]) < thr
    return df.loc[keep]


def curve(T, fn_keys, strict=False):
    """Sweep the flag threshold; return (removed FN, added FP) pairs."""
    T = T.sort_values("s", ascending=False)
    hum = (T.gt_instance_id // 1000).isin(HUMAN) & (T.gt_iou > 0)
    tp_mask = hum & (T.gt_cover >= 0.5 if strict else True)
    out = []
    for q in np.unique(np.quantile(T.s, np.linspace(0, 1, 81))):
        F = nms_keep(T[T.s >= q])
        h = (F.gt_instance_id // 1000).isin(HUMAN) & (F.gt_iou > 0)
        fp = int((~h).sum())
        tp = F[tp_mask.reindex(F.index).fillna(False).to_numpy()]
        removed = len(set(zip(tp.model, tp.stem, tp.gt_instance_id)) & fn_keys)
        out.append((removed, fp))
    return sorted(set(out))


def delta_at(cv, target_removed):
    ok = [(r, fp) for r, fp in cv if r >= target_removed]
    if not ok:
        return None
    r, fp = min(ok, key=lambda x: x[1])
    return round(fp / max(1, r), 3), r, fp


def main():
    v10a, v11, out = sys.argv[1:4]
    P11, _ = load(v11)
    fs = feature_sets(P11)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    S = P11[P11.det_src.isin(["rtdetr", "rtdetr_gamma"])]
    PC, OC = load(v10a)
    PC = PC[(PC.condition == "clean") & PC.det_src.isin(["rtdetr", "rtdetr_gamma"]) & (PC.det_label == "person")].copy()
    OC = OC[OC.condition == "clean"]
    full = pd.concat([pd.read_csv(os.path.join(v10a, f"objects_{m}.csv"), usecols=["instance_id", "stem", "condition", "omitted", "omitted_ml"])
                      .assign(model=m) for m in MODELS])
    full = full[full.condition == "clean"]
    res = {"reported_chan2020": REPORTED, "models": {}}
    if len(sys.argv) > 4 and sys.argv[4] == "--cv":   # in-domain 5-fold CV over Cityscapes val images, as in Chan et al.
        PA = load(v10a)[0]
        PA = PA[(PA.condition == "clean") & PA.det_src.isin(["rtdetr", "rtdetr_gamma"])]
        stems = np.array(sorted(PA.stem.unique()))
        fold = dict(zip(np.random.default_rng(0).permutation(stems), np.arange(len(stems)) % 5))
        for target in ("pos", "pos_co_stuff"):
            PC[f"s_{target}"] = np.nan
            for k in range(5):
                tr = PA[PA.stem.map(fold) != k]
                rng = np.random.default_rng(0)
                keep = tr[target].to_numpy() | (rng.random(len(tr)) < 0.25)
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
                clf.fit(tr[keep][cheap].fillna(-1).to_numpy(np.float32), tr[keep][target].to_numpy())
                te = PC.stem.map(fold) == k
                PC.loc[te, f"s_{target}"] = clf.predict_proba(PC[te][cheap].fillna(-1).to_numpy(np.float32))[:, 1]
    for target in (() if len(sys.argv) > 4 and sys.argv[4] == "--cv" else ("pos", "pos_co_stuff")):
        rng = np.random.default_rng(0)
        keep = S[target].to_numpy() | (rng.random(len(S)) < 0.25)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
        clf.fit(S[keep][cheap].fillna(-1).to_numpy(np.float32), S[keep][target].to_numpy())
        PC[f"s_{target}"] = clf.predict_proba(PC[cheap].fillna(-1).to_numpy(np.float32))[:, 1]
    print(f"{'model':24s} {'FN_base':>7s} {'FN_ML':>6s} | ours (generic target): Delta at FN -19% / -27% / -34% / -40%  | max removed")
    for m in MODELS:
        o = full[(full.model == m) & (full.instance_id // 1000).isin(HUMAN)]
        fn = o[o.omitted]
        fn_keys = set(zip(fn.model, fn.stem, fn.instance_id))
        r = {"FN_base": len(fn), "FN_ml_rule": int(o.omitted_ml.sum()), "n_gt_humans": len(o)}
        for target in ("pos", "pos_co_stuff"):
            for strict in (False, True):
                T = PC[PC.model == m].assign(s=PC[PC.model == m][f"s_{target}"])
                cv = curve(T, fn_keys, strict)
                key = f"{target}{'_strict' if strict else ''}"
                r[key] = {"max_removed": max(x[0] for x in cv),
                          "delta": {f"{p}%": delta_at(cv, int(np.ceil(p / 100 * len(fn)))) for p in (10, 19, 27, 34, 40)}}
        res["models"][m] = r
        d = r["pos"]["delta"]
        print(f"{m:24s} {r['FN_base']:7d} {r['FN_ml_rule']:6d} | " + " / ".join(
            f"{d[k][0]:.2f}" if d[k] else "  - " for k in ("19%", "27%", "34%", "40%")) + f"  | {r['pos']['max_removed']} "
              f"(strict: " + " / ".join(f"{r['pos_strict']['delta'][k][0]:.2f}" if r['pos_strict']['delta'][k] else "-" for k in ("19%", "27%", "34%")) + ")")
    json.dump(res, open(out, "w"), indent=1)
    print("reported (Chan et al. 2020): MobileNetV2 ML Delta 11.07 at -43%, MetaFusion best 1.68 at -27%, 2.59 at -34%; "
          "Xception ML 8.64 at -38%, MetaFusion best 2.53 at -19%, 2.91 at -25%")


if __name__ == "__main__":
    main()
