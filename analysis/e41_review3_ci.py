"""E41: unconditional bootstrap for the matched-false-alarm comparison (third simulated review, G-1 MAJOR 2).

The paired intervals of Table 4 (e10_e12.paired_ci) resample the source images of the TARGETS and keep each method's
threshold fixed at the value chosen on the full test set; images with only false alarms move the threshold but do not
enter the resampling. Here every resample draws ALL source images with replacement (the four segmenters of an image
stay together), re-estimates the threshold of every method at the same budget (strict false alarms per image x
segmenter unit) on the resampled set, and evaluates all methods on the same resampled images.

Scores are those of E38 (same code, same seeds): ACDC leave-one-condition-out over the four adverse conditions,
BDD100K and WildDash2 trained on all of ACDC; RT-DETR normal + gamma proposals; target = confident omission into
background (Section 3). Methods: verifier (ours), rule, MetaSeg-style meta-classifier, logistic regression on three
box features (logit3), logistic regression on the verifier's features (lr25), box-mask disagreement.

usage: python e41_review3_ci.py <v11> <v13> <v15> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e38_review2 import L3, fit, found_at, loco_scores, paired, prep, target_keys  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, feature_sets  # noqa: E402

N_BOOT = int(os.environ.get("E41_NBOOT", 2000))
FAS = (0.25, 0.5, 1.0, 2.0)
BASES = ("rule", "meta", "logit3", "lr25", "disagree")


def prepare(T, s, keys, pos):
    """per method: negatives (score, stem index) sorted descending; per target: best score of a covering proposal"""
    s = np.nan_to_num(np.asarray(s, float), nan=-np.inf)
    neg = ~pos
    order = np.argsort(-s[neg], kind="stable")
    best = {}
    P = T[pos]
    for k, v in zip(zip(P.model, P.stem, P.gt_instance_id), s[pos]):
        if v > best.get(k, -np.inf):
            best[k] = v
    return s[neg][order], order, np.array([best.get(k, -np.inf) for k in keys])


def main():
    v11, v13, v15, out = sys.argv[1:5]
    DA, OA = prep(v11)
    fs = feature_sets(DA)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta = [c for c in DA.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    TA = DA[DA.condition.isin(ADVERSE)].reset_index(drop=True)
    DB, OB = prep(v13)
    DW, OW = prep(v15)
    doms = {"ACDC": (TA, OA, True), "BDD100K": (DB, OB, False), "WildDash2": (DW, OW, False)}
    res = {"n_boot": N_BOOT, "domains": {}}
    for name, (T, O, acdc) in doms.items():
        S = {}
        for nm, cols, kind in (("ours", cheap, "gbm"), ("lr25", cheap, "lr"), ("logit3", L3, "lr"), ("meta", meta, "gbm")):
            S[nm] = loco_scores(DA, T, cols, kind, apply_to=T) if acdc else fit(DA, cols, kind)(T)
        S["rule"] = T.rule.to_numpy()
        S["disagree"] = T.disagree.to_numpy()
        keys, kstems = target_keys(O, acdc, "co_stuff")
        pos = T.pos_co_stuff.to_numpy()

        # source images: every image that has a proposal or a target
        stems = np.array(sorted(set(T.stem.astype(str)) | set(map(str, kstems))))
        sid = {u: i for i, u in enumerate(stems)}
        units = np.zeros(len(stems))
        for u, g in T.groupby("stem").unit.nunique().items():
            units[sid[str(u)]] = g
        t_stem = np.array([sid[str(u)] for u in T.stem])
        k_stem = np.array([sid[str(k[1])] for k in keys])

        prep_m = {}
        for m, s in S.items():
            s_sorted, order, best = prepare(T, s, keys, pos)
            prep_m[m] = (s_sorted, t_stem[~pos][order], best)

        def recall(m, c, fa):
            s_sorted, nst, best = prep_m[m]
            n_w = (c * units).sum()
            cw = np.cumsum(c[nst])
            i = min(int(np.searchsorted(cw, int(fa * n_w), side="right")), len(s_sorted) - 1)
            w = c[k_stem]
            return float((w * (best > s_sorted[i])).sum() / max(w.sum(), 1))

        ones = np.ones(len(stems))
        point = {fa: {m: recall(m, ones, fa) for m in S} for fa in FAS}
        # consistency with the protocol's read-off (found_at)
        chk = {m: round(float(found_at(T, S[m], keys, 1.0, pos).mean()), 4) for m in S}
        rng = np.random.default_rng(0)
        boot = {fa: {m: [] for m in S} for fa in FAS}
        for _ in range(N_BOOT):
            c = np.bincount(rng.integers(0, len(stems), len(stems)), minlength=len(stems)).astype(float)
            for fa in FAS:
                for m in S:
                    boot[fa][m].append(recall(m, c, fa))
        r = {"n_images": len(stems), "n_images_with_targets": int(len(set(k_stem))), "n_targets": len(keys),
             "recall_found_at_R@1": chk, "fa": {}}
        print(f"\n== {name}: {len(stems)} images ({len(set(k_stem))} with targets), {len(keys)} targets")
        for fa in FAS:
            r["fa"][str(fa)] = {"recall": {m: round(v, 4) for m, v in point[fa].items()}, "ours_minus": {}}
            for b in BASES:
                d = np.array(boot[fa]["ours"]) - np.array(boot[fa][b])
                lo, hi = np.percentile(d, [2.5, 97.5])
                cond = paired(T, S["ours"], S[b], keys, kstems, pos, fa=fa)
                r["fa"][str(fa)]["ours_minus"][b] = {"diff": round(point[fa]["ours"] - point[fa][b], 4),
                                                     "ci95_unconditional": [round(float(lo), 4), round(float(hi), 4)],
                                                     "p_le_0": round(float((d <= 0).mean()), 4),
                                                     "ci95_fixed_threshold": cond["ci95"]}
            print(f"  R@{fa}: " + " ".join(f"{m} {100*v:.1f}" for m, v in point[fa].items()))
            for b, v in r["fa"][str(fa)]["ours_minus"].items():
                print(f"     ours-{b:8s} {100*v['diff']:+5.1f}  uncond [{100*v['ci95_unconditional'][0]:+.1f}, "
                      f"{100*v['ci95_unconditional'][1]:+.1f}]  fixed-thr [{100*v['ci95_fixed_threshold'][0]:+.1f}, "
                      f"{100*v['ci95_fixed_threshold'][1]:+.1f}]", flush=True)
        res["domains"][name] = r
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
