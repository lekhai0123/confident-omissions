"""E38: offline analyses for the second simulated Q1 review (paper/review/review2.txt, 2026-11-22). No new inference.

Locked protocol (WRITING_KIT section 0): shared RT-DETR normal + gamma proposals, strict false alarms per image x
segmenter, matched-FA read-off, ACDC leave-one-condition-out, BDD100K / WildDash2 trained on all of ACDC, paired
bootstrap over source images (2000 resamples, seed 0).

1. (M2, P0) Hit-criterion sensitivity (Appendix C) with ALL strong baselines: rule, MetaSeg-style, logistic regression
   on three box features, box-mask disagreement; plus logistic regression on the verifier's 25 features (LR-25).
2. (P0, H2) Classifier attribution: GBM-25 (ours) vs LR-25 vs logistic-3, same proposals / target / training data;
   matched FA at 0.1 ... 2, count-match b = 1 (label-free), three domains and the Faster R-CNN proposals
   (ACDC v20a and BDD v20b, verifier trained with RT-DETR proposals = frozen, as Table 5).
3. (E1, seeds) Training-seed variability of GBM-25 and LR-25 (negative subsampling + model seed), 5 seeds.
4. (H5) Low budgets: R@0.1 and R@0.25 (in part 2).
5. (H4) Other targets with fixed protocol: pooled confident omissions (into background or another thing) and all
   omitted objects. (a) scores trained for the primary target kept fixed; (b) every learned method retrained on the
   target in question. Strict FA = every flag that hits no object of the target in question.

usage: python e38_review2.py <v11> <v13> <v15> <v20a> <v20b> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e10_e12 import paired_ci  # noqa: E402
from e27_matching import CRITERIA, hits_mask  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, feature_sets, load  # noqa: E402

DET = ["rtdetr", "rtdetr_gamma"]
L3 = ["det_score", "p_self_stuff_frac", "p_ent_mean"]
FAS = (0.1, 0.25, 0.5, 1.0, 2.0)
GRID = np.linspace(0.1, 2.0, 20)
X = lambda D, c: D[c].fillna(-1).to_numpy(np.float32)  # noqa: E731


# ---------------------------------------------------------------- models
def fit(tr, cols, kind, target="pos_co_stuff", seed=0):
    rng = np.random.default_rng(seed)
    y = tr[target].to_numpy()
    keep = y | (rng.random(len(tr)) < 0.25)
    if kind == "gbm":
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed)
    else:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    m.fit(X(tr[keep], cols), y[keep])
    return lambda D: m.predict_proba(X(D, cols))[:, 1]


def loco_scores(D, T, cols, kind, target="pos_co_stuff", seed=0, apply_to=None):
    """leave-one-condition-out on ACDC; apply_to: other proposals of the same images (e.g. Faster R-CNN) to score."""
    A = T if apply_to is None else apply_to
    s = np.full(len(A), np.nan)
    for h in ADVERSE:
        tr = D[~D.condition.isin([h, h + "_ref"])]
        te = (A.condition == h).to_numpy()
        s[te] = fit(tr, cols, kind, target, seed)(A[te])
    return s


# ---------------------------------------------------------------- evaluation (generic positive column)
def found_at(T, s, keys, fa, pos):
    s = np.nan_to_num(np.asarray(s, float), nan=-np.inf)
    n = T.unit.nunique()
    s_neg = np.sort(s[~pos])[::-1]
    t = s_neg[min(int(fa * n), len(s_neg) - 1)]
    h = T[(s > t) & pos]
    f = set(zip(h.model, h.stem, h.gt_instance_id))
    return np.array([k in f for k in keys])


def row(T, s, keys, pos):
    r = {f"R@{fa}": round(float(found_at(T, s, keys, fa, pos).mean()), 4) for fa in FAS}
    r["FROC[0,2]"] = round(float(np.mean([found_at(T, s, keys, fa, pos).mean() for fa in GRID])), 4)
    return r


def count_match(T, s, keys, pos, b=1.0):
    s = np.nan_to_num(np.asarray(s, float), nan=-np.inf)
    n = T.unit.nunique()
    thr = np.sort(s)[::-1][min(int(b * n), len(s) - 1)]
    flag = s > thr
    h = T[flag & pos]
    f = set(zip(h.model, h.stem, h.gt_instance_id))
    return {"recall": round(float(np.mean([k in f for k in keys])), 4), "fa_strict": round(float((flag & ~pos).sum() / n), 3)}


def paired(T, a, b, keys, stems, pos, fa=1.0):
    d, ci, p = paired_ci(found_at(T, a, keys, fa, pos).astype(float), found_at(T, b, keys, fa, pos).astype(float), stems)
    return {"diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]}


def fmt(r):
    return " ".join(f"{k[2:] if k.startswith('R@') else 'FROC'} {100*v:5.1f}" for k, v in r.items())


# ---------------------------------------------------------------- data
def prep(d):
    P, O = load(d)
    D = P[P.det_src.isin(DET)].copy()
    D["stem"] = D.stem.astype(str)
    O = O.copy()
    O["stem"] = O.stem.astype(str)
    flags = O[["model", "stem", "condition", "instance_id", "omitted", "confident_omission"]].rename(
        columns={"instance_id": "gt_instance_id", "omitted": "o_omitted", "confident_omission": "o_co"})
    D = D.merge(flags, on=["model", "stem", "condition", "gt_instance_id"], how="left")
    for c in ("o_omitted", "o_co"):
        D[c] = D[c].fillna(False).astype(bool)
    cov = D.gt_cover.to_numpy() >= 0.5
    D["pos_co_any"] = D.o_co.to_numpy() & cov
    D["pos_omitted"] = D.o_omitted.to_numpy() & cov
    D["disagree"] = D.p_self_stuff_frac + 1e-6 * D.det_score
    return D.reset_index(drop=True), O


def target_keys(O, adverse_only, col):
    o = O[O.condition.isin(ADVERSE)] if adverse_only else O
    o = o[o[col]] if col != "co_stuff" else o[o.co_stuff]
    return list(zip(o.model, o.stem, o.instance_id)), o.stem.to_numpy()


def main():
    v11, v13, v15, v20a, v20b, out = sys.argv[1:7]
    DA, OA = prep(v11)
    fs = feature_sets(DA)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta = [c for c in DA.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    print("features: ours/LR-25", len(cheap), "meta", len(meta), flush=True)
    TA = DA[DA.condition.isin(ADVERSE)].reset_index(drop=True)
    res = {"n_features": {"ours": len(cheap), "meta": len(meta)}}

    def learned_scores(T, target="pos_co_stuff", seed=0, acdc=False, frozen_from=None):
        """ours / LR-25 / logistic-3 / meta: ACDC LOCO (acdc=True), else trained on all of ACDC."""
        out_ = {}
        for nm, cols, kind in (("ours", cheap, "gbm"), ("lr25", cheap, "lr"), ("logit3", L3, "lr"), ("meta", meta, "gbm")):
            if acdc:
                out_[nm] = loco_scores(DA, T, cols, kind, target, seed, apply_to=T)
            else:
                out_[nm] = fit(DA, cols, kind, target, seed)(T)
        return out_

    # ---------------- domains (primary target)
    print("loading BDD / WD / Faster R-CNN runs", flush=True)
    DB, OB = prep(v13)
    DW, OW = prep(v15)
    DFa, OFa = prep(v20a)
    DFb, OFb = prep(v20b)
    TFa = DFa[DFa.condition.isin(ADVERSE)].reset_index(drop=True)
    doms = {"ACDC": (TA, OA, True), "BDD100K": (DB, OB, False), "WildDash2": (DW, OW, False),
            "ACDC, Faster R-CNN (frozen)": (TFa, OFa, True), "BDD100K, Faster R-CNN (frozen)": (DFb, OFb, False)}
    S = {}
    for name, (T, O, acdc) in doms.items():
        S[name] = learned_scores(T, acdc=acdc)
        S[name]["rule"] = T.rule.to_numpy()
        S[name]["disagree"] = T.disagree.to_numpy()
        print("  scored", name, flush=True)

    # ---------------- part 2 + 4: classifier attribution, low budgets, count-match
    res["P2_classifier"] = {}
    for name, (T, O, acdc) in doms.items():
        keys, stems = target_keys(O, acdc, "co_stuff")
        pos = T.pos_co_stuff.to_numpy()
        rows = {k: row(T, s, keys, pos) for k, s in S[name].items()}
        cm = {k: count_match(T, s, keys, pos) for k, s in S[name].items()}
        pr = {f"ours-{k}": paired(T, S[name]["ours"], S[name][k], keys, stems, pos) for k in S[name] if k != "ours"}
        pr["lr25-logit3"] = paired(T, S[name]["lr25"], S[name]["logit3"], keys, stems, pos)
        pr["ours-rule@0.25"] = paired(T, S[name]["ours"], S[name]["rule"], keys, stems, pos, fa=0.25)
        pr["lr25-rule@0.25"] = paired(T, S[name]["lr25"], S[name]["rule"], keys, stems, pos, fa=0.25)
        res["P2_classifier"][name] = {"n_targets": len(keys), "rows": rows, "count_match_b1": cm, "paired_at1": pr}
        print(f"\n== P2 {name} (targets {len(keys)})")
        for k in rows:
            print(f"  {k:9s} {fmt(rows[k])} | cm@1 R {100*cm[k]['recall']:5.1f} FA {cm[k]['fa_strict']:.2f}")
        for k, v in pr.items():
            print(f"  {k:16s} {100*v['diff']:+5.1f} [{100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}]")

    # ---------------- part 1: hit criteria with all strong baselines (three main domains)
    res["P1_matching"] = {}
    for name in ("ACDC", "BDD100K", "WildDash2"):
        T, O, acdc = doms[name]
        keys, stems = target_keys(O, acdc, "co_stuff")
        res["P1_matching"][name] = {}
        print(f"\n== P1 {name}")
        for crit in CRITERIA:
            hit = {k: hits_mask(T, np.asarray(s, float), crit) for k, s in S[name].items()}
            r1 = {k: round(float(found_at(T, s, keys, 1.0, hit[k]).mean()), 4) for k, s in S[name].items()}
            base = ["rule", "meta", "logit3", "disagree"]
            strongest = max(base, key=lambda k: r1[k])
            d, ci, _ = paired_ci(found_at(T, S[name]["ours"], keys, 1.0, hit["ours"]).astype(float),
                                 found_at(T, S[name][strongest], keys, 1.0, hit[strongest]).astype(float), stems)
            d2, ci2, _ = paired_ci(found_at(T, S[name]["ours"], keys, 1.0, hit["ours"]).astype(float),
                                   found_at(T, S[name]["lr25"], keys, 1.0, hit["lr25"]).astype(float), stems)
            res["P1_matching"][name][crit] = {"R@1": r1, "strongest_baseline": strongest,
                                              "ours_vs_strongest": {"diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]},
                                              "ours_vs_lr25": {"diff": round(d2, 4), "ci95": [round(ci2[0], 4), round(ci2[1], 4)]}}
            print(f"  {crit:13s} " + " ".join(f"{k}={100*v:.1f}" for k, v in r1.items())
                  + f" | vs {strongest} {100*d:+.1f} [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}] | vs lr25 {100*d2:+.1f} [{100*ci2[0]:+.1f}, {100*ci2[1]:+.1f}]",
                  flush=True)

    # ---------------- part 3: seeds
    res["P3_seeds"] = {}
    print("\n== P3 seeds (R@1)")
    for name in ("ACDC", "BDD100K", "WildDash2"):
        T, O, acdc = doms[name]
        keys, _ = target_keys(O, acdc, "co_stuff")
        pos = T.pos_co_stuff.to_numpy()
        r = {}
        for nm, kind in (("ours", "gbm"), ("lr25", "lr")):
            vals = []
            for seed in range(5):
                s = loco_scores(DA, T, cheap, kind, seed=seed, apply_to=T) if acdc else fit(DA, cheap, kind, seed=seed)(T)
                vals.append(float(found_at(T, s, keys, 1.0, pos).mean()))
            r[nm] = {"R@1": [round(v, 4) for v in vals], "mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals, ddof=1)), 4)}
        res["P3_seeds"][name] = r
        print(f"  {name:9s} " + " | ".join(f"{k} {100*v['mean']:.1f} +- {100*v['sd']:.1f} (min {100*min(v['R@1']):.1f}, max {100*max(v['R@1']):.1f})" for k, v in r.items()), flush=True)

    # ---------------- part 5: other targets
    res["P5_targets"] = {}
    for name in ("ACDC", "BDD100K", "WildDash2"):
        T, O, acdc = doms[name]
        res["P5_targets"][name] = {}
        for tcol, ocol in (("pos_co_any", "confident_omission"), ("pos_omitted", "omitted")):
            keys, stems = target_keys(O, acdc, ocol)
            pos = T[tcol].to_numpy()
            fixed = {k: row(T, s, keys, pos)["R@1.0"] for k, s in S[name].items()}
            Sr = learned_scores(T, target=tcol, acdc=acdc)
            Sr["rule"], Sr["disagree"] = S[name]["rule"], S[name]["disagree"]
            retr = {k: row(T, s, keys, pos)["R@1.0"] for k, s in Sr.items()}
            base = ["rule", "meta", "logit3", "disagree"]
            st = max(base, key=lambda k: retr[k])
            pr = paired(T, Sr["ours"], Sr[st], keys, stems, pos)
            prf = paired(T, S[name]["ours"], Sr[st], keys, stems, pos)
            res["P5_targets"][name][tcol] = {"n_targets": len(keys), "R@1_fixed_primary_scores": fixed,
                                             "R@1_retrained": retr, "strongest_retrained_baseline": st,
                                             "ours_retrained_vs_strongest": pr, "ours_fixed_vs_strongest_retrained": prf}
            print(f"\n== P5 {name} target {tcol} (n {len(keys)})")
            print("  fixed   " + " ".join(f"{k}={100*v:.1f}" for k, v in fixed.items()))
            print("  retrain " + " ".join(f"{k}={100*v:.1f}" for k, v in retr.items())
                  + f" | ours(retr) vs {st} {100*pr['diff']:+.1f} [{100*pr['ci95'][0]:+.1f}, {100*pr['ci95'][1]:+.1f}]"
                  + f" | ours(fixed) vs {st} {100*prf['diff']:+.1f} [{100*prf['ci95'][0]:+.1f}, {100*prf['ci95'][1]:+.1f}]", flush=True)

    res["note"] = {"bootstrap": "paired_ci: 2000 resamples of source images, seed 0",
                   "frozen_frcnn": "verifier and baselines trained on RT-DETR proposals (v11), LOCO on ACDC, all ACDC for BDD"}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
