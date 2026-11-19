"""E18: unified comparison table (main paper table) - every method scored under one protocol.

Setting "shared proposals": every scorer ranks the SAME candidate regions (RT-DETR normal + gamma proposals of the
monitored segmenter), so differences come from the scoring alone. This favours the baselines (they get the
detector's candidate regions for free). Each raw signal is evaluated with its mean and max aggregate and both
orientations; the best of the four is reported (again favourable to the baselines).
Native rows: Maag & Rottmann proposals (depth fg/bg), unscored and with their meta-classifier.

Metric: recall of confident omissions into background (all target objects, uncovered ones count as misses) at a
matched number of STRICT false alarms per image x monitored model (test-set threshold, identical rule for all
methods), at 0.5 / 1.0 / 2.0, plus the normalised FROC area over [0, 2]. Paired bootstrap over images: ours vs
the strongest baseline at 1.0.

ACDC: learned methods with leave-one-condition-out (v11); evaluation on the four adverse conditions.
BDD: learned methods trained on all of ACDC (v11), applied to BDD (v13) without retraining.

usage: python e18_comparison.py <v11 dir> <v13 dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e10_e12 import paired_ci  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, feature_sets, load  # noqa: E402

FA = (0.5, 1.0, 2.0)
GRID = np.linspace(0.1, 2.0, 20)
RAW = {  # family label -> column stem (p_<stem>_mean / _max)
    "Entropy (pixel uncertainty)": "ent",
    "Margin / MSP (pixel uncertainty)": "margin",
    "Energy (Liu et al. 2020)": "energy",
    "RbA score (Nayal et al. 2023)": "rba",
    "Mahalanobis, predicted class (Lee et al. 2018)": "maha_pred",
    "Mahalanobis, min over classes": "maha_min",
    "Ensemble disagreement (cross-model JS)": "xmodel_js",
    "Maag & Rottmann fg/bg on shared proposals": "fgbg_stuff",
}


def gbm(tr, cols, seed=0):
    rng = np.random.default_rng(seed)
    keep = tr.pos_co_stuff.to_numpy() | (rng.random(len(tr)) < 0.25)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed)
    m.fit(tr[keep][cols].fillna(-1).to_numpy(np.float32), tr[keep].pos_co_stuff.to_numpy())
    return m


def loco(D, cols):
    s = pd.Series(np.nan, index=D.index)
    for h in ADVERSE:
        tr = D[~D.condition.isin([h, h + "_ref"])]
        te = D.condition == h
        s[te] = gbm(tr, cols).predict_proba(D[te][cols].fillna(-1).to_numpy(np.float32))[:, 1]
    return s


def found_at(T, s, keys, fa):
    s = np.nan_to_num(np.asarray(s, float), nan=-np.inf)
    neg = ~T.pos_co_stuff.to_numpy()
    n = T.unit.nunique()
    s_neg = np.sort(s[neg])[::-1]
    t = s_neg[min(int(fa * n), len(s_neg) - 1)]
    hit = T[(s > t) & T.pos_co_stuff.to_numpy()]
    f = set(zip(hit.model, hit.stem, hit.gt_instance_id))
    return np.array([k in f for k in keys])


def score_row(T, s, keys):
    r = {f"R@{fa}": float(found_at(T, s, keys, fa).mean()) for fa in FA}
    r["FROC[0,2]"] = float(np.mean([found_at(T, s, keys, fa).mean() for fa in GRID]))
    return r


def best_raw(T, stem, keys):
    best = None
    for agg in ("mean", "max"):
        c = f"p_{stem}_{agg}"
        if c not in T or T[c].notna().mean() < 0.5:
            continue
        for sign in (1, -1):
            r = score_row(T, sign * T[c].to_numpy(), keys)
            if best is None or r["FROC[0,2]"] > best[0]["FROC[0,2]"]:
                best = (r, sign * T[c].to_numpy(), f"{'+' if sign > 0 else '-'}{agg}")
    return best


def table(T, keys, learned: dict, native: dict, stems):
    rows, scores = {}, {}
    for name, stem in RAW.items():
        b = best_raw(T, stem, keys)
        if b:
            rows[name] = dict(b[0], variant=b[2])
            scores[name] = b[1]
    rows["Detector confidence alone"] = score_row(T, T.det_score.to_numpy(), keys)
    scores["Detector confidence alone"] = T.det_score.to_numpy()
    rows["Rule: detector x segmenter-stuff x (1-entropy)"] = score_row(T, T.rule.to_numpy(), keys)
    scores["Rule: detector x segmenter-stuff x (1-entropy)"] = T.rule.to_numpy()
    for name, s in learned.items():
        rows[name] = score_row(T, s, keys)
        scores[name] = s
    for name, (Tn, sn) in native.items():
        rows[name] = score_row(Tn, sn, keys)
        rows[name]["native"] = True
    ours = "Ours (CO verifier, single model)"
    base = [n for n in scores if not n.startswith("Ours")]
    strongest = max(base, key=lambda n: rows[n]["R@1.0"])
    fa_o = found_at(T, scores[ours], keys, 1.0)
    fa_b = found_at(T, scores[strongest], keys, 1.0)
    d, ci, p = paired_ci(fa_o.astype(float), fa_b.astype(float), stems)
    paired = {"vs": strongest, "diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)], "p_one_sided": round(p, 4)}
    return rows, paired


def main():
    v11, v13, out = sys.argv[1:4]
    P, O = load(v11)
    P["src_maag"] = (P.det_src == "maag").astype(float)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    fgbg = [c for c in P.columns if c.startswith("p_fgbg")]
    meta_unc = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    meta_maag = meta_unc + [c for c in fgbg]
    det = P.det_src.isin(["rtdetr", "rtdetr_gamma"])
    D = P[det | (P.det_src == "maag")]
    keys_df = O[O.condition.isin(ADVERSE) & O.co_stuff]
    keys = list(zip(keys_df.model, keys_df.stem, keys_df.instance_id))
    stems = keys_df.stem.to_numpy()
    Dd = P[det].copy()
    learned = {
        "MetaSeg-style meta-classifier (uncertainty + geometry, GBM)": loco(Dd, meta_unc),
        "Ours (CO verifier, single model)": loco(Dd, cheap),
    }
    Dm = D.copy()
    s_om = loco(Dm, cheap + fgbg + ["src_maag"])
    Ma = P[P.det_src == "maag"].copy()
    native = {"Maag & Rottmann + meta-classifier (native)": (Ma[Ma.condition.isin(ADVERSE)], loco(Ma, meta_maag)[Ma.condition.isin(ADVERSE)].to_numpy())}
    T = Dd[Dd.condition.isin(ADVERSE)]
    learned_t = {k: v[T.index].to_numpy() for k, v in learned.items()}
    rows_acdc, paired_acdc = table(T, keys, learned_t, native, stems)
    Tm = Dm[Dm.condition.isin(ADVERSE)]
    rows_acdc["Ours + Maag proposals"] = score_row(Tm, s_om[Tm.index].to_numpy(), keys)

    # Maag fidelity / native operating point: all unscored proposals. Their proposals only exist where the segmenter
    # predicts no thing, so their natural target is omission into background (any confidence).
    Mad = Ma[Ma.condition.isin(ADVERSE)]
    om_bg = O[O.condition.isin(ADVERSE) & O.omitted & O.predicted_as.isin(__import__("e_suite_v8").STUFF)]
    fid = {"maag_props_per_unit": round(float(len(Mad) / max(1, T.unit.nunique())), 3),
           "maag_strict_fa_per_unit": round(float((~Mad.pos_co_stuff).sum() / max(1, T.unit.nunique())), 3)}
    for cv in (0.5, 0.25, 0.1):
        mt = Mad[Mad.gt_cover >= cv]
        cov = set(zip(mt.model, mt.stem, mt.gt_instance_id))
        fid[f"cover{cv}_co_stuff"] = round(float(np.mean([k in cov for k in keys])), 4)
        fid[f"cover{cv}_omitted_into_bg_any_conf"] = round(float(np.mean([k in cov for k in zip(om_bg.model, om_bg.stem, om_bg.instance_id)])), 4)
    rows_acdc["Maag & Rottmann proposals, unscored (native operating point)"] = {
        "R@native": fid["cover0.5_co_stuff"], "FA_native": fid["maag_strict_fa_per_unit"], "native": True}

    # BDD: learned on all ACDC, applied to BDD
    PB, OB = load(v13)
    TB = PB[PB.det_src.isin(["rtdetr", "rtdetr_gamma"])].copy()
    kb_df = OB[OB.co_stuff]
    kb = list(zip(kb_df.model, kb_df.stem, kb_df.instance_id))
    learned_b = {"MetaSeg-style meta-classifier (uncertainty + geometry, GBM)": gbm(Dd, meta_unc).predict_proba(TB[meta_unc].fillna(-1).to_numpy(np.float32))[:, 1],
                 "Ours (CO verifier, single model)": gbm(Dd, cheap).predict_proba(TB[cheap].fillna(-1).to_numpy(np.float32))[:, 1]}
    rows_bdd, paired_bdd = table(TB, kb, learned_b, {}, kb_df.stem.to_numpy())

    res = {"ACDC": rows_acdc, "ACDC_paired": paired_acdc, "BDD": rows_bdd, "BDD_paired": paired_bdd, "maag_fidelity": fid,
           "n_targets": {"ACDC": len(keys), "BDD": len(kb)}}
    json.dump(res, open(out, "w"), indent=1)
    names = list(rows_acdc)
    print(f"{'method':62s} | ACDC R@0.5 R@1 R@2 FROC | BDD R@0.5 R@1 R@2 FROC")
    for n in names:
        a = rows_acdc[n]; b = rows_bdd.get(n)
        if "R@native" in a:
            print(f"{n[:62]:62s} | R {100*a['R@native']:.1f} at {a['FA_native']:.2f} strict FA/img (all proposals)")
            continue
        bs = f"{100*b['R@0.5']:5.1f} {100*b['R@1.0']:5.1f} {100*b['R@2.0']:5.1f} {100*b['FROC[0,2]']:5.1f}" if b else "   -"
        print(f"{n[:62]:62s} | {100*a['R@0.5']:5.1f} {100*a['R@1.0']:5.1f} {100*a['R@2.0']:5.1f} {100*a['FROC[0,2]']:5.1f} | {bs}  {a.get('variant','')}")
    print("paired ACDC:", paired_acdc)
    print("paired BDD:", paired_bdd)
    print("Maag fidelity:", fid, " targets:", res["n_targets"])


if __name__ == "__main__":
    main()
