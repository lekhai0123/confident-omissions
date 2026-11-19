"""E19: ablation of the main configuration (v11, ACDC LOCO, strict false alarms, matched FA, all targets).

Rows: full single-model verifier; minus feature groups (detector cues, segmenter uncertainty, Mahalanobis,
geometry, segmenter thing-probability / stuff fraction); without the gamma-brightened detector pass; generic
training target (any omitted object instead of confident omission into background); logistic regression instead of
GBM; plus Maag proposals. Paired bootstrap vs full at 1.0 FA/img.

usage: python e19_ablation.py <v11 dir> <out json>
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
from e18_comparison import found_at, score_row  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, feature_sets, load  # noqa: E402


def loco(D, cols, target="pos_co_stuff", model="gbm"):
    s = pd.Series(np.nan, index=D.index)
    for h in ADVERSE:
        tr = D[~D.condition.isin([h, h + "_ref"])]
        te = D.condition == h
        rng = np.random.default_rng(0)
        keep = tr[target].to_numpy() | (rng.random(len(tr)) < 0.25)
        X, y = tr[keep][cols].fillna(-1).to_numpy(np.float32), tr[keep][target].to_numpy()
        m = (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
             if model == "gbm" else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")))
        m.fit(X, y)
        s[te] = m.predict_proba(D[te][cols].fillna(-1).to_numpy(np.float32))[:, 1]
    return s


def main():
    v11, out = sys.argv[1:3]
    P, O = load(v11)
    P["src_maag"] = (P.det_src == "maag").astype(float)
    P["pos_any"] = P.pos & (P.gt_cover >= 0.5) if "pos" in P else P.pos_co_stuff
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    groups = {
        "detector cues (det score, detector-thing/segmenter-stuff)": [c for c in cheap if c.startswith(("det_score", "p_det_stuff", "src_"))],
        "segmenter uncertainty (entropy, margin)": [c for c in cheap if c.startswith(("p_ent", "p_margin"))],
        "Mahalanobis": [c for c in cheap if c.startswith("p_maha")],
        "segmenter thing-probability / stuff fraction": [c for c in cheap if c.startswith(("p_thing_prob", "p_self_stuff"))],
        "geometry (area, box, position, fill)": [c for c in cheap if c in ("p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill")],
    }
    det = P.det_src.isin(["rtdetr", "rtdetr_gamma"])
    D = P[det].copy()
    keys_df = O[O.condition.isin(ADVERSE) & O.co_stuff]
    keys = list(zip(keys_df.model, keys_df.stem, keys_df.instance_id))
    stems = keys_df.stem.to_numpy()
    T = D[D.condition.isin(ADVERSE)]
    runs = {"Full (main configuration)": (D, loco(D, cheap))}
    for g, cols in groups.items():
        runs[f"- {g}"] = (D, loco(D, [c for c in cheap if c not in cols]))
    Dn = P[P.det_src == "rtdetr"].copy()
    runs["- gamma-brightened detector pass"] = (Dn, loco(Dn, [c for c in cheap if c != "src_rtdetr_gamma"]))
    runs["generic target (any omitted object)"] = (D, loco(D, cheap, target="pos_any"))
    runs["logistic regression instead of GBM"] = (D, loco(D, cheap, model="lr"))
    Dm = P[det | (P.det_src == "maag")].copy()
    fg = [c for c in P.columns if c.startswith("p_fgbg")]
    runs["+ Maag depth proposals and fg/bg features"] = (Dm, loco(Dm, cheap + fg + ["src_maag"]))
    rows = {}
    f_full = None
    for name, (Dx, s) in runs.items():
        Tx = Dx[Dx.condition.isin(ADVERSE)]
        sx = s[Tx.index].to_numpy()
        r = score_row(Tx, sx, keys)
        f = found_at(Tx, sx, keys, 1.0).astype(float)
        if f_full is None:
            f_full = f
        else:
            d, ci, p = paired_ci(f, f_full, stems)
            r.update({"delta_vs_full@1.0": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]})
        rows[name] = r
        extra = f"  Δ {100*r['delta_vs_full@1.0']:+.1f} [{100*r['ci95'][0]:+.1f}, {100*r['ci95'][1]:+.1f}]" if "ci95" in r else ""
        print(f"{name:62s} R@0.5 {100*r['R@0.5']:5.1f}  R@1 {100*r['R@1.0']:5.1f}  R@2 {100*r['R@2.0']:5.1f}  FROC {100*r['FROC[0,2]']:5.1f}{extra}", flush=True)
    json.dump({"rows": rows, "groups": groups}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
