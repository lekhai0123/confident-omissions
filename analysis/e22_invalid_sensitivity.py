"""E22: sensitivity of the ACDC comparison (E18) to the annotators' uncertain regions (ACDC invalid masks).

Main analyses keep every object as a target (conservative). Here objects whose pixels lie mostly (> 50%) in an
invalid region are removed from the targets, and proposals matched to such an object are ignored (neither a hit
nor a false alarm). Same scorers and protocol as E18 (shared proposals, strict false alarms, matched FA).

usage: python e22_invalid_sensitivity.py <v11 dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e10_e12 import paired_ci  # noqa: E402
from e18_comparison import best_raw, found_at, loco, score_row  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, feature_sets, load  # noqa: E402


def main():
    v11, out = sys.argv[1:3]
    P, O = load(v11)
    inv = pd.concat([pd.read_csv(os.path.join(v11, f"objects_{m}.csv"), usecols=["stem", "condition", "instance_id", "frac_invalid"])
                     .assign(model=m) for m in MODELS])
    O = O.merge(inv, on=["model", "stem", "condition", "instance_id"], how="left")
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta_unc = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    D = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"])].copy()
    s_ours, s_meta = loco(D, cheap), loco(D, meta_unc)
    T = D[D.condition.isin(ADVERSE)].copy()
    T["s_ours"], T["s_meta"] = s_ours[T.index], s_meta[T.index]
    res = {}
    for variant in ("all targets (main)", "without objects mostly in uncertain regions"):
        Ov = O[O.condition.isin(ADVERSE) & O.co_stuff]
        Tv = T
        if variant.startswith("without"):
            bad = O[O.condition.isin(ADVERSE) & (O.frac_invalid.fillna(0) > 0.5)]
            bad_keys = set(zip(bad.model, bad.stem, bad.condition, bad.instance_id))
            Ov = Ov[~pd.Series(list(zip(Ov.model, Ov.stem, Ov.condition, Ov.instance_id)), index=Ov.index).isin(bad_keys)]
            drop = pd.Series(list(zip(T.model, T.stem, T.condition, T.gt_instance_id)), index=T.index).isin(bad_keys) & (T.gt_cover >= 0.5)
            Tv = T[~drop]
        keys = list(zip(Ov.model, Ov.stem, Ov.instance_id))
        rows = {"n_targets": len(keys)}
        rows["Ours"] = score_row(Tv, Tv.s_ours.to_numpy(), keys)
        rows["MetaSeg-style"] = score_row(Tv, Tv.s_meta.to_numpy(), keys)
        rows["Rule"] = score_row(Tv, Tv.rule.to_numpy(), keys)
        b = best_raw(Tv, "xmodel_js", keys)
        rows["Ensemble disagreement"] = b[0]
        d, ci, p = paired_ci(found_at(Tv, Tv.s_ours.to_numpy(), keys, 1.0).astype(float),
                             found_at(Tv, Tv.rule.to_numpy(), keys, 1.0).astype(float), Ov.stem.to_numpy())
        rows["ours_minus_rule@1"] = {"diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]}
        res[variant] = rows
        print(f"== {variant}: {len(keys)} targets")
        for k in ("Ours", "MetaSeg-style", "Rule", "Ensemble disagreement"):
            r = rows[k]
            print(f"  {k:22s} R@0.5 {100*r['R@0.5']:5.1f}  R@1 {100*r['R@1.0']:5.1f}  R@2 {100*r['R@2.0']:5.1f}  FROC {100*r['FROC[0,2]']:5.1f}")
        print(f"  ours - rule @1: {100*d:+.1f} [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
