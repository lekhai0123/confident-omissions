"""E38b: simple scorers on the unseen-CNN rows of Table 5 (review2 J5.2: strongest simple baselines in every setting).

Same data and training as E25 (BDD100K, v19b; trained on all of ACDC v11) and E31 (ACDC, v19c; leave-one-condition-out
on v11); the segmenter one-hot is zero; cross-model disagreement is left out (between the two CNNs). Adds the logistic
regression on three box features and LR on the verifier's features (LR-25) to the verifier, the rule and the
MetaSeg-style baseline: recall at 1 matched strict false alarm per image, count-match b = 1, paired differences.

usage: python e38b_cnn_simple.py <v11> <v19b> <v19c> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e25_cnn_unseen import CNNS, load_tags  # noqa: E402
from e38_review2 import L3, count_match, fit, found_at, paired  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, feature_sets, load  # noqa: E402

DET = ["rtdetr", "rtdetr_gamma"]


def main():
    v11, v19b, v19c, out = sys.argv[1:5]
    P, _ = load(v11)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    Dd = P[P.det_src.isin(DET)]
    res = {}
    for name, d, acdc in (("BDD100K CNNs", v19b, False), ("ACDC CNNs", v19c, True)):
        C, OC = load_tags(d, CNNS)
        T = C[C.det_src.isin(DET) & (C.condition.isin(ADVERSE) if acdc else True)].copy().reset_index(drop=True)
        for c in ("p_xmodel_js_mean", "p_xmodel_js_max"):
            T[c] = np.nan
        S = {}
        for nm, cols, kind in (("ours", cheap, "gbm"), ("lr25", cheap, "lr"), ("logit3", L3, "lr"), ("meta", meta, "gbm")):
            if acdc:
                s = np.zeros(len(T))
                for h in ADVERSE:
                    te = (T.condition == h).to_numpy()
                    s[te] = fit(Dd[~Dd.condition.isin([h, h + "_ref"])], cols, kind)(T[te])
            else:
                s = fit(Dd, cols, kind)(T)
            S[nm] = s
        S["rule"] = T.rule.to_numpy()
        res[name] = {}
        for scope, tags in (("both", CNNS), *((t, [t]) for t in CNNS)):
            sel = T.model.isin(tags).to_numpy()
            Ts = T[sel]
            k = OC[OC.co_stuff & OC.model.isin(tags) & (OC.condition.isin(ADVERSE) if acdc else True)]
            keys, stems = list(zip(k.model, k.stem, k.instance_id)), k.stem.to_numpy()
            pos = Ts.pos_co_stuff.to_numpy()
            r1 = {m: round(float(found_at(Ts, s[sel], keys, 1.0, pos).mean()), 4) for m, s in S.items()}
            r05 = {m: round(float(found_at(Ts, s[sel], keys, 0.5, pos).mean()), 4) for m, s in S.items()}
            cm = {m: count_match(Ts, s[sel], keys, pos) for m, s in S.items()}
            base = ["rule", "meta", "logit3"]
            st = max(base, key=lambda m: r1[m])
            pr = {f"ours-{m}": paired(Ts, S["ours"][sel], S[m][sel], keys, stems, pos) for m in base + ["lr25"]}
            res[name][scope] = {"n_targets": len(keys), "R@1": r1, "R@0.5": r05, "count_match_b1": cm,
                                "strongest_baseline": st, "paired_at1": pr}
            print(f"{name:13s} {scope:20s} n {len(keys):4d} | R@1 " + " ".join(f"{m}={100*v:.1f}" for m, v in r1.items())
                  + f" | strongest {st} | " + " ".join(f"{m} {100*v['diff']:+.1f} [{100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}]" for m, v in pr.items()),
                  flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
