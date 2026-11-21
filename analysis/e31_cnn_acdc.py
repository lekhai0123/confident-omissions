"""E31: per-pixel CNNs (DeepLabV3+ R101 / R18) on ACDC (v19c), WRITING_KIT section 0 (CNN part, ACDC).

1. Failure statistics on ACDC (Table 1 columns): mIoU on the adverse conditions (mean over fog/night/rain/snow),
   share of omitted thing objects (adverse), confident share on reference and adverse images.
2. Unseen segmenter on ACDC: the verifier and the MetaSeg-style baseline are trained on the four v11 segmenters,
   leave-one-condition-out (held-out condition and its reference frames excluded), and applied to the CNNs'
   proposals of the held-out condition; segmenter one-hot = 0. Same E18 protocol (shared proposals, matched strict
   false alarms on the test set, best-of-four for raw scores) plus count-match (b = 0.5 / 1 / 2).

usage: python e31_cnn_acdc.py <v11 dir> <v19c dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e18_comparison import gbm, table  # noqa: E402
from e23_wilddash import count_match  # noqa: E402
from e25_cnn_unseen import CNNS, load_tags  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, STUFF, feature_sets, load  # noqa: E402


def main():
    v11, v19c, out = sys.argv[1:4]
    P, _ = load(v11)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta_unc = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    Dd = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"])]
    X = lambda D, c: D[c].fillna(-1).to_numpy(np.float32)  # noqa: E731

    C, OC = load_tags(v19c, CNNS)
    summ = json.load(open(os.path.join(v19c, "summary.json"), encoding="utf-8"))
    stats = {}
    for m in CNNS:
        o = OC[OC.model == m]
        adv, ref = o[o.condition.isin(ADVERSE)], o[o.condition.isin([c + "_ref" for c in ADVERSE])]
        mi = summ["models"][m]["miou_by_condition"]
        stats[m] = {"miou_adv_mean": round(100 * float(np.mean([mi[c]["argmax"]["miou"] for c in ADVERSE])), 1),
                    "miou_ref_mean": round(100 * float(np.mean([mi[c + "_ref"]["argmax"]["miou"] for c in ADVERSE])), 1),
                    "omitted_adv": round(100 * float(adv.omitted.mean()), 1),
                    "cs_adv": round(100 * float(adv.co_stuff.sum() / max(1, adv.omitted.sum())), 1),
                    "cs_ref": round(100 * float(ref.co_stuff.sum() / max(1, ref.omitted.sum())), 1),
                    "n_co_stuff_adv": int(adv.co_stuff.sum()),
                    "cs_adv_by_condition": {c: round(100 * float(adv[adv.condition == c].co_stuff.sum() /
                                                                 max(1, adv[adv.condition == c].omitted.sum())), 1) for c in ADVERSE}}
        print(m, stats[m], flush=True)

    T = C[C.det_src.isin(["rtdetr", "rtdetr_gamma"]) & C.condition.isin(ADVERSE)].copy()
    s_ours, s_meta = np.zeros(len(T)), np.zeros(len(T))
    for h in ADVERSE:
        tr = Dd[~Dd.condition.isin([h, h + "_ref"])]
        te = (T.condition == h).to_numpy()
        s_ours[te] = gbm(tr, cheap).predict_proba(X(T[te], cheap))[:, 1]
        s_meta[te] = gbm(tr, meta_unc).predict_proba(X(T[te], meta_unc))[:, 1]
        print("fold", h, int(te.sum()), flush=True)
    T["p_xmodel_js_mean"] = np.nan        # cross-model disagreement between the two CNNs is not comparable
    T["p_xmodel_js_max"] = np.nan
    res = {"failure_stats": stats, "per_scope": {}}
    for scope, tags in (("both CNNs", CNNS), *((t, [t]) for t in CNNS)):
        sel = T.model.isin(tags).to_numpy()
        Ts = T[sel]
        kdf = OC[OC.co_stuff & OC.model.isin(tags) & OC.condition.isin(ADVERSE)]
        keys = list(zip(kdf.model, kdf.stem, kdf.instance_id))
        rows, paired = table(Ts, keys, {"MetaSeg-style meta-classifier (uncertainty + geometry, GBM)": s_meta[sel],
                                        "Ours (CO verifier, single model)": s_ours[sel]}, {}, kdf.stem.to_numpy())
        cov = set(zip(*[Ts[Ts.pos_co_stuff][c] for c in ("model", "stem", "gt_instance_id")]))
        r = {"n_targets": len(keys), "proposal_coverage": round(float(np.mean([k in cov for k in keys])), 4), "table": rows,
             "paired": paired, "count_match": {"ours": count_match(Ts, s_ours[sel], keys), "rule": count_match(Ts, Ts.rule.to_numpy(), keys),
                                               "metaseg": count_match(Ts, s_meta[sel], keys)}}
        res["per_scope"][scope] = r
        print(f"\n== {scope}: targets {len(keys)}, coverage {r['proposal_coverage']:.3f}")
        for n, a in rows.items():
            print(f"{n[:62]:62s} | {100*a['R@0.5']:5.1f} {100*a['R@1.0']:5.1f} {100*a['R@2.0']:5.1f} {100*a['FROC[0,2]']:5.1f}")
        print("paired:", paired)
        for n, cm in r["count_match"].items():
            print(f"count-match {n:8s} " + "  ".join(f"[{k}] R {100*v['recall']:.1f} FA {v['fa_strict']:.2f}" for k, v in cm.items()))
    res["note"] = "verifier trained on the 4 v11 segmenters, leave-one-condition-out; CNNs unseen (one-hot zeros); ACDC 3009 images (v19c)"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
