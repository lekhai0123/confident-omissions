"""E26: detector swap on BDD100K (v20b: torchvision Faster R-CNN R50-FPN v2 instead of RT-DETR), WRITING_KIT section 0.

Same 3000 BDD images, same four segmenters, same proposal budget and COCO class mapping; only the detector differs
(the proposal slots keep the names rtdetr / rtdetr_gamma for the normal and the gamma pass).

(b) FROZEN verifier: trained on ACDC with RT-DETR proposals (v11), applied unchanged to Faster R-CNN proposals.
    Same for the MetaSeg-style baseline; raw signals and the rule are recomputed on the new proposals.
(a) RETRAINED verifier: the locked protocol asks for ACDC LOCO with the new proposals; that needs the ACDC run
    (own accounts only, pending). As a supplementary within-BDD check we report a 2-fold-by-image retraining on
    Faster R-CNN proposals next to the same 2-fold retraining on RT-DETR proposals (v13).
Also proposal coverage of the targets and the E18 rows for RT-DETR on the same images (reference).

usage: python e26_detswap.py <v11 dir> <v13 dir (RT-DETR)> <v20b dir (Faster R-CNN)> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e18_comparison import gbm, score_row, table  # noqa: E402
from e23_wilddash import count_match  # noqa: E402
from e_suite_v8 import MODELS, feature_sets, load  # noqa: E402

DET = ["rtdetr", "rtdetr_gamma"]


def two_fold(T, cols):
    stems = np.array(sorted(T.stem.unique()))
    half = set(np.random.default_rng(0).permutation(stems)[: len(stems) // 2])
    s = np.zeros(len(T))
    for fold in (True, False):
        tr = T.stem.isin(half).to_numpy() == fold
        s[~tr] = gbm(T[tr], cols).predict_proba(T[~tr][cols].fillna(-1).to_numpy(np.float32))[:, 1]
    return s


def main():
    v11, v13, v20b, out = sys.argv[1:5]
    P, _ = load(v11)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta_unc = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    Dd = P[P.det_src.isin(DET)]
    m_ours, m_meta = gbm(Dd, cheap), gbm(Dd, meta_unc)
    X = lambda D, c: D[c].fillna(-1).to_numpy(np.float32)  # noqa: E731

    res = {}
    for name, d in (("RT-DETR (v13, reference)", v13), ("Faster R-CNN (v20b)", v20b)):
        PB, OB = load(d)
        T = PB[PB.det_src.isin(DET)].copy()
        kdf = OB[OB.co_stuff]
        keys = list(zip(kdf.model, kdf.stem, kdf.instance_id))
        s_ours, s_meta = m_ours.predict_proba(X(T, cheap))[:, 1], m_meta.predict_proba(X(T, meta_unc))[:, 1]
        rows, paired = table(T, keys, {"MetaSeg-style meta-classifier (uncertainty + geometry, GBM)": s_meta,
                                       "Ours (CO verifier, single model)": s_ours}, {}, kdf.stem.to_numpy())
        s_2f = two_fold(T, cheap)
        rows["Ours, retrained on this detector (BDD 2-fold, supplementary)"] = score_row(T, s_2f, keys)
        cov = set(zip(*[T[T.pos_co_stuff][c] for c in ("model", "stem", "gt_instance_id")]))
        props_per_unit = len(T) / T.unit.nunique()
        r = {"n_targets": len(keys), "units": int(T.unit.nunique()), "proposals_per_unit": round(float(props_per_unit), 1),
             "proposal_coverage": round(float(np.mean([k in cov for k in keys])), 4), "table": rows, "paired": paired,
             "count_match": {"ours_frozen": count_match(T, s_ours, keys), "ours_2fold": count_match(T, s_2f, keys),
                             "rule": count_match(T, T.rule.to_numpy(), keys), "metaseg": count_match(T, s_meta, keys)}}
        res[name] = r
        print(f"\n== {name}: targets {len(keys)}, proposals/unit {props_per_unit:.1f}, coverage {r['proposal_coverage']:.3f}")
        for n, a in rows.items():
            print(f"{n[:62]:62s} | {100*a['R@0.5']:5.1f} {100*a['R@1.0']:5.1f} {100*a['R@2.0']:5.1f} {100*a['FROC[0,2]']:5.1f}  {a.get('variant', '')}")
        print("paired:", paired)
        for n, cm in r["count_match"].items():
            print(f"count-match {n:12s} " + "  ".join(f"[{k}] R {100*v['recall']:.1f} FA {v['fa_strict']:.2f}" for k, v in cm.items()))
    res["note"] = ("frozen = trained on ACDC with RT-DETR proposals (v11); same 3000 BDD images; ACDC LOCO with the new "
                   "detector (protocol (a)) pending")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
