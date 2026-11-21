"""E33: detector swap on ACDC (v20a: torchvision Faster R-CNN R50-FPN v2 instead of RT-DETR), WRITING_KIT section 0.

Same 3009 ACDC images, four segmenters, proposal budget and COCO class mapping as v11; only the detector differs.
(a) RETRAINED: verifier (and MetaSeg-style baseline) trained leave-one-condition-out on the Faster R-CNN proposals.
(b) FROZEN: verifier trained on the RT-DETR proposals of v11, leave-one-condition-out (held-out condition and its
    reference frames excluded), applied unchanged to the Faster R-CNN proposals of the held-out condition.
E18 protocol on the adverse conditions (shared proposals, matched strict false alarms, best-of-four raw scores),
count-match thresholds, proposal coverage; the RT-DETR rows (v11) are recomputed with the same code for reference.

usage: python e33_detswap_acdc.py <v11 dir> <v20a dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e18_comparison import gbm, loco, score_row, table  # noqa: E402
from e23_wilddash import count_match  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, feature_sets, load  # noqa: E402

DET = ["rtdetr", "rtdetr_gamma"]


def main():
    v11, v20a, out = sys.argv[1:4]
    P11, O11 = load(v11)
    fs = feature_sets(P11)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta_unc = [c for c in P11.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    X = lambda D, c: D[c].fillna(-1).to_numpy(np.float32)  # noqa: E731
    D11 = P11[P11.det_src.isin(DET)].copy()
    P20, O20 = load(v20a)
    D20 = P20[P20.det_src.isin(DET)].copy()
    k11 = O11[O11.condition.isin(ADVERSE) & O11.co_stuff]
    k20 = O20[O20.condition.isin(ADVERSE) & O20.co_stuff]
    print("targets RT-DETR run", len(k11), "Faster R-CNN run", len(k20), flush=True)
    keys = list(zip(k20.model, k20.stem, k20.instance_id))
    stems = k20.stem.to_numpy()
    res = {}

    # reference: RT-DETR (v11), as E18
    T11 = D11[D11.condition.isin(ADVERSE)]
    keys11 = list(zip(k11.model, k11.stem, k11.instance_id))
    l11 = {"MetaSeg-style meta-classifier (uncertainty + geometry, GBM)": loco(D11, meta_unc)[T11.index].to_numpy(),
           "Ours (CO verifier, single model)": loco(D11, cheap)[T11.index].to_numpy()}
    rows11, paired11 = table(T11, keys11, l11, {}, k11.stem.to_numpy())

    # Faster R-CNN (v20a)
    T20 = D20[D20.condition.isin(ADVERSE)].copy()
    s_frozen, s_meta_frozen = np.zeros(len(T20)), np.zeros(len(T20))
    for h in ADVERSE:
        tr = D11[~D11.condition.isin([h, h + "_ref"])]
        te = (T20.condition == h).to_numpy()
        s_frozen[te] = gbm(tr, cheap).predict_proba(X(T20[te], cheap))[:, 1]
        s_meta_frozen[te] = gbm(tr, meta_unc).predict_proba(X(T20[te], meta_unc))[:, 1]
    s_retr = loco(D20, cheap)[T20.index].to_numpy()
    s_meta_retr = loco(D20, meta_unc)[T20.index].to_numpy()
    learned = {"MetaSeg-style meta-classifier (uncertainty + geometry, GBM)": s_meta_retr,
               "Ours (CO verifier, single model)": s_retr}
    rows20, paired20 = table(T20, keys, learned, {}, stems)
    rows20["Ours, frozen (trained with RT-DETR proposals)"] = score_row(T20, s_frozen, keys)
    rows20["MetaSeg-style, frozen"] = score_row(T20, s_meta_frozen, keys)
    # paired: frozen ours vs strongest baseline
    from e10_e12 import paired_ci
    from e18_comparison import found_at
    base = {n: r for n, r in rows20.items() if not n.startswith("Ours")}
    strongest = max(base, key=lambda n: base[n]["R@1.0"])
    score_of = {"Rule: detector x segmenter-stuff x (1-entropy)": T20.rule.to_numpy(),
                "MetaSeg-style meta-classifier (uncertainty + geometry, GBM)": s_meta_retr, "MetaSeg-style, frozen": s_meta_frozen}
    pf = None
    if strongest in score_of:
        a = found_at(T20, s_frozen, keys, 1.0).astype(float)
        b = found_at(T20, score_of[strongest], keys, 1.0).astype(float)
        d, ci, p = paired_ci(a, b, stems)
        pf = {"vs": strongest, "diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)], "p_one_sided": round(p, 4)}
    a = found_at(T20, s_frozen, keys, 1.0).astype(float)
    b = found_at(T20, s_retr, keys, 1.0).astype(float)
    d, ci, p = paired_ci(a, b, stems)
    p_fr = {"frozen_minus_retrained": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]}

    def cov(T, kk):
        c = set(zip(*[T[T.pos_co_stuff][x] for x in ("model", "stem", "gt_instance_id")]))
        return round(float(np.mean([k in c for k in kk])), 4)

    res = {"targets": {"rtdetr": len(keys11), "frcnn": len(keys)},
           "proposals_per_unit": {"rtdetr": round(len(T11) / T11.unit.nunique(), 1), "frcnn": round(len(T20) / T20.unit.nunique(), 1)},
           "coverage": {"rtdetr": cov(T11, keys11), "frcnn": cov(T20, keys)},
           "rtdetr_table": rows11, "rtdetr_paired": paired11, "frcnn_table": rows20, "frcnn_paired_retrained": paired20,
           "frcnn_paired_frozen": pf, "frozen_vs_retrained": p_fr,
           "count_match": {"frozen": count_match(T20, s_frozen, keys), "retrained": count_match(T20, s_retr, keys),
                           "rule": count_match(T20, T20.rule.to_numpy(), keys), "metaseg_retrained": count_match(T20, s_meta_retr, keys)}}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("proposals/unit", res["proposals_per_unit"], "coverage", res["coverage"])
    names = sorted(set(rows11) | set(rows20), key=lambda n: (n.startswith("Ours"), n))
    print(f"{'method':62s} | RT-DETR R@0.5 R@1 R@2 | Faster R-CNN R@0.5 R@1 R@2")
    for n in names:
        a, b = rows11.get(n), rows20.get(n)
        fa = f"{100*a['R@0.5']:5.1f} {100*a['R@1.0']:5.1f} {100*a['R@2.0']:5.1f}" if a else "   -              "
        fb = f"{100*b['R@0.5']:5.1f} {100*b['R@1.0']:5.1f} {100*b['R@2.0']:5.1f}" if b else "   -"
        print(f"{n[:62]:62s} | {fa} | {fb}")
    print("paired RT-DETR:", paired11)
    print("paired FRCNN retrained:", paired20)
    print("paired FRCNN frozen:", pf)
    print("frozen - retrained:", p_fr)
    for n, cm in res["count_match"].items():
        print(f"count-match {n:18s} " + "  ".join(f"[{k}] R {100*v['recall']:.1f} FA {v['fa_strict']:.2f}" for k, v in cm.items()))


if __name__ == "__main__":
    main()
