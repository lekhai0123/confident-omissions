"""E14: cross-dataset operating point. Verifier trained on Cityscapes-C (v10a, cheap config), applied to ACDC adverse
(v11, same proposal configuration). Threshold strategies for a nominal false-alarm budget b per image:
  source     threshold calibrated on Cityscapes-C (no target information)
  rate-match unsupervised: target threshold such that flagged proposals per target image equal the source's flagged
             proposals per image at budget b (uses target images, no target labels)
  count-match unsupervised: target threshold giving exactly b flagged proposals per target image (no labels;
             true alarms are rare, so false alarms ~ b)
  oracle     threshold set on target labels (upper bound for the operating point)
Reports achieved false alarms per image (lenient and strict) and recall of confident omissions into background.

usage: python e14_calibration.py <v10a dir> <v11 dir>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6_transfer import BASE, NEG_RATE, load_source  # noqa: E402
from e_suite_v8 import load as load_target  # noqa: E402

BUDGETS = (0.25, 0.5, 1.0, 2.0)


def main():
    src, tgt = sys.argv[1], sys.argv[2]
    S, n_units_src = load_source(src)
    P, O = load_target(tgt)
    T = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"]) & P.condition.isin(["fog", "night", "rain", "snow"])].copy()
    Ot = O[O.condition.isin(["fog", "night", "rain", "snow"]) & O.co_stuff]
    keys = list(zip(Ot.model, Ot.stem, Ot.instance_id))
    n_units_tgt = P[P.condition.isin(["fog", "night", "rain", "snow"])].unit.nunique()
    rng = np.random.default_rng(0)
    keep = S.pos_co_stuff.to_numpy() | (rng.random(len(S)) < 0.5)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
    m.fit(S[keep][BASE].fillna(-1).to_numpy(np.float32), S[keep].pos_co_stuff.to_numpy())
    s_src = m.predict_proba(S[BASE].fillna(-1).to_numpy(np.float32))[:, 1]
    s_tgt = m.predict_proba(T[BASE].fillna(-1).to_numpy(np.float32))[:, 1]
    # source flagged-per-unit curve (negatives were subsampled at NEG_RATE; positives kept in full)
    w_src = np.where(S.pos.to_numpy(), 1.0, 1.0 / NEG_RATE)
    neg_src = np.sort(s_src[~S.pos.to_numpy()])[::-1]
    out = {}
    s_tgt_sorted = np.sort(s_tgt)[::-1]
    neg_tgt = np.sort(s_tgt[~T.pos.to_numpy()])[::-1]

    def evaluate(thr):
        flag = s_tgt > thr
        hit = T[flag & T.pos_co_stuff.to_numpy()]
        found = set(zip(hit.model, hit.stem, hit.gt_instance_id))
        return {"fa_lenient": round(float((flag & ~T.pos.to_numpy()).sum() / n_units_tgt), 3),
                "fa_strict": round(float((flag & ~T.pos_co_stuff.to_numpy()).sum() / n_units_tgt), 3),
                "flags_per_image": round(float(flag.sum() / n_units_tgt), 3),
                "recall": round(float(np.mean([k in found for k in keys])), 4)}

    for b in BUDGETS:
        t_src = neg_src[min(int(b * n_units_src * NEG_RATE), len(neg_src) - 1)]
        flags_src = float((w_src * (s_src > t_src)).sum() / n_units_src)      # flagged proposals per source image
        t_rate = s_tgt_sorted[min(int(flags_src * n_units_tgt), len(s_tgt_sorted) - 1)]
        t_orc = neg_tgt[min(int(b * n_units_tgt), len(neg_tgt) - 1)]
        t_cnt = s_tgt_sorted[min(int(b * n_units_tgt), len(s_tgt_sorted) - 1)]   # unlabeled: b flags per target image
        out[str(b)] = {"source": evaluate(t_src), "rate_match": evaluate(t_rate), "oracle": evaluate(t_orc), "count_match": evaluate(t_cnt),
                       "source_flags_per_image": round(flags_src, 3)}
        r = out[str(b)]
        print(f"budget {b:4}:  source FA {r['source']['fa_lenient']:.2f} R {r['source']['recall']:.3f} | "
              f"rate-match FA {r['rate_match']['fa_lenient']:.2f} (strict {r['rate_match']['fa_strict']:.2f}) R {r['rate_match']['recall']:.3f} | "
              f"count-match FA {r['count_match']['fa_lenient']:.2f} (strict {r['count_match']['fa_strict']:.2f}) R {r['count_match']['recall']:.3f} | "
              f"oracle FA {r['oracle']['fa_lenient']:.2f} R {r['oracle']['recall']:.3f}", flush=True)
    with open(os.path.join(tgt, "e14_calibration.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
