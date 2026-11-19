"""E1: Maag & Rottmann (2022) re-implementation vs our verifier, same protocol (v11, ACDC adverse, LOCO by condition,
target = confident omission into background, false alarms per image x monitored model, thresholds calibrated on the
training conditions).

Configurations
  maag_unscored      all Maag overlooked segments flagged (their aggregation without meta classifier)
  maag_meta          Maag segments scored by a MetaSeg-style meta classifier (dispersion + geometry + depth-fg
                     probability), trained on our target (the most favourable variant for them)
  ours_cheap         RT-DETR normal + gamma proposals, segmenter + detector features
  ours_cheap+energy  + unnormalised evidence (energy / RbA)
  ours+maag          union of proposals, + depth-fg feature (+ energy)
  ours_full          + cross-model proposals and features

usage: python e1_maag.py <v11 dir>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e_suite_v8 import ADVERSE, MODELS, feature_sets, load, run_protocol  # noqa: E402


def main():
    d = sys.argv[1]
    P, O = load(d)
    P["src_maag"] = (P.det_src == "maag").astype(float)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    energy = [c for c in P.columns if c.startswith(("p_energy", "p_rba"))]
    fgbg = [c for c in P.columns if c.startswith("p_fgbg")]
    meta = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_fgbg", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    det = lambda X: X.det_src.isin(["rtdetr", "rtdetr_gamma"])
    out = {}

    def show(name, r):
        fr = r["froc"]
        print(f"{name:26s} " + "  ".join(f"[{fa}] FA {fr[fa]['fa_per_unit']:.2f} R {fr[fa]['recall']:.3f}" for fa in ("0.5", "1.0", "2.0"))
              + f"  CI@1 {fr['1.0'].get('recall_ci95')}  b5 {fr['1.0']['recall_segformer-b5']} m2f {fr['1.0']['recall_mask2former-swin-small']}",
              flush=True)
        out[name] = r

    # Maag without meta classifier: one operating point
    T = P[P.condition.isin(ADVERSE) & (P.det_src == "maag")]
    Ot = O[O.condition.isin(ADVERSE) & O.co_stuff]
    hit = T[T.pos_co_stuff]
    found = set(zip(hit.model, hit.stem, hit.gt_instance_id))
    units = P[P.condition.isin(ADVERSE)].unit.nunique()
    rec = float(np.mean([k in found for k in zip(Ot.model, Ot.stem, Ot.instance_id)]))
    fa = float((~T.pos).sum() / units)
    out["maag_unscored"] = {"recall": round(rec, 4), "fa_per_unit": round(fa, 3),
                            "segments_per_unit": round(len(T) / units, 2)}
    print(f"maag_unscored              FA {fa:.2f}  R {rec:.3f}  ({len(T) / units:.2f} segments/img)", flush=True)
    cov = {s: float(np.mean([k in set(zip(P[(P.det_src.isin(s)) & P.pos_co_stuff].model, P[(P.det_src.isin(s)) & P.pos_co_stuff].stem,
                                           P[(P.det_src.isin(s)) & P.pos_co_stuff].gt_instance_id))
                              for k in zip(Ot.model, Ot.stem, Ot.instance_id)]))
           for s in (("maag",), ("rtdetr", "rtdetr_gamma"), ("rtdetr", "rtdetr_gamma", "maag"))}
    out["coverage"] = {"+".join(k): round(v, 3) for k, v in cov.items()}
    print("coverage of CO-into-background:", out["coverage"], flush=True)

    show("maag_meta", run_protocol(P, O, meta, prop_filter=lambda X: X.det_src == "maag"))
    show("ours_cheap", run_protocol(P, O, cheap, prop_filter=det))
    show("ours_cheap+energy", run_protocol(P, O, cheap + energy, prop_filter=det))
    show("ours+maag (+fgbg)", run_protocol(P, O, cheap + fgbg + ["src_maag"],
                                           prop_filter=lambda X: X.det_src.isin(["rtdetr", "rtdetr_gamma", "maag"])))
    show("ours+maag (+fgbg,+energy)", run_protocol(P, O, cheap + fgbg + energy + ["src_maag"],
                                                   prop_filter=lambda X: X.det_src.isin(["rtdetr", "rtdetr_gamma", "maag"])))
    show("ours_full (+xmodel)", run_protocol(P, O, fs["full"] + ["src_maag"]))
    with open(os.path.join(d, "e1_maag.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
