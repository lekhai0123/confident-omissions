"""E23: WildDash2 as an independent test set (v15), under the protocol locked in WRITING_KIT section 0.

- Verifier and every learned baseline are trained on ALL of ACDC (v11) and applied to WildDash without retraining.
- Thresholds: count-match b = 0.5 / 1 / 2 flags per image x model (no target labels), and the matched strict-FA
  (oracle) operating points of E18 for reference; recall over all confident omissions into background.
- Baselines: every row of Table 3 (E18), same shared-proposal protocol, same best-of-four choice for raw signals.
- Nothing here feeds back into feature / threshold / Mahalanobis choices (decision locked: keep).

Also: failure-mode statistics on WildDash per segmenter (omitted share, confident-into-background share).

usage: python e23_wilddash.py <v11 dir> <v15 dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e18_comparison import gbm, score_row, table  # noqa: E402
from e_suite_v8 import MODELS, STUFF, feature_sets, load  # noqa: E402

BUDGETS = (0.5, 1.0, 2.0)


def count_match(T, s, keys):
    """Label-free threshold: flag the top b x (#units) proposals; report recall and realised strict FA."""
    s = np.nan_to_num(np.asarray(s, float), nan=-np.inf)
    n = T.unit.nunique()
    srt = np.sort(s)[::-1]
    pos = T.pos_co_stuff.to_numpy()
    out = {}
    for b in BUDGETS:
        thr = srt[min(int(b * n), len(srt) - 1)]
        flag = s > thr
        hit = T[flag & pos]
        found = set(zip(hit.model, hit.stem, hit.gt_instance_id))
        out[f"cm@{b}"] = {"recall": round(float(np.mean([k in found for k in keys])), 4),
                          "fa_strict": round(float((flag & ~pos).sum() / n), 3),
                          "flags_per_unit": round(float(flag.sum() / n), 3)}
    return out


def main():
    v11, v15, out = sys.argv[1:4]
    P, _ = load(v11)
    P["src_maag"] = (P.det_src == "maag").astype(float)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    fgbg = [c for c in P.columns if c.startswith("p_fgbg")]
    meta_unc = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    meta_maag = meta_unc + fgbg
    det = ["rtdetr", "rtdetr_gamma"]
    Dd = P[P.det_src.isin(det)]
    Dm = P[P.det_src.isin(det + ["maag"])]
    Ma = P[P.det_src == "maag"]
    print(f"ACDC training rows: {len(Dd)} (positives {int(Dd.pos_co_stuff.sum())}); features {len(cheap)}", flush=True)

    W, OW = load(v15)
    W["src_maag"] = (W.det_src == "maag").astype(float)
    T = W[W.det_src.isin(det)].copy()
    Tm = W[W.det_src.isin(det + ["maag"])].copy()
    Wa = W[W.det_src == "maag"].copy()
    kdf = OW[OW.co_stuff]
    keys = list(zip(kdf.model, kdf.stem, kdf.instance_id))
    X = lambda D, c: D[c].fillna(-1).to_numpy(np.float32)  # noqa: E731

    # 1. failure-mode statistics
    stats = {}
    for m in MODELS:
        o = OW[OW.model == m]
        om = int(o.omitted.sum())
        stats[m] = {"objects": int(len(o)), "omitted_share": round(float(o.omitted.mean()), 4),
                    "confident_share_of_omissions": round(float(o.confident_omission.sum() / max(1, om)), 4),
                    "co_stuff_share_of_omissions": round(float(o.co_stuff.sum() / max(1, om)), 4),
                    "n_co_stuff": int(o.co_stuff.sum())}
        print(m, stats[m], flush=True)

    # 2. method table (E18 protocol, matched strict FA on the test set; identical rule for every method)
    s_ours = gbm(Dd, cheap).predict_proba(X(T, cheap))[:, 1]
    s_meta = gbm(Dd, meta_unc).predict_proba(X(T, meta_unc))[:, 1]
    learned = {"MetaSeg-style meta-classifier (uncertainty + geometry, GBM)": s_meta,
               "Ours (CO verifier, single model)": s_ours}
    s_maag = gbm(Ma, meta_maag).predict_proba(X(Wa, meta_maag))[:, 1]
    native = {"Maag & Rottmann + meta-classifier (native)": (Wa, s_maag)}
    rows, paired = table(T, keys, learned, native, kdf.stem.to_numpy())
    s_om = gbm(Dm, cheap + fgbg + ["src_maag"]).predict_proba(X(Tm, cheap + fgbg + ["src_maag"]))[:, 1]
    rows["Ours + Maag proposals"] = score_row(Tm, s_om, keys)
    n_units = T.unit.nunique()
    cov = set(zip(*[Wa[Wa.gt_cover >= 0.5][c] for c in ("model", "stem", "gt_instance_id")]))
    rows["Maag & Rottmann proposals, unscored (native operating point)"] = {
        "R@native": round(float(np.mean([k in cov for k in keys])), 4),
        "FA_native": round(float((~Wa.pos_co_stuff).sum() / max(1, n_units)), 3), "native": True}

    # proposal coverage (upper bound of any scorer on shared proposals)
    covp = set(zip(*[T[T.pos_co_stuff][c] for c in ("model", "stem", "gt_instance_id")]))
    coverage = round(float(np.mean([k in covp for k in keys])), 4)

    # 3. count-match (label-free) for the learned rows and the rule
    cm = {"Ours (CO verifier, single model)": count_match(T, s_ours, keys),
          "MetaSeg-style meta-classifier (uncertainty + geometry, GBM)": count_match(T, s_meta, keys),
          "Rule: detector x segmenter-stuff x (1-entropy)": count_match(T, T.rule.to_numpy(), keys)}
    per_model = {}
    for m in MODELS:
        sel = (T.model == m).to_numpy()
        km = [k for k in keys if k[0] == m]
        per_model[m] = {"cm": count_match(T[sel], s_ours[sel], km), "oracle": score_row(T[sel], s_ours[sel], km),
                        "n_targets": len(km)}

    res = {"n_images_processed": int(W.stem.nunique()), "n_units": int(n_units), "n_targets": len(keys),
           "proposal_coverage": coverage, "failure_stats": stats, "table": rows, "paired": paired,
           "count_match": cm, "ours_per_model": per_model,
           "note": "train = all ACDC (v11), no retraining; WildDash run stopped at 2760/4256 images (time budget)"}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)

    print(f"\nimages {res['n_images_processed']}  units {n_units}  targets {len(keys)}  proposal coverage {coverage:.3f}")
    print(f"{'method':62s} | R@0.5  R@1   R@2  FROC")
    for n, a in rows.items():
        if "R@native" in a:
            print(f"{n[:62]:62s} | R {100*a['R@native']:.1f} at {a['FA_native']:.2f} strict FA/img")
            continue
        print(f"{n[:62]:62s} | {100*a['R@0.5']:5.1f} {100*a['R@1.0']:5.1f} {100*a['R@2.0']:5.1f} {100*a['FROC[0,2]']:5.1f}  {a.get('variant', '')}")
    print("paired:", paired)
    for n, r in cm.items():
        print(f"count-match {n[:40]:40s} " + "  ".join(f"[{k}] R {100*v['recall']:.1f} FA {v['fa_strict']:.2f}" for k, v in r.items()))
    for m, r in per_model.items():
        print(f"  {m:24s} n={r['n_targets']:4d} cm@1 R {100*r['cm']['cm@1.0']['recall']:.1f} FA {r['cm']['cm@1.0']['fa_strict']:.2f} | oracle R@1 {100*r['oracle']['R@1.0']:.1f}")


if __name__ == "__main__":
    main()
