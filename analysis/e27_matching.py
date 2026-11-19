"""E27: sensitivity of the comparison (Table 3 / E18 + E23) to the hit criterion. Scores are NOT recomputed differently:
every method is trained exactly as in E18 / E23 (target pos_co_stuff = cover >= 0.5); only the evaluation changes.

Hit criteria (a flag is a true positive only if it satisfies the criterion; every other flag is a strict false alarm):
  cover50        : proposal covers >= 50% of a CO-bg object's pixels (paper protocol; extra flags on an already-hit
                   object are not counted as false alarms)
  cover50_1to1   : as cover50, but one-to-one: per (unit, object) only the highest-scoring covering flag is a hit,
                   every further flag on the same object is a false alarm (COCO-style duplicate penalty)
  iou30_1to1     : one-to-one, and IoU(flagged box region, object mask) >= 0.3
  iou50_1to1     : one-to-one, and IoU(flagged box region, object mask) >= 0.5
The IoU is between the rasterised box and the object's MASK (stored as gt_iou); for a tight box this is at most the
box-to-box IoU, so these criteria are stricter than box IoU thresholds. Each proposal is matched to the object it
covers most (gt_instance_id), so one box can never hit two objects.
Duplicates are defined per method by its own score ranking, so the labelling does not depend on the threshold.

usage: python e27_matching.py <v11> <v13> <v15> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e18_comparison as E  # noqa: E402
from e10_e12 import paired_ci  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, feature_sets, load  # noqa: E402

CRITERIA = ("cover50", "cover50_1to1", "iou30_1to1", "iou50_1to1")
FA = (0.5, 1.0, 2.0)


def hits_mask(T, s, crit):
    """Boolean array: proposal counts as a hit under crit (given this method's scores s)."""
    base = T.co_stuff.to_numpy() & (T.gt_instance_id.to_numpy() >= 0)
    if crit == "cover50":
        return base & (T.gt_cover.to_numpy() >= 0.5)
    thr = {"cover50_1to1": None, "iou30_1to1": 0.3, "iou50_1to1": 0.5}[crit]
    ok = base & ((T.gt_cover.to_numpy() >= 0.5) if thr is None else (T.gt_iou.to_numpy() >= thr))
    # one-to-one: keep only the top-scoring qualifying flag per (unit, object)
    d = pd.DataFrame({"u": T.unit.to_numpy(), "g": T.gt_instance_id.to_numpy(), "s": np.nan_to_num(s, nan=-np.inf),
                      "ok": ok, "i": np.arange(len(T))})
    d = d[d.ok].sort_values("s", ascending=False)
    first = d.drop_duplicates(["u", "g"]).i.to_numpy()
    out = np.zeros(len(T), bool)
    out[first] = True
    return out


def found_at(T, s, keys, fa, crit, hit=None):
    s = np.nan_to_num(np.asarray(s, float), nan=-np.inf)
    hit = hits_mask(T, s, crit) if hit is None else hit
    n = T.unit.nunique()
    s_neg = np.sort(s[~hit])[::-1]
    t = s_neg[min(int(fa * n), len(s_neg) - 1)]
    h = T[(s > t) & hit]
    f = set(zip(h.model, h.stem, h.gt_instance_id))
    return np.array([k in f for k in keys])


def row(T, s, keys, crit):
    hit = hits_mask(T, np.asarray(s, float), crit)
    return {f"R@{fa}": round(float(found_at(T, s, keys, fa, crit, hit).mean()), 4) for fa in FA}


def evaluate(T, keys, stems, methods, crit):
    rows = {n: row(T, s, keys, crit) for n, s in methods.items()}
    ours = "Ours"
    base = [n for n in methods if n != ours and not n.startswith("Ours")]
    strongest = max(base, key=lambda n: rows[n]["R@1.0"])
    a = found_at(T, methods[ours], keys, 1.0, crit).astype(float)
    b = found_at(T, methods[strongest], keys, 1.0, crit).astype(float)
    d, ci, p = paired_ci(a, b, stems)
    return rows, {"vs": strongest, "diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)], "p": round(p, 4)}


def raw_methods(T, keys):
    m = {}
    for name, stem in E.RAW.items():
        b = E.best_raw(T, stem, keys)
        if b:
            m[name] = b[1]       # variant chosen under the paper protocol, then kept fixed
    m["Detector confidence alone"] = T.det_score.to_numpy()
    m["Rule"] = T.rule.to_numpy()
    return m


def main():
    v11, v13, v15, out = sys.argv[1:5]
    P, O = load(v11)
    P["src_maag"] = (P.det_src == "maag").astype(float)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta_unc = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    det = P.det_src.isin(["rtdetr", "rtdetr_gamma"])
    Dd = P[det].copy()
    res = {}

    # ACDC (LOCO)
    T = Dd[Dd.condition.isin(ADVERSE)]
    kdf = O[O.condition.isin(ADVERSE) & O.co_stuff]
    keys = list(zip(kdf.model, kdf.stem, kdf.instance_id))
    m = raw_methods(T, keys)
    m["MetaSeg-style"] = E.loco(Dd, meta_unc)[T.index].to_numpy()
    m["Ours"] = E.loco(Dd, cheap)[T.index].to_numpy()
    datasets = {"ACDC": (T, keys, kdf.stem.to_numpy(), m)}

    m_ours, m_meta = E.gbm(Dd, cheap), E.gbm(Dd, meta_unc)
    for name, d in (("BDD100K", v13), ("WildDash2", v15)):
        PB, OB = load(d)
        TB = PB[PB.det_src.isin(["rtdetr", "rtdetr_gamma"])].copy()
        kb = OB[OB.co_stuff]
        mb = raw_methods(TB, list(zip(kb.model, kb.stem, kb.instance_id)))
        mb["MetaSeg-style"] = m_meta.predict_proba(TB[meta_unc].fillna(-1).to_numpy(np.float32))[:, 1]
        mb["Ours"] = m_ours.predict_proba(TB[cheap].fillna(-1).to_numpy(np.float32))[:, 1]
        datasets[name] = (TB, list(zip(kb.model, kb.stem, kb.instance_id)), kb.stem.to_numpy(), mb)

    for name, (T, keys, stems, m) in datasets.items():
        res[name] = {}
        for crit in CRITERIA:
            rows, paired = evaluate(T, keys, stems, m, crit)
            res[name][crit] = {"rows": rows, "paired": paired}
            o = rows["Ours"]
            top = sorted(((n, r["R@1.0"]) for n, r in rows.items() if n != "Ours"), key=lambda x: -x[1])[:2]
            print(f"{name:9s} {crit:13s} ours {100*o['R@0.5']:5.1f} {100*o['R@1.0']:5.1f} {100*o['R@2.0']:5.1f} | "
                  + " ".join(f"{n[:14]}={100*v:.1f}" for n, v in top)
                  + f" | diff vs {paired['vs'][:14]} {100*paired['diff']:+.1f} [{100*paired['ci95'][0]:+.1f}, {100*paired['ci95'][1]:+.1f}]",
                  flush=True)
        # how often a flagged object attracts duplicates / how tight hits are (ours, cover50 hits at 1 FA)
        T2 = T.assign(s=m["Ours"])
        cov = T2[T2.co_stuff & (T2.gt_cover >= 0.5)]
        res[name]["iou_of_cover50_hits"] = {"median": round(float(cov.gt_iou.median()), 3),
                                            "share_ge_0.3": round(float((cov.gt_iou >= 0.3).mean()), 3),
                                            "share_ge_0.5": round(float((cov.gt_iou >= 0.5).mean()), 3)}
        print("   mask-IoU of covering proposals:", res[name]["iou_of_cover50_hits"])
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
