"""E25: per-pixel CNNs (DeepLabV3+ R101 / R18) as UNSEEN segmenters on BDD100K (v19b), protocol of WRITING_KIT section 0 (b).

The verifier and every learned baseline are trained on all of ACDC with the four v11 segmenters (SegFormer-B0/B2/B5,
Mask2Former-S) and applied, without retraining, to the CNN's proposals on BDD100K. The segmenter one-hot is all
zeros (unseen model). Same E18 protocol (shared proposals, matched strict false alarms per image x model on the test
set, best of four variants for raw signals) plus label-free count-match thresholds (b = 0.5 / 1 / 2).
Cross-model disagreement is not comparable here (it would be between the two CNNs) and is left out.

Also the failure statistics of the CNNs on BDD (omitted share, confident-into-background share of omissions).

usage: python e25_cnn_unseen.py <v11 dir> <v19b dir> <v13 dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_v8 import attach  # noqa: E402
from e18_comparison import gbm, table  # noqa: E402
from e23_wilddash import count_match  # noqa: E402
from e_suite_v8 import MODELS, STUFF, feature_sets, load  # noqa: E402

CNNS = ["deeplabv3plus-r101", "deeplabv3plus-r18"]


def load_tags(d, tags):
    frames, objs = [], []
    for tag in tags:
        obj = pd.read_csv(os.path.join(d, f"objects_{tag}.csv"),
                          usecols=["stem", "condition", "instance_id", "omitted", "confident_omission", "area", "predicted_as"])
        obj["model"] = tag
        obj["co_stuff"] = obj.confident_omission & obj.predicted_as.isin(STUFF)
        p = attach(pd.read_csv(os.path.join(d, f"proposals_{tag}.csv")), obj)
        cs = obj[["stem", "condition", "instance_id", "co_stuff"]].rename(columns={"instance_id": "gt_instance_id"})
        p = p.merge(cs, on=["stem", "condition", "gt_instance_id"], how="left")
        p["co_stuff"] = p.co_stuff.fillna(False).astype(bool)
        p["pos_co_stuff"] = p.co_stuff & (p.gt_cover >= 0.5)
        p["model"] = tag
        for t in MODELS:
            p[f"m_{t}"] = 0.0          # unseen segmenter
        frames.append(p)
        objs.append(obj)
    P = pd.concat(frames, ignore_index=True)
    P["unit"] = P.stem + "|" + P.model
    P["rule"] = P.det_score * P.p_self_stuff_frac * (1 - P.p_ent_mean)
    return P, pd.concat(objs, ignore_index=True)


def main():
    v11, v19b, v13, out = sys.argv[1:5]
    P, _ = load(v11)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta_unc = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    Dd = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"])]
    m_ours, m_meta = gbm(Dd, cheap), gbm(Dd, meta_unc)
    X = lambda D, c: D[c].fillna(-1).to_numpy(np.float32)  # noqa: E731

    C, OC = load_tags(v19b, CNNS)
    missing = [c for c in cheap + meta_unc if c not in C.columns]
    assert not missing, missing

    stats = {}
    for m in CNNS:
        o = OC[OC.model == m]
        om = int(o.omitted.sum())
        stats[m] = {"objects": int(len(o)), "omitted_share": round(float(o.omitted.mean()), 4),
                    "co_stuff_share_of_omissions": round(float(o.co_stuff.sum() / max(1, om)), 4), "n_co_stuff": int(o.co_stuff.sum())}
        print(m, stats[m], flush=True)

    res = {"failure_stats": stats, "per_model": {}}
    for scope, tags in (("both CNNs", CNNS), *((t, [t]) for t in CNNS)):
        T = C[C.det_src.isin(["rtdetr", "rtdetr_gamma"]) & C.model.isin(tags)].copy()
        kdf = OC[OC.co_stuff & OC.model.isin(tags)]
        keys = list(zip(kdf.model, kdf.stem, kdf.instance_id))
        s_ours = m_ours.predict_proba(X(T, cheap))[:, 1]
        s_meta = m_meta.predict_proba(X(T, meta_unc))[:, 1]
        T["p_xmodel_js_mean"] = np.nan          # not comparable (two CNNs) -> dropped by best_raw
        T["p_xmodel_js_max"] = np.nan
        rows, paired = table(T, keys, {"MetaSeg-style meta-classifier (uncertainty + geometry, GBM)": s_meta,
                                       "Ours (CO verifier, single model)": s_ours}, {}, kdf.stem.to_numpy())
        cov = set(zip(*[T[T.pos_co_stuff][c] for c in ("model", "stem", "gt_instance_id")]))
        r = {"n_targets": len(keys), "units": int(T.unit.nunique()), "proposal_coverage": round(float(np.mean([k in cov for k in keys])), 4),
             "table": rows, "paired": paired,
             "count_match": {"ours": count_match(T, s_ours, keys), "metaseg": count_match(T, s_meta, keys),
                             "rule": count_match(T, T.rule.to_numpy(), keys)}}
        res["per_model"][scope] = r
        print(f"\n== {scope}: targets {len(keys)}, units {r['units']}, proposal coverage {r['proposal_coverage']:.3f}")
        for n, a in rows.items():
            print(f"{n[:62]:62s} | {100*a['R@0.5']:5.1f} {100*a['R@1.0']:5.1f} {100*a['R@2.0']:5.1f} {100*a['FROC[0,2]']:5.1f}  {a.get('variant', '')}")
        print("paired:", paired)
        for n, cm in r["count_match"].items():
            print(f"count-match {n:8s} " + "  ".join(f"[{k}] R {100*v['recall']:.1f} FA {v['fa_strict']:.2f}" for k, v in cm.items()))

    # reference: the seen SegFormer / Mask2Former models on BDD with the same ACDC-trained verifier (E18 BDD column)
    PB, OB = load(v13)
    TB = PB[PB.det_src.isin(["rtdetr", "rtdetr_gamma"])]
    kb = OB[OB.co_stuff]
    s = m_ours.predict_proba(X(TB, cheap))[:, 1]
    res["reference_seen_models_bdd"] = count_match(TB, s, list(zip(kb.model, kb.stem, kb.instance_id)))
    print("reference (4 seen models on BDD) count-match:", res["reference_seen_models_bdd"])
    res["note"] = "train = all ACDC, 4 v11 segmenters; CNNs unseen (one-hot zeros); BDD 3000 images (v19b)"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
