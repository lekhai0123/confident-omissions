"""E15: second real-world domain (BDD100K, v13; objects = connected components of thing classes).

1. Confident omissions into background on BDD (share of omissions, per model; per weather / time of day for the
   934 frames with BDD100K attributes).
2. Verifier transfer to BDD with shared cheap features (segmenter + RT-DETR normal/gamma proposals):
   ACDC (v11, all conditions) -> BDD, Cityscapes-C (v10a) -> BDD, BDD -> BDD (2-fold by image, reference), and the
   hand rule. Thresholds: count-match (unlabeled target: b flags per image) and oracle (target labels).
   False alarms per image x model, lenient and strict. Recall over all confident omissions into background.

usage: python e15_bdd.py <v13 dir> <v11 dir> <v10a dir> <attributes json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6_transfer import BASE, load_source  # noqa: E402
from e_suite_v8 import MODELS, load  # noqa: E402

BUDGETS = (0.5, 1.0, 2.0)


def fit(tr, target="pos_co_stuff", seed=0):
    rng = np.random.default_rng(seed)
    keep = tr[target].to_numpy() | (rng.random(len(tr)) < 0.25)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed)
    m.fit(tr[keep][BASE].fillna(-1).to_numpy(np.float32), tr[keep][target].to_numpy())
    return m


def evaluate(T, s, keys, attrs=None):
    n_units = T.unit.nunique()
    srt = np.sort(s)[::-1]
    res = {}
    for b in BUDGETS:
        for mode, thr in (("count_match", srt[min(int(b * n_units), len(srt) - 1)]),
                          ("oracle", np.sort(s[~T.pos.to_numpy()])[::-1][min(int(b * n_units), int((~T.pos).sum()) - 1)])):
            flag = s > thr
            hit = T[flag & T.pos_co_stuff.to_numpy()]
            found = set(zip(hit.model, hit.stem, hit.gt_instance_id))
            h = np.array([k in found for k in keys])
            r = {"fa_lenient": round(float((flag & ~T.pos.to_numpy()).sum() / n_units), 3),
                 "fa_strict": round(float((flag & ~T.pos_co_stuff.to_numpy()).sum() / n_units), 3),
                 "recall": round(float(h.mean()), 4)}
            for mm in MODELS:
                sel = np.array([k[0] == mm for k in keys])
                r[f"recall_{mm}"] = round(float(h[sel].mean()), 4)
            if attrs is not None:
                for grp, stems in attrs.items():
                    sel = np.array([k[1] in stems for k in keys])
                    if sel.sum() >= 10:
                        r[f"recall_{grp}"] = [round(float(h[sel].mean()), 4), int(sel.sum())]
            res[f"{mode}@{b}"] = r
    return res


def main():
    v13, v11, v10a, attr_path = sys.argv[1:5]
    P, O = load(v13)
    T = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"])].copy()
    Ot = O[O.co_stuff]
    keys = list(zip(Ot.model, Ot.stem, Ot.instance_id))
    with open(attr_path) as f:
        at = json.load(f)
    attrs = {"rainy": {k for k, v in at.items() if v["weather"] == "rainy"},
             "snowy": {k for k, v in at.items() if v["weather"] == "snowy"},
             "night": {k for k, v in at.items() if v["timeofday"] == "night"},
             "dawn_dusk": {k for k, v in at.items() if v["timeofday"] == "dawn/dusk"},
             "clear_day": {k for k, v in at.items() if v["weather"] == "clear" and v["timeofday"] == "daytime"}}
    out = {"n_co_stuff": len(keys), "units": int(T.unit.nunique())}

    # 1. failure-mode statistics on BDD
    stats = {}
    for m in MODELS:
        o = O[O.model == m]
        om = o.omitted.sum()
        r = {"omitted_share": round(float(o.omitted.mean()), 4), "co_stuff_share_of_omissions": round(float(o.co_stuff.sum() / max(1, om)), 4),
             "n_co_stuff": int(o.co_stuff.sum())}
        for grp, stems in attrs.items():
            g = o[o.stem.isin(stems)]
            if g.omitted.sum() >= 20:
                r[f"{grp}"] = {"omitted_share": round(float(g.omitted.mean()), 4),
                               "co_stuff_share": round(float(g.co_stuff.sum() / g.omitted.sum()), 4), "n_omitted": int(g.omitted.sum())}
        stats[m] = r
        print(f"{m:24s} omitted {r['omitted_share']:.3f}  confident-into-background share {r['co_stuff_share_of_omissions']:.3f} "
              f"(n {r['n_co_stuff']})  " + " ".join(f"{g}={r[g]['co_stuff_share']:.3f}" for g in attrs if g in r), flush=True)
    out["failure_stats"] = stats

    # 2. verifier transfer
    P11, _ = load(v11)
    S11 = P11[P11.det_src.isin(["rtdetr", "rtdetr_gamma"])]
    S10, _ = load_source(v10a)
    runs = {}
    m = fit(S11)
    runs["ACDC -> BDD"] = m.predict_proba(T[BASE].fillna(-1).to_numpy(np.float32))[:, 1]
    m = fit(S10)
    runs["Cityscapes-C -> BDD"] = m.predict_proba(T[BASE].fillna(-1).to_numpy(np.float32))[:, 1]
    stems = np.array(sorted(T.stem.unique()))
    half = set(np.random.default_rng(0).permutation(stems)[: len(stems) // 2])
    s_in = np.zeros(len(T))
    for fold in (True, False):
        trm = T.stem.isin(half).to_numpy() == fold
        mm = fit(T[trm])
        s_in[~trm] = mm.predict_proba(T[~trm][BASE].fillna(-1).to_numpy(np.float32))[:, 1]
    runs["BDD -> BDD (2-fold)"] = s_in
    runs["rule"] = T.rule.fillna(0).to_numpy()
    for name, s in runs.items():
        res = evaluate(T, s, keys, attrs)
        out[name] = res
        print(f"{name:22s} " + "  ".join(f"[{k}] FA {v['fa_lenient']:.2f}/{v['fa_strict']:.2f} R {v['recall']:.3f}"
                                        for k, v in res.items() if k in ("count_match@0.5", "count_match@1.0", "oracle@1.0"))
              + "  | per model @cm1: " + " ".join(f"{mm.split('-')[-1]}={res['count_match@1.0'][f'recall_{mm}']}" for mm in MODELS)
              + "  | " + " ".join(f"{g}={res['count_match@1.0'].get(f'recall_{g}')}" for g in attrs), flush=True)
    with open(os.path.join(v13, "e15_bdd.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
