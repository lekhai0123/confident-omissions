"""E10 + E12 on the main configuration (v11: monitored segmenter + RT-DETR normal/gamma proposals, no TTA).

E10  strict false alarms: any flag that does not cover a confident omission into background (instead of "does not
     cover any omitted object"); distribution of false alarms per image x model (mean, median, P95, P99, share of
     units without false alarms).
E12  5 seeds (negative subsampling + GBM seed): mean ± sd of recall at calibrated budgets; paired bootstrap (over
     images) of recall differences at matched achieved false-alarm rates (test-set threshold, identical for both
     methods' budget) for ours vs rule, ours vs Maag+meta, ours+Maag vs ours.

usage: python e10_e12.py <v11 dir>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e_suite_v8 import ADVERSE, MODELS, feature_sets, load  # noqa: E402

FA_LEVELS = (0.25, 0.5, 1.0, 2.0)
SEEDS = (0, 1, 2, 3, 4)


def fold_scores(P, cols, prop_filter, seed, raw=None):
    """LOCO scores for adverse conditions. Returns test frame with column s and train-calibrated thresholds
    under lenient and strict negatives."""
    rng = np.random.default_rng(seed)
    out, thr = [], {}
    D = P[prop_filter(P)]
    for h in ADVERSE:
        tr = D[~D.condition.isin([h, h + "_ref"])]
        te = D[D.condition == h].copy()
        if raw is None:
            keep = tr.pos_co_stuff.to_numpy() | (rng.random(len(tr)) < 0.25)
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed)
            m.fit(tr[keep][cols].fillna(-1).to_numpy(np.float32), tr[keep].pos_co_stuff.to_numpy())
            s_tr = m.predict_proba(tr[cols].fillna(-1).to_numpy(np.float32))[:, 1]
            te["s"] = m.predict_proba(te[cols].fillna(-1).to_numpy(np.float32))[:, 1]
        else:
            s_tr = tr[raw].fillna(0).to_numpy()
            te["s"] = te[raw].fillna(0).to_numpy()
        n = tr.unit.nunique()
        for mode, negmask in (("lenient", ~tr.pos.to_numpy()), ("strict", ~tr.pos_co_stuff.to_numpy())):
            neg = np.sort(s_tr[negmask])[::-1]
            for fa in FA_LEVELS:
                thr[(h, mode, fa)] = neg[min(int(fa * n), len(neg) - 1)]
        out.append(te)
    return pd.concat(out), thr


def evaluate(T, thr, O, units_by_cond):
    keys_df = O[O.condition.isin(ADVERSE) & O.co_stuff]
    keys = list(zip(keys_df.model, keys_df.stem, keys_df.instance_id))
    res = {}
    for mode in ("lenient", "strict"):
        for fa in FA_LEVELS:
            flag = np.zeros(len(T), bool)
            for h in ADVERSE:
                sel = (T.condition == h).to_numpy()
                flag[sel] = T.s.to_numpy()[sel] > thr[(h, mode, fa)]
            neg = ~(T.pos if mode == "lenient" else T.pos_co_stuff).to_numpy()
            fa_units = pd.Series((flag & neg).astype(int), index=T.index).groupby(T.unit).sum()
            all_units = sum(units_by_cond.values())
            fa_units = fa_units.reindex(pd.Index(sorted(set(T.unit))), fill_value=0)
            n_zero_units = all_units - len(fa_units)       # units without any proposal
            dist = np.r_[fa_units.to_numpy(), np.zeros(max(0, n_zero_units))]
            hit = T[flag & T.pos_co_stuff.to_numpy()]
            found = set(zip(hit.model, hit.stem, hit.gt_instance_id))
            h_arr = np.array([k in found for k in keys])
            r = {"recall": float(h_arr.mean()), "fa_mean": float(dist.mean()), "fa_median": float(np.median(dist)),
                 "fa_p95": float(np.percentile(dist, 95)), "fa_p99": float(np.percentile(dist, 99)),
                 "share_units_no_fa": float((dist == 0).mean())}
            for m in MODELS:
                sm = np.array([k[0] == m for k in keys])
                r[f"recall_{m}"] = float(h_arr[sm].mean())
            res[(mode, fa)] = r
    return res, keys


def matched_found(T, keys, fa, mode):
    """Found flags at a test-set threshold giving exactly `fa` false alarms per unit (pooled)."""
    neg = ~(T.pos if mode == "lenient" else T.pos_co_stuff).to_numpy()
    n = T.unit.nunique()
    s_neg = np.sort(T.s.to_numpy()[neg])[::-1]
    t = s_neg[min(int(fa * n), len(s_neg) - 1)]
    hit = T[(T.s.to_numpy() > t) & T.pos_co_stuff.to_numpy()]
    found = set(zip(hit.model, hit.stem, hit.gt_instance_id))
    return np.array([k in found for k in keys])


def paired_ci(a, b, stems, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    us = np.unique(stems)
    idx = {u: np.where(stems == u)[0] for u in us}
    d = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        d.append(a[ii].mean() - b[ii].mean())
    return float(a.mean() - b.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float((np.array(d) <= 0).mean())


def main():
    d = sys.argv[1]
    P, O = load(d)
    P["src_maag"] = (P.det_src == "maag").astype(float)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    fgbg = [c for c in P.columns if c.startswith("p_fgbg")]
    meta = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_fgbg", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    det = lambda X: X.det_src.isin(["rtdetr", "rtdetr_gamma"])
    detm = lambda X: X.det_src.isin(["rtdetr", "rtdetr_gamma", "maag"])
    units_by_cond = {h: P[P.condition == h].unit.nunique() for h in ADVERSE}
    configs = {"ours": (cheap, det, None), "ours+maag": (cheap + fgbg + ["src_maag"], detm, None),
               "rule": (None, det, "rule"), "maag_meta": (meta, lambda X: X.det_src == "maag", None)}
    summary, stores = {}, {}
    for name, (cols, filt, raw) in configs.items():
        seeds = SEEDS if raw is None else (0,)
        per_seed = []
        for sd in seeds:
            T, thr = fold_scores(P, cols, filt, sd, raw)
            res, keys = evaluate(T, thr, O, units_by_cond)
            per_seed.append(res)
            if sd == 0:
                stores[name] = (T, keys)
        agg = {}
        for k in per_seed[0]:
            agg[f"{k[0]}@{k[1]}"] = {m: [round(float(np.mean([r[k][m] for r in per_seed])), 4),
                                        round(float(np.std([r[k][m] for r in per_seed])), 4)] for m in per_seed[0][k]}
        summary[name] = agg
        for mode in ("lenient", "strict"):
            line = "  ".join(f"@{fa}: R {agg[f'{mode}@{fa}']['recall'][0]:.3f}±{agg[f'{mode}@{fa}']['recall'][1]:.3f} "
                             f"FA {agg[f'{mode}@{fa}']['fa_mean'][0]:.2f} (med {agg[f'{mode}@{fa}']['fa_median'][0]:.0f}, "
                             f"P95 {agg[f'{mode}@{fa}']['fa_p95'][0]:.0f}, P99 {agg[f'{mode}@{fa}']['fa_p99'][0]:.0f}, "
                             f"0-FA {agg[f'{mode}@{fa}']['share_units_no_fa'][0]:.2f})" for fa in (0.5, 1.0))
            print(f"{name:10s} {mode:8s} {line}", flush=True)
        a1 = agg["strict@1.0"]
        print(f"{'':10s} per-model strict@1.0: " + " ".join(f"{m.split('-')[-1]}={a1[f'recall_{m}'][0]:.3f}±{a1[f'recall_{m}'][1]:.3f}"
                                                         for m in MODELS), flush=True)
    # paired comparisons at matched achieved false alarms (seed 0 scores)
    keys_df = O[O.condition.isin(ADVERSE) & O.co_stuff]
    stems = keys_df.stem.to_numpy()
    comps = {}
    for mode in ("lenient", "strict"):
        for fa in (0.5, 1.0):
            f = {n: matched_found(stores[n][0], stores[n][1], fa, mode) for n in stores}
            for a, b in (("ours", "rule"), ("ours", "maag_meta"), ("ours+maag", "ours")):
                diff, ci, p = paired_ci(f[a], f[b], stems)
                comps[f"{a} - {b} | {mode} @{fa}"] = {"diff": round(diff, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)],
                                                      "p_one_sided": round(p, 4), "recall_a": round(float(f[a].mean()), 4),
                                                      "recall_b": round(float(f[b].mean()), 4)}
                print(f"paired {a:9s} - {b:9s} {mode:8s} @{fa}: {f[a].mean():.3f} vs {f[b].mean():.3f}  diff {diff:+.3f} "
                      f"CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.4f}", flush=True)
    with open(os.path.join(d, "e10_e12.json"), "w") as fh:
        json.dump({"summary": summary, "paired": comps}, fh, indent=1)


if __name__ == "__main__":
    main()
