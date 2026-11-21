"""E34: offline analyses for the Q1 review of 2026-11-22 (no new inference; stored proposals and objects only).

Protocol as in E18/E23 (locked, WRITING_KIT section 0): shared RT-DETR normal + gamma proposals, strict false alarms
per image x model, matched-FA read-off for method comparison, recall over all confident omissions into background.
ACDC = leave-one-condition-out on v11; BDD100K (v13) and WildDash2 (v15) = trained on all of ACDC, no retraining.

1. Simple baselines on the same boxes (review section 5): logistic regression on three box features (detector score,
   segmenter stuff share, mean entropy or mean margin), and box-mask disagreement scores (stuff share of the box,
   alone and times the detector score; cf. BoxMask, Hekimoglu et al.). Paired bootstrap vs the verifier at 1 FA.
2. Ablation control (review section 6): the generic training target (any omitted object) with its positives
   subsampled to the number of confident-omission positives, so that rarity is matched.
3. Label-free threshold on a future batch (review M6b): set the count-match threshold on an unlabelled batch A,
   freeze it, apply it to a disjoint batch B (random halves of the frames, 50 repeats; plus the official WildDash
   val/train split and, for ACDC, split by recording sequence). Realised flags and strict FA per unit, recall.
4. Recall and proposal coverage per class group and object size (review section 8), at matched 1 FA.
5. Denominators (review section 1): confident omissions into background per eligible object, not only per omission.
6. Operating load: false alarms per recovered target at count-match b = 1.

usage: python e34_review_offline.py <v11 dir> <v13 dir> <v15 dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e10_e12 import paired_ci  # noqa: E402
from e18_comparison import found_at, gbm, score_row  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, STUFF, feature_sets, load  # noqa: E402

DET = ["rtdetr", "rtdetr_gamma"]
GROUPS = {"person": ["person"], "rider": ["rider"], "car": ["car"], "large vehicle": ["truck", "bus", "train"],
          "two-wheeler": ["motorcycle", "bicycle"]}
SIZE_BINS = [(0, 1024), (1024, 4096), (4096, 16384), (16384, 10 ** 9)]   # pixels at the evaluated resolution


def logit(tr, cols, target="pos_co_stuff", seed=0):
    rng = np.random.default_rng(seed)
    keep = tr[target].to_numpy() | (rng.random(len(tr)) < 0.25)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit(tr[keep][cols].fillna(-1).to_numpy(np.float32), tr[keep][target].to_numpy())
    return m


def gbm_matched(tr, cols, target, n_pos, seed=0):
    """GBM with the positives of `target` subsampled to n_pos (rarity matched to the primary target)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    pos = np.where(tr[target].to_numpy())[0]
    pos = rng.choice(pos, min(n_pos, len(pos)), replace=False)
    neg = np.where(~tr[target].to_numpy() & (rng.random(len(tr)) < 0.25))[0]
    idx = np.concatenate([pos, neg])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed)
    m.fit(tr.iloc[idx][cols].fillna(-1).to_numpy(np.float32), tr.iloc[idx][target].to_numpy())
    return m


def predict(m, D, cols):
    return m.predict_proba(D[cols].fillna(-1).to_numpy(np.float32))[:, 1]


def loco(D, fit):
    s = pd.Series(np.nan, index=D.index)
    for h in ADVERSE:
        tr = D[~D.condition.isin([h, h + "_ref"])]
        te = D.condition == h
        s[te] = fit(tr)(D[te])
    return s


def keys_of(df):
    return list(zip(df.model, df.stem, df.instance_id))


def count_match_ab(T, s, keys_df, split_fn, n_rep=50, budgets=(0.5, 1.0, 2.0), seed=0):
    """Threshold from the unlabelled batch A at b flags/unit, frozen, applied to batch B."""
    rng = np.random.default_rng(seed)
    s = np.nan_to_num(np.asarray(s, float), nan=-np.inf)
    pos = T.pos_co_stuff.to_numpy()
    out = {b: {"flags": [], "fa": [], "recall": [], "recall_sameB": []} for b in budgets}
    for r in range(n_rep):
        a_stems = split_fn(rng, r)
        if a_stems is None:
            break
        inA = T.stem.isin(a_stems).to_numpy()
        inB = ~inA
        kB = keys_df[~keys_df.stem.isin(a_stems)]
        kBs = set(keys_of(kB))
        nA, nB = T[inA].unit.nunique(), T[inB].unit.nunique()
        sA, sB = np.sort(s[inA])[::-1], np.sort(s[inB])[::-1]
        for b in budgets:
            thr = sA[min(int(b * nA), len(sA) - 1)]
            for nm, t in (("frozen", thr), ("sameB", sB[min(int(b * nB), len(sB) - 1)])):
                flag = (s > t) & inB
                hit = T[flag & pos]
                found = set(zip(hit.model, hit.stem, hit.gt_instance_id)) & kBs
                rec = len(found) / max(1, len(kBs))
                if nm == "frozen":
                    out[b]["flags"].append(flag.sum() / nB)
                    out[b]["fa"].append((flag & ~pos).sum() / nB)
                    out[b]["recall"].append(rec)
                else:
                    out[b]["recall_sameB"].append(rec)
    res = {}
    for b, v in out.items():
        if not v["flags"]:
            continue
        f = lambda x: [round(float(np.mean(x)), 3), round(float(np.percentile(x, 5)), 3), round(float(np.percentile(x, 95)), 3)]  # noqa: E731
        res[f"b={b}"] = {"flags_per_unit": f(v["flags"]), "fa_strict_per_unit": f(v["fa"]), "recall": f(v["recall"]),
                         "recall_if_set_on_B": f(v["recall_sameB"]), "n_splits": len(v["flags"])}
    return res


def group_recall(T, s, keys_df, fa=1.0):
    hit = found_at(T, s, list(keys_of(keys_df)), fa)
    cov = set(zip(*[T[T.pos_co_stuff][c] for c in ("model", "stem", "gt_instance_id")]))
    covered = np.array([k in cov for k in keys_of(keys_df)])
    out = {}
    for g, cls in GROUPS.items():
        sel = keys_df["class"].isin(cls).to_numpy()
        out[g] = {"n": int(sel.sum()), "recall": round(float(hit[sel].mean()), 3) if sel.any() else None,
                  "coverage": round(float(covered[sel].mean()), 3) if sel.any() else None}
    for lo, hi in SIZE_BINS:
        sel = ((keys_df.area >= lo) & (keys_df.area < hi)).to_numpy()
        out[f"area {lo}-{hi}"] = {"n": int(sel.sum()), "recall": round(float(hit[sel].mean()), 3) if sel.any() else None,
                                  "coverage": round(float(covered[sel].mean()), 3) if sel.any() else None}
    return out


def load_objects_with_class(d):
    """objects with class column (load() keeps only a subset of columns)."""
    fr = []
    for tag in MODELS:
        o = pd.read_csv(os.path.join(d, f"objects_{tag}.csv"),
                        usecols=["stem", "condition", "instance_id", "class", "area", "omitted", "confident_omission", "predicted_as"])
        o["model"] = tag
        o["co_stuff"] = o.confident_omission & o.predicted_as.isin(STUFF)
        o["stem"] = o.stem.astype(str)
        fr.append(o)
    return pd.concat(fr, ignore_index=True)


def main():
    v11, v13, v15, out = sys.argv[1:5]
    P, O = load(v11)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta_unc = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    L3E = ["det_score", "p_self_stuff_frac", "p_ent_mean"]
    L3M = ["det_score", "p_self_stuff_frac", "p_margin_mean"]
    Dd = P[P.det_src.isin(DET)].copy()
    Dd["disagree"] = Dd.p_self_stuff_frac
    Dd["disagree_det"] = Dd.p_self_stuff_frac * Dd.det_score
    Dd["pos_any"] = Dd.pos & (Dd.gt_cover >= 0.5)
    n_pos_primary = int(Dd[~Dd.condition.isin(["fog", "fog_ref"])].pos_co_stuff.sum())

    OA = load_objects_with_class(v11)
    res = {}
    domains = {}
    # ---- ACDC (LOCO)
    sc = {"ours": loco(Dd, lambda tr: (lambda m: (lambda D: predict(m, D, cheap)))(gbm(tr, cheap))),
          "meta": loco(Dd, lambda tr: (lambda m: (lambda D: predict(m, D, meta_unc)))(gbm(tr, meta_unc))),
          "logit3_ent": loco(Dd, lambda tr: (lambda m: (lambda D: predict(m, D, L3E)))(logit(tr, L3E))),
          "logit3_margin": loco(Dd, lambda tr: (lambda m: (lambda D: predict(m, D, L3M)))(logit(tr, L3M))),
          "generic_matched": loco(Dd, lambda tr: (lambda m: (lambda D: predict(m, D, cheap)))(
              gbm_matched(tr, cheap, "pos_any", int(tr.pos_co_stuff.sum()))))}
    TA = Dd[Dd.condition.isin(ADVERSE)]
    domains["ACDC"] = (TA, {k: v[TA.index].to_numpy() for k, v in sc.items()},
                       OA[OA.condition.isin(ADVERSE) & OA.co_stuff].reset_index(drop=True), OA[OA.condition.isin(ADVERSE)])
    # ---- BDD and WildDash (trained on all ACDC)
    m_ours, m_meta = gbm(Dd, cheap), gbm(Dd, meta_unc)
    m_l3e, m_l3m = logit(Dd, L3E), logit(Dd, L3M)
    for name, d in (("BDD100K", v13), ("WildDash2", v15)):
        PX, _ = load(d)
        T = PX[PX.det_src.isin(DET)].copy()
        T["disagree"] = T.p_self_stuff_frac
        T["disagree_det"] = T.p_self_stuff_frac * T.det_score
        OX = load_objects_with_class(d)
        domains[name] = (T, {"ours": predict(m_ours, T, cheap), "meta": predict(m_meta, T, meta_unc),
                             "logit3_ent": predict(m_l3e, T, L3E), "logit3_margin": predict(m_l3m, T, L3M)},
                         OX[OX.co_stuff].reset_index(drop=True), OX)

    for name, (T, S, K, Oall) in domains.items():
        T = T.copy()
        T["stem"] = T.stem.astype(str)
        keys = keys_of(K)
        stems = K.stem.to_numpy()
        # stuff share is exactly 1 for many boxes: ties are broken by the detector score (otherwise no box exceeds the threshold)
        S = dict(S, rule=T.rule.to_numpy(), disagree=T.disagree.to_numpy() + 1e-6 * T.det_score.to_numpy(),
                 disagree_det=T.disagree_det.to_numpy())
        rows = {k: {kk: round(v, 4) for kk, v in score_row(T, s, keys).items()} for k, s in S.items()}
        hit_o = found_at(T, S["ours"], keys, 1.0).astype(float)
        paired = {}
        for k in S:
            if k == "ours":
                continue
            d_, ci, p = paired_ci(hit_o, found_at(T, S[k], keys, 1.0).astype(float), stems)
            paired[k] = {"diff": round(d_, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]}
        r = {"n_targets": len(keys), "rows": rows, "paired_vs_ours_at1": paired}
        # 3. count-match on a future batch
        frames = np.array(sorted(T.stem.unique()))

        def halves(rng, i, frames=frames):
            return set(rng.permutation(frames)[: len(frames) // 2])
        cm = {"random_halves": {k: count_match_ab(T, S[k], K, halves) for k in ("ours", "rule", "meta")}}
        if name == "WildDash2":
            val = set(T[T.condition == "wd_val"].stem)
            cm["official_split_A=val"] = {k: count_match_ab(T, S[k], K, lambda rng, i: val if i == 0 else None, n_rep=1)
                                          for k in ("ours", "rule")}
        if name == "ACDC":
            seq = pd.Series(frames).str.split("_frame").str[0].to_numpy()
            useq = np.unique(seq)

            def by_seq(rng, i, frames=frames, seq=seq, useq=useq):
                a = set(rng.permutation(useq)[: len(useq) // 2])
                return set(frames[np.isin(seq, list(a))])
            cm["by_sequence"] = {k: count_match_ab(T, S[k], K, by_seq) for k in ("ours", "rule")}
            r["n_sequences"] = int(len(useq))
        r["count_match_future_batch"] = cm
        # 4. per group
        r["per_group"] = {k: group_recall(T, S[k], K) for k in ("ours", "rule", "meta")}
        # 5. denominators
        den = {}
        for m in MODELS + ["all"]:
            o = Oall if m == "all" else Oall[Oall.model == m]
            den[m] = {"eligible": int(len(o)), "omitted": int(o.omitted.sum()), "co_stuff": int(o.co_stuff.sum()),
                      "omitted_share": round(float(o.omitted.mean()), 4),
                      "co_stuff_per_omitted": round(float(o.co_stuff.sum() / max(1, o.omitted.sum())), 4),
                      "co_stuff_per_eligible": round(float(o.co_stuff.mean()), 5)}
        r["denominators"] = den
        # 6. operating load at count-match b = 1 (all frames)
        s = np.nan_to_num(S["ours"], nan=-np.inf)
        n = T.unit.nunique()
        thr = np.sort(s)[::-1][int(n)]
        flag = s > thr
        pos = T.pos_co_stuff.to_numpy()
        hit = T[flag & pos]
        found = set(zip(hit.model, hit.stem, hit.gt_instance_id)) & set(keys)
        r["load_cm1"] = {"units": int(n), "flags": int(flag.sum()), "fa_strict": int((flag & ~pos).sum()),
                         "found_targets": len(found), "fa_per_found": round(float((flag & ~pos).sum() / max(1, len(found))), 1),
                         "found_per_1000_units": round(1000 * len(found) / n, 2)}
        res[name] = r
        print(f"\n== {name}: targets {len(keys)}")
        for k, v in rows.items():
            print(f"  {k:16s} R@0.5 {100*v['R@0.5']:5.1f}  R@1 {100*v['R@1.0']:5.1f}  R@2 {100*v['R@2.0']:5.1f}  FROC {100*v['FROC[0,2]']:5.1f}"
                  + (f"   ours-this {100*paired[k]['diff']:+.1f} [{100*paired[k]['ci95'][0]:+.1f}, {100*paired[k]['ci95'][1]:+.1f}]" if k in paired else ""))
        for split, dd in cm.items():
            for k, v in dd.items():
                b1 = v.get("b=1.0")
                if b1:
                    print(f"  cm {split:22s} {k:6s} b=1: flags {b1['flags_per_unit']}  FA {b1['fa_strict_per_unit']}  R {b1['recall']}  (set on B: {b1['recall_if_set_on_B']})")
        print("  load:", r["load_cm1"])
        print("  denominators all:", den["all"])
    res["note"] = {"n_pos_primary_ACDC_example_fold": n_pos_primary,
                   "bootstrap": "paired_ci: 2000 resamples of source images, seed 0"}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
