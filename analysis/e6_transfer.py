"""E6: cross-dataset generalisation. Train the cheap verifier (monitored segmenter + RT-DETR normal/gamma proposals)
on Cityscapes-C (v10a, synthetic corruptions, 17 conditions) and test on ACDC (v8, real adverse weather).
Threshold chosen on the source data only (false alarms per image x condition x model on Cityscapes-C).
Target: confident omission into background. Compared with the in-domain ACDC result (leave-one-condition-out).

usage: python e6_transfer.py <v10a dir> <v8 dir>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e_suite_v8 import STUFF, MODELS, load as load_v8  # noqa: E402

NEG_RATE = 0.10
RNG = np.random.default_rng(0)
FA_GRID = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
BASE = ["p_ent_mean", "p_ent_max", "p_margin_mean", "p_margin_max", "p_maha_min_mean", "p_maha_min_max",
        "p_maha_pred_mean", "p_maha_pred_max", "p_det_stuff_mean", "p_det_stuff_max", "p_thing_prob_mean",
        "p_thing_prob_max", "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill",
        "det_score", "src_rtdetr", "src_rtdetr_gamma"] + [f"m_{t}" for t in MODELS]


def load_source(d: str):
    """Cityscapes-C proposals (detector sources only): all positives + NEG_RATE of the rest."""
    parts, n_units, n_neg_total = [], 0, 0
    for tag in MODELS:
        obj = pd.read_csv(os.path.join(d, f"objects_{tag}.csv"),
                          usecols=["stem", "condition", "instance_id", "omitted", "confident_omission", "predicted_as"])
        obj["co_stuff"] = obj.confident_omission & obj.predicted_as.isin(STUFF)
        key = obj.set_index(["stem", "condition", "instance_id"])[["omitted", "co_stuff"]]
        for ch in pd.read_csv(os.path.join(d, f"proposals_{tag}.csv"), chunksize=400_000):
            ch = ch[ch.det_src.isin(["rtdetr", "rtdetr_gamma"])]
            ch = ch.join(key, on=["stem", "condition", "gt_instance_id"])
            ch[["omitted", "co_stuff"]] = ch[["omitted", "co_stuff"]].fillna(False).astype(bool)
            ch["pos"] = ch.omitted & (ch.gt_cover >= 0.5)
            ch["pos_co_stuff"] = ch.co_stuff & (ch.gt_cover >= 0.5)
            ch["src_rtdetr"] = (ch.det_src == "rtdetr").astype(float)
            ch["src_rtdetr_gamma"] = (ch.det_src == "rtdetr_gamma").astype(float)
            for t in MODELS:
                ch[f"m_{t}"] = float(tag == t)
            n_neg_total += int((~ch.pos).sum())
            keep = ch.pos.to_numpy() | (RNG.random(len(ch)) < NEG_RATE)
            parts.append(ch.loc[keep, BASE + ["pos", "pos_co_stuff", "condition", "stem"]])
        n_units += obj.groupby(["stem", "condition"]).ngroups
        print(f"  loaded {tag}", flush=True)
    S = pd.concat(parts, ignore_index=True)
    return S, n_units


def main():
    src_dir, tgt_dir = sys.argv[1], sys.argv[2]
    S, n_units_src = load_source(src_dir)
    print(f"source rows {len(S):,} (positives CO-stuff {int(S.pos_co_stuff.sum()):,}); units {n_units_src:,}", flush=True)
    P, O = load_v8(tgt_dir)
    T = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"]) & P.condition.isin(["fog", "night", "rain", "snow"])].copy()
    Ot = O[O.condition.isin(["fog", "night", "rain", "snow"]) & O.co_stuff]
    keys = list(zip(Ot.model, Ot.stem, Ot.condition, Ot.instance_id))
    n_units_tgt = T.groupby(["stem", "model"]).ngroups
    out = {}
    for name, sel in [("cityscapes_c_all", np.ones(len(S), bool)),
                      ("cityscapes_c_no_noise", ~S.condition.str.contains("noise").to_numpy()),
                      ("cityscapes_clean_only", (S.condition == "clean").to_numpy())]:
        tr = S[sel]
        keep = tr.pos_co_stuff.to_numpy() | (RNG.random(len(tr)) < 0.5)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
        m.fit(tr[keep][BASE].fillna(-1).to_numpy(np.float32), tr[keep].pos_co_stuff.to_numpy())
        s_tr = m.predict_proba(tr[BASE].fillna(-1).to_numpy(np.float32))[:, 1]
        neg = np.sort(s_tr[~tr.pos.to_numpy()])[::-1]
        units = n_units_src if name != "cityscapes_clean_only" else S[S.condition == "clean"].groupby("stem").ngroups * 4
        if name == "cityscapes_c_no_noise":
            units = int(n_units_src * (sel.mean()))
        s_te = m.predict_proba(T[BASE].fillna(-1).to_numpy(np.float32))[:, 1]
        res = {"positives_train": int(tr.pos_co_stuff.sum()), "froc": {}}
        for fa in FA_GRID:
            k = int(fa * units * NEG_RATE)            # negatives were subsampled at NEG_RATE
            thr = neg[min(k, len(neg) - 1)]
            flag = s_te > thr
            fa_t = (flag & ~T.pos.to_numpy()).sum() / n_units_tgt
            hit = T[flag & T.pos_co_stuff.to_numpy()]
            found = set(zip(hit.model, hit.stem, hit.condition, hit.gt_instance_id))
            h = np.array([kk in found for kk in keys])
            r = {"fa_target": round(float(fa_t), 3), "recall": round(float(h.mean()), 4)}
            for t in MODELS:
                sm = np.array([kk[0] == t for kk in keys])
                r[f"recall_{t}"] = round(float(h[sm].mean()), 4)
            res["froc"][str(fa)] = r
        # threshold re-chosen on the target (oracle operating point) to separate ranking from calibration shift
        neg_t = np.sort(s_te[~T.pos.to_numpy()])[::-1]
        res["oracle"] = {}
        for fa in (0.5, 1.0):
            thr = neg_t[min(int(fa * n_units_tgt), len(neg_t) - 1)]
            hit = T[(s_te > thr) & T.pos_co_stuff.to_numpy()]
            found = set(zip(hit.model, hit.stem, hit.condition, hit.gt_instance_id))
            res["oracle"][str(fa)] = round(float(np.mean([kk in found for kk in keys])), 4)
        out[name] = res
        f1 = res["froc"]
        print(f"{name:24s} " + "  ".join(f"cal{fa}: FA {f1[str(fa)]['fa_target']:.2f} R {f1[str(fa)]['recall']:.3f}"
                                         for fa in (0.3, 0.5, 1.0, 2.0))
              + f"  | oracle R@0.5 {res['oracle']['0.5']:.3f} R@1 {res['oracle']['1.0']:.3f}"
              + "  | per model @cal1: " + " ".join(f"{t.split('-')[-1]}={f1['1.0'][f'recall_{t}']}" for t in MODELS), flush=True)
    with open(os.path.join(tgt_dir, "e6_transfer.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
