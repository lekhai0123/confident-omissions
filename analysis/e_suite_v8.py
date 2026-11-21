"""Offline experiment suite on v8 (ACDC, dense proposals). Primary target: confident omission INTO BACKGROUND
(object predicted as a stuff class). Unit of false alarms: image x monitored model.

E2  ablations: training target (all omissions / all CO / CO-to-stuff), feature groups, cheap configurations
    (single segmenter + one detector, no cross-model evidence, detector-only / cross-model-only proposals).
E3  FROC: thresholds calibrated on the training conditions over a grid of false-alarm budgets; pooled recall vs
    achieved false alarms per image; bootstrap CI (over images) at 0.5 / 1 / 2.
E7  unseen model: leave-one-model-out x leave-one-condition-out (train on 3 models and 3 conditions + their refs).

usage: python e_suite_v8.py <v8 results dir> [--quick]
Writes e_suite_v8.json and fig_froc_v8.png/.pdf to the results dir.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_v8 import ADVERSE, attach  # noqa: E402

STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
MODELS = ["segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"]
FA_GRID = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
RNG = np.random.default_rng(0)


def load(d: str):
    frames, objs = [], []
    for tag in MODELS:
        obj = pd.read_csv(os.path.join(d, f"objects_{tag}.csv"),
                          usecols=["stem", "condition", "instance_id", "omitted", "confident_omission", "area", "predicted_as"])
        obj["model"] = tag
        obj["co_stuff"] = obj.confident_omission & obj.predicted_as.isin(STUFF)
        raw = pd.read_csv(os.path.join(d, f"proposals_{tag}.csv"))
        raw["row"] = np.arange(len(raw))   # row in the proposals CSV (joins the relabel pass); not a feature
        p = attach(raw, obj)
        cs = obj[["stem", "condition", "instance_id", "co_stuff"]].rename(columns={"instance_id": "gt_instance_id"})
        p = p.merge(cs, on=["stem", "condition", "gt_instance_id"], how="left")
        p["co_stuff"] = p.co_stuff.fillna(False).astype(bool)
        p["pos_co_stuff"] = p.co_stuff & (p.gt_cover >= 0.5)
        p["model"] = tag
        for t in MODELS:
            p[f"m_{t}"] = float(tag == t)
        frames.append(p)
        objs.append(obj)
    P = pd.concat(frames, ignore_index=True)
    P["unit"] = P.stem + "|" + P.model
    P["rule"] = P.det_score * P.p_self_stuff_frac * (1 - P.p_ent_mean)
    return P, pd.concat(objs, ignore_index=True)


def feature_sets(P: pd.DataFrame) -> dict:
    allc = [c for c in P.columns if c.startswith(("p_", "clip_p_", "m_"))] + [
        "det_score", "src_rtdetr", "src_rtdetr_gamma", "src_owl", "src_xmodel", "clip_thing_prob", "clip_top_is_thing",
        "zoom_done", "zoom_thing_frac", "zoom_thing_prob", "zoom_top_thing"]
    soft = [c for c in allc if c.startswith(("p_ent", "p_margin"))]
    clip = [c for c in allc if c.startswith("clip_")]
    zoom = [c for c in allc if c.startswith("zoom_")]
    xmod = [c for c in allc if c.startswith(("p_xmodel_js", "p_n_other_thing", "p_fused"))] + ["src_xmodel"]
    maha = [c for c in allc if c.startswith("p_maha")]
    return {"full": allc, "no_softmax": [c for c in allc if c not in soft], "no_clip": [c for c in allc if c not in clip],
            "no_zoom": [c for c in allc if c not in zoom], "no_clip_zoom": [c for c in allc if c not in clip + zoom],
            "no_crossmodel": [c for c in allc if c not in xmod], "no_mahalanobis": [c for c in allc if c not in maha],
            "cheap_single_model": [c for c in allc if c not in xmod + clip + zoom]}


def fit_scores(tr: pd.DataFrame, te: pd.DataFrame, cols: list, target: str):
    keep = tr[target].to_numpy() | (RNG.random(len(tr)) < 0.25)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
    m.fit(tr[keep][cols].fillna(-1).to_numpy(np.float32), tr[keep][target].to_numpy())
    return (m.predict_proba(tr[cols].fillna(-1).to_numpy(np.float32))[:, 1],
            m.predict_proba(te[cols].fillna(-1).to_numpy(np.float32))[:, 1])


def thresholds(s_tr: np.ndarray, tr: pd.DataFrame) -> dict:
    neg = np.sort(s_tr[~tr.pos.to_numpy()])[::-1]
    n = tr.unit.nunique()
    return {fa: neg[min(int(fa * n), len(neg) - 1)] for fa in FA_GRID}


def flags_at(te: pd.DataFrame, s_te: np.ndarray, thr: float):
    """(set of found CO-to-stuff objects, number of false alarms, units)."""
    flag = s_te > thr
    fa = int((flag & ~te.pos.to_numpy()).sum())
    hit = te[flag & te.pos_co_stuff.to_numpy()]
    return set(zip(hit.model, hit.stem, hit.gt_instance_id)), fa


def run_protocol(P, O, cols, target="pos_co_stuff", prop_filter=None, lomo=False, raw_score=None):
    """LOCO (optionally x LOMO). Returns per-FA pooled recall, achieved FA/unit, per-model recall, and per-object
    found flags at FA 1.0 for bootstrapping."""
    Pf = P if prop_filter is None else P[prop_filter(P)]
    folds = [(h, None) for h in ADVERSE] if not lomo else [(h, m) for h in ADVERSE for m in MODELS]
    found = {fa: set() for fa in FA_GRID}
    fa_cnt = {fa: 0 for fa in FA_GRID}
    units = 0
    targets = []
    for held, hm in folds:
        trm = ~Pf.condition.isin([held, held + "_ref"])
        tem = Pf.condition == held
        if hm is not None:
            trm &= Pf.model != hm
            tem &= Pf.model == hm
        tr, te = Pf[trm], Pf[tem]
        if raw_score is None:
            s_tr, s_te = fit_scores(tr, te, cols, target)
        else:
            s_tr, s_te = tr[raw_score].fillna(0).to_numpy(), te[raw_score].fillna(0).to_numpy()
        th = thresholds(s_tr, tr)
        for fa, t in th.items():
            f, n_fa = flags_at(te, s_te, t)
            found[fa] |= f
            fa_cnt[fa] += n_fa
        units += P[(P.condition == held) & ((P.model == hm) if hm is not None else True)].unit.nunique()
        o = O[(O.condition == held) & O.co_stuff]
        if hm is not None:
            o = o[o.model == hm]
        targets.append(o)
    T = pd.concat(targets, ignore_index=True)
    keys = list(zip(T.model, T.stem, T.instance_id))
    res = {"n_co_stuff": len(keys), "units": units, "froc": {}}
    for fa in FA_GRID:
        hit = np.array([k in found[fa] for k in keys])
        r = {"recall": round(float(hit.mean()), 4), "fa_per_unit": round(fa_cnt[fa] / units, 3)}
        for m in MODELS:
            sel = (T.model == m).to_numpy()
            r[f"recall_{m}"] = round(float(hit[sel].mean()), 4) if sel.any() else None
        res["froc"][str(fa)] = r
        if fa in (0.5, 1.0, 2.0):   # bootstrap over images
            stems = T.stem.to_numpy()
            us = np.unique(stems)
            idx = {u: np.where(stems == u)[0] for u in us}
            bs = [hit[np.concatenate([idx[u] for u in RNG.choice(us, len(us))])].mean() for _ in range(500)]
            r["recall_ci95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    return res


def main():
    d = sys.argv[1]
    quick = "--quick" in sys.argv
    P, O = load(d)
    fs = feature_sets(P)
    print(f"proposals {len(P):,}; CO-to-stuff objects (adverse) {int(O[O.condition.isin(ADVERSE)].co_stuff.sum())}", flush=True)
    out = {}

    def show(name, r):
        f1 = r["froc"]["1.0"]
        print(f"  {name:34s} recall@cal-FA1 {f1['recall']:.3f} {f1.get('recall_ci95')}  achieved FA {f1['fa_per_unit']:.2f}  "
              + " ".join(f"{m.split('-')[-1]}={f1[f'recall_{m}']}" for m in MODELS), flush=True)
        out[name] = r

    print("\nE2a training target", flush=True)
    for tgt in ["pos_co_stuff", "pos_co", "pos"]:
        show(f"target={tgt}", run_protocol(P, O, fs["full"], target=tgt))
    print("\nE2b feature groups (target CO-to-stuff)", flush=True)
    for name in ([] if quick else ["no_softmax", "no_clip", "no_zoom", "no_clip_zoom", "no_crossmodel", "no_mahalanobis"]):
        show(f"features={name}", run_protocol(P, O, fs[name]))
    print("\nE2c cheap configurations", flush=True)
    show("cheap: 1 segmenter + RT-DETR only", run_protocol(P, O, fs["cheap_single_model"],
                                                           prop_filter=lambda X: X.det_src == "rtdetr"))
    show("cheap: 1 segmenter + RT-DETR(+gamma)", run_protocol(P, O, fs["cheap_single_model"],
                                                              prop_filter=lambda X: X.det_src.isin(["rtdetr", "rtdetr_gamma"])))
    show("no cross-model (all detectors)", run_protocol(P, O, fs["no_crossmodel"], prop_filter=lambda X: X.det_src != "xmodel"))
    show("cross-model proposals only", run_protocol(P, O, fs["full"], prop_filter=lambda X: X.det_src == "xmodel"))
    print("\nbaselines (raw scores, same protocol)", flush=True)
    for b in ["rule", "det_score", "p_xmodel_js_mean", "clip_thing_prob", "p_maha_pred_mean"]:
        show(f"baseline={b}", run_protocol(P, O, None, raw_score=b))
    print("\nE7 unseen model (LOMO x LOCO)", flush=True)
    show("LOMOxLOCO full", run_protocol(P, O, fs["full"], lomo=True))
    show("LOMOxLOCO cheap 1 seg + RT-DETR(+gamma)", run_protocol(P, O, fs["cheap_single_model"], lomo=True,
                                                                 prop_filter=lambda X: X.det_src.isin(["rtdetr", "rtdetr_gamma"])))
    with open(os.path.join(d, "e_suite_v8.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    curves = [("target=pos_co_stuff", "verifier (full)", "#2a78d6", "o"),
              ("cheap: 1 segmenter + RT-DETR(+gamma)", "verifier, 1 segmenter + detector", "#1baf7a", "s"),
              ("LOMOxLOCO full", "verifier, unseen model", "#eda100", "D"),
              ("baseline=rule", "rule: det x stuff x (1-entropy)", "#eb6834", "^"),
              ("baseline=p_xmodel_js_mean", "cross-model JS", "#8a8a85", "v")]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key, lab, col, mk in curves:
        if key not in out:
            continue
        fr = out[key]["froc"]
        xs = [fr[str(f)]["fa_per_unit"] for f in FA_GRID]
        ys = [100 * fr[str(f)]["recall"] for f in FA_GRID]
        ax.plot(xs, ys, color=col, marker=mk, ms=4, lw=1.6, label=lab)
    ax.set_xlabel("false alarms per image (achieved on held-out condition)")
    ax.set_ylabel("recall of confident omissions into background (%)")
    ax.set_xlim(0, 3.2)
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"fig_froc_v8.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
