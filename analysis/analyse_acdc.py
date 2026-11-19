"""Analysis of the v7 ACDC run (real fog / night / rain / snow + labelled normal-condition references).

usage: python analyse_acdc.py <results dir>
Writes <results dir>/analysis_acdc.json.

A. Rates per condition: omitted / confident-omission (CO) share of thing objects, adverse vs reference.
B. Signal recall at the 2% / 10% pixel budget (hit >= 0.5) for CO objects and for other omitted objects;
   false alarms on well-segmented objects.
C. Proposal recall: share of omitted / CO objects covered (gt_cover >= 0.5) by at least one GT-free proposal.
D. Proposal verifier: gradient boosting on GT-free proposal features, leave-one-adverse-condition-out
   (train on the other conditions incl. their references, test on the held-out adverse condition).
   Threshold calibrated on training images to N false alarms per image; reports object recall.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SIGNALS = ["ent", "margin", "tta_js", "maha_min", "maha_pred", "xmodel_js", "depth_fg", "det_stuff", "fused"]
ADVERSE = ["fog", "night", "rain", "snow"]
FA_PER_IMAGE = (0.5, 1.0, 2.0)
COVER = 0.5


def rates(obj: pd.DataFrame) -> dict:
    out = {}
    for c, g in obj.groupby("condition"):
        out[c] = {"n_objects": int(len(g)), "n_images": int(g.stem.nunique()),
                  "omitted_pct": round(100 * g.omitted.mean(), 2),
                  "co_pct": round(100 * g.confident_omission.mean(), 2), "n_co": int(g.confident_omission.sum()),
                  "omitted_conf08_pct": round(100 * (g.omitted & (g.mean_conf >= 0.8)).mean(), 2)}
    return out


def signal_recall(obj: pd.DataFrame) -> dict:
    co = obj[obj.confident_omission]
    om = obj[obj.omitted & ~obj.confident_omission]
    good = obj[obj.frac_true >= 0.5]
    out = {}
    for s in SIGNALS:
        out[s] = {}
        for b in ("hit02", "hit10"):
            col = f"{s}_{b}"
            out[s][b] = {"recall_co": round(float((co[col] >= 0.5).mean()), 3) if len(co) else None,
                         "recall_other_omitted": round(float((om[col] >= 0.5).mean()), 3) if len(om) else None,
                         "flag_good": round(float((good[col] >= 0.5).mean()), 3)}
    return out


def attach_targets(props: pd.DataFrame, obj: pd.DataFrame) -> pd.DataFrame:
    key = ["stem", "condition", "instance_id"]
    t = obj[key + ["omitted", "confident_omission"]].rename(columns={"instance_id": "gt_instance_id"})
    p = props.merge(t, on=["stem", "condition", "gt_instance_id"], how="left")
    p[["omitted", "confident_omission"]] = p[["omitted", "confident_omission"]].fillna(False).astype(bool)
    p["pos"] = p.omitted & (p.gt_cover >= COVER)
    p["pos_co"] = p.confident_omission & (p.gt_cover >= COVER)
    return p


def proposal_recall(p: pd.DataFrame, obj: pd.DataFrame) -> dict:
    out = {}
    n_img = p.groupby("condition").stem.nunique()
    for c, g in obj.groupby("condition"):
        pc = p[p.condition == c]
        hit = pc[pc.gt_cover >= COVER][["stem", "gt_instance_id", "source"]]
        k = set(zip(hit.stem, hit.gt_instance_id))
        kd = set(zip(hit[hit.source == "det"].stem, hit[hit.source == "det"].gt_instance_id))
        kx = set(zip(hit[hit.source == "xmodel"].stem, hit[hit.source == "xmodel"].gt_instance_id))
        r = {"props_per_image": round(len(pc) / max(1, n_img.get(c, 1)), 1)}
        for name, sel in (("co", g.confident_omission), ("omitted", g.omitted)):
            gg = g[sel]
            ids = list(zip(gg.stem, gg.instance_id))
            if ids:
                r[f"recall_{name}"] = round(np.mean([i in k for i in ids]), 3)
                r[f"recall_{name}_det"] = round(np.mean([i in kd for i in ids]), 3)
                r[f"recall_{name}_xmodel"] = round(np.mean([i in kx for i in ids]), 3)
                r[f"n_{name}"] = len(ids)
        out[c] = r
    return out


def feature_cols(p: pd.DataFrame) -> list[str]:
    return [c for c in p.columns if c.startswith("p_")] + ["det_score", "is_det"]


def object_recall_at_fa(p_te: pd.DataFrame, score: np.ndarray, thr: float, target: str) -> tuple[float, float]:
    flagged = score > thr
    n_img = p_te.stem.nunique()
    fa = float((flagged & ~p_te.pos.to_numpy()).sum() / max(1, n_img))
    tgt = p_te[p_te[target]]
    if not len(tgt):
        return float("nan"), fa
    found = set(zip(tgt.stem[flagged[p_te[target].to_numpy()]], tgt.gt_instance_id[flagged[p_te[target].to_numpy()]]))
    return float(len(found)), fa


def verifier(p: pd.DataFrame, obj: pd.DataFrame) -> dict:
    p = p.copy()
    p["is_det"] = (p.source == "det").astype(float)
    cols = feature_cols(p)
    out = {}
    for held in ADVERSE:
        tr = p[~p.condition.isin([held, held + "_ref"])]
        te = p[p.condition == held]
        o_te = obj[obj.condition == held]
        if te.pos.sum() < 5:
            continue
        X_tr, X_te = tr[cols].fillna(0).to_numpy(np.float32), te[cols].fillna(0).to_numpy(np.float32)
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
        model.fit(X_tr, tr.pos.to_numpy())
        scores = {"verifier": model.predict_proba(X_te)[:, 1], "verifier_tr": model.predict_proba(X_tr)[:, 1]}
        res = {"n_test_images": int(te.stem.nunique()), "n_omitted": int(o_te.omitted.sum()),
               "n_co": int(o_te.confident_omission.sum()),
               "auroc_proposal": round(float(roc_auc_score(te.pos, scores["verifier"])), 3)}
        # single-feature baselines ranked the same way (higher = more suspicious)
        base = {"det_score": "det_score", "xmodel_js": "p_xmodel_js_mean", "entropy": "p_ent_mean",
                "fused": "p_fused_mean"}
        for name in ["verifier"] + list(base):
            s_te = scores["verifier"] if name == "verifier" else te[base[name]].fillna(0).to_numpy()
            s_tr = scores["verifier_tr"] if name == "verifier" else tr[base[name]].fillna(0).to_numpy()
            neg_tr = np.sort(s_tr[~tr.pos.to_numpy()])[::-1]
            n_img_tr = tr.stem.nunique()
            res[name] = {}
            for fa in FA_PER_IMAGE:
                k = int(fa * n_img_tr)
                thr = neg_tr[min(k, len(neg_tr) - 1)]
                r = {}
                for tgt, n in (("pos", res["n_omitted"]), ("pos_co", res["n_co"])):
                    found, fa_te = object_recall_at_fa(te, s_te, thr, tgt)
                    r["recall_omitted" if tgt == "pos" else "recall_co"] = round(found / n, 3) if n else None
                    r["fa_per_image_test"] = round(fa_te, 2)
                res[name][f"fa{fa}"] = r
        out[held] = res
    return out


def main():
    d = sys.argv[1]
    report = {}
    for f in sorted(glob.glob(os.path.join(d, "objects_*.csv"))):
        tag = os.path.basename(f)[len("objects_"):-4]
        obj = pd.read_csv(f)
        obj_valid = obj[obj.frac_invalid.fillna(0) <= 0.5]
        adv = obj[obj.condition.isin(ADVERSE)]
        props = attach_targets(pd.read_csv(os.path.join(d, f"proposals_{tag}.csv")), obj)
        report[tag] = {"rates": rates(obj), "rates_valid_only": rates(obj_valid),
                       "signal_recall_adverse": signal_recall(adv),
                       "signal_recall_ref": signal_recall(obj[~obj.condition.isin(ADVERSE)]),
                       "proposal_recall": proposal_recall(props, obj), "verifier_loco": verifier(props, obj)}
        print(f"\n== {tag}")
        for c, r in report[tag]["rates"].items():
            print(f"  {c:10s} n={r['n_objects']:5d} omitted={r['omitted_pct']:5.1f}% co={r['co_pct']:4.2f}% ({r['n_co']})")
        sr = report[tag]["signal_recall_adverse"]
        print("  recall@10% CO / other-omitted / good:",
              {s: (v["hit10"]["recall_co"], v["hit10"]["recall_other_omitted"], v["hit10"]["flag_good"]) for s, v in sr.items()})
        for c, r in report[tag]["proposal_recall"].items():
            print(f"  props {c:10s}", r)
        for c, r in report[tag]["verifier_loco"].items():
            print(f"  verifier {c}: auroc={r['auroc_proposal']}",
                  {n: (r[n]["fa1.0"]["recall_omitted"], r[n]["fa1.0"]["recall_co"], r[n]["fa1.0"]["fa_per_image_test"])
                   for n in ("verifier", "det_score", "xmodel_js", "entropy", "fused")})
    with open(os.path.join(d, "analysis_acdc.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
