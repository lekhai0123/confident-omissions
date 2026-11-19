"""Milestone 1a: upper bound for a proposal verifier on v6 (regions = ground-truth object masks).

Question: if a proposal covered an object perfectly, could GT-free signals inside it separate a confident
omission from a well-segmented object? Logistic regression / gradient boosting on per-object signal
statistics, leave-one-corruption-family-out, trained and tested on disjoint source images.
Reports AUROC and recall of confident omissions at 5% / 10% false-positive rate (on well-segmented objects).

usage: python oracle_region_v6.py <results dir>
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_v6 import family  # noqa: E402

SOFTMAX = ["ent", "margin"]
NON_SOFTMAX = ["tta_js", "maha_min", "maha_pred", "xmodel_js", "depth_fg", "det_stuff"]


def recall_at_fpr(y, s, fpr):
    neg = np.sort(s[y == 0])[::-1]
    thr = neg[min(int(np.floor(fpr * len(neg))), len(neg) - 1)]
    return float((s[y == 1] > thr).mean())


def run(obj: pd.DataFrame, seed: int = 0) -> dict:
    obj = obj[~obj.condition.str.startswith("gaussian_noise")]
    obj = obj[obj.confident_omission | (obj.frac_true >= 0.5)].copy()
    obj["fam"] = obj.condition.map(family)
    y = obj.confident_omission.astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    stems = obj.stem.unique()
    train_stems = set(rng.choice(stems, len(stems) // 2, replace=False))
    is_tr = obj.stem.isin(train_stems).to_numpy()
    sets = {
        "non_softmax": [f"{s}_{k}" for s in NON_SOFTMAX for k in ("mean", "hit02", "hit10")],
        "all": [f"{s}_{k}" for s in SOFTMAX + NON_SOFTMAX for k in ("mean", "hit02", "hit10")],
    }
    out = {}
    for name, cols in sets.items():
        X = obj[cols].fillna(0).to_numpy(dtype=np.float32)
        out[name] = {}
        for held in ["noise", "blur", "digital", "weather", "dark"]:
            tr = is_tr & (obj.fam != held).to_numpy()
            te = ~is_tr & (obj.fam == held).to_numpy()
            if y[te].sum() < 15 or y[tr].sum() < 30:
                continue
            res = {"n_co_test": int(y[te].sum())}
            for mname, model in (("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                                 ("gbm", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced"))):
                model.fit(X[tr], y[tr])
                p = model.predict_proba(X[te])[:, 1]
                res[mname] = {"auroc": round(float(roc_auc_score(y[te], p)), 3),
                              "recall_fpr05": round(recall_at_fpr(y[te], p, 0.05), 3),
                              "recall_fpr10": round(recall_at_fpr(y[te], p, 0.10), 3)}
            out[name][held] = res
    return out


def main():
    d = sys.argv[1]
    report = {}
    for f in sorted(glob.glob(os.path.join(d, "objects_*.csv"))):
        tag = os.path.basename(f)[len("objects_"):-4]
        report[tag] = run(pd.read_csv(f))
        print(f"\n== {tag}")
        for fs, r in report[tag].items():
            print(f"  {fs}:", {h: (v["gbm"]["auroc"], v["gbm"]["recall_fpr05"], v["gbm"]["recall_fpr10"]) for h, v in r.items()})
    with open(os.path.join(d, "oracle_region_v6.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
