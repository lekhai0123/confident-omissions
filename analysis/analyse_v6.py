"""Offline analysis of the v6 run (P4 feasibility: 4 models, 17 conditions, GT-free signals).

usage: python analyse_v6.py <results dir> [--boot 500]
Writes <results dir>/analysis_v6.json and prints a compact report.

1. Recall at a fixed alarm budget: share of confident-omission (CO) objects with >= half of their pixels
   inside the image's top-10% pixels of a signal (hit10 >= 0.5). Control: the same share for objects the
   model segments well (false alarms on good objects). CIs: bootstrap over source images.
2. Pixel level: AUROC CO pixels vs correct pixels; leave-one-corruption-family-out (LOCO) logistic fusion
   trained on half of the images, tested on the other half within the held-out family.
3. MetaFusion ML rule recovery, cross-model overlap of CO objects.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

THINGS = {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
SIGNALS = ["ent", "margin", "tta_js", "maha_min", "maha_pred", "xmodel_js", "depth_fg", "det_stuff", "fused"]
NON_SOFTMAX = ["tta_js", "maha_min", "maha_pred", "xmodel_js", "depth_fg", "det_stuff"]
FAMILY = {"gaussian_noise": "noise", "shot_noise": "noise", "gaussian_blur": "blur", "defocus_blur": "blur",
          "motion_blur": "blur", "jpeg": "digital", "contrast": "digital", "brightness": "weather",
          "fog": "weather", "dark": "dark"}


def family(cond: str) -> str:
    return "clean" if cond == "clean" else FAMILY[cond.rsplit("-", 1)[0]]


def boot_share(flag, groups, n_boot, rng):
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=flag.astype(float), minlength=len(ug))
    cnts = np.bincount(inv, minlength=len(ug)).astype(float)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        c = cnts[pick].sum()
        if c:
            vals.append(sums[pick].sum() / c)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def recall_table(obj, n_boot, rng, budget="hit10"):
    co, good = obj[obj.confident_omission], obj[obj.frac_true >= 0.5]
    out = {}
    for s in SIGNALS:
        col = f"{s}_{budget}"
        if col not in obj or not len(co):
            continue
        f_co = (co[col] >= 0.5).to_numpy()
        out[s] = {"recall_co": float(f_co.mean()),
                  "recall_co_ci95": boot_share(f_co, co.stem.to_numpy(), n_boot, rng) if n_boot else None,
                  "flag_rate_good_objects": float((good[col] >= 0.5).mean()) if len(good) else None}
    return out


def object_auroc(obj):
    sub = obj[obj.confident_omission | (obj.frac_true >= 0.5)]
    y = sub.confident_omission.astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return {}
    return {s: float(roc_auc_score(y, sub[f"{s}_mean"])) for s in SIGNALS if f"{s}_mean" in sub}


def pixel_analysis(npz_path, seed=0):
    z = np.load(npz_path)
    names = [str(s) for s in z["signals"]]
    conds = [str(c) for c in z["conditions"]]
    X = np.nan_to_num(z["scores"].astype(np.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    kind, img, cond = z["kind"], z["image"], z["condition"]
    fam = np.array([family(c) for c in conds])[cond]
    gnoise = np.array([c.startswith("gaussian_noise") for c in conds])[cond]
    keep = np.isin(kind, [0, 1])
    y = (kind == 1).astype(int)
    res = {"n_pixels": int(keep.sum()), "n_co_pixels": int(y.sum())}
    m = keep & ~gnoise
    res["auroc_co_vs_correct_no_gnoise"] = {n: float(roc_auc_score(y[m], X[m, i])) for i, n in enumerate(names)}
    rng = np.random.default_rng(seed)
    uimg = np.unique(img)
    train_img = rng.choice(uimg, len(uimg) // 2, replace=False)
    is_train_img = np.isin(img, train_img)
    feats = {"non_softmax": [names.index(n) for n in NON_SOFTMAX if n in names],
             "all_signals": [i for i, n in enumerate(names) if n != "fused"]}
    loco = {}
    for fset, cols in feats.items():
        loco[fset] = {}
        for held in ["noise", "blur", "digital", "weather", "dark"]:
            tr = keep & is_train_img & (fam != held)
            te = keep & ~is_train_img & (fam == held)
            if y[te].sum() < 20 or y[tr].sum() < 20:
                continue
            pos, neg = np.flatnonzero(tr & (y == 1)), np.flatnonzero(tr & (y == 0))
            neg = rng.choice(neg, min(len(neg), 5 * len(pos)), replace=False)
            idx = np.concatenate([pos, neg])
            sc = StandardScaler().fit(X[idx][:, cols])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[idx][:, cols]), y[idx])
            p = clf.predict_proba(sc.transform(X[te][:, cols]))[:, 1]
            best = max(((n, float(roc_auc_score(y[te], X[te, i]))) for i, n in enumerate(names)), key=lambda kv: kv[1])
            loco[fset][held] = {"auroc": float(roc_auc_score(y[te], p)), "n_co_pixels": int(y[te].sum()),
                                "best_single": best,
                                "coef": dict(zip([names[c] for c in cols], np.round(clf.coef_[0], 3).tolist()))}
    res["loco_fusion"] = loco
    return res


def analyse_model(obj, n_boot, rng):
    obj = obj.copy()
    obj["family"] = obj.condition.map(family)
    obj["true_miss"] = obj.confident_omission & ~obj.predicted_as.isin(THINGS)
    ng = obj[~obj.condition.str.startswith("gaussian_noise")]
    res = {"events": {}, "ml_rule": {}, "recall_hit10_no_gnoise": recall_table(ng, n_boot, rng),
           "recall_hit02_no_gnoise": recall_table(ng, 0, rng, "hit02"),
           "recall_hit10_by_family": {}, "object_auroc_no_gnoise": object_auroc(ng)}
    for name, d in (("all", obj), ("no_gnoise", ng), ("clean", obj[obj.condition == "clean"])):
        co = d[d.confident_omission]
        res["events"][name] = {"co": int(len(co)),
                               "unique_objects": int(co[["stem", "instance_id"]].drop_duplicates().shape[0]),
                               "true_miss_unique": int(d[d.true_miss][["stem", "instance_id"]].drop_duplicates().shape[0]),
                               "source_images": int(co.stem.nunique())}
        if len(co):
            res["ml_rule"][name] = {"recovered_share": float((~co.omitted_ml).mean()),
                                    "recovered_majority": float((co.frac_true_ml >= 0.5).mean()),
                                    "median_p_true": float(co.mean_p_true.median())}
    res["events"]["by_condition"] = obj[obj.confident_omission].condition.value_counts().to_dict()
    for fam, d in ng.groupby("family"):
        if d.confident_omission.sum() >= 10:
            res["recall_hit10_by_family"][fam] = {s: v["recall_co"] for s, v in recall_table(d, 0, rng).items()}
            res["recall_hit10_by_family"][fam]["n_co"] = int(d.confident_omission.sum())
    return res


def cross_model(objs):
    key = ["stem", "instance_id", "condition"]
    sets = {t: set(map(tuple, o[key].to_numpy().tolist())) for t, o in objs.items()}
    return {f"{a}_in_{b}": round(len(sets[a] & sets[b]) / len(sets[a]), 3)
            for a in sets for b in sets if a != b and sets[a]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    report, objs = {}, {}
    for f in sorted(glob.glob(os.path.join(a.results, "objects_*.csv"))):
        tag = os.path.basename(f)[len("objects_"):-4]
        obj = pd.read_csv(f)
        objs[tag] = obj[obj.confident_omission]
        r = analyse_model(obj, a.boot, rng)
        pix = os.path.join(a.results, f"pixels_{tag}.npz")
        if os.path.exists(pix):
            r["pixels"] = pixel_analysis(pix)
        report[tag] = r
        print(f"\n===== {tag}")
        print("events no-gnoise:", json.dumps(r["events"]["no_gnoise"]), "| clean:", json.dumps(r["events"]["clean"]))
        print("ML rule (no gnoise):", json.dumps(r["ml_rule"].get("no_gnoise")))
        print("recall of CO at top-10% budget (no gnoise) [flag rate on well-segmented objects]:")
        for s, v in r["recall_hit10_no_gnoise"].items():
            ci = v["recall_co_ci95"]
            print(f"  {s:10s} {v['recall_co']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]   good {v['flag_rate_good_objects']:.3f}")
        print("recall top-2%:", {s: round(v["recall_co"], 3) for s, v in r["recall_hit02_no_gnoise"].items()})
        for k, v in r["recall_hit10_by_family"].items():
            print(f"  family {k:8s}", {s: (round(x, 2) if isinstance(x, float) else x) for s, x in v.items()})
        print("object AUROC CO vs well-segmented:", {s: round(x, 3) for s, x in r["object_auroc_no_gnoise"].items()})
        if "pixels" in r:
            print("pixel AUROC CO vs correct (no gnoise):", {s: round(x, 3) for s, x in r["pixels"]["auroc_co_vs_correct_no_gnoise"].items()})
            for fset, d in r["pixels"]["loco_fusion"].items():
                print(f"  LOCO {fset}:", {k: (round(v['auroc'], 3), v["best_single"][0], round(v["best_single"][1], 3)) for k, v in d.items()})
    report["cross_model"] = cross_model(objs)
    print("\ncross-model CO overlap:", json.dumps(report["cross_model"]))
    with open(os.path.join(a.results, "analysis_v6.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
