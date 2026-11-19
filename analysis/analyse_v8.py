"""v8 (ACDC, denser proposals + second-opinion features): proposal recall and verifier ablation.

usage: python analyse_v8.py <v8 results dir>
Writes <dir>/analysis_v8.json.

Proposal recall: share of omitted / confident-omission (CO) objects covered (gt_cover >= 0.5) by >= 1 proposal,
overall and by detector source; also with a tighter match (cover >= 0.5 and IoU >= 0.2).
Verifier: gradient boosting, leave-one-adverse-condition-out (train on the other conditions incl. refs), feature
sets: v7 (segmenter signals + geometry), +CLIP, +zoom, all. Object recall at N false alarms per image, with the
threshold (a) calibrated on the training conditions and (b) set on the test condition (oracle operating point).
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

ADVERSE = ["fog", "night", "rain", "snow"]
FA = (0.5, 1.0, 2.0)
COVER = 0.5


def attach(props: pd.DataFrame, obj: pd.DataFrame) -> pd.DataFrame:
    t = obj[["stem", "condition", "instance_id", "omitted", "confident_omission", "area"]].rename(
        columns={"instance_id": "gt_instance_id", "area": "gt_area"})
    p = props.merge(t, on=["stem", "condition", "gt_instance_id"], how="left")
    p[["omitted", "confident_omission"]] = p[["omitted", "confident_omission"]].fillna(False).astype(bool)
    p["pos"] = p.omitted & (p.gt_cover >= COVER)
    p["pos_co"] = p.confident_omission & (p.gt_cover >= COVER)
    p["det_src"] = p.det_src.fillna("xmodel")
    for s in ("rtdetr", "rtdetr_gamma", "owl", "xmodel"):
        p[f"src_{s}"] = (p.det_src == s).astype(float)
    if "zoom_top" in p:
        p["zoom_top_thing"] = (p.zoom_top >= 11).astype(float)
    return p


def recall_table(p: pd.DataFrame, obj: pd.DataFrame) -> dict:
    out = {}
    for c, g in obj.groupby("condition"):
        pc = p[p.condition == c]
        r = {"props_per_image": round(len(pc) / max(1, pc.stem.nunique()), 1)}
        for match, sel in (("cover", pc.gt_cover >= COVER), ("cover_iou", (pc.gt_cover >= COVER) & (pc.gt_iou >= 0.2))):
            hit = pc[sel]
            keys = {s: set(zip(hit[hit.det_src == s].stem, hit[hit.det_src == s].gt_instance_id))
                    for s in ("rtdetr", "rtdetr_gamma", "owl", "xmodel")}
            allk = set(zip(hit.stem, hit.gt_instance_id))
            for name, msk in (("co", g.confident_omission), ("omitted", g.omitted)):
                ids = list(zip(g[msk].stem, g[msk].instance_id))
                if not ids:
                    continue
                r[f"{match}_recall_{name}"] = round(float(np.mean([i in allk for i in ids])), 3)
                if match == "cover":
                    for s, k in keys.items():
                        r[f"recall_{name}_{s}"] = round(float(np.mean([i in k for i in ids])), 3)
                    r[f"n_{name}"] = len(ids)
        out[c] = r
    return out


def found_at(te: pd.DataFrame, score: np.ndarray, thr: float, target: str) -> tuple[int, float]:
    flag = score > thr
    fa = float((flag & ~te.pos.to_numpy()).sum() / max(1, te.stem.nunique()))
    m = flag & te[target].to_numpy()
    return len(set(zip(te.stem[m], te.gt_instance_id[m]))), fa


def verifier(p: pd.DataFrame, obj: pd.DataFrame) -> dict:
    base = [c for c in p.columns if c.startswith("p_")] + ["det_score", "src_rtdetr", "src_rtdetr_gamma", "src_owl", "src_xmodel"]
    clip = [c for c in p.columns if c.startswith("clip_p_")] + ["clip_thing_prob", "clip_top_is_thing"]
    zoom = ["zoom_done", "zoom_thing_frac", "zoom_thing_prob", "zoom_top_thing"]
    sets = {"v7_features": base, "plus_clip": base + clip, "plus_zoom": base + zoom, "all": base + clip + zoom}
    sets = {k: [c for c in v if c in p.columns] for k, v in sets.items()}
    out = {}
    for held in ADVERSE:
        tr = p[~p.condition.isin([held, held + "_ref"])]
        te = p[p.condition == held]
        o = obj[obj.condition == held]
        n_om, n_co = int(o.omitted.sum()), int(o.confident_omission.sum())
        res = {"n_images": int(te.stem.nunique()), "n_omitted": n_om, "n_co": n_co}
        n_tr, n_te = tr.stem.nunique(), te.stem.nunique()
        for name, cols in sets.items():
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
            m.fit(tr[cols].fillna(-1).to_numpy(np.float32), tr.pos.to_numpy())
            s_tr = m.predict_proba(tr[cols].fillna(-1).to_numpy(np.float32))[:, 1]
            s_te = m.predict_proba(te[cols].fillna(-1).to_numpy(np.float32))[:, 1]
            r = {"auroc": round(float(roc_auc_score(te.pos, s_te)), 3)}
            neg_tr = np.sort(s_tr[~tr.pos.to_numpy()])[::-1]
            neg_te = np.sort(s_te[~te.pos.to_numpy()])[::-1]
            for fa in FA:
                for mode, neg, n in (("cal", neg_tr, n_tr), ("oracle", neg_te, n_te)):
                    thr = neg[min(int(fa * n), len(neg) - 1)]
                    f_om, fa_te = found_at(te, s_te, thr, "pos")
                    f_co, _ = found_at(te, s_te, thr, "pos_co")
                    r[f"{mode}_fa{fa}"] = {"recall_omitted": round(f_om / max(1, n_om), 3),
                                           "recall_co": round(f_co / max(1, n_co), 3), "fa_test": round(fa_te, 2)}
            res[name] = r
        out[held] = res
    return out


def main():
    d = sys.argv[1]
    report = {}
    for f in sorted(glob.glob(os.path.join(d, "objects_*.csv"))):
        tag = os.path.basename(f)[len("objects_"):-4]
        obj = pd.read_csv(f)
        p = attach(pd.read_csv(os.path.join(d, f"proposals_{tag}.csv")), obj)
        report[tag] = {"proposal_recall": recall_table(p, obj), "verifier_loco": verifier(p, obj)}
        print(f"\n== {tag}")
        for c, r in report[tag]["proposal_recall"].items():
            print(f"  {c:10s} props/img {r['props_per_image']:5.1f}  CO {r.get('cover_recall_co')} (iou {r.get('cover_iou_recall_co')})"
                  f"  omitted {r.get('cover_recall_omitted')}  CO by src rt {r.get('recall_co_rtdetr')} gam {r.get('recall_co_rtdetr_gamma')}"
                  f" owl {r.get('recall_co_owl')} xm {r.get('recall_co_xmodel')}")
        for c, r in report[tag]["verifier_loco"].items():
            print(f"  verifier {c:6s} n_co {r['n_co']:3d}", {k: (v["auroc"], v["cal_fa1.0"]["recall_co"], v["oracle_fa1.0"]["recall_co"],
                                                             v["oracle_fa1.0"]["recall_omitted"]) for k, v in r.items() if isinstance(v, dict)})
    with open(os.path.join(d, "analysis_v8.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
