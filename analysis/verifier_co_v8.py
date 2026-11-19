"""v8: verifier trained specifically for confident omissions, pooled over the 4 models (more positives).
Leave-one-adverse-condition-out; negatives subsampled 25% for training only; oracle and calibrated thresholds."""
import glob, os, sys, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_v8 import attach, found_at, ADVERSE
d = sys.argv[1]
frames = []
for f in sorted(glob.glob(os.path.join(d, "objects_*.csv"))):
    tag = os.path.basename(f)[8:-4]
    obj = pd.read_csv(f); obj["model"] = tag
    p = attach(pd.read_csv(os.path.join(d, f"proposals_{tag}.csv")), obj)
    p["stem"] = p.stem + "|" + tag            # image x model unit for false alarms
    for t in ("segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"):
        p[f"m_{t}"] = float(tag == t)
    frames.append(p)
P = pd.concat(frames, ignore_index=True)
cols = [c for c in P.columns if c.startswith(("p_", "clip_p_", "m_"))] + ["det_score", "src_rtdetr", "src_rtdetr_gamma", "src_owl",
        "src_xmodel", "clip_thing_prob", "clip_top_is_thing", "zoom_done", "zoom_thing_frac", "zoom_thing_prob", "zoom_top_thing"]
no_soft = [c for c in cols if not c.startswith(("p_ent", "p_margin"))]
rng = np.random.default_rng(0)
out = {}
for target in ("pos_co", "pos"):
    for fname, fc in (("all", cols), ("no_softmax", no_soft)):
        for held in ADVERSE:
            tr = P[~P.condition.isin([held, held + "_ref"])]
            keep = tr[target].to_numpy() | (rng.random(len(tr)) < 0.25)
            trs = tr[keep]
            te = P[P.condition == held]
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
            m.fit(trs[fc].fillna(-1).to_numpy(np.float32), trs[target].to_numpy())
            s = m.predict_proba(te[fc].fillna(-1).to_numpy(np.float32))[:, 1]
            neg = np.sort(s[~te.pos.to_numpy()])[::-1]
            n_img = te.stem.nunique()
            n_co = int(te[te.pos_co].drop_duplicates(["stem", "gt_instance_id"]).shape[0])
            r = {"auroc_co": round(float(roc_auc_score(te.pos_co, s)), 3), "n_co_covered": n_co}
            for fa in (0.5, 1.0, 2.0):
                thr = neg[min(int(fa * n_img), len(neg) - 1)]
                f_co, _ = found_at(te, s, thr, "pos_co")
                r[f"oracle_fa{fa}_co_recall_of_covered"] = round(f_co / max(1, n_co), 3)
            out.setdefault(f"{target}/{fname}", {})[held] = r
            print(target, fname, held, r, flush=True)
json.dump(out, open(os.path.join(d, "verifier_co_v8.json"), "w"), indent=1)
