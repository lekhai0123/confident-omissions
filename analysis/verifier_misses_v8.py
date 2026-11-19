"""Why does the CO verifier miss? For each CO object in held-out adverse conditions (LOCO, calibrated threshold at
1 FA/image-model): covered by a proposal? best verifier score; size, class, frac_invalid, mean_conf, predicted_as.
Bootstrap CI (over images) of the pooled recall."""
import glob, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_v8 import attach, ADVERSE
d = sys.argv[1]
frames, objs = [], []
for f in sorted(glob.glob(os.path.join(d, "objects_*.csv"))):
    tag = os.path.basename(f)[8:-4]
    obj = pd.read_csv(f); obj["model"] = tag
    p = attach(pd.read_csv(os.path.join(d, f"proposals_{tag}.csv")), obj); p["model"] = tag
    for t in ("segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"):
        p[f"m_{t}"] = float(tag == t)
    frames.append(p); objs.append(obj)
P = pd.concat(frames, ignore_index=True); O = pd.concat(objs, ignore_index=True)
P["unit"] = P.stem + "|" + P.model
cols = [c for c in P.columns if c.startswith(("p_", "clip_p_", "m_"))] + ["det_score", "src_rtdetr", "src_rtdetr_gamma", "src_owl",
        "src_xmodel", "clip_thing_prob", "clip_top_is_thing", "zoom_done", "zoom_thing_frac", "zoom_thing_prob", "zoom_top_thing"]
rng = np.random.default_rng(0)
rows = []
for held in ADVERSE:
    tr = P[~P.condition.isin([held, held + "_ref"])]
    te = P[P.condition == held].copy()
    keep = tr.pos_co.to_numpy() | (rng.random(len(tr)) < 0.25)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
    m.fit(tr[keep][cols].fillna(-1).to_numpy(np.float32), tr[keep].pos_co.to_numpy())
    s_tr = m.predict_proba(tr[cols].fillna(-1).to_numpy(np.float32))[:, 1]
    te["s"] = m.predict_proba(te[cols].fillna(-1).to_numpy(np.float32))[:, 1]
    neg = np.sort(s_tr[~tr.pos.to_numpy()])[::-1]
    thr = neg[min(int(1.0 * tr.unit.nunique()), len(neg) - 1)]
    best = te[te.pos_co].groupby(["model", "stem", "gt_instance_id"]).s.max()
    co = O[(O.condition == held) & O.confident_omission].copy()
    key = list(zip(co.model, co.stem, co.instance_id))
    co["covered"] = [k in best.index for k in key]
    co["best_s"] = [best.get(k, np.nan) for k in key]
    co["found"] = co.best_s > thr
    rows.append(co)
C = pd.concat(rows, ignore_index=True)
C["size"] = pd.cut(C.area, [0, 256, 1024, 4096, 1e9], labels=["<256", "256-1k", "1k-4k", ">4k"])
C["invalid"] = C.frac_invalid.fillna(0) > 0.5
print("pooled recall", round(C.found.mean(), 3), "covered", round(C.covered.mean(), 3), "found|covered", round(C[C.covered].found.mean(), 3))
st = C.stem.unique(); idx = {s: np.where(C.stem.to_numpy() == s)[0] for s in st}
bs = [C.found.to_numpy()[np.concatenate([idx[s] for s in rng.choice(st, len(st))])].mean() for _ in range(1000)]
print("95% CI (bootstrap over images):", np.round(np.percentile(bs, [2.5, 97.5]), 3))
for col in ["condition", "model", "size", "class", "invalid", "predicted_as"]:
    t = C.groupby(col, observed=True).agg(n=("found", "size"), covered=("covered", "mean"), found=("found", "mean")).round(3)
    print(f"\n-- by {col}\n{t.sort_values('n', ascending=False).head(10).to_string()}")
C["conf_bin"] = pd.cut(C.mean_conf, [0, 0.8, 0.9, 0.95, 1.0])
print("\n-- by mean_conf\n", C.groupby("conf_bin", observed=True).agg(n=("found", "size"), covered=("covered", "mean"), found=("found", "mean")).round(3).to_string())
C.to_csv(os.path.join(d, "verifier_misses_v8.csv"), index=False)
