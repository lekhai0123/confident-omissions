"""Robustness check of the CO-specific pooled verifier (v8): calibrated vs oracle threshold, recall over ALL CO
objects (incl. uncovered), per model, simple baselines at the same operating point. False alarm = flagged proposal
not covering any omitted object, counted per image x model."""
import glob, os, sys, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_v8 import attach, ADVERSE
d = sys.argv[1]
frames, objs = [], []
for f in sorted(glob.glob(os.path.join(d, "objects_*.csv"))):
    tag = os.path.basename(f)[8:-4]
    obj = pd.read_csv(f, usecols=["stem", "condition", "instance_id", "omitted", "confident_omission", "area"]); obj["model"] = tag
    p = attach(pd.read_csv(os.path.join(d, f"proposals_{tag}.csv")), obj)
    p["model"] = tag
    for t in ("segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"):
        p[f"m_{t}"] = float(tag == t)
    frames.append(p); objs.append(obj)
P = pd.concat(frames, ignore_index=True); O = pd.concat(objs, ignore_index=True)
P["unit"] = P.stem + "|" + P.model
P["rule"] = P.det_score * P.p_self_stuff_frac * (1 - P.p_ent_mean)          # detector says thing, model says stuff, confidently
cols = [c for c in P.columns if c.startswith(("p_", "clip_p_", "m_"))] + ["det_score", "src_rtdetr", "src_rtdetr_gamma", "src_owl",
        "src_xmodel", "clip_thing_prob", "clip_top_is_thing", "zoom_done", "zoom_thing_frac", "zoom_thing_prob", "zoom_top_thing"]
no_soft = [c for c in cols if not c.startswith(("p_ent", "p_margin"))]
rng = np.random.default_rng(0)

def evaluate(te, s, thr, o_te):
    flag = s > thr
    fa = (flag & ~te.pos.to_numpy()).sum() / te.unit.nunique()
    hit = te[flag & te.pos_co.to_numpy()]
    found = set(zip(hit.model, hit.stem, hit.gt_instance_id))
    co = o_te[o_te.confident_omission]
    res = {"fa_per_image_model": round(float(fa), 2), "recall_all_co": round(len(found) / max(1, len(co)), 3), "n_co": int(len(co))}
    for m, g in co.groupby("model"):
        ids = list(zip(g.model, g.stem, g.instance_id))
        res[f"recall_{m}"] = round(float(np.mean([i in found for i in ids])), 3)
    return res

out = {}
for held in ADVERSE:
    tr = P[~P.condition.isin([held, held + "_ref"])]
    te = P[P.condition == held]
    o_te = O[O.condition == held]
    keep = tr.pos_co.to_numpy() | (rng.random(len(tr)) < 0.25)
    scores = {}
    for name, fc in (("verifier_all", cols), ("verifier_no_softmax", no_soft)):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
        m.fit(tr[keep][fc].fillna(-1).to_numpy(np.float32), tr[keep].pos_co.to_numpy())
        scores[name] = (m.predict_proba(tr[fc].fillna(-1).to_numpy(np.float32))[:, 1], m.predict_proba(te[fc].fillna(-1).to_numpy(np.float32))[:, 1])
    for b in ("rule", "det_score", "p_xmodel_js_mean", "clip_thing_prob"):
        scores[b] = (tr[b].fillna(0).to_numpy(), te[b].fillna(0).to_numpy())
    for name, (s_tr, s_te) in scores.items():
        for fa in (0.5, 1.0, 2.0):
            for mode, s_ref, ref in (("cal", s_tr, tr), ("oracle", s_te, te)):
                neg = np.sort(s_ref[~ref.pos.to_numpy()])[::-1]
                thr = neg[min(int(fa * ref.unit.nunique()), len(neg) - 1)]
                out.setdefault(held, {}).setdefault(name, {})[f"{mode}_fa{fa}"] = evaluate(te, s_te, thr, o_te)
    print(held, {n: (v["cal_fa1.0"]["recall_all_co"], v["cal_fa1.0"]["fa_per_image_model"], v["oracle_fa1.0"]["recall_all_co"]) for n, v in out[held].items()}, flush=True)
    print("   per model (verifier_all, cal fa1):", {k: v for k, v in out[held]["verifier_all"]["cal_fa1.0"].items() if k.startswith("recall_")}, flush=True)
json.dump(out, open(os.path.join(d, "verifier_co_v8_check.json"), "w"), indent=1)
