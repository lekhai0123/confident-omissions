"""E9: qualitative figure for the cheap verifier (1 segmenter + RT-DETR with gamma pass).
Verifier trained on v8 (4 models, all conditions) EXCLUDING the frames shown; threshold calibrated at 1 FA/image.
Panels: image with GT confident omissions into background (yellow), flagged proposals (green = covers an omitted
object, red = false alarm); right: the segmenter's prediction."""
import os, sys
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebook"))
from e_suite_v8 import load, feature_sets, STUFF, MODELS   # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier
import pilot   # noqa: E402

v8, loc = sys.argv[1], sys.argv[2]
L = pd.read_csv(os.path.join(loc, "proposals_segformer-b5.csv"))
LO = pd.read_csv(os.path.join(loc, "objects_segformer-b5.csv"))
stems = set(L.stem)
P, O = load(v8)
cols = [c for c in feature_sets(P)["cheap_single_model"] if c in L.columns or c.startswith("m_")]
missing = [c for c in feature_sets(P)["cheap_single_model"] if c not in cols]
print("features used", len(cols), "missing in local run:", missing)
tr = P[P.det_src.isin(["rtdetr", "rtdetr_gamma"]) & ~P.stem.isin(stems)]
rng = np.random.default_rng(0)
keep = tr.pos_co_stuff.to_numpy() | (rng.random(len(tr)) < 0.25)
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
m.fit(tr[keep][cols].fillna(-1).to_numpy(np.float32), tr[keep].pos_co_stuff.to_numpy())
s_tr = m.predict_proba(tr[cols].fillna(-1).to_numpy(np.float32))[:, 1]
neg = np.sort(s_tr[~tr.pos.to_numpy()])[::-1]
thr = neg[min(int(1.0 * tr.unit.nunique()), len(neg) - 1)]
for t in MODELS:
    L[f"m_{t}"] = float(t == "segformer-b5")
L["s"] = m.predict_proba(L[cols].fillna(-1).to_numpy(np.float32))[:, 1]
LO["co_stuff"] = LO.confident_omission & LO.predicted_as.isin(STUFF)
co = LO[LO.co_stuff][["stem", "instance_id"]].rename(columns={"instance_id": "gt_instance_id"}).assign(is_co=True)
L = L.merge(co, on=["stem", "gt_instance_id"], how="left")
L["is_co"] = L.is_co.fillna(False).astype(bool) & (L.gt_cover >= 0.5)
om = LO[LO.omitted][["stem", "instance_id"]].rename(columns={"instance_id": "gt_instance_id"}).assign(is_om=True)
L = L.merge(om, on=["stem", "gt_instance_id"], how="left")
L["is_om"] = L.is_om.fillna(False).astype(bool) & (L.gt_cover >= 0.5)
F = L[L.s > thr]
print(f"threshold {thr:.3f}; flagged {len(F)} on {L.stem.nunique()} images ({len(F)/L.stem.nunique():.2f}/img); "
      f"false alarms {int((~F.is_om).sum())}; CO found {F[F.is_co].drop_duplicates(['stem','gt_instance_id']).shape[0]} "
      f"of {int(LO.co_stuff.sum())}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
items = {it["stem"]: it for it in pilot.list_acdc(sys.argv[3], ("train", "val"), False) if it["stem"] in stems}
order = LO[LO.co_stuff].sort_values("area", ascending=False).drop_duplicates("stem")
order = order.groupby("condition").head(2).stem.tolist()[:8]
fig, axes = plt.subplots(len(order), 2, figsize=(10, 2.9 * len(order)))
for row, stem in zip(axes, order):
    it = items[stem]
    x, inst, _ = pilot.load_acdc(it, 1.0)
    img = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ids = set(LO[(LO.stem == stem) & LO.co_stuff].instance_id)
    row[0].imshow(img)
    for iid in ids:
        row[0].contour(inst == iid, levels=[0.5], colors=["#ffd400"], linewidths=1.6)
    for _, r in F[F.stem == stem].iterrows():
        col = "#1baf7a" if r.is_om else "#ff3b30"
        row[0].add_patch(Rectangle((2 * r.box_x0, 2 * r.box_y0), 2 * (r.box_x1 - r.box_x0), 2 * (r.box_y1 - r.box_y0),
                                   fill=False, ec=col, lw=1.6))
    row[0].set_title(f"{it['cond']}  {stem}", fontsize=8)
    pred = np.load(os.path.join(loc, "pred_cache", f"{stem}.npy")) if os.path.exists(os.path.join(loc, "pred_cache", f"{stem}.npy")) else None
    if pred is None:
        row[1].imshow(img); row[1].set_title("(prediction not cached)", fontsize=8)
    else:
        row[1].imshow(pilot.colorize(pred) if hasattr(pilot, "colorize") else pred); row[1].set_title("SegFormer-B5 prediction", fontsize=8)
    for a in row:
        a.axis("off")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(loc, f"fig_qualitative_cheap.{ext}"), dpi=130, bbox_inches="tight")
print("saved")
