"""E16 helper: depth-fg probability for every GT thing instance on 500 ACDC adverse frames (local; depth + fg/bg net only)."""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "notebook"); import pilot
root, ck, out = sys.argv[1], sys.argv[2], sys.argv[3]
items = [it for it in pilot.list_acdc(root, ("train", "val"), False)]
rng = np.random.default_rng(0)
items = [items[i] for i in rng.permutation(len(items))[:500]]
depth = pilot.DepthCue("depth-anything/Depth-Anything-V2-Small-hf", "cuda")
net = pilot.FgBgNet("cuda", ck)
rows = []
for it in items:
    x, inst, _ = pilot.load_acdc(it, 1.0)
    H2, W2 = (x.shape[-2] + 1) // 2, (x.shape[-1] + 1) // 2
    fg = net.prob(pilot.depth_input(depth, x.cuda(), (H2, W2)))[0].cpu().numpy()
    for iid, t, mk in pilot.instance_objects(inst, 64):
        m2 = mk[::2, ::2][:H2, :W2]
        if m2.sum():
            rows.append({"stem": it["stem"], "condition": it["cond"], "instance_id": iid, "fg_mean": float(fg[m2].mean())})
pd.DataFrame(rows).to_csv(out, index=False); print(len(rows), "objects")
