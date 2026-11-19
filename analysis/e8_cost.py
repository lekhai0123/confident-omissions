"""E8: inference cost per component on ACDC frames (local GPU, fp16 autocast as in the pipeline)."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebook"))
import pilot
root = sys.argv[1]
dev = "cuda"
items = pilot.list_acdc(root, ("val",), False)[:6]
xs = [pilot.load_acdc(it, 1.0)[0].to(dev) for it in items]
def params(m): return sum(p.numel() for p in m.parameters()) / 1e6
def timeit(fn, n_warm=1):
    for x in xs[:n_warm]: fn(x)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    for x in xs[n_warm:]: fn(x)
    torch.cuda.synchronize()
    return (time.time() - t) / (len(xs) - n_warm) * 1000, torch.cuda.max_memory_allocated() / 2**30
rows = []
for name in ["nvidia/segformer-b0-finetuned-cityscapes-1024-1024", "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
             "facebook/mask2former-swin-small-cityscapes-semantic"]:
    seg = pilot.Segmenter(name, dev)
    ms, mem = timeit(lambda x: seg(x))
    rows.append((seg.tag + " forward", ms, mem, params(seg.model)))
    ms, mem = timeit(lambda x: seg(x.flip(-1), want_feat=False))
    rows.append((seg.tag + " flip view (TTA)", ms, mem, 0))
    del seg; torch.cuda.empty_cache()
det = pilot.DetectorCue(["PekingU/rtdetr_r50vd_coco_o365"], dev)
H2, W2 = xs[0].shape[-2] // 2, xs[0].shape[-1] // 2
ms, mem = timeit(lambda x: det(x, (H2, W2), box_thr=0.05)); rows.append(("RT-DETR R50 (one pass)", ms, mem, params(det.model)))
ms, mem = timeit(lambda x: det(x.clamp(0, 1) ** 0.5, (H2, W2), box_thr=0.05)); rows.append(("RT-DETR gamma pass", ms, mem, 0))
# proposals + features for one model (synthetic maps of the right size)
pred_h = torch.randint(0, 19, (H2, W2), device=dev); probs_h = torch.softmax(torch.randn(19, H2, W2, device=dev), 0)
stack = torch.rand(7, H2, W2, device=dev); masks = [torch.rand(H2, W2, device=dev) > 0.999 for _ in range(15)]
det(xs[0], (H2, W2), box_thr=0.05); boxes = det.last_boxes
def feat(_):
    props = pilot.proposal_masks(pred_h, {}, boxes, 16, max_props=90, max_det=60)
    pilot.proposal_records(props, probs_h, pred_h, {}, stack, list("abcdefg"), masks, list(range(15)))
ms, mem = timeit(feat); rows.append(("proposals + features (~60 boxes)", ms, mem, 0))
from sklearn.ensemble import HistGradientBoostingClassifier
X = np.random.rand(20000, 25).astype(np.float32); y = np.random.rand(20000) > 0.9
gb = HistGradientBoostingClassifier(max_iter=300).fit(X, y); Xp = np.random.rand(60, 25).astype(np.float32)
t = time.time(); [gb.predict_proba(Xp) for _ in range(50)]; rows.append(("verifier (GBM, 60 proposals)", (time.time() - t) / 50 * 1000, 0, 0))
print(f"GPU: {torch.cuda.get_device_name(0)}; frames {xs[0].shape[-1]}x{xs[0].shape[-2]}")
print(f"{'component':38s} {'ms/frame':>9s} {'peak GB':>8s} {'params M':>9s}")
for r in rows: print(f"{r[0]:38s} {r[1]:9.0f} {r[2]:8.2f} {r[3]:9.1f}")
