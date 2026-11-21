"""Pilot for topic 01: how often do confident whole-object omissions occur in semantic segmentation,
and do pixel uncertainty maps flag them?

Frozen SegFormer (Cityscapes) on Cityscapes val, clean and corrupted. For every ground-truth instance
(person, rider, car, truck, bus, train, motorcycle, bicycle) it records whether the object was omitted
(at most --omit-frac of its pixels get the true class), whether the omission was confident (most of the
object's pixels are confidently wrong), and how uncertain the model was inside the object relative to
the rest of the image. Outputs: objects.csv, images.csv, summary.json in --out.

Go/no-go (PLAN.md): enough confident omissions, and a clear share of them not flagged by uncertainty.

Local smoke test:  python pilot.py --smoke
Kaggle:            runs with defaults (data found under /kaggle/input, output to /kaggle/working)
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

# Cityscapes labelId -> trainId (19 evaluation classes); everything else is ignored (255)
LABELID_TO_TRAINID = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
                      24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
INSTANCE_LABELIDS = {24, 25, 26, 27, 28, 31, 32, 33}
TRAINID_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                 "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
                 "motorcycle", "bicycle"]
N_CLASSES = 19
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# corruption strengths (ImageNet-C / imagecorruptions conventions, severities 1..5)
NOISE_SIGMA = [0.08, 0.12, 0.18, 0.26, 0.38]
BLUR_SIGMA = [1.0, 2.0, 3.0, 4.0, 6.0]
BRIGHTNESS = [0.1, 0.2, 0.3, 0.4, 0.5]
CONTRAST = [0.4, 0.3, 0.2, 0.1, 0.05]
JPEG_QUALITY = [25, 18, 15, 10, 7]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- data
def find_root(explicit: str | None) -> str:
    """Directory that contains images/val and gtFine/val."""
    candidates = [explicit] if explicit else []
    candidates += [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "cityscapes_sample", "Cityspaces")]
    candidates += [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/images/val", recursive=True)]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "images", "val")) and os.path.isdir(os.path.join(c, "gtFine", "val")):
            return os.path.abspath(c)
    raise SystemExit(f"Cityscapes root not found; tried: {candidates}")


def list_val(root: str) -> list[tuple[str, str, str]]:
    """[(stem, image path, instanceIds path)] sorted by stem."""
    out = []
    for img in sorted(glob.glob(os.path.join(root, "images", "val", "*", "*_leftImg8bit.png"))):
        city = os.path.basename(os.path.dirname(img))
        stem = os.path.basename(img).replace("_leftImg8bit.png", "")
        inst = os.path.join(root, "gtFine", "val", city, f"{stem}_gtFine_instanceIds.png")
        if os.path.exists(inst):
            out.append((stem, img, inst))
    return out


def labels_from_instances(inst: np.ndarray) -> np.ndarray:
    """trainId map from an instanceIds map (ids >= 1000 encode labelId * 1000 + instance)."""
    label_ids = np.where(inst >= 1000, inst // 1000, inst)
    lut = np.full(256, 255, dtype=np.uint8)
    for lid, tid in LABELID_TO_TRAINID.items():
        lut[lid] = tid
    return lut[np.clip(label_ids, 0, 255)]


def instance_objects(inst: np.ndarray, min_area: int) -> list[tuple[int, int, np.ndarray]]:
    """[(instance id, trainId, mask)] for non-crowd instances of the instance classes with area >= min_area."""
    objs = []
    for iid in np.unique(inst):
        if iid < 1000 or (iid // 1000) not in INSTANCE_LABELIDS:
            continue  # stuff, crowd regions (id < 1000) or non-instance classes
        mask = inst == iid
        if int(mask.sum()) >= min_area:
            objs.append((int(iid), LABELID_TO_TRAINID[int(iid // 1000)], mask))
    return objs


# ---------------------------------------------------------------- corruptions (x: 1x3xHxW in [0,1])
def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = int(3 * sigma + 0.5)
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, 1, -1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k.repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k.transpose(2, 3).repeat(3, 1, 1, 1), groups=3)


def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    arr = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1)[None].to(x.device)


def corrupt(x: torch.Tensor, name: str, severity: int, gen: torch.Generator) -> torch.Tensor:
    s = severity - 1
    if name == "clean":
        return x
    if name == "gaussian_noise":
        noise = torch.randn(x.shape, generator=gen, device="cpu").to(x.device)
        return (x + NOISE_SIGMA[s] * noise).clamp(0, 1)
    if name == "gaussian_blur":
        return gaussian_blur(x, BLUR_SIGMA[s]).clamp(0, 1)
    if name == "brightness":
        return (x + BRIGHTNESS[s]).clamp(0, 1)
    if name == "contrast":
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - mean) * CONTRAST[s] + mean).clamp(0, 1)
    if name == "jpeg":
        return jpeg(x, JPEG_QUALITY[s])
    raise ValueError(name)


# ---------------------------------------------------------------- model
def load_model(name: str, device: str):
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(name)
    return model.eval().to(device)


@torch.no_grad()
def predict_logits(model, x: torch.Tensor, device: str) -> torch.Tensor:
    """Full-resolution logits (1 x C x H x W, float32) for an image tensor in [0,1]."""
    inp = ((x - MEAN.to(x.device)) / STD.to(x.device)).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        logits = model(pixel_values=inp).logits
    return F.interpolate(logits.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, iters: int = 100) -> float:
    """Single temperature minimising NLL on held-out pixels (logits N x C, labels N)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().item())


# ---------------------------------------------------------------- analysis
def object_records(pred: np.ndarray, conf: np.ndarray, ent: np.ndarray, labels: np.ndarray,
                   objs, conf_thr: float, omit_frac: float) -> list[dict]:
    valid = labels != 255
    conf_err = (pred != labels) & valid & (conf >= conf_thr)
    ent_valid = np.sort(ent[valid].ravel()) if valid.any() else np.sort(ent.ravel())
    rows = []
    for iid, tid, mask in objs:
        n = int(mask.sum())
        frac_true = float((pred[mask] == tid).mean())
        frac_conf_err = float(conf_err[mask].mean())
        mean_ent = float(ent[mask].mean())
        values, counts = np.unique(pred[mask], return_counts=True)
        # joint criterion (review 2026-11-22): pixels that are BOTH predicted as stuff (trainIds 0-10, as in the
        # paper) AND confident; ties of the majority class are flagged (np.unique + argmax picks the lower id)
        p_o, c_o = pred[mask].astype(np.int64), conf[mask] >= conf_thr
        st_o = p_o < 11
        cnt = np.bincount(p_o.ravel(), minlength=N_CLASSES)[:N_CLASSES]
        top_st = int(cnt[:11].argmax())
        srt = np.sort(cnt)[::-1]
        joint = {"frac_stuff": float(st_o.mean()), "frac_conf_stuff": float((st_o & c_o).mean()),
                 "top_stuff": TRAINID_NAMES[top_st], "frac_conf_top_stuff": float(((p_o == top_st) & c_o).mean()),
                 "maj_share": float(srt[0] / n), "maj_tie": bool(srt[0] == srt[1] and srt[0] > 0),
                 "frac_conf_thing_err": float(((p_o >= 11) & (p_o != tid) & c_o).mean())}
        rows.append({
            **joint,
            "instance_id": iid, "class": TRAINID_NAMES[tid], "area": n,
            "frac_true": frac_true, "omitted": frac_true <= omit_frac,
            "frac_conf_err": frac_conf_err,
            "confident_omission": frac_true <= omit_frac and frac_conf_err >= 0.5,
            "mean_conf": float(conf[mask].mean()), "mean_entropy": mean_ent,
            "max_entropy": float(ent[mask].max()),
            # share of the image's labelled pixels that are less uncertain than this object on average
            "entropy_percentile": float(np.searchsorted(ent_valid, mean_ent) / max(1, ent_valid.size)),
            "predicted_as": TRAINID_NAMES[int(values[counts.argmax()])] if int(values[counts.argmax()]) < N_CLASSES else "ignore",
        })
    return rows


def image_scores(ent: np.ndarray, conf: np.ndarray, window: int) -> dict:
    t = torch.from_numpy(ent)[None, None]
    pooled = F.avg_pool2d(t, kernel_size=window, stride=max(1, window // 2))
    return {"mean_entropy": float(ent.mean()), "p95_entropy": float(np.quantile(ent, 0.95)),
            "max_window_entropy": float(pooled.max()), "frac_low_conf": float((conf < 0.5).mean())}


def class_histogram(pred: torch.Tensor) -> torch.Tensor:
    h = torch.bincount(pred.reshape(-1), minlength=N_CLASSES)[:N_CLASSES].float()
    return h / h.sum().clamp_min(1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p, q = p.clamp_min(1e-8), q.clamp_min(1e-8)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum())


def quality_features(x: torch.Tensor) -> dict:
    """Image-quality cues of the (possibly corrupted) input, x: 1x3xHxW in [0,1]."""
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device).view(1, 1, 3, 3)
    return {"q_brightness": float(gray.mean()), "q_contrast": float(gray.std()),
            "q_sharpness": float(F.conv2d(gray, lap).var())}


def object_extras(probs: torch.Tensor, pred_ml: torch.Tensor, masks: list[torch.Tensor], tids: list[int],
                  omit_frac: float) -> list[dict]:
    """True-class probability inside each object and whether MetaFusion's maximum-likelihood rule
    (argmax p(y|x) / p(y), which favours rare classes) recovers it."""
    rows = []
    for m, tid in zip(masks, tids):
        p_obj = probs[:, m]                                  # C x n
        p_true = p_obj[tid]
        top2 = p_obj.topk(2, dim=0).indices
        frac_ml = float((pred_ml[m] == tid).float().mean())
        rows.append({"mean_p_true": float(p_true.mean()), "max_p_true": float(p_true.max()),
                     "frac_true_in_top2": float((top2 == tid).any(0).float().mean()),
                     "frac_true_ml": frac_ml, "omitted_ml": frac_ml <= omit_frac})
    return rows


CITYSCAPES_PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                               [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                               [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                               [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=np.uint8)


def save_panel(path: str, x: torch.Tensor, labels: np.ndarray, pred: np.ndarray, factor: int = 4) -> None:
    """Corrupted input | ground truth | prediction, downsampled for a visual audit."""
    img = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[::factor, ::factor]
    lab = labels[::factor, ::factor]
    gt = np.where(lab[..., None] == 255, 0, CITYSCAPES_PALETTE[np.minimum(lab, N_CLASSES - 1)])
    pr = CITYSCAPES_PALETTE[np.minimum(pred[::factor, ::factor], N_CLASSES - 1)]
    Image.fromarray(np.concatenate([img, gt.astype(np.uint8), pr], axis=1)).save(path)


def summarise(objects, images, args) -> dict:
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    obj, img = pd.DataFrame(objects), pd.DataFrame(images)
    out = {"by_condition": {}}
    for cond, g in obj.groupby("condition"):
        co = g[g.confident_omission]
        rest = g[~g.omitted]
        gi = img[img.condition == cond]
        entry = {
            "n_images": int(gi.shape[0]), "n_objects": int(g.shape[0]),
            "n_omitted": int(g.omitted.sum()), "n_confident_omissions": int(co.shape[0]),
            "images_with_confident_omission": int(gi.has_confident_omission.sum()),
            "confident_omissions_by_class": co["class"].value_counts().to_dict(),
            "median_entropy_percentile_confident_omissions": float(co.entropy_percentile.median()) if len(co) else None,
            "median_entropy_percentile_detected_objects": float(rest.entropy_percentile.median()) if len(rest) else None,
            # "invisible": the omitted object is on average no more uncertain than half of the image
            "share_confident_omissions_below_median_uncertainty": float((co.entropy_percentile < 0.5).mean()) if len(co) else None,
            "image_level_detection_of_confident_omission": {},
        }
        if "omitted_ml" in co:
            entry["confident_omissions_recovered_by_ml_rule"] = float((~co.omitted_ml).mean()) if len(co) else None
            entry["median_mean_p_true_confident_omissions"] = float(co.mean_p_true.median()) if len(co) else None
            entry["omitted_objects_ml_rule"] = int(g.omitted_ml.sum())
        y = gi.has_confident_omission.astype(int).to_numpy()
        if 0 < y.sum() < len(y):
            for score in ("mean_entropy", "p95_entropy", "max_window_entropy", "frac_low_conf", "one_minus_mean_margin",
                          "class_shift_js", "pred_frac_building", "q_sharpness_inv"):
                if score not in gi:
                    continue
                entry["image_level_detection_of_confident_omission"][score] = {
                    "auroc": float(roc_auc_score(y, gi[score])), "auprc": float(average_precision_score(y, gi[score])),
                    "positive_rate": float(y.mean())}
        out["by_condition"][cond] = entry
    total = obj[obj.confident_omission]
    out["totals"] = {"n_objects": int(obj.shape[0]), "n_confident_omissions": int(total.shape[0]),
                     "source_images_with_confident_omission": int(img[img.has_confident_omission].stem.nunique()),
                     "share_below_median_uncertainty": float((total.entropy_percentile < 0.5).mean()) if len(total) else None}
    return out


# ---------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root")
    p.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "pilot_out")
    p.add_argument("--models", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
                   help="comma-separated; each model gets its own objects/images CSV")
    p.add_argument("--n-panels", type=int, default=30, help="visual-audit panels per model (class-collapse cases)")
    p.add_argument("--collapse-frac", type=float, default=0.6, help="predicted building share that counts as collapse")
    p.add_argument("--corruptions", default="clean,gaussian_noise,gaussian_blur,contrast,jpeg,brightness")
    p.add_argument("--severities", default="3,5")
    p.add_argument("--max-images", type=int, default=0, help="0 = all val images")
    p.add_argument("--n-dev", type=int, default=50, help="clean images used only to fit the temperature")
    p.add_argument("--scale", type=float, default=1.0, help="resize factor (1.0 = 1024x2048)")
    p.add_argument("--min-area", type=int, default=64, help="minimum object area in pixels at scale 1.0")
    p.add_argument("--conf-thr", type=float, default=0.9)
    p.add_argument("--omit-frac", type=float, default=0.05)
    p.add_argument("--window", type=int, default=65, help="window for the spatial entropy score (pixels at scale 1.0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny local run: 3 images, quarter resolution, 2 conditions")
    # --mode p4 (v6): several models + ground-truth-free signals, image-major loop with a time budget
    p.add_argument("--mode", choices=["pilot", "p4", "maag_train", "cost", "relabel"], default="pilot")
    p.add_argument("--ref-dir", default="", help="relabel: folder with <set>/summary.json, images.csv, props_<tag>.csv.gz")
    p.add_argument("--relabel-sets", default="acdc", help="relabel: comma-separated sub-folders of --ref-dir")
    p.add_argument("--cost-frames", type=int, default=20)
    p.add_argument("--conditions", default=V6_CONDITIONS, help="p4: comma-separated name-severity list")
    p.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf", help="p4: '' disables")
    p.add_argument("--det-models", default="PekingU/rtdetr_r50vd_coco_o365,PekingU/rtdetr_r50vd,facebook/detr-resnet-50",
                   help="p4: first one that loads is used; '' disables")
    p.add_argument("--no-tta", dest="tta", action="store_false", help="p4: skip the flip view")
    p.add_argument("--maha-per-class", type=int, default=100, help="p4: dev feature samples per class per image")
    p.add_argument("--pix-per-object", type=int, default=30)
    p.add_argument("--pix-correct", type=int, default=100)
    p.add_argument("--pix-error", type=int, default=50)
    p.add_argument("--save-every", type=int, default=20, help="p4: write all outputs every N images")
    p.add_argument("--max-minutes", type=float, default=650, help="p4: stop starting new images after this")
    p.add_argument("--dataset", choices=["cityscapes", "acdc", "both", "bdd", "wilddash"], default="cityscapes",
                   help="p4 evaluation data (dev statistics always come from clean Cityscapes val)")
    p.add_argument("--acdc-root")
    p.add_argument("--bdd-root")
    p.add_argument("--bdd-splits", default="val,train")
    p.add_argument("--wd-root")
    p.add_argument("--corruption-lib", choices=["ours", "imagecorruptions"], default="ours")
    p.add_argument("--cost-no-gt", action="store_true", help="cost: proposal features without the GT matching")
    p.add_argument("--ml-props", action="store_true", help="add MetaFusion ML-rule thing segments as proposals")
    p.add_argument("--fgbg-ckpt", default="", help="p4: Maag-style depth fg/bg checkpoint (path or file name under /kaggle/input)")
    p.add_argument("--train-iters", type=int, default=8000, help="maag_train")
    p.add_argument("--train-minutes", type=float, default=75, help="maag_train: stop training after this")
    p.add_argument("--train-hw", type=int, nargs=2, default=[512, 1024], help="maag_train: H W")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--val-images", type=int, default=100)
    p.add_argument("--acdc-stems", default="", help="p4/acdc: only these frame stems (comma separated)")
    p.add_argument("--no-proposals", dest="proposals", action="store_false", help="p4: skip missed-object proposals")
    p.add_argument("--det-thr", type=float, default=0.3, help="p4: detector score threshold for proposal boxes")
    p.add_argument("--det-gamma", action="store_true", help="p4: extra detector pass on a gamma-brightened image")
    p.add_argument("--owl-model", default="", help="p4: open-vocabulary detector for proposals ('' = off)")
    p.add_argument("--owl-thr", type=float, default=0.1)
    p.add_argument("--clip-model", default="", help="p4: CLIP crop re-recognition of proposals ('' = off)")
    p.add_argument("--zoom-k", type=int, default=0, help="p4: proposals per model re-segmented on a zoomed crop")
    p.add_argument("--zoom-size", type=int, default=512)
    p.add_argument("--max-props", type=int, default=60)
    p.add_argument("--max-det-props", type=int, default=30)
    p.add_argument("--max-owl-props", type=int, default=20)
    p.add_argument("--acdc-splits", default="train,val")
    p.add_argument("--no-acdc-ref", dest="acdc_ref", action="store_false", help="skip labelled normal-condition refs")
    args = p.parse_args(argv)
    if args.smoke:
        args.max_images, args.n_dev, args.scale = 3, 1, 0.25
        args.corruptions, args.severities = "clean,gaussian_noise", "3"
        args.n_panels, args.collapse_frac = 2, 0.0
        args.max_images = 2 if args.mode == "p4" else 3
        args.save_every = 1
    return args


def load_pair(img_path: str, inst_path: str, scale: float):
    img = Image.open(img_path).convert("RGB")
    inst = Image.open(inst_path)
    if scale != 1.0:
        size = (int(img.width * scale), int(img.height * scale))
        img, inst = img.resize(size, Image.BILINEAR), inst.resize(size, Image.NEAREST)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return x, np.asarray(inst, dtype=np.int32)


def train_class_prior(root: str, max_images: int = 600, seed: int = 0) -> torch.Tensor | None:
    """Pixel class frequencies of the Cityscapes train split (MetaFusion's ML rule uses training priors)."""
    paths = sorted(glob.glob(os.path.join(root, "gtFine", "train", "*", "*_gtFine_labelTrainIds.png")))
    use_lut = False
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "gtFine", "train", "*", "*_gtFine_labelIds.png")))
        use_lut = True
    if not paths:
        return None
    rng = np.random.default_rng(seed)
    paths = [paths[i] for i in rng.permutation(len(paths))[:max_images]]
    lut = np.full(256, 255, dtype=np.uint8)
    for lid, tid in LABELID_TO_TRAINID.items():
        lut[lid] = tid
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for p in paths:
        lab = np.asarray(Image.open(p))
        lab = lut[lab] if use_lut else lab
        counts += np.bincount(lab[lab < N_CLASSES].ravel(), minlength=N_CLASSES)[:N_CLASSES]
    counts += 1
    return torch.from_numpy(counts / counts.sum()).float()


def run_model(model_name: str, args, dev_items, eval_items, conditions, device,
              train_prior: torch.Tensor | None = None) -> tuple[list, list, dict]:
    """Evaluate one frozen model on every eval image x condition; returns objects, images, extras."""
    tag = model_name.split("/")[-1].split("-finetuned")[0]
    model = load_model(model_name, device)
    min_area = max(1, int(round(args.min_area * args.scale ** 2)))
    window = max(3, int(round(args.window * args.scale)) | 1)

    # 1) clean development images: temperature, class prior (for the ML rule), clean prediction histogram
    sample_logits, sample_labels = [], []
    label_counts = torch.zeros(N_CLASSES)
    dev_hist = torch.zeros(N_CLASSES)
    for stem, img_path, inst_path in dev_items:
        x, inst = load_pair(img_path, inst_path, args.scale)
        labels = torch.from_numpy(labels_from_instances(inst).astype(np.int64))
        logits_full = predict_logits(model, x, device)[0]
        dev_hist += class_histogram(logits_full.argmax(0)).cpu()
        logits = logits_full.permute(1, 2, 0).reshape(-1, N_CLASSES).cpu()
        valid = (labels.view(-1) != 255).nonzero().view(-1)
        label_counts += torch.bincount(labels.view(-1)[valid], minlength=N_CLASSES)[:N_CLASSES].float()
        pick = valid[torch.randperm(valid.numel(), generator=torch.Generator().manual_seed(args.seed))[:20000]]
        sample_logits.append(logits[pick]), sample_labels.append(labels.view(-1)[pick])
    temperature = fit_temperature(torch.cat(sample_logits), torch.cat(sample_labels))
    prior = train_prior if train_prior is not None else (label_counts + 1) / (label_counts + 1).sum()
    dev_hist = dev_hist / dev_hist.sum()
    log(f"[{tag}] temperature {temperature:.3f} (fitted on {len(dev_items)} clean dev images)")
    prior_d, dev_hist_d = prior.to(device), dev_hist.to(device)

    # 2) evaluation images x conditions
    objects, images = [], []
    conf_mats = {}
    panel_dir = os.path.join(args.out, "panels")
    os.makedirs(panel_dir, exist_ok=True)
    n_panels = 0
    started = time.time()
    gen = torch.Generator().manual_seed(args.seed)
    for k, (stem, img_path, inst_path) in enumerate(eval_items, start=1):
        x, inst = load_pair(img_path, inst_path, args.scale)
        labels = labels_from_instances(inst)
        objs = instance_objects(inst, min_area)
        masks_d = [torch.from_numpy(m).to(device) for _, _, m in objs]
        tids = [t for _, t, _ in objs]
        labels_d = torch.from_numpy(labels.astype(np.int64)).to(device)
        valid_d = labels_d != 255
        x = x.to(device)
        for cond, sev in conditions:
            xc = corrupt(x, cond, sev, gen)
            probs = torch.softmax(predict_logits(model, xc, device)[0] / temperature, dim=0)
            top2 = probs.topk(2, dim=0).values
            conf_t, pred_t = probs.max(dim=0)
            ent_t = -(probs * probs.clamp_min(1e-12).log()).sum(dim=0) / np.log(N_CLASSES)
            pred_ml = (probs / prior_d[:, None, None]).argmax(0)
            name = cond if cond == "clean" else f"{cond}-{sev}"
            for rule, pr in (("argmax", pred_t), ("ml", pred_ml)):
                cm = torch.bincount(labels_d[valid_d] * N_CLASSES + pr[valid_d], minlength=N_CLASSES ** 2)
                key = (name, rule)
                conf_mats[key] = conf_mats.get(key, 0) + cm.reshape(N_CLASSES, N_CLASSES).cpu().numpy()
            pred, conf, ent = pred_t.cpu().numpy(), conf_t.cpu().numpy(), ent_t.cpu().numpy()
            rows = object_records(pred, conf, ent, labels, objs, args.conf_thr, args.omit_frac)
            for r, extra in zip(rows, object_extras(probs, pred_ml, masks_d, tids, args.omit_frac)):
                r.update(extra)
                r.update({"stem": stem, "condition": name, "model": tag})
            objects.extend(rows)
            hist = class_histogram(pred_t)
            q = quality_features(xc)
            img_row = {"stem": stem, "condition": name, "model": tag, "n_objects": len(rows),
                       "has_omission": any(r["omitted"] for r in rows),
                       "has_confident_omission": any(r["confident_omission"] for r in rows),
                       "pixel_acc": float((pred_t[valid_d] == labels_d[valid_d]).float().mean()) if bool(valid_d.any()) else None,
                       **image_scores(ent, conf, window),
                       "max_entropy": float(ent.max()),
                       "one_minus_mean_margin": float(1 - (top2[0] - top2[1]).mean()),
                       "class_shift_js": js_divergence(hist, dev_hist_d),
                       "pred_frac_building": float(hist[2]), **q, "q_sharpness_inv": -q["q_sharpness"],
                       **{f"hist_{c}": float(hist[c]) for c in range(N_CLASSES)}}
            images.append(img_row)
            collapse = cond != "clean" and float(hist[2]) >= args.collapse_frac
            if n_panels < args.n_panels and (collapse or (cond == "clean" and k <= 3)):
                save_panel(os.path.join(panel_dir, f"{tag}_{stem}_{name}.png"), xc, labels, pred)
                n_panels += 1
        if k == 1 or k % 25 == 0 or k == len(eval_items):
            rate = (time.time() - started) / k
            log(f"[{tag}] {k}/{len(eval_items)} images, {rate:.1f}s/image, eta {rate * (len(eval_items) - k) / 60:.1f} min, "
                f"confident omissions so far {sum(o['confident_omission'] for o in objects)}")
    miou = {}
    for (name, rule), cm in conf_mats.items():
        tp = np.diag(cm).astype(float)
        iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
        miou.setdefault(name, {})[rule] = {"miou": float(iou.mean()),
                                           "iou": {TRAINID_NAMES[c]: round(float(iou[c]), 4) for c in range(N_CLASSES)}}
    extras = {"temperature": temperature, "miou_by_condition": miou, "runtime_min": round((time.time() - started) / 60, 2),
              "class_prior": prior.tolist(), "class_prior_source": "train" if train_prior is not None else "dev", "clean_pred_hist_dev": dev_hist.tolist(), "n_panels": n_panels}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return objects, images, extras


def run_cost(args) -> dict:
    """E8 on the run hardware (Table 6): per-frame cost of the monitored segmenters and of every method's overhead.
    Real Cityscapes val frames (2048x1024), fp16 autocast as in the pipeline, CUDA-synchronised wall time, peak memory
    above the loaded models. GT matching inside proposal_records is evaluation-only and is included (slight
    overestimate of our cost)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    root = find_root(args.data_root)
    pairs = list_split(root, "val")[: args.cost_frames + 2]
    xs = [load_pair(i, l, 1.0)[0].to(dev) for i, l in pairs]
    H, W = xs[0].shape[-2:]
    H2, W2 = H // 2, W // 2
    rows = []

    def params(m):
        return sum(p_.numel() for p_ in m.parameters()) / 1e6

    def timeit(name, fn, n_params=0.0, group=""):
        for x in xs[:2]:
            fn(x)
        torch.cuda.synchronize()
        base_mem = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for x in xs[2:]:
            fn(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / (len(xs) - 2) * 1000
        gb = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 30
        rows.append({"component": name, "group": group, "ms": round(ms, 1), "peak_extra_gb": round(gb, 3),
                     "params_m": round(n_params, 1)})
        log(f"[cost] {name:48s} {ms:8.1f} ms  {gb:6.3f} GB  {n_params:6.1f} M")
        return ms

    segs = {}
    for name in args.models.split(","):
        sg = Segmenter(name, dev)
        timeit(f"{sg.tag} forward", lambda x, sg=sg: sg(x), params(sg.model), "segmenter")
        segs[sg.tag] = sg
    tags = list(segs)
    m0 = segs[tags[0]]
    seg0 = rows[0]["ms"]
    logits, feat = m0(xs[2])
    probs = torch.softmax(logits[0], 0)
    pred = probs.argmax(0)

    def unc(x):
        lg, _ = m0(x, want_feat=False)
        p_ = torch.softmax(lg[0], 0)
        top2 = p_.topk(2, dim=0).values
        return (-(p_ * p_.clamp_min(1e-12).log()).sum(0), 1 - (top2[0] - top2[1]), -torch.logsumexp(lg[0], 0))
    t_unc = timeit(f"{tags[0]} forward + entropy/margin/MSP/energy", unc, 0, "uncertainty") - seg0
    fd = FeatureDensity(100)
    gen = torch.Generator().manual_seed(0)
    lab = torch.randint(0, N_CLASSES, (H, W), device=dev)
    fd.add(feat, lab, gen)
    fd.add(feat, lab, gen)
    fd.fit()
    pred_low = F.interpolate(pred[None, None].float(), size=feat.shape[-2:], mode="nearest")[0, 0].long()
    t_maha = timeit("Mahalanobis (class-conditional, shared cov.)", lambda x: fd.score(feat, pred_low), 0, "mahalanobis")
    others = [segs[t] for t in tags[1:]]

    def js(x):
        ps = [torch.softmax(s_(x, want_feat=False)[0][0], 0) for s_ in [m0] + others]
        mean = torch.stack(ps).mean(0)
        return sum((q * (q.clamp_min(1e-12).log() - mean.clamp_min(1e-12).log())).sum(0) for q in ps)
    t_ens = timeit(f"all {len(tags)} segmenters + JS disagreement", js, 0, "ensemble") - seg0
    det = DetectorCue([args.det_models.split(",")[0]], dev)
    t_d1 = timeit("RT-DETR R50, normal pass", lambda x: det(x, (H2, W2), box_thr=args.det_thr), params(det.model), "detector")
    t_d2 = timeit("RT-DETR R50, gamma-brightened pass", lambda x: det(x.clamp(0, 1) ** 0.5, (H2, W2), box_thr=args.det_thr),
                  0, "detector")
    det(xs[2], (H2, W2), box_thr=args.det_thr)
    boxes = det.last_boxes
    pred_h = pred[::2, ::2][:H2, :W2]
    probs_h = F.avg_pool2d(probs[None], 2)[0][:, :H2, :W2]
    stack = torch.rand(7, H2, W2, device=dev)
    masks = [] if args.cost_no_gt else [torch.rand(H2, W2, device=dev) > 0.999 for _ in range(15)]
    t_feat = timeit("proposals + features (~60 boxes)", lambda x: proposal_records(
        proposal_masks(pred_h, {}, boxes, 16, max_props=90, max_det=60), probs_h, pred_h, {}, stack, list("abcdefg"),
        masks, list(range(len(masks)))), 0, "proposals")
    X = np.random.rand(20000, 30).astype(np.float32)
    y = np.random.rand(20000) > 0.9
    gb = HistGradientBoostingClassifier(max_iter=300).fit(X, y)
    Xp = np.random.rand(60, 30).astype(np.float32)
    t = time.perf_counter()
    for _ in range(50):
        gb.predict_proba(Xp)
    t_gbm = (time.perf_counter() - t) / 50 * 1000
    rows.append({"component": "verifier GBM (60 proposals)", "group": "verifier", "ms": round(t_gbm, 2), "peak_extra_gb": 0,
                 "params_m": 0})
    depth = DepthCue("depth-anything/Depth-Anything-V2-Small-hf", dev)
    fg = FgBgNet(dev, None)
    t_dep = timeit("Maag: Depth-Anything-V2-S depth", lambda x: depth(x, (H2, W2)), params(depth.model), "maag")
    dimg = depth_input(depth, xs[2], (H2, W2))
    thing_d = torch.tensor(THING_TRAIN_IDS, device=dev)

    def maag_cc(x):
        cand = ((fg.prob(dimg)[0] > 0.5) & ~torch.isin(pred_h, thing_d)).cpu().numpy()
        return ndimage.label(cand)
    t_fg = timeit("Maag: fg/bg net (MiT-B2) + connected components", maag_cc, params(fg.model), "maag")
    ours = t_d1 + t_d2 + t_feat + t_gbm
    summary = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "frame": [int(W), int(H)], "n_frames": len(xs) - 2,
        "monitored_model": tags[0],
        "overhead_ms": {"entropy / MSP / energy (training-free)": round(t_unc, 1),
                        "Mahalanobis": round(t_maha, 1),
                        f"ensemble disagreement ({len(tags)} models)": round(t_ens, 1),
                        "Maag & Rottmann (depth + fg/bg + components + meta GBM)": round(t_dep + t_fg + t_feat / 4 + t_gbm, 1),
                        "rule (detector x2 + proposals)": round(ours - t_gbm, 1),
                        "MetaSeg-style on shared proposals": round(ours, 1),
                        "ours (detector x2 + proposals + GBM)": round(ours, 1)},
        "segmenter_ms": {r["component"].replace(" forward", ""): r["ms"] for r in rows if r["group"] == "segmenter"},
        "rows": rows}
    with open(os.path.join(args.out, "cost.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    log(json.dumps(summary["overhead_ms"], indent=1))
    log(json.dumps(summary["segmenter_ms"], indent=1))
    return summary


def main(argv=None) -> dict:
    import pandas as pd

    args = parse_args(argv)
    if args.mode == "p4":
        return run_p4(args)
    if args.mode == "maag_train":
        return run_maag_train(args)
    if args.mode == "cost":
        return run_cost(args)
    if args.mode == "relabel":
        return run_relabel(args)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"python {platform.python_version()} torch {torch.__version__} device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    root = find_root(args.data_root)
    items = list_val(root)
    if not items:
        raise SystemExit(f"no val images with instanceIds under {root}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(items))
    dev_items = [items[i] for i in order[: args.n_dev]]
    eval_items = [items[i] for i in order[args.n_dev:]] or dev_items  # smoke run with 1 image: reuse
    if args.max_images:
        eval_items = eval_items[: args.max_images]
    conditions = [("clean", 0)] + [(c, int(s)) for c in args.corruptions.split(",") if c != "clean"
                                   for s in args.severities.split(",")]
    log(f"root {root}: {len(items)} images; dev {len(dev_items)}, eval {len(eval_items)}; conditions {len(conditions)}")

    summary = {"settings": vars(args), "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "models": {}, "status": "running"}
    train_prior = train_class_prior(root)
    log(f"class prior from {'train labels' if train_prior is not None else 'dev labels (no train split found)'}")
    for model_name in [m for m in args.models.split(",") if m]:
        objects, images, extras = run_model(model_name, args, dev_items, eval_items, conditions, device, train_prior)
        tag = objects[0]["model"] if objects else model_name.split("/")[-1]
        pd.DataFrame(objects).to_csv(os.path.join(args.out, f"objects_{tag}.csv"), index=False)
        pd.DataFrame(images).to_csv(os.path.join(args.out, f"images_{tag}.csv"), index=False)
        summary["models"][tag] = {**summarise(objects, images, args), **extras}
        with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        log(f"[{tag}] done: {summary['models'][tag]['totals']}")
    summary["status"] = "done"
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


# ================================================================ v6 = P4 feasibility (--mode p4)
# Several frozen segmenters on the same images x conditions, plus ground-truth-free signals that do not
# rely on the (possibly collapsed) softmax: feature density, cross-model and flip disagreement, a depth
# "sticks out but is called stuff" cue and an object detector that sees a thing where the segmenter
# predicts stuff. Evaluation is per object (signal mean inside the object, share of the object inside
# the top-q pixels of the image = alarm budget) and on sampled pixels (offline AUROC / fusion with LOCO).
STUFF_IDS = [0, 1, 2, 3, 4, 8, 9, 10]          # road .. fence, vegetation, terrain, sky
COCO_THINGS = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck"}
BASE_SIGNALS = ["ent", "margin", "energy", "rba", "tta_js", "maha_min", "maha_pred", "xmodel_js", "depth_fg", "det_stuff"]
FUSED_FROM = ["maha_pred", "xmodel_js", "depth_fg", "det_stuff", "tta_js"]
ALARM_BUDGETS = (0.02, 0.10)

# extra corruptions (ImageNet-C formulas; "dark" is a custom low-light proxy: less light + photon noise)
SHOT_PHOTONS = [60, 25, 12, 5, 3]
DEFOCUS = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
MOTION = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]
FOG = [(1.5, 2.0), (2.0, 2.0), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)]
DARK = [0.5, 0.35, 0.25, 0.15, 0.08]
V6_CONDITIONS = ("clean,gaussian_noise-3,gaussian_noise-5,shot_noise-3,gaussian_blur-3,gaussian_blur-5,defocus_blur-3,"
                 "motion_blur-3,motion_blur-5,jpeg-3,jpeg-5,contrast-5,brightness-5,fog-3,fog-5,dark-3,dark-5")


def plasma_fractal(mapsize: int, wibbledecay: float, rng: np.random.Generator) -> np.ndarray:
    """Diamond-square fractal in [0, 1] (as in ImageNet-C's fog)."""
    assert mapsize & (mapsize - 1) == 0
    arr = np.zeros((mapsize, mapsize), dtype=np.float64)
    step, wibble = mapsize, 100.0

    def wmean(a):
        return a / 4 + wibble * rng.uniform(-wibble, wibble, a.shape)

    while step >= 2:
        corner = arr[0:mapsize:step, 0:mapsize:step]
        sq = corner + np.roll(corner, -1, axis=0)
        sq = sq + np.roll(sq, -1, axis=1)
        arr[step // 2:mapsize:step, step // 2:mapsize:step] = wmean(sq)
        dr = arr[step // 2:mapsize:step, step // 2:mapsize:step]
        ul = arr[0:mapsize:step, 0:mapsize:step]
        arr[0:mapsize:step, step // 2:mapsize:step] = wmean(dr + np.roll(dr, 1, axis=0) + ul + np.roll(ul, -1, axis=1))
        arr[step // 2:mapsize:step, 0:mapsize:step] = wmean(dr + np.roll(dr, 1, axis=1) + ul + np.roll(ul, -1, axis=0))
        step //= 2
        wibble /= wibbledecay
    arr -= arr.min()
    return arr / max(arr.max(), 1e-12)


def disk_kernel(radius: float, alias_blur: float) -> np.ndarray:
    r = 8 if radius <= 8 else int(radius)
    L = np.arange(-r, r + 1)
    X, Y = np.meshgrid(L, L)
    k = ((X ** 2 + Y ** 2) <= radius ** 2).astype(np.float64)
    k = ndimage.gaussian_filter(k / k.sum(), alias_blur)
    return (k / k.sum()).astype(np.float32)


def motion_kernel(radius: int, sigma: float, angle_deg: float) -> np.ndarray:
    """One-sided line kernel with Gaussian weights along the motion direction."""
    size = 2 * radius + 1
    k = np.zeros((size, size), dtype=np.float64)
    t = np.arange(0, radius + 1)
    w = np.exp(-t ** 2 / (2 * sigma ** 2))
    a = np.deg2rad(angle_deg)
    xs = np.clip(np.round(radius + t * np.cos(a)).astype(int), 0, size - 1)
    ys = np.clip(np.round(radius - t * np.sin(a)).astype(int), 0, size - 1)
    np.add.at(k, (ys, xs), w)
    return (k / k.sum()).astype(np.float32)


def conv_rgb(x: torch.Tensor, kernel: np.ndarray) -> torch.Tensor:
    k = torch.from_numpy(kernel).to(x.device, x.dtype)
    ph, pw = k.shape[0] // 2, k.shape[1] // 2
    xp = F.pad(x, (pw, pw, ph, ph), mode="reflect")
    return F.conv2d(xp, k[None, None].repeat(3, 1, 1, 1), groups=3)


def corrupt_v6(x: torch.Tensor, name: str, severity: int, gen: torch.Generator, rng: np.random.Generator) -> torch.Tensor:
    s = severity - 1
    if name == "shot_noise":
        c = SHOT_PHOTONS[s]
        return (torch.poisson((x.cpu() * c).clamp_min(0), generator=gen) / c).clamp(0, 1).to(x.device)
    if name == "defocus_blur":
        return conv_rgb(x, disk_kernel(*DEFOCUS[s])).clamp(0, 1)
    if name == "motion_blur":
        r, sg = MOTION[s]
        return conv_rgb(x, motion_kernel(r, sg, float(rng.uniform(-45, 45)))).clamp(0, 1)
    if name == "fog":
        c = FOG[s]
        h, w = x.shape[-2:]
        size = 1 << int(np.ceil(np.log2(max(h, w))))
        pf = torch.from_numpy(plasma_fractal(size, c[1], rng)[:h, :w]).to(x.device, x.dtype)
        max_val = x.max()
        return ((x + c[0] * pf) * max_val / (max_val + c[0])).clamp(0, 1)
    if name == "dark":
        k = DARK[s]
        y = torch.poisson((x.cpu() * k * 255).clamp_min(0), generator=gen) / 255
        y = y + 0.01 * torch.randn(y.shape, generator=gen)
        return y.clamp(0, 1).to(x.device)
    return corrupt(x, name, severity, gen)


IC_NAMES = {"gaussian_noise": "gaussian_noise", "shot_noise": "shot_noise", "defocus_blur": "defocus_blur",
            "motion_blur": "motion_blur", "fog": "fog", "contrast": "contrast", "brightness": "brightness",
            "jpeg": "jpeg_compression"}   # gaussian_blur: package incompatible with scikit-image >= 0.21 -> ours


def corrupt_lib(x: torch.Tensor, name: str, severity: int, seed: int) -> torch.Tensor:
    """imagecorruptions (Michaelis et al. 2019) on a uint8 HxWx3 image; numpy RNG seeded per image and condition."""
    if not hasattr(np, "float_"):      # imagecorruptions 1.1.x still uses NumPy-1 aliases
        np.float_ = np.float64
    try:
        from imagecorruptions import corrupt as ic_corrupt
    except ImportError:   # Kaggle: install on first use (internet enabled)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imagecorruptions"])
        from imagecorruptions import corrupt as ic_corrupt
    arr = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    np.random.seed(seed % (2 ** 32))
    y = ic_corrupt(arr, corruption_name=IC_NAMES[name], severity=severity)
    y = np.asarray(y)[: arr.shape[0], : arr.shape[1]].astype(np.float32) / 255.0
    return torch.from_numpy(y).permute(2, 0, 1)[None].to(x.device)


def parse_conditions(spec: str) -> list[tuple[str, int]]:
    out = []
    for c in [c.strip() for c in spec.split(",") if c.strip()]:
        if c == "clean":
            out.append(("clean", 0))
        else:
            name, sev = c.rsplit("-", 1)
            out.append((name, int(sev)))
    return out


def js_map(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-pixel Jensen-Shannon divergence of two C x H x W probability maps."""
    def h(a):
        return -(a * a.clamp_min(1e-12).log()).sum(0)
    return (h((p + q) / 2) - 0.5 * (h(p) + h(q))).clamp_min(0)


MMSEG_URLS = {
    "deeplabv3plus-r101": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/"
                          "deeplabv3plus_r101-d8_512x1024_80k_cityscapes/"
                          "deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth",
    "deeplabv3plus-r18": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/"
                         "deeplabv3plus_r18-d8_512x1024_80k_cityscapes/"
                         "deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth",
}
MMSEG_ARCH = {"deeplabv3plus-r101": dict(depth=101, head=dict(cin=2048, ch=512, c1_in=256, c1_ch=48)),
              "deeplabv3plus-r18": dict(depth=18, head=dict(cin=512, ch=128, c1_in=64, c1_ch=12))}


class _ConvBN(torch.nn.Module):
    """mmcv ConvModule (conv -> BN -> ReLU) with its parameter names (conv.*, bn.*)."""

    def __init__(self, cin, cout, k, stride=1, padding=0, dilation=1, groups=1, act=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(cin, cout, k, stride, padding, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x, inplace=True) if self.act else x


class _DWSep(torch.nn.Module):
    """mmcv DepthwiseSeparableConvModule."""

    def __init__(self, cin, cout, k=3, padding=1, dilation=1):
        super().__init__()
        self.depthwise_conv = _ConvBN(cin, cin, k, padding=padding, dilation=dilation, groups=cin)
        self.pointwise_conv = _ConvBN(cin, cout, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class _Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, cin, planes, stride=1, dilation=1, downsample=False):
        super().__init__()
        cout = planes * 4
        self.conv1 = torch.nn.Conv2d(cin, planes, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, cout, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(cout)
        self.downsample = (torch.nn.Sequential(torch.nn.Conv2d(cin, cout, 1, stride, bias=False), torch.nn.BatchNorm2d(cout))
                           if downsample else None)

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        return F.relu(self.bn3(self.conv3(y)) + idt, inplace=True)


class _BasicBlock(torch.nn.Module):
    """mmseg BasicBlock: conv1 carries stride and dilation, conv2 is a plain 3x3."""
    expansion = 1

    def __init__(self, cin, planes, stride=1, dilation=1, downsample=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(cin, planes, 3, stride, dilation, dilation, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = (torch.nn.Sequential(torch.nn.Conv2d(cin, planes, 1, stride, bias=False), torch.nn.BatchNorm2d(planes))
                           if downsample else None)

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(y)) + idt, inplace=True)


class _ResNetV1cD8(torch.nn.Module):
    """mmseg ResNetV1c (deep stem), dilated (strides 1,2,1,1; dilations 1,1,2,4; contract_dilation)."""

    def __init__(self, depth: int = 101):
        super().__init__()
        blocks, B = {18: ((2, 2, 2, 2), _BasicBlock), 50: ((3, 4, 6, 3), _Bottleneck), 101: ((3, 4, 23, 3), _Bottleneck)}[depth]
        C = torch.nn
        self.stem = C.Sequential(C.Conv2d(3, 32, 3, 2, 1, bias=False), C.BatchNorm2d(32), C.ReLU(inplace=True),
                                 C.Conv2d(32, 32, 3, 1, 1, bias=False), C.BatchNorm2d(32), C.ReLU(inplace=True),
                                 C.Conv2d(32, 64, 3, 1, 1, bias=False), C.BatchNorm2d(64), C.ReLU(inplace=True))
        cin = 64
        for i, (n, planes, stride, dil) in enumerate(zip(blocks, (64, 128, 256, 512), (1, 2, 1, 1), (1, 1, 2, 4))):
            first_dil = dil // 2 if dil > 1 else dil
            layers = [B(cin, planes, stride, first_dil, downsample=(stride != 1 or cin != planes * B.expansion))]
            cin = planes * B.expansion
            layers += [B(cin, planes, 1, dil) for _ in range(1, n)]
            setattr(self, f"layer{i + 1}", C.Sequential(*layers))

    def forward(self, x):
        x = F.max_pool2d(self.stem(x), 3, 2, 1)
        c1 = self.layer1(x)
        c4 = self.layer4(self.layer3(self.layer2(c1)))
        return c1, c4


class _DLV3PHead(torch.nn.Module):
    def __init__(self, cin=2048, ch=512, c1_in=256, c1_ch=48, n_cls=19, dilations=(1, 12, 24, 36)):
        super().__init__()
        self.image_pool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), _ConvBN(cin, ch, 1))
        self.aspp_modules = torch.nn.ModuleList(
            [_ConvBN(cin, ch, 1) if d == 1 else _DWSep(cin, ch, 3, padding=d, dilation=d) for d in dilations])
        self.bottleneck = _ConvBN((len(dilations) + 1) * ch, ch, 3, padding=1)
        self.c1_bottleneck = _ConvBN(c1_in, c1_ch, 1)
        self.sep_bottleneck = torch.nn.Sequential(_DWSep(ch + c1_ch, ch), _DWSep(ch, ch))
        self.conv_seg = torch.nn.Conv2d(ch, n_cls, 1)

    def forward(self, c1, x):
        outs = [F.interpolate(self.image_pool(x), size=x.shape[2:], mode="bilinear", align_corners=False)]
        outs += [m(x) for m in self.aspp_modules]
        y = self.bottleneck(torch.cat(outs, 1))
        y = F.interpolate(y, size=c1.shape[2:], mode="bilinear", align_corners=False)
        feat = self.sep_bottleneck(torch.cat([y, self.c1_bottleneck(c1)], 1))
        return self.conv_seg(feat), feat


class MMSegDeepLabV3Plus(torch.nn.Module):
    """DeepLabV3+ R101-D8 with the OpenMMLab Cityscapes weights (strict load; auxiliary FCN head dropped)."""

    def __init__(self, url: str, depth: int = 101, head: dict | None = None):
        super().__init__()
        self.backbone = _ResNetV1cD8(depth)
        self.decode_head = _DLV3PHead(**(head or {}))
        ck = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=False)
        sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith("auxiliary_head")}
        self.load_state_dict(sd, strict=True)

    def forward(self, x):
        c1, c4 = self.backbone(x)
        return self.decode_head(c1, c4)


class Segmenter:
    """Frozen Cityscapes segmenter (SegFormer or Mask2Former) -> full-res logits + decoder features."""

    def __init__(self, name: str, device: str):
        self.name, self.device = name, device
        self.last_unc = None
        if name.startswith("mmseg:"):      # per-pixel CNN (OpenMMLab weights, plain PyTorch re-implementation)
            self.kind, self.tag = "mmseg", name.split(":", 1)[1]
            self.model = MMSegDeepLabV3Plus(MMSEG_URLS[self.tag], **MMSEG_ARCH[self.tag]).eval().to(device)
            self._feat = None
            return
        self.kind = "m2f" if ("mask2former" in name or "maskformer" in name) else "segformer"
        base = name.split("/")[-1]
        res_sfx = "" if base.endswith("1024-1024") or "cityscapes-" not in base else "-" + base.split("cityscapes-")[-1]
        self.tag = (base.split("-finetuned")[0] + res_sfx if self.kind == "segformer"
                    else base.replace("-cityscapes-semantic", "").replace("-cityscapes", ""))
        self.last_unc = None
        if "maskformer" in name and "mask2former" not in name:
            from transformers import MaskFormerForInstanceSegmentation as M
        elif self.kind == "m2f":
            from transformers import Mask2FormerForUniversalSegmentation as M
        else:
            from transformers import SegformerForSemanticSegmentation as M
        self.model = M.from_pretrained(name).eval().to(device)
        self._feat = None
        if self.kind == "segformer":
            self.model.decode_head.classifier.register_forward_pre_hook(self._hook)

    def _hook(self, module, inp):
        self._feat = inp[0]

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, want_feat: bool = True):
        inp = (x - MEAN.to(x.device)) / STD.to(x.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            out = self.model(inp) if self.kind == "mmseg" else self.model(pixel_values=inp)
        if self.kind == "mmseg":
            low, feat = out[0].float(), out[1]
            e_low = -torch.logsumexp(low, 1, keepdim=True)
            r_low = e_low
        elif self.kind == "segformer":
            low, feat = out.logits.float(), self._feat
            e_low = -torch.logsumexp(low, 1, keepdim=True)          # energy of the per-pixel logits
            r_low = e_low
        else:
            cls = out.class_queries_logits.float().softmax(-1)[..., :-1]
            masks = out.masks_queries_logits.float().sigmoid()
            sem = torch.einsum("bqc,bqhw->bchw", cls, masks)
            # evidence before normalisation (total query vote; RbA, Nayal et al. 2023) - lost in the normalised probs
            e_low = -sem.sum(1, keepdim=True).clamp_min(1e-6).log()
            r_low = -torch.tanh(sem).sum(1, keepdim=True)
            low = (sem / sem.sum(1, keepdim=True).clamp_min(1e-6)).clamp_min(1e-6).log()
            feat = out.pixel_decoder_last_hidden_state
        self._feat = None
        if want_feat:
            self.last_unc = F.interpolate(torch.cat([e_low, r_low], 1), size=x.shape[-2:], mode="bilinear",
                                          align_corners=False)
        logits = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits, (feat.float() if want_feat and feat is not None else None)


class FeatureDensity:
    """Class-conditional Gaussians with a shared covariance, fitted on clean dev features (Mahalanobis)."""

    def __init__(self, per_class: int = 100):
        self.per_class = per_class
        self.samples: dict[int, list] = {c: [] for c in range(N_CLASSES)}
        self.L = self.mz = self.present = None

    def add(self, feat: torch.Tensor, labels: torch.Tensor, gen: torch.Generator) -> None:
        h, w = feat.shape[-2:]
        lab = F.interpolate(labels[None, None].float(), size=(h, w), mode="nearest")[0, 0].long().view(-1)
        f = feat[0].permute(1, 2, 0).reshape(-1, feat.shape[1])
        for c in lab.unique().tolist():
            if c >= N_CLASSES:
                continue
            idx = (lab == c).nonzero().view(-1)
            idx = idx[torch.randperm(idx.numel(), generator=gen)[: self.per_class].to(idx.device)]
            self.samples[c].append(f[idx].float().cpu())

    def fit(self, eps: float = 1e-3) -> int:
        means, centred = [], []
        for c in range(N_CLASSES):
            if self.samples[c]:
                X = torch.cat(self.samples[c]).double()
                mu = X.mean(0)
                centred.append(X - mu)
                means.append(mu)
            else:
                means.append(None)
        Xc = torch.cat(centred)
        d = Xc.shape[1]
        cov = Xc.T @ Xc / len(Xc)
        cov += eps * cov.diagonal().mean() * torch.eye(d, dtype=cov.dtype)
        L = torch.linalg.cholesky(torch.linalg.inv(cov))      # precision = L L^T
        self.present = torch.tensor([m is not None for m in means])
        M = torch.stack([m if m is not None else torch.zeros(d, dtype=torch.float64) for m in means])
        self.L, self.mz = L.float(), (M @ L).float()
        self.samples = {}
        return d

    @torch.no_grad()
    def score(self, feat: torch.Tensor, pred_low: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance to the closest class and to the predicted class (both h x w, at feature resolution)."""
        h, w = feat.shape[-2:]
        L, mz, present = self.L.to(feat.device), self.mz.to(feat.device), self.present.to(feat.device)
        z = feat[0].permute(1, 2, 0).reshape(-1, feat.shape[1]).float() @ L
        d = (z * z).sum(1, keepdim=True) - 2 * z @ mz.T + (mz * mz).sum(1)[None]
        d[:, ~present] = 1e12
        d = d.clamp_min(0)
        dmin = d.min(1).values.sqrt().view(h, w)
        pl = pred_low.reshape(-1).clamp(0, N_CLASSES - 1)
        dpred = d.gather(1, pl[:, None]).view(h, w).sqrt()
        dpred = torch.where(present[pl].view(h, w), dpred, dmin)   # class unseen in dev: fall back to dmin
        return dmin, dpred


def depth_tophat(disp: torch.Tensor, kernels: tuple[int, ...]) -> torch.Tensor:
    """How much a pixel is nearer than its surroundings (white top-hat of normalised disparity).
    disp: h x w, larger = nearer. Returns h x w >= 0."""
    flat = disp.flatten().float()
    lo, hi = torch.quantile(flat[:: max(1, flat.numel() // 100000)], torch.tensor([0.01, 0.99], device=disp.device))
    d = ((disp.float() - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)[None, None]
    out = torch.zeros_like(d)
    for k in kernels:
        k = max(3, int(k) | 1)
        eroded = -F.max_pool2d(-d, k, 1, k // 2)
        opened = F.max_pool2d(eroded, k, 1, k // 2)
        out = torch.maximum(out, d - opened)
    return out[0, 0]


class DepthCue:
    def __init__(self, name: str, device: str):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        arr = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        inputs = self.proc(images=arr, return_tensors="pt").to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            pd_ = self.model(**inputs).predicted_depth
        return F.interpolate(pd_.float()[:, None], size=out_hw, mode="bilinear", align_corners=False)[0, 0]


class DetectorCue:
    """COCO object detector; map = highest score of a thing box covering the pixel."""

    def __init__(self, names: list[str], device: str):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        last = None
        self.tv = False
        if names and names[0].startswith("torchvision:"):   # CNN two-stage detector (COCO), detector-dependence check
            from torchvision.models import detection as tvd
            arch = names[0].split(":", 1)[1]
            weights = {"fasterrcnn_resnet50_fpn_v2": tvd.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1}[arch]
            self.model = getattr(tvd, arch)(weights=weights, box_score_thresh=0.01, box_detections_per_img=100).eval().to(device)
            self.name, self.tv, self.device = names[0], True, device
            cats = weights.meta["categories"]
            self.tv_labels = {i: c for i, c in enumerate(cats)}
            self.keep = {i for i, c in self.tv_labels.items() if str(c).lower() in COCO_THINGS}
            return
        for n in names:
            try:
                self.proc = AutoImageProcessor.from_pretrained(n)
                self.model = AutoModelForObjectDetection.from_pretrained(n).eval().to(device)
                self.name = n
                break
            except Exception as e:  # noqa: BLE001 - try the next detector
                last = e
                log(f"detector {n} unavailable: {type(e).__name__}: {e}")
        else:
            raise RuntimeError(f"no detector could be loaded: {last}")
        self.device = device
        self.keep = {int(i) for i, lab in self.model.config.id2label.items() if str(lab).lower() in COCO_THINGS}

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, out_hw: tuple[int, int], thr: float = 0.3, box_thr: float | None = None,
                 src: str = "rtdetr") -> torch.Tensor:
        lo = thr if box_thr is None else min(thr, box_thr)
        if self.tv:
            o = self.model([x[0].clamp(0, 1).to(self.device)])[0]
            sy, sx = out_hw[0] / x.shape[-2], out_hw[1] / x.shape[-1]
            keep = o["scores"] >= lo
            res = {"scores": o["scores"][keep], "labels": o["labels"][keep],
                   "boxes": o["boxes"][keep] * torch.tensor([sx, sy, sx, sy], device=o["boxes"].device)}
        else:
            arr = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            inputs = self.proc(images=arr, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            res = self.proc.post_process_object_detection(out, threshold=lo, target_sizes=[out_hw])[0]
        m = torch.zeros(out_hw, device=self.device)
        H, W = out_hw
        self.last_boxes = []
        for s, lab, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            if int(lab) not in self.keep:
                continue
            x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
            x1, y1 = min(W, int(np.ceil(b[2])) + 1), min(H, int(np.ceil(b[3])) + 1)
            if x1 > x0 and y1 > y0:
                if s >= thr:
                    m[y0:y1, x0:x1] = m[y0:y1, x0:x1].clamp_min(float(s))
                name_ = self.tv_labels[int(lab)] if self.tv else self.model.config.id2label[int(lab)]
                self.last_boxes.append((float(s), str(name_), x0, y0, x1, y1, src))
        return m


def zscore(m: torch.Tensor) -> torch.Tensor:
    """Robust per-image standardisation (median / MAD, std if MAD is 0), clipped to +-10."""
    m = m.float()
    med = m.median()
    scale = (m - med).abs().median() * 1.4826
    if float(scale) < 1e-6:
        scale = m.std()
    return ((m - med) / scale.clamp_min(1e-6)).clamp(-10, 10)


def budget_thresholds(maps: torch.Tensor, budgets=ALARM_BUDGETS) -> dict[float, torch.Tensor]:
    """Per-signal threshold so that a share q of the image's pixels is flagged (top-q alarm budget)."""
    S = maps.shape[0]
    flat = maps.reshape(S, -1)
    n = flat.shape[1]
    out = {}
    for q in budgets:
        k = min(n, max(1, int(round((1 - q) * n))))
        out[q] = flat.kthvalue(k, dim=1).values
    return out


def object_signal_stats(masks: list[torch.Tensor], maps: torch.Tensor, names: list[str],
                        budgets=ALARM_BUDGETS) -> list[dict]:
    """Per object: mean of every signal inside it and the share of its pixels inside each alarm budget."""
    if not masks:
        return []
    S = maps.shape[0]
    flat = maps.reshape(S, -1).float()
    Mh = torch.stack([m.reshape(-1) for m in masks]).float()
    area = Mh.sum(1, keepdim=True).clamp_min(1)
    means = (Mh @ flat.T) / area
    cols = {f"{n}_mean": means[:, i] for i, n in enumerate(names)}
    for q, thr in budget_thresholds(maps, budgets).items():
        hits = (Mh @ (flat > thr[:, None]).float().T) / area
        tag = f"hit{int(round(q * 100)):02d}"
        cols.update({f"{n}_{tag}": hits[:, i] for i, n in enumerate(names)})
    keys = list(cols)
    vals = torch.stack([cols[k] for k in keys], 1).cpu().numpy()
    return [{k: float(v) for k, v in zip(keys, row)} for row in vals]


def downsample_mask(m: torch.Tensor) -> torch.Tensor:
    h = m[::2, ::2]
    if bool(h.any()):
        return h
    return F.max_pool2d(m.float()[None, None], 2, ceil_mode=True)[0, 0][: h.shape[0], : h.shape[1]] > 0


def save_panel_v6(path: str, x: torch.Tensor, labels: np.ndarray, pred: np.ndarray, heat: list[torch.Tensor],
                  factor: int = 4) -> None:
    img = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[::factor, ::factor]
    lab = labels[::factor, ::factor]
    gt = np.where(lab[..., None] == 255, 0, CITYSCAPES_PALETTE[np.minimum(lab, N_CLASSES - 1)]).astype(np.uint8)
    pr = CITYSCAPES_PALETTE[np.minimum(pred[::factor, ::factor], N_CLASSES - 1)]
    tiles = [img, gt, pr]
    for hm in heat:
        h = F.interpolate(hm[None, None].float(), size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0]
        h = ((h - h.min()) / (h.max() - h.min()).clamp_min(1e-6) * 255).byte().cpu().numpy()
        tiles.append(np.repeat(h[..., None], 3, axis=2))
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


THING_TRAIN_IDS = list(range(11, 19))


def proposal_masks(pred_h: torch.Tensor, other_preds: dict, det_boxes: list, min_area: int,
                   max_props: int = 60, max_comps_per_model: int = 20, max_det: int | None = None) -> list:
    """Ground-truth-free candidates for an omitted object: detector thing boxes, and connected regions that
    another model labels as a thing while this model labels them as stuff."""
    H, W = pred_h.shape
    dev = pred_h.device
    stuff_self = torch.isin(pred_h, torch.tensor(STUFF_IDS, device=dev))
    thing_t = torch.tensor(THING_TRAIN_IDS, device=dev)
    props = []
    for b in sorted(det_boxes, key=lambda b: -b[0])[: max_props // 2 if max_det is None else max_det]:
        s, lab, x0, y0, x1, y1 = b[:6]
        m = torch.zeros(H, W, dtype=torch.bool, device=dev)
        m[y0:y1, x0:x1] = True
        if int(m.sum()) >= min_area:
            props.append((m, {"source": "det", "det_score": s, "det_label": lab, "other_model": "",
                              "det_src": b[6] if len(b) > 6 else "rtdetr"}))
    comps = []
    for t, op in other_preds.items():
        cand = (torch.isin(op, thing_t) & stuff_self).cpu().numpy()
        lab, n = ndimage.label(cand)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        for i in np.argsort(-sizes)[:max_comps_per_model]:
            if sizes[i] < min_area:
                break
            comps.append((int(sizes[i]), t, lab == i + 1))
    comps.sort(key=lambda c: -c[0])
    for _, t, m in comps[: max(0, max_props - len(props))]:
        props.append((torch.from_numpy(m).to(dev), {"source": "xmodel", "det_score": 0.0, "det_label": "", "other_model": t,
                                                     "det_src": ""}))
    return props


def proposal_records(props: list, probs_h: torch.Tensor, pred_h: torch.Tensor, other_preds: dict,
                     stack: torch.Tensor, names: list[str], masks_h: list, obj_ids: list) -> list[dict]:
    """Features of every proposal (GT-free) plus its best ground-truth match (for training / evaluation only)."""
    if not props:
        return []
    H, W = pred_h.shape
    dev = pred_h.device
    Pm = torch.stack([m.reshape(-1) for m, _ in props]).float()               # K x N
    area = Pm.sum(1).clamp_min(1)
    thing_t = torch.tensor(THING_TRAIN_IDS, device=dev)
    thing_prob = probs_h[THING_TRAIN_IDS].float().sum(0).reshape(-1)
    stuff_self = torch.isin(pred_h, torch.tensor(STUFF_IDS, device=dev)).float().reshape(-1)
    n_other = torch.stack([torch.isin(op, thing_t) for op in other_preds.values()]).float().sum(0).reshape(-1) \
        if other_preds else torch.zeros_like(stuff_self)
    flat = stack.reshape(stack.shape[0], -1).float()
    cols = {f"p_{n}_mean": (Pm @ flat[i]) / area for i, n in enumerate(names)}
    for i, n in enumerate(names):
        cols[f"p_{n}_max"] = (Pm * flat[i][None]).amax(1)
    cols["p_thing_prob_mean"] = (Pm @ thing_prob) / area
    cols["p_thing_prob_max"] = (Pm * thing_prob[None]).amax(1)
    cols["p_self_stuff_frac"] = (Pm @ stuff_self) / area
    cols["p_n_other_thing_mean"] = (Pm @ n_other) / area
    cols["p_area_frac"] = area / (H * W)
    m3 = Pm.view(-1, H, W) > 0
    rows_any, cols_any = m3.any(2).float(), m3.any(1).float()
    ar_h, ar_w = torch.arange(H, device=dev).float(), torch.arange(W, device=dev).float()
    y0 = torch.where(rows_any > 0, ar_h[None], torch.full_like(rows_any, H)).amin(1)
    y1 = torch.where(rows_any > 0, ar_h[None], torch.full_like(rows_any, -1)).amax(1)
    x0 = torch.where(cols_any > 0, ar_w[None], torch.full_like(cols_any, W)).amin(1)
    x1 = torch.where(cols_any > 0, ar_w[None], torch.full_like(cols_any, -1)).amax(1)
    cols["p_box_h"], cols["p_box_w"] = (y1 - y0 + 1) / H, (x1 - x0 + 1) / W
    cols["p_center_y"] = (y0 + y1) / 2 / H
    cols["p_fill"] = area / ((y1 - y0 + 1) * (x1 - x0 + 1)).clamp_min(1)
    box_cols = {"box_x0": x0, "box_y0": y0, "box_x1": x1 + 1, "box_y1": y1 + 1}   # half-res, for figures only
    keys = list(cols)
    feats = torch.stack([cols[k] for k in keys], 1).cpu().numpy()
    if masks_h:
        G = torch.stack([m.reshape(-1) for m in masks_h]).float()
        inter = Pm @ G.T
        g_area = G.sum(1).clamp_min(1)
        cover = inter / g_area[None]
        iou = inter / (area[:, None] + g_area[None] - inter)
        best = cover.argmax(1)
        best_cover = cover.gather(1, best[:, None])[:, 0].cpu().numpy()
        best_iou = iou.gather(1, best[:, None])[:, 0].cpu().numpy()
        best = best.cpu().numpy()
    out = []
    for k, (_, meta) in enumerate(props):
        r = {**meta, **{kk: float(v) for kk, v in zip(keys, feats[k])},
             **{bk: int(bv[k]) for bk, bv in box_cols.items()}}
        if masks_h:
            r.update({"gt_instance_id": int(obj_ids[best[k]]), "gt_cover": float(best_cover[k]), "gt_iou": float(best_iou[k])})
        else:
            r.update({"gt_instance_id": -1, "gt_cover": 0.0, "gt_iou": 0.0})
        out.append(r)
    return out


OWL_QUERIES = ["a photo of a person", "a photo of a rider on a bicycle", "a photo of a car", "a photo of a truck",
               "a photo of a bus", "a photo of a train", "a photo of a motorcycle", "a photo of a bicycle"]
OWL_LABELS = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def box_iou_pair(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_nms(boxes: list, iou_thr: float) -> list:
    """Greedy NMS on (score, label, x0, y0, x1, y1, ...) tuples, highest score first."""
    kept = []
    for b in sorted(boxes, key=lambda t: -t[0]):
        if all(box_iou_pair(b[2:6], k[2:6]) <= iou_thr for k in kept):
            kept.append(b)
    return kept


def merge_boxes(groups: list, quotas: list, iou_thr: float = 0.7, cross_iou: float = 0.8) -> list:
    """NMS within each detector group, a quota per group (scores are not comparable across detectors),
    then drop boxes that duplicate an already kept box of an earlier group."""
    out = []
    for boxes, q in zip(groups, quotas):
        for b in box_nms(boxes, iou_thr)[:q]:
            if all(box_iou_pair(b[2:6], o[2:6]) <= cross_iou for o in out):
                out.append(b)
    return out


def mask_box(m: torch.Tensor) -> tuple:
    ys = m.any(1).nonzero().view(-1)
    xs = m.any(0).nonzero().view(-1)
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def square_crop(box, H: int, W: int, factor: float, min_side: int) -> tuple:
    """Square window around a box (context factor), shifted to stay inside the image."""
    x0, y0, x1, y1 = box
    side = int(round(max(min_side, factor * max(x1 - x0, y1 - y0))))
    side = max(2, min(side, H, W))
    cx0 = min(max(0, int(round((x0 + x1) / 2 - side / 2))), W - side)
    cy0 = min(max(0, int(round((y0 + y1) / 2 - side / 2))), H - side)
    return cx0, cy0, cx0 + side, cy0 + side


def crop_batch(x: torch.Tensor, crops: list, size: int) -> torch.Tensor:
    return torch.cat([F.interpolate(x[:, :, c[1]:c[3], c[0]:c[2]], size=(size, size), mode="bilinear",
                                    align_corners=False, antialias=True) for c in crops])


class OwlCue:
    """OWLv2 open-vocabulary detector for the Cityscapes thing classes; boxes in out_hw pixel coordinates."""

    def __init__(self, name: str, device: str):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.proc = Owlv2Processor.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name).eval().to(device)
        tok = self.proc(text=[OWL_QUERIES], return_tensors="pt")
        self.text = {k: v.to(device) for k, v in tok.items() if k in ("input_ids", "attention_mask")}
        size = self.proc.image_processor.size
        self.size = int(size.get("height", 960)) if isinstance(size, dict) else 960
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, out_hw: tuple[int, int], thr: float) -> list:
        _, _, H, W = x.shape
        L = max(H, W)
        sq = torch.full((1, 3, L, L), 0.5, device=x.device)        # OWLv2 pads bottom/right with grey
        sq[:, :, :H, :W] = x.clamp(0, 1)
        inp = F.interpolate(sq, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)
        inp = (inp - CLIP_MEAN.to(x.device)) / CLIP_STD.to(x.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            out = self.model(pixel_values=inp, **self.text)
        scores, lab = out.logits[0].float().sigmoid().max(-1)
        keep = (scores >= thr).nonzero().view(-1)
        boxes = out.pred_boxes[0][keep].float()
        f = out_hw[0] / H * L                                      # normalised (padded square) -> out_hw pixels
        res = []
        for s, li, (cx, cy, w, h) in zip(scores[keep].tolist(), lab[keep].tolist(), boxes.tolist()):
            x0, y0 = max(0, int((cx - w / 2) * f)), max(0, int((cy - h / 2) * f))
            x1, y1 = min(out_hw[1], int(np.ceil((cx + w / 2) * f)) + 1), min(out_hw[0], int(np.ceil((cy + h / 2) * f)) + 1)
            if x1 > x0 and y1 > y0:
                res.append((float(s), OWL_LABELS[li], x0, y0, x1, y1, "owl"))
        return res


class ClipCue:
    """CLIP zero-shot re-recognition of a crop over the 19 Cityscapes classes (independent of the segmenter)."""

    def __init__(self, name: str, device: str):
        from transformers import AutoTokenizer, CLIPModel
        self.model = CLIPModel.from_pretrained(name).eval().to(device)
        tok = AutoTokenizer.from_pretrained(name)
        t = tok([f"a photo of a {n}" for n in TRAINID_NAMES], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            tf = self.model.text_projection(self.model.text_model(**t).pooler_output)
        self.tf = F.normalize(tf.float(), dim=-1)
        self.scale = float(self.model.logit_scale.exp())
        self.size = int(self.model.config.vision_config.image_size)
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, boxes_full: list, chunk: int = 64) -> np.ndarray:
        H, W = x.shape[-2:]
        crops = [square_crop(b, H, W, 1.2, 32) for b in boxes_full]
        out = []
        for i in range(0, len(crops), chunk):
            inp = (crop_batch(x.clamp(0, 1), crops[i:i + chunk], self.size) - CLIP_MEAN.to(x.device)) / CLIP_STD.to(x.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                f = self.model.visual_projection(self.model.vision_model(pixel_values=inp).pooler_output)
            out.append(torch.softmax(self.scale * F.normalize(f.float(), dim=-1) @ self.tf.T, dim=-1).cpu())
        return torch.cat(out).numpy()


@torch.no_grad()
def zoom_resegment(seg_model, x: torch.Tensor, boxes_full: list, T: float, size: int, chunk: int = 8) -> list:
    """Re-run the same segmenter on an enlarged square crop around each box; thing evidence inside the box."""
    H, W = x.shape[-2:]
    thing = torch.tensor(THING_TRAIN_IDS, device=x.device)
    out = []
    for i in range(0, len(boxes_full), chunk):
        bs = boxes_full[i:i + chunk]
        crops = [square_crop(b, H, W, 1.5, 128) for b in bs]
        logits, _ = seg_model(crop_batch(x, crops, size), want_feat=False)
        probs = torch.softmax(logits / T, dim=1)
        del logits
        pred = probs.argmax(1)
        tp = probs[:, THING_TRAIN_IDS].sum(1)
        for j, (b, c) in enumerate(zip(bs, crops)):
            f = size / (c[2] - c[0])
            ix0, iy0 = min(size - 1, max(0, int((b[0] - c[0]) * f))), min(size - 1, max(0, int((b[1] - c[1]) * f)))
            ix1 = min(size, max(ix0 + 1, int(np.ceil((b[2] - c[0]) * f))))
            iy1 = min(size, max(iy0 + 1, int(np.ceil((b[3] - c[1]) * f))))
            rp, rt = pred[j, iy0:iy1, ix0:ix1], tp[j, iy0:iy1, ix0:ix1]
            isth = torch.isin(rp, thing)
            top = int(torch.mode(rp[isth]).values) if bool(isth.any()) else -1
            out.append((float(isth.float().mean()), float(rt.mean()), top))
    return out


def second_opinion(props: list, xc: torch.Tensor, pred_h: torch.Tensor, seg_model, T: float, clip, clip_cache: dict,
                   zoom_k: int, zoom_size: int) -> None:
    """Adds CLIP and zoom-in features to each proposal's metadata (in place)."""
    H, W = xc.shape[-2:]
    boxes_f = []
    for mk, _ in props:
        a, b, c, d = mask_box(mk)
        boxes_f.append((min(W - 1, 2 * a), min(H - 1, 2 * b), min(W, max(2 * c, 2 * a + 1)), min(H, max(2 * d, 2 * b + 1))))
    if clip is not None:
        need = [b for b in dict.fromkeys(boxes_f) if b not in clip_cache]
        if need:
            for b, pr in zip(need, clip(xc, need)):
                clip_cache[b] = pr
        for (_, meta), b in zip(props, boxes_f):
            pr = clip_cache[b]
            top = int(pr.argmax())
            meta.update({"clip_thing_prob": float(pr[THING_TRAIN_IDS].sum()), "clip_top_class": top,
                         "clip_top_is_thing": float(top in THING_TRAIN_IDS),
                         **{f"clip_p_{TRAINID_NAMES[c].replace(' ', '_')}": float(pr[c]) for c in range(N_CLASSES)}})
    if zoom_k:
        stuff = torch.isin(pred_h, torch.tensor(STUFF_IDS, device=pred_h.device))
        score = [(float((mk & stuff).sum()) / max(1, int(mk.sum())), int(mk.sum())) for mk, _ in props]
        order = sorted(range(len(props)), key=lambda i: (-score[i][0], -score[i][1]))[:zoom_k]
        for _, meta in props:
            meta.update({"zoom_done": 0.0, "zoom_thing_frac": float("nan"), "zoom_thing_prob": float("nan"), "zoom_top": -1})
        for i, (fr, tp, top) in zip(order, zoom_resegment(seg_model, xc, [boxes_f[i] for i in order], T, zoom_size)):
            props[i][1].update({"zoom_done": 1.0, "zoom_thing_frac": fr, "zoom_thing_prob": tp, "zoom_top": top})


def depth_input(depth_cue, x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Per-image normalised disparity (2nd-98th percentile) as a 1x3xHxW image in [0, 1]."""
    d = depth_cue(x, hw)
    flat = d.flatten()[:: max(1, d.numel() // 200000)].float()
    lo, hi = torch.quantile(flat, torch.tensor([0.02, 0.98], device=d.device))
    d = ((d - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    return d[None, None].expand(1, 3, *hw).contiguous()


class FgBgNet:
    """Maag & Rottmann (2022)-style foreground (road users) / background segmentation from a depth heatmap.
    SegFormer (ImageNet-initialised MiT) instead of their DeepLabv3+ WRN38; Depth-Anything-V2 instead of BTS/Monodepth2."""

    def __init__(self, device: str, ckpt: str | None = None, backbone: str = "nvidia/mit-b2"):
        from transformers import SegformerForSemanticSegmentation
        self.model = SegformerForSemanticSegmentation.from_pretrained(backbone, num_labels=2,
                                                                      ignore_mismatched_sizes=True).to(device)
        if ckpt:
            sd = torch.load(ckpt, map_location=device)
            self.model.load_state_dict({k: v.float() for k, v in sd.items()})
        self.device = device

    def logits(self, dimg: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            out = self.model(pixel_values=(dimg - 0.5) / 0.25).logits
        return F.interpolate(out.float(), size=dimg.shape[-2:], mode="bilinear", align_corners=False)

    @torch.no_grad()
    def prob(self, dimg: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return torch.softmax(self.logits(dimg), 1)[:, 1]


def list_split(root: str, split: str) -> list[tuple[str, str]]:
    """[(image path, trainId label path or labelIds path)] for a Cityscapes split."""
    out = []
    for img in sorted(glob.glob(os.path.join(root, "images", split, "*", "*_leftImg8bit.png"))):
        city = os.path.basename(os.path.dirname(img))
        stem = os.path.basename(img).replace("_leftImg8bit.png", "")
        for suf in ("_gtFine_labelTrainIds.png", "_gtFine_labelIds.png", "_gtFine_instanceIds.png"):
            lab = os.path.join(root, "gtFine", split, city, stem + suf)
            if os.path.exists(lab):
                out.append((img, lab))
                break
    return out


def fg_target(lab_path: str, size: tuple[int, int]) -> np.ndarray:
    """1 = road user (trainIds 11-18), 0 = other valid class, 255 = ignore; resized (nearest) to size=(W, H)."""
    a = np.asarray(Image.open(lab_path).resize(size, Image.NEAREST)).astype(np.int64)
    if lab_path.endswith("_labelTrainIds.png"):
        tid = a
    else:
        lut = np.full(256 * 1000, 255, dtype=np.int64)
        tid = np.full(a.shape, 255, dtype=np.int64)
        ids = np.where(a >= 1000, a // 1000, a)
        for lid, t in LABELID_TO_TRAINID.items():
            tid[ids == lid] = t
        del lut
    out = np.where(tid == 255, 255, np.where((tid >= 11) & (tid <= 18), 1, 0))
    return out.astype(np.uint8)


def run_maag_train(args) -> dict:
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = find_root(args.data_root)
    train = list_split(root, "train") or list_split(root, "val")   # local smoke data has no train split
    val = list_split(root, "val")[: args.val_images]
    if args.max_images:
        train, val = train[: args.max_images], val[: max(1, args.max_images // 2)]
    H, W = args.train_hw
    depth = DepthCue(args.depth_model or "depth-anything/Depth-Anything-V2-Small-hf", device)
    log(f"[maag] {len(train)} train / {len(val)} val images at {W}x{H}; device {device}")

    def prep(items):
        D, Y = [], []
        for img_p, lab_p in items:
            x = torch.from_numpy(np.asarray(Image.open(img_p).convert("RGB").resize((W, H), Image.BILINEAR),
                                            dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            D.append((depth_input(depth, x, (H, W))[0, 0] * 255).round().byte().cpu().numpy())
            Y.append(fg_target(lab_p, (W, H)))
        return np.stack(D), np.stack(Y)
    Dtr, Ytr = prep(train)
    Dva, Yva = prep(val)
    log(f"[maag] depth precomputed in {(time.time() - t0) / 60:.1f} min; fg pixel share {float((Ytr == 1).mean()):.3f}")
    del depth
    torch.cuda.empty_cache()
    net = FgBgNet(device)
    head = [p_ for n, p_ in net.model.named_parameters() if n.startswith("decode_head")]
    body = [p_ for n, p_ in net.model.named_parameters() if not n.startswith("decode_head")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = np.random.default_rng(args.seed)
    crop = min(args.crop, W)
    t_train, it, losses = time.time(), 0, []
    net.model.train()
    while it < args.train_iters and (time.time() - t_train) < args.train_minutes * 60:
        idx = rng.integers(0, len(Dtr), args.batch)
        xs = rng.integers(0, W - crop + 1, args.batch)
        d = np.stack([Dtr[i][:, x0:x0 + crop] for i, x0 in zip(idx, xs)])
        y = np.stack([Ytr[i][:, x0:x0 + crop] for i, x0 in zip(idx, xs)])
        if rng.random() < 0.5:
            d, y = d[:, :, ::-1], y[:, :, ::-1]
        dt = torch.from_numpy(np.ascontiguousarray(d)).to(device).float().div(255)[:, None].expand(-1, 3, -1, -1)
        yt = torch.from_numpy(np.ascontiguousarray(y)).to(device).long()
        frac = it / args.train_iters
        for g, base in zip(opt.param_groups, (args.lr, args.lr * 10)):
            g["lr"] = base * min(1.0, (it + 1) / 200) * (1 - frac) ** 0.9
        loss = F.cross_entropy(net.logits(dt), yt, ignore_index=255)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss))
        it += 1
        if it % 500 == 0 or it == 1:
            log(f"[maag] iter {it}, loss {np.mean(losses[-500:]):.4f}, {(time.time() - t_train) / 60:.1f} min")
    inter = np.zeros(2)
    union = np.zeros(2)
    for d, y in zip(Dva, Yva):
        pr = (net.prob(torch.from_numpy(d).to(device).float().div(255)[None, None].expand(1, 3, -1, -1))[0] > 0.5).cpu().numpy()
        v = y != 255
        for c in (0, 1):
            inter[c] += ((pr == c) & (y == c) & v).sum()
            union[c] += (((pr == c) | (y == c)) & v).sum()
    iou = inter / np.maximum(union, 1)
    sd = {k: v.half() for k, v in net.model.state_dict().items()}
    torch.save(sd, os.path.join(args.out, "maag_fgbg_mitb2.pt"))
    summ = {"iters": it, "train_minutes": round((time.time() - t_train) / 60, 1), "val_iou_bg": float(iou[0]),
            "val_iou_fg": float(iou[1]), "val_miou": float(iou.mean()), "train_images": len(train), "hw": [H, W],
            "settings": vars(args)}
    with open(os.path.join(args.out, "maag_train_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=str)
    log(f"[maag] done: {it} iters, val IoU fg {iou[1]:.3f} bg {iou[0]:.3f}, total {(time.time() - t0) / 60:.1f} min")
    return summ


def find_bdd_root(explicit: str | None) -> str:
    cands = [explicit] if explicit else []
    cands += [os.path.dirname(os.path.dirname(os.path.dirname(p_)))
              for p_ in glob.glob("/kaggle/input/**/labels/val/*_train_id.png", recursive=True)[:1]]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "labels")) and os.path.isdir(os.path.join(c, "images")):
            return os.path.abspath(c)
    raise SystemExit(f"BDD100K root not found; tried: {cands}")


def list_bdd(root: str, splits=("val", "train")) -> list[dict]:
    items = []
    for split in splits:
        for lab in sorted(glob.glob(os.path.join(root, "labels", split, "*_train_id.png"))):
            stem = os.path.basename(lab).split("_")[0]
            img = os.path.join(root, "images", split, stem + ".jpg")
            if os.path.exists(img):
                items.append({"stem": stem, "cond": f"bdd_{split}", "split": split, "img": img, "lab": lab})
    return items


TRAINID_TO_LABELID = {}
for _lid, _tid in LABELID_TO_TRAINID.items():
    TRAINID_TO_LABELID.setdefault(_tid, _lid)


def semantic_to_instances(tid: np.ndarray) -> np.ndarray:
    """trainId map -> Cityscapes-style instanceIds: each connected component of a thing class (trainIds 11-18) is an
    instance (labelId * 1000 + k); stuff pixels get their labelId; 255 -> 0. Adjacent same-class objects merge."""
    inst = np.zeros(tid.shape, dtype=np.int32)
    for t in np.unique(tid):
        if t == 255 or int(t) not in TRAINID_TO_LABELID:
            continue
        lid = TRAINID_TO_LABELID[int(t)]
        m = tid == t
        if 11 <= t <= 18:
            lab, n = ndimage.label(m)
            for k in range(1, n + 1):
                inst[lab == k] = lid * 1000 + (k - 1)
        else:
            inst[m] = lid
    return inst


def load_bdd(item: dict, scale: float):
    img = Image.open(item["img"]).convert("RGB")
    tid = np.asarray(Image.open(item["lab"]))
    if scale != 1.0:
        size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(size, Image.BILINEAR)
        tid = np.asarray(Image.fromarray(tid).resize(size, Image.NEAREST))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return x, semantic_to_instances(tid), None


def find_wd_root(explicit: str | None) -> str:
    cands = [explicit] if explicit else []
    cands += [os.path.dirname(os.path.dirname(p_))
              for p_ in glob.glob("/kaggle/input/**/labels/*_instanceIds.png", recursive=True)[:1]]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "labels")) and os.path.isdir(os.path.join(c, "images")):
            return os.path.abspath(c)
    raise SystemExit(f"WildDash2 root not found; tried: {cands}")


def list_wd(root: str) -> list[dict]:
    """WildDash2 public GT frames; cond = wd_val / wd_train by the official random split (both are test data here)."""
    val = set()
    vp = os.path.join(root, "validation.txt")
    if os.path.exists(vp):
        val = set(open(vp).read().split())
    items = []
    for lab in sorted(glob.glob(os.path.join(root, "labels", "*_instanceIds.png"))):
        stem = os.path.basename(lab)[: -len("_instanceIds.png")]
        img = os.path.join(root, "images", stem + ".jpg")
        if os.path.exists(img):
            items.append({"stem": stem, "cond": "wd_val" if stem in val else "wd_train", "img": img, "lab": lab})
    return items


def load_wd(item: dict, scale: float):
    img = Image.open(item["img"]).convert("RGB")
    inst = np.asarray(Image.open(item["lab"]), dtype=np.int32)
    if scale != 1.0:
        size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(size, Image.BILINEAR)
        inst = np.asarray(Image.fromarray(inst).resize(size, Image.NEAREST), dtype=np.int32)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return x, inst, None


ACDC_CONDS = ["fog", "night", "rain", "snow"]


def find_acdc_root(explicit: str | None) -> str:
    """Directory that contains rgb_anon/ and gt_panoptic/ (ACDC)."""
    cands = [explicit] if explicit else []
    cands += [os.path.dirname(os.path.dirname(p))
              for p in glob.glob("/kaggle/input/**/gt_panoptic/train_gt_panoptic.json", recursive=True)]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "rgb_anon")) and os.path.isdir(os.path.join(c, "gt_panoptic")):
            return os.path.abspath(c)
    raise SystemExit(f"ACDC root not found; tried: {cands}")


def list_acdc(root: str, splits=("train", "val"), with_ref: bool = True) -> list[dict]:
    """Labelled ACDC images: adverse train/val and the labelled normal-condition references."""
    items = []
    for split in splits:
        for kind, sp, js in (("adverse", split, f"{split}_gt_panoptic.json"),
                             ("ref", f"{split}_ref", f"{split}_gt_ref_panoptic.json")):
            path = os.path.join(root, "gt_panoptic", js)
            if (kind == "ref" and not with_ref) or not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                ann = json.load(f)["annotations"]
            for a in ann:
                cond, _, seq, base = a["file_name"].split("/")
                stem = base.split("_gt_")[0]
                img = os.path.join(root, "rgb_anon", cond, sp, seq,
                                   stem + ("_rgb_ref_anon.png" if kind == "ref" else "_rgb_anon.png"))
                pan = os.path.join(root, "gt_panoptic", a["file_name"])
                inv = os.path.join(root, "gt", cond, sp, seq, stem + "_gt_invIds.png") if kind == "adverse" else None
                if os.path.exists(img) and os.path.exists(pan):
                    items.append({"stem": stem, "cond": cond if kind == "adverse" else cond + "_ref", "split": split,
                                  "img": img, "pan": pan, "segments": a["segments_info"],
                                  "inv": inv if inv and os.path.exists(inv) else None})
    return items


def panoptic_to_instances(pan_rgb: np.ndarray, segments: list[dict]) -> np.ndarray:
    """COCO-panoptic PNG (id = R + 256 G + 256^2 B) -> Cityscapes-style instanceIds map
    (labelId * 1000 + k for non-crowd things, labelId otherwise, 0 = unlabelled)."""
    a = pan_rgb.astype(np.int64)
    ids = a[..., 0] + 256 * a[..., 1] + 65536 * a[..., 2]
    inst = np.zeros(ids.shape, dtype=np.int32)
    counters: dict[int, int] = {}
    for sgm in segments:
        cat, m = int(sgm["category_id"]), ids == sgm["id"]
        if cat in INSTANCE_LABELIDS and not sgm.get("iscrowd", 0):
            k = counters.get(cat, 0)
            counters[cat] = k + 1
            inst[m] = cat * 1000 + k
        else:
            inst[m] = cat
    return inst


def load_acdc(item: dict, scale: float):
    img = Image.open(item["img"]).convert("RGB")
    inst = panoptic_to_instances(np.asarray(Image.open(item["pan"]).convert("RGB")), item["segments"])
    inv = np.asarray(Image.open(item["inv"])) > 0 if item["inv"] else None
    if scale != 1.0:
        size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(size, Image.BILINEAR)
        inst = np.asarray(Image.fromarray(inst).resize(size, Image.NEAREST), dtype=np.int32)
        if inv is not None:
            inv = np.asarray(Image.fromarray(inv.astype(np.uint8)).resize(size, Image.NEAREST)) > 0
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return x, inst, inv


def run_p4(args) -> dict:
    import pandas as pd

    t_start = time.time()
    deadline = t_start + args.max_minutes * 60
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[p4] python {platform.python_version()} torch {torch.__version__} device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    try:
        import transformers
        log(f"[p4] transformers {transformers.__version__}")
    except Exception:  # noqa: BLE001
        pass
    root = find_root(args.data_root)
    items = list_val(root)
    rng0 = np.random.default_rng(args.seed)
    order = rng0.permutation(len(items))
    dev_items = [items[i] for i in order[: args.n_dev]]
    eval_items = [items[i] for i in order[args.n_dev:]] or dev_items
    conditions = parse_conditions(args.conditions)
    cond_names = [c if c == "clean" else f"{c}-{s}" for c, s in conditions]
    if args.dataset == "acdc":   # real adverse conditions: one pass per image, the condition is the ACDC label
        acdc = list_acdc(find_acdc_root(args.acdc_root), tuple(args.acdc_splits.split(",")), args.acdc_ref)
        eval_items = [acdc[i] for i in np.random.default_rng(args.seed).permutation(len(acdc))]
        if args.acdc_stems:   # restrict to chosen frames (figures / cost measurement)
            want = set(args.acdc_stems.split(","))
            eval_items = [it for it in eval_items if it["stem"] in want and not it["cond"].endswith("_ref")]
        conditions = [("clean", 0)]
        cond_names = ACDC_CONDS + [c + "_ref" for c in ACDC_CONDS]
    elif args.dataset == "bdd":   # second real-world domain: BDD100K 10K (val + train subset), one pass per image
        bdd = list_bdd(find_bdd_root(args.bdd_root), tuple(args.bdd_splits.split(",")))
        eval_items = [bdd[i] for i in np.random.default_rng(args.seed).permutation(len(bdd))]
        conditions = [("clean", 0)]
        cond_names = [f"bdd_{s}" for s in args.bdd_splits.split(",")]
    elif args.dataset == "wilddash":   # third real-world domain: WildDash2 public GT (panoptic -> instanceIds)
        wd = list_wd(find_wd_root(args.wd_root))
        eval_items = [wd[i] for i in np.random.default_rng(args.seed).permutation(len(wd))]
        conditions = [("clean", 0)]
        cond_names = ["wd_val", "wd_train"]
    elif args.dataset == "both":  # Cityscapes(-C) conditions and ACDC images, interleaved so a cut-off keeps both
        acdc = list_acdc(find_acdc_root(args.acdc_root), tuple(args.acdc_splits.split(",")), args.acdc_ref)
        cs_part = [("cityscapes", it) for it in (eval_items[: args.max_images] if args.max_images else eval_items)]
        ac_part = [("acdc", it) for it in (acdc[: args.max_images] if args.max_images else acdc)]
        both = cs_part + ac_part
        eval_items = [both[i] for i in np.random.default_rng(args.seed).permutation(len(both))]
        cond_names = cond_names + ACDC_CONDS + [c + "_ref" for c in ACDC_CONDS]
    if args.max_images and args.dataset != "both":
        eval_items = eval_items[: args.max_images]
    log(f"[p4] {len(items)} images; dev {len(dev_items)}, eval up to {len(eval_items)}; {len(conditions)} conditions; "
        f"budget {args.max_minutes} min")
    train_prior = train_class_prior(root)

    seg = [Segmenter(n, device) for n in args.models.split(",") if n]
    tags = [m.tag for m in seg]
    depth = det = None
    status = {}
    if args.depth_model:
        try:
            depth = DepthCue(args.depth_model, device)
            status["depth"] = args.depth_model
        except Exception as e:  # noqa: BLE001
            status["depth"] = f"unavailable: {type(e).__name__}: {e}"
    if args.det_models:
        try:
            det = DetectorCue(args.det_models.split(","), device)
            status["detector"] = det.name
        except Exception as e:  # noqa: BLE001
            status["detector"] = f"unavailable: {type(e).__name__}: {e}"
    owl = clip = None
    if args.owl_model:
        try:
            owl = OwlCue(args.owl_model, device)
            status["owl"] = args.owl_model
        except Exception as e:  # noqa: BLE001
            status["owl"] = f"unavailable: {type(e).__name__}: {e}"
    if args.clip_model:
        try:
            clip = ClipCue(args.clip_model, device)
            status["clip"] = args.clip_model
        except Exception as e:  # noqa: BLE001
            status["clip"] = f"unavailable: {type(e).__name__}: {e}"
    fgbg = fg_depth = None
    if args.fgbg_ckpt:
        ck = args.fgbg_ckpt if os.path.exists(args.fgbg_ckpt) else next(iter(glob.glob(f"/kaggle/input/**/{args.fgbg_ckpt}", recursive=True)), None)
        if ck:
            fgbg = FgBgNet(device, ck)
            fg_depth = depth if depth is not None else DepthCue("depth-anything/Depth-Anything-V2-Small-hf", device)
            status["fgbg"] = ck
        else:
            status["fgbg"] = f"checkpoint not found: {args.fgbg_ckpt}"
    log(f"[p4] models {tags}; cues {status}")
    signals = [s for s in BASE_SIGNALS
               if not (s == "xmodel_js" and len(seg) < 2) and not (s == "depth_fg" and depth is None)
               and not (s == "det_stuff" and det is None) and not (s == "tta_js" and not args.tta)]
    if fgbg is not None:
        signals.append("fgbg_stuff")
    fused_from = [s for s in FUSED_FROM if s in signals]
    signals.append("fused")

    # ---- clean dev images: temperature, clean prediction histogram, feature density per model
    st = {}
    gen = torch.Generator().manual_seed(args.seed)
    for m in seg:
        lg_s, lb_s, hist = [], [], torch.zeros(N_CLASSES)
        dens = FeatureDensity(args.maha_per_class)
        for stem, img_path, inst_path in dev_items:
            x, inst = load_pair(img_path, inst_path, args.scale)
            labels = torch.from_numpy(labels_from_instances(inst).astype(np.int64))
            logits, feat = m(x.to(device))
            hist += class_histogram(logits[0].argmax(0)).cpu()
            dens.add(feat, labels.to(device), gen)
            lf = logits[0].permute(1, 2, 0).reshape(-1, N_CLASSES).cpu()
            valid = (labels.view(-1) != 255).nonzero().view(-1)
            pick = valid[torch.randperm(valid.numel(), generator=gen)[:20000]]
            lg_s.append(lf[pick]), lb_s.append(labels.view(-1)[pick])
        T = fit_temperature(torch.cat(lg_s), torch.cat(lb_s))
        dim = dens.fit()
        st[m.tag] = {"T": T, "dev_hist": (hist / hist.sum()).to(device), "dens": dens, "feat_dim": dim,
                     "conf_mats": {}, "objects": [], "images": [], "pix": [], "pix_meta": [], "props": [], "n_panels": 0,
                     "panel_conds": {}}
        log(f"[p4] [{m.tag}] T {T:.3f}, feature dim {dim}")
    bad = [m.tag for m in seg if st[m.tag]["T"] > 5.0]   # a broken checkpoint load gives a huge temperature
    if bad and len(bad) < len(seg):
        log(f"[p4] WARNING dropping models with implausible temperature (weights probably not loaded): {bad}")
        seg = [m for m in seg if m.tag not in bad]
        tags = [m.tag for m in seg]
        for b in bad:
            del st[b]
    prior_d = (train_prior if train_prior is not None else torch.full((N_CLASSES,), 1 / N_CLASSES)).to(device)
    s = args.scale
    min_area = max(1, int(round(args.min_area * s ** 2)))
    window = max(3, int(round(args.window * s)) | 1)
    tophat_k = tuple(max(3, int(round(k * s / 4)) | 1) for k in (61, 181))   # full-res px -> quarter-res grid

    def write_outputs(final: bool, n_done: int, sec_per_image: float):
        summ = {"settings": vars(args), "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "status": "done" if final else "running", "n_eval_images_done": n_done,
                "sec_per_image": round(sec_per_image, 2), "conditions": cond_names, "signals": signals,
                "fused_from": fused_from, "cues": status, "elapsed_min": round((time.time() - t_start) / 60, 1),
                "models": {}}
        for tag in tags:
            S_ = st[tag]
            pd.DataFrame(S_["objects"]).to_csv(os.path.join(args.out, f"objects_{tag}.csv"), index=False)
            pd.DataFrame(S_["images"]).to_csv(os.path.join(args.out, f"images_{tag}.csv"), index=False)
            if S_["props"]:
                pd.DataFrame(S_["props"]).to_csv(os.path.join(args.out, f"proposals_{tag}.csv"), index=False)
            if S_["pix"]:
                meta = np.concatenate(S_["pix_meta"], 0)
                np.savez(os.path.join(args.out, f"pixels_{tag}.npz"), scores=np.concatenate(S_["pix"], 0),
                         image=meta[:, 0], condition=meta[:, 1], kind=meta[:, 2], true_class=meta[:, 3],
                         pred_class=meta[:, 4], signals=np.array(signals), conditions=np.array(cond_names))
            miou = {}
            for (name, rule), cm in S_["conf_mats"].items():
                tp = np.diag(cm).astype(float)
                iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
                miou.setdefault(name, {})[rule] = {"miou": float(iou.mean()),
                                                   "iou": {TRAINID_NAMES[c]: round(float(iou[c]), 4) for c in range(N_CLASSES)}}
            obj = pd.DataFrame(S_["objects"])
            summ["models"][tag] = {"temperature": S_["T"], "feature_dim": S_["feat_dim"], "miou_by_condition": miou,
                                   "n_objects": int(len(obj)),
                                   "n_confident_omissions": int(obj.confident_omission.sum()) if len(obj) else 0,
                                   "class_prior": prior_d.cpu().tolist(), "clean_pred_hist_dev": S_["dev_hist"].cpu().tolist()}
        with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2, default=str)
        return summ

    panel_dir = os.path.join(args.out, "panels")
    os.makedirs(panel_dir, exist_ok=True)
    t_loop = time.time()
    n_done = 0
    for k, item in enumerate(eval_items):
        if n_done:
            per = (time.time() - t_loop) / n_done
            if time.time() + 1.3 * per > deadline:
                log(f"[p4] stopping before image {k}: time budget ({per:.0f} s/image)")
                break
        src_ds = args.dataset
        if src_ds == "both":
            src_ds, item = item
        if src_ds == "acdc":
            stem = item["stem"]
            x, inst, inv = load_acdc(item, s)
            item_conds = [(cond_names.index(item["cond"]), "clean", 0)]
        elif src_ds == "bdd":
            stem = item["stem"]
            x, inst, inv = load_bdd(item, s)
            item_conds = [(cond_names.index(item["cond"]), "clean", 0)]
        elif src_ds == "wilddash":
            stem = item["stem"]
            x, inst, inv = load_wd(item, s)
            item_conds = [(cond_names.index(item["cond"]), "clean", 0)]
        else:
            stem, img_path, inst_path = item
            x, inst = load_pair(img_path, inst_path, s)
            inv = None
            item_conds = [(ci, c, sv) for ci, (c, sv) in enumerate(conditions)]
        labels = labels_from_instances(inst)
        objs = instance_objects(inst, min_area)
        masks_d = [torch.from_numpy(mk).to(device) for _, _, mk in objs]
        masks_h = [downsample_mask(mk) for mk in masks_d]
        tids = [t for _, t, _ in objs]
        labels_d = torch.from_numpy(labels.astype(np.int64)).to(device)
        valid_d = labels_d != 255
        lab_h = labels_d[::2, ::2]
        valid_h = lab_h != 255
        x = x.to(device)
        H2, W2 = lab_h.shape
        inv_d = torch.from_numpy(inv).to(device) if inv is not None else None
        for ci, cond, sev in item_conds:
            name = cond_names[ci]
            g = torch.Generator().manual_seed(args.seed * 1_000_003 + k * 101 + ci)
            r = np.random.default_rng(args.seed * 1_000_003 + k * 101 + ci)
            if args.corruption_lib == "imagecorruptions" and cond in IC_NAMES:
                xc = corrupt_lib(x, cond, sev, args.seed * 1_000_003 + k * 101 + ci)
            else:
                xc = corrupt_v6(x, cond, sev, g, r)
            per_model = {}
            for m in seg:
                S_ = st[m.tag]
                logits, feat = m(xc)
                probs = torch.softmax(logits[0] / S_["T"], dim=0)
                del logits
                conf_t, pred_t = probs.max(dim=0)
                top2 = probs.topk(2, dim=0).values
                ent_t = -(probs * probs.clamp_min(1e-12).log()).sum(dim=0) / np.log(N_CLASSES)
                margin_t = 1 - (top2[0] - top2[1])
                pred_ml = (probs / prior_d[:, None, None]).argmax(0)
                for rule, pr in (("argmax", pred_t), ("ml", pred_ml)):
                    cm = torch.bincount(labels_d[valid_d] * N_CLASSES + pr[valid_d], minlength=N_CLASSES ** 2)
                    S_["conf_mats"][(name, rule)] = S_["conf_mats"].get((name, rule), 0) + cm.reshape(N_CLASSES, N_CLASSES).cpu().numpy()
                pred, conf, ent = pred_t.cpu().numpy(), conf_t.cpu().numpy(), ent_t.cpu().numpy()
                rows = object_records(pred, conf, ent, labels, objs, args.conf_thr, args.omit_frac)
                for row, extra in zip(rows, object_extras(probs, pred_ml, masks_d, tids, args.omit_frac)):
                    row.update(extra)
                    row.update({"stem": stem, "condition": name, "model": m.tag})
                if inv_d is not None:
                    for row, mk in zip(rows, masks_d):
                        row["frac_invalid"] = float((inv_d & mk).float().sum() / mk.float().sum().clamp_min(1))
                if src_ds == "acdc":
                    for row in rows:
                        row["split"] = item["split"]
                probs_h = F.avg_pool2d(probs[None], 2)[0][:, :H2, :W2]
                maps = {"ent": F.avg_pool2d(ent_t[None, None], 2)[0, 0][:H2, :W2],
                        "margin": F.avg_pool2d(margin_t[None, None], 2)[0, 0][:H2, :W2]}
                unc_h = F.avg_pool2d(m.last_unc, 2)[0][:, :H2, :W2]
                maps["energy"], maps["rba"] = unc_h[0], unc_h[1]
                if args.tta:
                    lf, _ = m(xc.flip(-1), want_feat=False)
                    pf = torch.softmax(lf[0].flip(-1) / S_["T"], dim=0)
                    del lf
                    maps["tta_js"] = js_map(probs_h, F.avg_pool2d(pf[None], 2)[0][:, :H2, :W2])
                    del pf
                fh, fw = feat.shape[-2:]
                pred_low = F.interpolate(pred_t[None, None].float(), size=(fh, fw), mode="nearest")[0, 0].long()
                dmin, dpred = S_["dens"].score(feat, pred_low)
                maps["maha_min"] = F.interpolate(dmin[None, None], size=(H2, W2), mode="bilinear", align_corners=False)[0, 0]
                maps["maha_pred"] = F.interpolate(dpred[None, None], size=(H2, W2), mode="bilinear", align_corners=False)[0, 0]
                del feat, probs
                per_model[m.tag] = {"rows": rows, "maps": maps, "probs_h": probs_h.half(), "pred_h": pred_t[::2, ::2],
                                    "pred_full": pred, "hist": class_histogram(pred_t),
                                    "margin_mean": float(margin_t.mean()), "ent_np": ent, "conf_np": conf,
                                    "pixel_acc": float((pred_t[valid_d] == labels_d[valid_d]).float().mean()) if bool(valid_d.any()) else None}
            tophat = None
            if depth is not None:
                disp = depth(xc, (max(1, H2 // 2), max(1, W2 // 2)))
                tophat = F.interpolate(depth_tophat(disp, tophat_k)[None, None], size=(H2, W2), mode="bilinear",
                                       align_corners=False)[0, 0]
            det_map = det(xc, (H2, W2), box_thr=args.det_thr) if det is not None else None
            groups, quotas = [], []
            if det is not None:
                rt = list(det.last_boxes)
                if args.det_gamma:
                    det(xc.clamp(0, 1) ** 0.5, (H2, W2), box_thr=args.det_thr, src="rtdetr_gamma")
                    rt += det.last_boxes
                groups.append(rt)
                quotas.append(args.max_det_props if owl is None else args.max_det_props - args.max_owl_props)
            if owl is not None:
                groups.append(owl(xc, (H2, W2), args.owl_thr))
                quotas.append(args.max_owl_props)
            img_boxes = merge_boxes(groups, quotas)
            fg_h = None
            if fgbg is not None:   # depth -> foreground probability at half resolution (training scale)
                fg_h = fgbg.prob(depth_input(fg_depth, xc, (H2, W2)))[0]
            clip_cache = {}
            q = quality_features(xc)
            for m in seg:
                S_, P = st[m.tag], per_model[m.tag]
                maps = P["maps"]
                if "xmodel_js" in signals:
                    others = torch.stack([per_model[t]["probs_h"].float() for t in tags if t != m.tag]).mean(0)
                    maps["xmodel_js"] = js_map(P["probs_h"].float(), others)
                stuff = torch.isin(P["pred_h"], torch.tensor(STUFF_IDS, device=device)).float()
                if tophat is not None:
                    maps["depth_fg"] = tophat * stuff
                if det_map is not None:
                    maps["det_stuff"] = det_map * stuff
                if fg_h is not None:
                    maps["fgbg_stuff"] = fg_h * stuff
                maps["fused"] = torch.stack([zscore(maps[f]) for f in fused_from]).mean(0) if fused_from else maps["ent"]
                stack = torch.stack([maps[n].float() for n in signals])
                for row, extra in zip(P["rows"], object_signal_stats(masks_h, stack, signals)):
                    row.update(extra)
                S_["objects"].extend(P["rows"])
                if args.proposals:
                    others_pred = {t: per_model[t]["pred_h"] for t in tags if t != m.tag}
                    props = proposal_masks(P["pred_h"], others_pred, img_boxes, max(4, min_area // 4),
                                           max_props=args.max_props, max_det=args.max_det_props)
                    if fg_h is not None:   # Maag & Rottmann: foreground by depth where this model predicts no road user
                        cand = ((fg_h > 0.5) & ~torch.isin(P["pred_h"], torch.tensor(THING_TRAIN_IDS, device=device))).cpu().numpy()
                        lab_c, n_c = ndimage.label(cand)
                        if n_c:
                            sizes = np.bincount(lab_c.ravel())[1:]
                            for ci_ in np.argsort(-sizes)[:20]:
                                if sizes[ci_] < max(4, min_area // 4):
                                    break
                                props.append((torch.from_numpy(lab_c == ci_ + 1).to(device),
                                              {"source": "maag", "det_score": 0.0, "det_label": "", "other_model": "",
                                               "det_src": "maag"}))
                    if args.ml_props:   # MetaFusion (Chan et al. 2020): ML rule (probs / class prior) creates extra thing segments
                        pred_ml_h = (P["probs_h"] / prior_d[:, None, None]).argmax(0)
                        thing_d = torch.tensor(THING_TRAIN_IDS, device=device)
                        cand = (torch.isin(pred_ml_h, thing_d) & ~torch.isin(P["pred_h"], thing_d)).cpu().numpy()
                        lab_c, n_c = ndimage.label(cand)
                        if n_c:
                            sizes = np.bincount(lab_c.ravel())[1:]
                            for ci_ in np.argsort(-sizes)[:20]:
                                if sizes[ci_] < max(4, min_area // 4):
                                    break
                                props.append((torch.from_numpy(lab_c == ci_ + 1).to(device),
                                              {"source": "ml", "det_score": 0.0, "det_label": "", "other_model": "",
                                               "det_src": "ml"}))
                    if props and (clip is not None or args.zoom_k):
                        second_opinion(props, xc, P["pred_h"], m, S_["T"], clip, clip_cache, args.zoom_k, args.zoom_size)
                    obj_ids = [iid for iid, _, _ in objs]
                    for pr in proposal_records(props, P["probs_h"], P["pred_h"], others_pred, stack, signals, masks_h, obj_ids):
                        pr.update({"stem": stem, "condition": name, "model": m.tag})
                        S_["props"].append(pr)
                flat = stack.reshape(len(signals), -1)
                p99 = flat.kthvalue(max(1, int(0.99 * flat.shape[1])), dim=1).values
                hist = P["hist"]
                img_row = {"stem": stem, "condition": name, "model": m.tag, "n_objects": len(P["rows"]),
                           "has_omission": any(rw["omitted"] for rw in P["rows"]),
                           "has_confident_omission": any(rw["confident_omission"] for rw in P["rows"]),
                           "pixel_acc": P["pixel_acc"], **image_scores(P["ent_np"], P["conf_np"], window),
                           "one_minus_mean_margin": P["margin_mean"],
                           "class_shift_js": js_divergence(hist, S_["dev_hist"]), "pred_frac_building": float(hist[2]),
                           "pred_max_class_frac": float(hist.max()), **q,
                           **{f"img_{n}_mean": float(flat[i].mean()) for i, n in enumerate(signals)},
                           **{f"img_{n}_p99": float(p99[i]) for i, n in enumerate(signals)},
                           **{f"hist_{c}": float(hist[c]) for c in range(N_CLASSES)}}
                S_["images"].append(img_row)
                # sampled pixels (half-res): 0 correct, 1 confident-omission object, 2 other error, 3 other omitted object
                co_union = torch.zeros_like(valid_h)
                picks, kinds = [], []
                for rw, mh in zip(P["rows"], masks_h):
                    if rw["omitted"]:
                        idx = mh.reshape(-1).nonzero().view(-1)
                        idx = idx[torch.randperm(idx.numel(), generator=g)[: args.pix_per_object].to(device)]
                        picks.append(idx)
                        kinds.append(torch.full_like(idx, 1 if rw["confident_omission"] else 3))
                        if rw["confident_omission"]:
                            co_union |= mh
                correct = ((P["pred_h"] == lab_h) & valid_h).reshape(-1).nonzero().view(-1)
                wrong = ((P["pred_h"] != lab_h) & valid_h & ~co_union).reshape(-1).nonzero().view(-1)
                for pool, n_take, kind in ((correct, args.pix_correct, 0), (wrong, args.pix_error, 2)):
                    idx = pool[torch.randperm(pool.numel(), generator=g)[:n_take].to(device)]
                    picks.append(idx)
                    kinds.append(torch.full_like(idx, kind))
                if picks:
                    idx = torch.cat(picks)
                    S_["pix"].append(flat[:, idx].T.half().cpu().numpy())
                    S_["pix_meta"].append(np.stack([np.full(idx.numel(), k), np.full(idx.numel(), ci),
                                                    torch.cat(kinds).cpu().numpy(), lab_h.reshape(-1)[idx].cpu().numpy(),
                                                    P["pred_h"].reshape(-1)[idx].cpu().numpy()], 1).astype(np.int16))
                if (S_["n_panels"] < args.n_panels and img_row["has_confident_omission"]
                        and S_["panel_conds"].get(name, 0) < max(1, args.n_panels // len(cond_names) + 1)):
                    heat = [maps["fused"]] + ([maps["depth_fg"]] if "depth_fg" in maps else []) + [maps["maha_pred"]]
                    save_panel_v6(os.path.join(panel_dir, f"{m.tag}_{stem}_{name}.png"), xc, labels, P["pred_full"], heat)
                    S_["n_panels"] += 1
                    S_["panel_conds"][name] = S_["panel_conds"].get(name, 0) + 1
            del per_model
        n_done += 1
        per = (time.time() - t_loop) / n_done
        if n_done == 1 or n_done % args.save_every == 0:
            n_co = {t: sum(o["confident_omission"] for o in st[t]["objects"]) for t in tags}
            log(f"[p4] {n_done}/{len(eval_items)} images, {per:.1f} s/image, elapsed {(time.time() - t_start) / 60:.0f} min, "
                f"confident omissions {n_co}")
            write_outputs(False, n_done, per)
    summ = write_outputs(True, n_done, (time.time() - t_loop) / max(1, n_done))
    log(f"[p4] done: {n_done} images in {(time.time() - t_start) / 60:.0f} min")
    return summ


def _box_sums(ii: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Sums over half-open boxes (x0, y0, x1, y1) from a zero-padded integral image ii (... x H+1 x W+1)."""
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return ii[..., y1, x1] - ii[..., y0, x1] - ii[..., y1, x0] + ii[..., y0, x0]


def _integral(m: torch.Tensor) -> torch.Tensor:
    return F.pad(m.double().cumsum(-2).cumsum(-1), (1, 0, 1, 0))


def run_relabel(args) -> dict:
    """Re-label pass for the review of 2026-11-22. Segmenters only (no detector): temperatures, image list and the
    detector boxes come from a reference p4 run (--ref-dir/<set>). Per model and image it writes
    (1) the object records with the joint criterion (pixels both predicted as stuff and confident, see object_records),
    (2) the maximum softmax probability (MSP, mean / max) inside each stored detector box, a re-computed match of the
        box to the true instances (sanity check against the stored gt_instance_id / gt_cover), and
    (3) for datasets with true instances (ACDC, WildDash2): objects and box matches under connected-component units,
        i.e. the object definition used for BDD100K (semantic_to_instances)."""
    import pandas as pd

    t_start = time.time()
    deadline = t_start + args.max_minutes * 60
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_root = args.ref_dir
    log(f"[relabel] device {device}; ref {ref_root}; sets {args.relabel_sets}")
    report = {}
    for ds in [s for s in args.relabel_sets.split(",") if s]:
        rd = os.path.join(ref_root, ds)
        if not os.path.exists(os.path.join(rd, "summary.json")):
            # Kaggle unpacks uploaded archives (also .csv.gz -> .csv) and flattens single-folder datasets: find the
            # reference by the dataset named in its summary.json
            hits = []
            for p_ in glob.glob(os.path.join(ref_root or "/kaggle/input", "**", "summary.json"), recursive=True):
                dd = os.path.dirname(p_)
                if not glob.glob(os.path.join(dd, "props_*.csv*")):
                    continue
                try:
                    with open(p_, encoding="utf-8") as f:
                        if json.load(f)["settings"]["dataset"] == ds:
                            hits.append(dd)
                except Exception:  # noqa: BLE001
                    continue
            if not hits:
                log(f"[relabel] {ds}: reference not found, skipped; summary files: "
                    f"{glob.glob(os.path.join(ref_root or '/kaggle/input', '**', 'summary.json'), recursive=True)[:10]}")
                continue
            rd = hits[0]
        log(f"[relabel] {ds}: reference {rd}")
        with open(os.path.join(rd, "summary.json"), encoding="utf-8") as f:
            ref = json.load(f)
        st_ = ref["settings"]
        temps = {t: v["temperature"] for t, v in ref["models"].items()}
        keys = pd.read_csv(os.path.join(rd, "images.csv"))
        keyset = set(zip(keys.stem.astype(str), keys.condition))
        groups = {}
        for tag in temps:
            pf = os.path.join(rd, f"props_{tag}.csv.gz")
            if not os.path.exists(pf):
                pf = pf[:-3]
            if os.path.exists(pf):
                d = pd.read_csv(pf)
                d["stem"] = d.stem.astype(str)
                groups[tag] = {k: g[["row", "box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy()
                               for k, g in d.groupby(["stem", "condition"])}
        dset = st_["dataset"]
        if dset == "acdc":
            items = list_acdc(find_acdc_root(args.acdc_root), tuple(st_["acdc_splits"].split(",")), st_["acdc_ref"])
        elif dset == "bdd":
            items = list_bdd(find_bdd_root(args.bdd_root), tuple(st_["bdd_splits"].split(",")))
        elif dset == "wilddash":
            items = list_wd(find_wd_root(args.wd_root))
        else:
            raise SystemExit(f"relabel: unsupported dataset {dset}")
        items = [items[i] for i in np.random.default_rng(st_["seed"]).permutation(len(items))]
        items = [it for it in items if (str(it["stem"]), it["cond"]) in keyset]
        if args.max_images:
            items = items[: args.max_images]
        seg = [Segmenter(n, device) for n in st_["models"].split(",") if n]
        seg = [m for m in seg if m.tag in temps]
        s = st_["scale"]
        min_area = max(1, int(round(st_["min_area"] * s ** 2)))
        out_dir = os.path.join(args.out, ds)
        os.makedirs(out_dir, exist_ok=True)
        with_cc = dset in ("acdc", "wilddash")
        res = {m.tag: {"obj": [], "props": [], "cc": []} for m in seg}
        log(f"[relabel] {ds}: {len(items)} images (ref {len(keyset)}), models {[m.tag for m in seg]}, scale {s}, cc {with_cc}")

        def flush(final: bool, n_done: int):
            for tag, R in res.items():
                pd.DataFrame(R["obj"]).to_csv(os.path.join(out_dir, f"objects_rl_{tag}.csv"), index=False)
                pd.DataFrame(R["props"]).to_csv(os.path.join(out_dir, f"props_rl_{tag}.csv"), index=False)
                if with_cc:
                    pd.DataFrame(R["cc"]).to_csv(os.path.join(out_dir, f"objects_cc_{tag}.csv"), index=False)
            info = {"status": "done" if final else "running", "n_done": n_done, "n_items": len(items),
                    "elapsed_min": round((time.time() - t_start) / 60, 1), "temperatures": temps, "min_area": min_area}
            with open(os.path.join(out_dir, "summary_rl.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            return info

        n_done, t_loop = 0, time.time()
        for k, item in enumerate(items):
            if n_done and time.time() + 1.3 * (time.time() - t_loop) / n_done > deadline:
                log(f"[relabel] {ds}: stopping at {k} (time budget)")
                break
            loader = {"acdc": load_acdc, "bdd": load_bdd, "wilddash": load_wd}[dset]
            x, inst, _ = loader(item, s)
            stem, cond = str(item["stem"]), item["cond"]
            labels = labels_from_instances(inst)
            objs = instance_objects(inst, min_area)
            cc_objs = instance_objects(semantic_to_instances(labels), min_area) if with_cc else []
            x = x.to(device)
            H2, W2 = labels[::2, ::2].shape
            # half-res masks (same downsampling as the proposal matching of p4) and their integral images
            def mask_ii(ob):
                if not ob:
                    return None, None
                G = torch.stack([downsample_mask(torch.from_numpy(mk).to(device)) for _, _, mk in ob]).float()
                return _integral(G), G.sum((1, 2))
            ii_inst, a_inst = mask_ii(objs)
            ii_cc, a_cc = mask_ii(cc_objs)
            for m in seg:
                T = temps[m.tag]
                logits, _ = m(x, want_feat=False)
                probs = torch.softmax(logits[0] / T, dim=0)
                del logits
                conf_t, pred_t = probs.max(dim=0)
                ent_t = -(probs * probs.clamp_min(1e-12).log()).sum(dim=0) / np.log(N_CLASSES)
                del probs
                pred, conf, ent = pred_t.cpu().numpy(), conf_t.cpu().numpy(), ent_t.cpu().numpy()
                keep = ("instance_id", "class", "area", "frac_true", "omitted", "frac_conf_err", "confident_omission",
                        "predicted_as", "frac_stuff", "frac_conf_stuff", "top_stuff", "frac_conf_top_stuff", "maj_share",
                        "maj_tie", "frac_conf_thing_err", "mean_conf")
                for rw in object_records(pred, conf, ent, labels, objs, st_["conf_thr"], st_["omit_frac"]):
                    res[m.tag]["obj"].append({**{kk: rw[kk] for kk in keep}, "stem": stem, "condition": cond, "model": m.tag})
                if with_cc:
                    for rw in object_records(pred, conf, ent, labels, cc_objs, st_["conf_thr"], st_["omit_frac"]):
                        res[m.tag]["cc"].append({**{kk: rw[kk] for kk in keep}, "stem": stem, "condition": cond, "model": m.tag})
                g = groups.get(m.tag, {}).get((stem, cond))
                if g is not None and len(g):
                    conf_h = F.avg_pool2d(conf_t[None, None], 2)[0, 0][:H2, :W2]
                    bx = torch.from_numpy(g[:, 1:5].astype(np.int64)).to(device)
                    bx[:, [0, 2]] = bx[:, [0, 2]].clamp(0, W2)
                    bx[:, [1, 3]] = bx[:, [1, 3]].clamp(0, H2)
                    barea = ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])).clamp_min(1).double()
                    msp_mean = (_box_sums(_integral(conf_h), bx) / barea).cpu().numpy()
                    msp_max = [float(conf_h[b[1]:b[3], b[0]:b[2]].max()) if (b[3] > b[1] and b[2] > b[0]) else float("nan")
                               for b in bx.tolist()]
                    out = {"row": g[:, 0], "p_msp_mean": msp_mean, "p_msp_max": np.array(msp_max)}
                    for nm, ii, ga in (("chk", ii_inst, a_inst), ("cc", ii_cc, a_cc)):
                        if ii is None:
                            continue
                        inter = _box_sums(ii, bx).T                       # boxes x objects
                        cover = inter / ga[None].double().clamp_min(1)
                        best = cover.argmax(1)
                        bc = cover.gather(1, best[:, None])[:, 0]
                        bi = inter.gather(1, best[:, None])[:, 0]
                        biou = bi / (barea + ga.double()[best] - bi)
                        ids = np.array([(objs if nm == "chk" else cc_objs)[j][0] for j in best.cpu().numpy()])
                        out.update({f"{nm}_id": ids, f"{nm}_cover": bc.cpu().numpy(), f"{nm}_iou": biou.cpu().numpy()})
                    res[m.tag]["props"].extend(pd.DataFrame(out).to_dict("records"))
                del conf_t, pred_t, ent_t
            n_done += 1
            if n_done == 1 or n_done % args.save_every == 0:
                log(f"[relabel] {ds}: {n_done}/{len(items)} images, {(time.time() - t_loop) / n_done:.2f} s/image")
                flush(False, n_done)
        report[ds] = flush(True, n_done)
        log(f"[relabel] {ds} done: {n_done} images")
        del seg
        if device == "cuda":
            torch.cuda.empty_cache()
    return report


# Kaggle scripts get no command-line arguments: the arguments for a Kaggle run are set here.
# v4/v5 = feasibility (B0 + B2). v6 = P4 feasibility on Cityscapes-C (4 models, 17 conditions).
# v7 = same models and GT-free signals on real adverse conditions (ACDC train+val + labelled normal refs).
# v8 = ACDC with denser proposals (RT-DETR >= 0.05 + gamma pass + OWLv2) and second-opinion features (CLIP, zoom).
# v9 = architecture test: per-pixel (SegFormer B0/B2/B5) vs mask classification (MaskFormer-R101, Mask2Former T/S/B/L),
#      unnormalised evidence signals (energy, RbA), 11 Cityscapes(-C) conditions + ACDC in one run, objects only.
KAGGLE_RUN_ARGS: list[str] = ["--mode", "p4", "--dataset", "both", "--max-minutes", "650", "--n-panels", "0",
                              "--conditions", "clean,gaussian_noise-3,shot_noise-3,defocus_blur-3,motion_blur-3,"
                                              "fog-3,fog-5,dark-3,dark-5,brightness-5,contrast-5",
                              "--no-tta", "--no-proposals", "--depth-model", "", "--det-models", "",
                              "--models",
                              "nvidia/segformer-b0-finetuned-cityscapes-1024-1024,"
                              "nvidia/segformer-b2-finetuned-cityscapes-1024-1024,"
                              "nvidia/segformer-b5-finetuned-cityscapes-1024-1024,"
                              "facebook/maskformer-resnet101-cityscapes,"
                              "facebook/mask2former-swin-tiny-cityscapes-semantic,"
                              "facebook/mask2former-swin-small-cityscapes-semantic,"
                              "facebook/mask2former-swin-base-IN21k-cityscapes-semantic,"
                              "facebook/mask2former-swin-large-cityscapes-semantic"]

if __name__ == "__main__":
    cli = sys.argv[1:] if len(sys.argv) > 1 and not sys.argv[1].startswith("-f") else []
    main(cli if cli else (KAGGLE_RUN_ARGS if os.path.isdir("/kaggle/working") else []))
