"""E35: stability of the synthetic-vs-real rank agreement (review of 2026-11-22, major 2 and section 7).

Same data and pairs as Table 2 / E32 (15 segmenters; 12 = without the three SegFormer-B0 resolution variants).
For each pair (real ACDC condition ~ synthetic corruption):
  - delta_rho = rho(accuracy) - rho(confident share), with a 95% bootstrap CI over images (stems resampled
    independently in the real and in the synthetic condition, jointly across models). Accuracy = mIoU (fixed: per-image
    confusion counts are not stored; its sampling variance over ~400-500 images is small) and, fully bootstrapped,
    the omitted share (object level, same images).
  - leave-one-family-out rho (SegFormer, Mask(2)Former, DeepLabV3+).
  - model bootstrap: CI of rho when the segmenters themselves are resampled.
Within-domain repeatability (the reviewer's alternative explanation): split the frames of ONE condition into two
random halves (200 repeats) and compute rho between the halves, for the confident share and the omitted share.
If the within-domain rho is high while the cross-domain rho is low, the disagreement is not just counting noise.

usage: python e35_rank_stability.py <15-model merged dir> <common dir> <out json>
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
PAIRS = {"fog": ["fog-3", "fog-5"], "night": ["dark-3", "dark-5", "brightness-5"], "rain": ["motion_blur-3", "defocus_blur-3"],
         "snow": ["gaussian_noise-3", "shot_noise-3", "contrast-5"]}
VARIANTS = {"segformer-b0-512-1024", "segformer-b0-640-1280", "segformer-b0-768-768"}
N_BOOT = 1000


def bc_ci(boot, obs):
    """Bias-corrected percentile interval (Efron): Spearman rho of noisy counts is biased towards 0 under resampling."""
    b = np.asarray(boot, float)
    b = b[~np.isnan(b)]
    z0 = norm.ppf(np.clip((b < obs).mean() + 0.5 * (b == obs).mean(), 1e-3, 1 - 1e-3))
    lo, hi = norm.cdf(2 * z0 - 1.96), norm.cdf(2 * z0 + 1.96)
    return [round(float(np.percentile(b, 100 * lo)), 3), round(float(np.percentile(b, 100 * hi)), 3)]


def family(t):
    return "segformer" if t.startswith("segformer") else ("deeplab" if t.startswith("deeplab") else "maskformer")


def main():
    d, common_dir, out_path = sys.argv[1:4]
    summ = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    common = pd.read_csv(os.path.join(common_dir, "images_segformer-b0.csv"), usecols=["stem", "condition"]).drop_duplicates()
    conds = sorted(set(PAIRS) | {c for v in PAIRS.values() for c in v})
    agg, miou = {}, {}
    for f in sorted(glob.glob(os.path.join(d, "objects_*.csv"))):
        tag = os.path.basename(f)[len("objects_"):-4]
        o = pd.read_csv(f, usecols=["stem", "condition", "omitted", "confident_omission", "predicted_as"])
        o = o[o.condition.isin(conds)].merge(common, on=["stem", "condition"])
        o["co_bg"] = o.confident_omission & o.predicted_as.isin(STUFF)
        g = o.groupby(["condition", "stem"]).agg(n_obj=("omitted", "size"), n_om=("omitted", "sum"), n_co=("co_bg", "sum"))
        agg[tag] = {c: g.loc[c] for c in conds}
        miou[tag] = {c: summ["models"][tag]["miou_by_condition"][c]["argmax"]["miou"] for c in conds}
    rng = np.random.default_rng(0)
    report = {}
    for set_name, tags in (("15", sorted(agg)), ("12", sorted(t for t in agg if t not in VARIANTS))):
        stems = {c: agg[tags[0]][c].index.to_numpy() for c in conds}
        # arrays (models x stems) aligned on the first model's stems
        A = {c: {k: np.stack([agg[t][c].reindex(stems[c])[k].fillna(0).to_numpy() for t in tags]) for k in ("n_obj", "n_om", "n_co")}
             for c in conds}

        def stats(c, idx=None, models=None):
            a = A[c]
            sel = slice(None) if idx is None else idx
            mm = slice(None) if models is None else models
            no, nm, nc = a["n_obj"][mm][:, sel].sum(1), a["n_om"][mm][:, sel].sum(1), a["n_co"][mm][:, sel].sum(1)
            return nm / np.maximum(no, 1), nc / np.maximum(nm, 1), nc, nm

        fam = np.array([family(t) for t in tags])
        res = {"models": tags, "pairs": {}}
        for real, syns in PAIRS.items():
            for s in syns:
                om_r, cs_r, nc_r, _ = stats(real)
                om_s, cs_s, nc_s, _ = stats(s)
                m_r = np.array([miou[t][real] for t in tags]); m_s = np.array([miou[t][s] for t in tags])
                rm, rc, ro = spearmanr(m_r, m_s)[0], spearmanr(cs_r, cs_s)[0], spearmanr(om_r, om_s)[0]
                d_m, d_o = [], []
                for _ in range(N_BOOT):
                    ir = rng.integers(0, len(stems[real]), len(stems[real]))
                    is_ = rng.integers(0, len(stems[s]), len(stems[s]))
                    o1, c1, _, _ = stats(real, ir)
                    o2, c2, _, _ = stats(s, is_)
                    rcb = spearmanr(c1, c2)[0]
                    d_m.append(rm - rcb)
                    d_o.append(spearmanr(o1, o2)[0] - rcb)
                mb = []
                for _ in range(N_BOOT):
                    mi = rng.integers(0, len(tags), len(tags))
                    if len(set(mi)) < 4:
                        continue
                    mb.append((spearmanr(m_r[mi], m_s[mi])[0], spearmanr(cs_r[mi], cs_s[mi])[0]))
                mb = np.array(mb)
                lofo = {}
                for f in np.unique(fam):
                    keep = fam != f
                    if keep.sum() >= 4:
                        lofo[f"without_{f}"] = {"n": int(keep.sum()), "rho_miou": round(float(spearmanr(m_r[keep], m_s[keep])[0]), 3),
                                                "rho_conf": round(float(spearmanr(cs_r[keep], cs_s[keep])[0]), 3)}
                pct = lambda x: [round(float(np.nanpercentile(x, 2.5)), 3), round(float(np.nanpercentile(x, 97.5)), 3)]  # noqa: E731
                res["pairs"][f"{real}~{s}"] = {
                    "rho_miou": round(float(rm), 3), "rho_conf": round(float(rc), 3), "rho_omitted": round(float(ro), 3),
                    "delta_miou_minus_conf": round(float(rm - rc), 3), "delta_miou_ci95_pct": pct(d_m),
                    "delta_miou_ci95": bc_ci(d_m, rm - rc),
                    "delta_omitted_minus_conf": round(float(ro - rc), 3), "delta_omitted_ci95_pct": pct(d_o),
                    "delta_omitted_ci95": bc_ci(d_o, ro - rc),
                    "model_boot_rho_miou_ci95": pct(mb[:, 0]), "model_boot_rho_conf_ci95": pct(mb[:, 1]),
                    "lofo": lofo, "targets_min_per_cell": int(min(nc_r.min(), nc_s.min())),
                    "targets_median_per_cell": [int(np.median(nc_r)), int(np.median(nc_s))]}
        # within-domain repeatability
        rep = {}
        for c in conds:
            n = len(stems[c])
            rr, ro_ = [], []
            for _ in range(200):
                perm = rng.permutation(n)
                a, b = perm[: n // 2], perm[n // 2:]
                oa, ca, _, _ = stats(c, a)
                ob, cb, _, _ = stats(c, b)
                rr.append(spearmanr(ca, cb)[0])
                ro_.append(spearmanr(oa, ob)[0])
            rep[c] = {"n_frames": int(n), "split_half_rho_conf": round(float(np.nanmean(rr)), 3),
                      "split_half_rho_conf_5_95": [round(float(np.nanpercentile(rr, 5)), 3), round(float(np.nanpercentile(rr, 95)), 3)],
                      "split_half_rho_omitted": round(float(np.nanmean(ro_)), 3),
                      "spearman_brown_conf": round(float(2 * np.nanmean(rr) / (1 + np.nanmean(rr))), 3)}
        res["within_domain_split_half"] = rep
        for k, v in res["pairs"].items():   # attenuation ceiling: rho cannot exceed sqrt(rel_real * rel_syn) from noise alone
            real, syn = k.split("~")
            v["noise_ceiling_conf"] = round(float(np.sqrt(max(0, rep[real]["spearman_brown_conf"]) * max(0, rep[syn]["spearman_brown_conf"]))), 3)
        P = res["pairs"].values()
        res["summary"] = {"delta_miou_ci_excludes_0": int(sum(v["delta_miou_ci95"][0] > 0 for v in P)),
                          "delta_omitted_ci_excludes_0": int(sum(v["delta_omitted_ci95"][0] > 0 for v in P))}
        report[set_name] = res
        print(f"\n== {set_name} segmenters")
        for k, v in res["pairs"].items():
            print(f"  {k:24s} rho mIoU {v['rho_miou']:+.2f} conf {v['rho_conf']:+.2f} | d(mIoU-conf) {v['delta_miou_minus_conf']:+.2f} "
                  f"CI {v['delta_miou_ci95']} | d(om-conf) {v['delta_omitted_minus_conf']:+.2f} CI {v['delta_omitted_ci95']} | "
                  f"model-boot conf {v['model_boot_rho_conf_ci95']} | ceiling {v['noise_ceiling_conf']} | min/cell {v['targets_min_per_cell']}")
            print("      LOFO:", {f: (x["rho_miou"], x["rho_conf"]) for f, x in v["lofo"].items()})
        print("  split-half:", {c: (v["split_half_rho_conf"], v["split_half_rho_omitted"]) for c, v in rep.items()})
        print("  summary:", res["summary"])
    json.dump(report, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
