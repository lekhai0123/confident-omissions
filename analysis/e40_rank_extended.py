"""E40: synthetic-vs-real rank agreement with four further independently trained checkpoints and a fully paired
bootstrap (review2 M4 / H3 / P1). Same pairs, common frames and statistics as Table 2 / E32 / E35.

New checkpoints (run with the same code, conditions and frames): PSPNet R101 and FCN R101 (OpenMMLab, CNNs with other
decode heads), OneFormer Swin-L (universal segmenter) and EoMT-L (plain ViT, DINOv2), giving 19 configurations,
16 independent checkpoints, five families (SegFormer, MaskFormer/Mask2Former, per-pixel CNN, OneFormer, EoMT).

Per-image confusion counts (tp_c, gt_c, pr_c) make mIoU resamplable: the bootstrap now resamples frames for BOTH
correlations (images independently in the real and the synthetic condition, jointly across segmenters). For the 15
original segmenters the counts come from an argmax-only re-run (pilot.py --mode confusion) on the same frames with the
same corruption seeds; deterministic corruptions reproduce the stored images exactly, the random ones (noise, fog,
dark, motion blur) are a new draw with the same parameters. A two-level bootstrap (segmenters and frames) gives the
interval when the set of segmenters is also treated as a sample.

usage: python e40_rank_extended.py <15-model merged dir> <common dir> <new cs dir> <new acdc dir>
                                   <confusion cs T dir> <confusion cs C dir> <confusion acdc dir> <out json>
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e35_rank_stability import PAIRS, STUFF, VARIANTS, bc_ci  # noqa: E402

N_BOOT = int(os.environ.get("E40_NBOOT", 1000))
NEW = ["pspnet-r101", "fcn-r101", "oneformer-swin-large", "eomt-large-1024"]
C19 = 19


def family(t):
    if t.startswith("segformer"):
        return "SegFormer"
    if t.startswith(("deeplab", "pspnet", "fcn")):
        return "CNN"
    if t.startswith("oneformer"):
        return "OneFormer"
    if t.startswith("eomt"):
        return "EoMT"
    return "MaskFormer"


def cm_cols(kind):
    return [f"{kind}_{c}" for c in range(C19)]


def main():
    merged, common_dir, new_cs, new_ac, conf_T, conf_C, conf_A, out = sys.argv[1:9]
    common = pd.read_csv(os.path.join(common_dir, "images_segformer-b0.csv"), usecols=["stem", "condition"]).drop_duplicates()
    common["stem"] = common.stem.astype(str)
    conds = sorted(set(PAIRS) | {c for v in PAIRS.values() for c in v})
    common = common[common.condition.isin(conds)]
    stems = {c: np.sort(common[common.condition == c].stem.unique()) for c in conds}

    def per_stem_objects(o):
        o = o[o.condition.isin(conds)].copy()
        o["stem"] = o.stem.astype(str)
        o = o.merge(common, on=["stem", "condition"])
        o["co_bg"] = o.confident_omission & o.predicted_as.isin(STUFF)
        return o.groupby(["condition", "stem"]).agg(n_obj=("omitted", "size"), n_om=("omitted", "sum"), n_co=("co_bg", "sum"))

    def per_stem_conf(df):
        df = df.copy()
        df["stem"] = df.stem.astype(str)
        df = df[df.condition.isin(conds)].merge(common, on=["stem", "condition"])
        return df.groupby(["condition", "stem"])[cm_cols("tp") + cm_cols("gt") + cm_cols("pr")].sum()

    OBJ, CM, miou_check = {}, {}, {}
    usecols = ["stem", "condition", "omitted", "confident_omission", "predicted_as"]
    for f in sorted(glob.glob(os.path.join(merged, "objects_*.csv"))):
        tag = os.path.basename(f)[len("objects_"):-4]
        OBJ[tag] = per_stem_objects(pd.read_csv(f, usecols=usecols))
    for d in (conf_T, conf_C, conf_A):
        for f in glob.glob(os.path.join(d, "confusion_*.csv")):
            tag = os.path.basename(f)[len("confusion_"):-4]
            c = per_stem_conf(pd.read_csv(f))
            CM[tag] = c if tag not in CM else pd.concat([CM[tag], c])
    for tag in NEW:
        o = pd.concat([pd.read_csv(os.path.join(d, f"objects_{tag}.csv"), usecols=usecols) for d in (new_cs, new_ac)
                       if os.path.exists(os.path.join(d, f"objects_{tag}.csv"))], ignore_index=True)
        OBJ[tag] = per_stem_objects(o)
        im = pd.concat([pd.read_csv(os.path.join(d, f"images_{tag}.csv")) for d in (new_cs, new_ac)
                        if os.path.exists(os.path.join(d, f"images_{tag}.csv"))], ignore_index=True)
        CM[tag] = per_stem_conf(im)
    tags = sorted(t for t in OBJ if t in CM)
    missing = sorted(set(OBJ) ^ set(CM))
    print("segmenters:", len(tags), "| missing one side:", missing)

    # arrays: condition -> (models x stems)
    def at(df, c, st):
        """rows of condition c reindexed to the common stems (all-NaN if the condition is absent)"""
        if c in df.index.get_level_values(0):
            return df.loc[c].reindex(st)
        return pd.DataFrame(np.nan, index=st, columns=df.columns)

    A = {}
    for c in conds:
        st = stems[c]
        o = {k: np.stack([at(OBJ[t], c, st)[k].fillna(0).to_numpy() for t in tags]) for k in ("n_obj", "n_om", "n_co")}
        cov = np.array([at(CM[t], c, st).notna().all(axis=1).mean() for t in tags])
        m = {k: np.stack([at(CM[t], c, st)[cm_cols(k)].fillna(0).to_numpy() for t in tags])
             for k in ("tp", "gt", "pr")}   # models x stems x 19
        A[c] = {**o, **m, "coverage": cov}
    for c in conds:
        low = [(t, round(float(v), 3)) for t, v in zip(tags, A[c]["coverage"]) if v < 0.99]
        if low:
            print(f"  WARNING {c}: confusion coverage < 99% for {low}")

    def stats(c, idx=None, mods=None):
        a = A[c]
        sl = slice(None) if idx is None else idx
        mm = slice(None) if mods is None else mods
        nm, nc = a["n_om"][mm][:, sl].sum(1), a["n_co"][mm][:, sl].sum(1)
        tp, gt, pr = a["tp"][mm][:, sl].sum(1), a["gt"][mm][:, sl].sum(1), a["pr"][mm][:, sl].sum(1)
        iou = tp / np.maximum(gt + pr - tp, 1)
        miou = np.where(gt > 0, iou, np.nan)
        return np.nanmean(miou, 1), nc / np.maximum(nm, 1)

    # sanity: pooled mIoU from the re-run counts vs the stored mIoU of the original runs
    summ = json.load(open(os.path.join(merged, "summary.json"), encoding="utf-8"))
    for c in conds:
        mi, _ = stats(c)
        for t, v in zip(tags, mi):
            if t in summ["models"]:
                ref = summ["models"][t]["miou_by_condition"][c]["argmax"]["miou"]
                miou_check.setdefault(c, {})[t] = round(float(v - ref), 4)
    worst = max((abs(v), c, t) for c, d in miou_check.items() for t, v in d.items())
    print("max |mIoU(re-run, common frames) - mIoU(stored, all frames)|:", worst)

    rng = np.random.default_rng(0)
    fam = np.array([family(t) for t in tags])
    sets = {"19": tags, "16": [t for t in tags if t not in VARIANTS], "15_original": [t for t in tags if t not in NEW],
            "12_original": [t for t in tags if t not in NEW and t not in VARIANTS]}
    report = {"models": tags, "families": dict(zip(tags, fam.tolist())), "miou_rerun_minus_stored": miou_check, "sets": {}}
    for set_name, sel_tags in sets.items():
        mods = np.array([tags.index(t) for t in sel_tags])
        res = {}
        for real, syns in PAIRS.items():
            for s in syns:
                m_r, c_r = stats(real, mods=mods)
                m_s, c_s = stats(s, mods=mods)
                rm, rc = spearmanr(m_r, m_s)[0], spearmanr(c_r, c_s)[0]
                d_img, d_two = [], []
                for b in range(N_BOOT):
                    ir = rng.integers(0, len(stems[real]), len(stems[real]))
                    is_ = rng.integers(0, len(stems[s]), len(stems[s]))
                    a1, b1 = stats(real, ir, mods)
                    a2, b2 = stats(s, is_, mods)
                    d_img.append(spearmanr(a1, a2)[0] - spearmanr(b1, b2)[0])
                    mi = rng.integers(0, len(mods), len(mods))      # two-level: segmenters and frames
                    if len(set(mi)) >= 5:
                        d_two.append(spearmanr(a1[mi], a2[mi])[0] - spearmanr(b1[mi], b2[mi])[0])
                lofo = {}
                for f in np.unique(fam[mods]):
                    keep = fam[mods] != f
                    if keep.sum() >= 5:
                        lofo[f"without_{f}"] = {"n": int(keep.sum()), "rho_miou": round(float(spearmanr(m_r[keep], m_s[keep])[0]), 3),
                                                "rho_conf": round(float(spearmanr(c_r[keep], c_s[keep])[0]), 3)}
                res[f"{real}~{s}"] = {"rho_miou": round(float(rm), 3), "p_miou": round(float(spearmanr(m_r, m_s)[1]), 4),
                                      "rho_conf": round(float(rc), 3), "p_conf": round(float(spearmanr(c_r, c_s)[1]), 4),
                                      "delta": round(float(rm - rc), 3), "delta_ci95_images": bc_ci(d_img, rm - rc),
                                      "delta_ci95_images_and_models": bc_ci(d_two, rm - rc), "lofo": lofo}
        # within-domain split-half repeatability of the confident-share ranking and the noise ceiling (as in E35)
        rep = {}
        for c in conds:
            n = len(stems[c])
            rr = []
            for _ in range(200):
                perm = rng.permutation(n)
                rr.append(spearmanr(stats(c, perm[: n // 2], mods)[1], stats(c, perm[n // 2:], mods)[1])[0])
            r = float(np.nanmean(rr))
            rep[c] = {"split_half_rho_conf": round(r, 3), "spearman_brown_conf": round(2 * r / (1 + r), 3)}
        for k, v in res.items():
            real, syn = k.split("~")
            v["noise_ceiling_conf"] = round(float(np.sqrt(max(0, rep[real]["spearman_brown_conf"])
                                                          * max(0, rep[syn]["spearman_brown_conf"]))), 3)
        P = res.values()
        summary = {"n": len(mods), "split_half": rep,
                   "miou_sig_uncorr": int(sum(v["p_miou"] < 0.05 and v["rho_miou"] > 0 for v in P)),
                   "conf_sig_uncorr": int(sum(v["p_conf"] < 0.05 and v["rho_conf"] > 0 for v in P)),
                   "delta_ci_excl0_images": int(sum(v["delta_ci95_images"][0] > 0 for v in P)),
                   "delta_ci_excl0_images_and_models": int(sum(v["delta_ci95_images_and_models"][0] > 0 for v in P)),
                   "miou_sig_bonf": int(sum(v["p_miou"] < 0.005 and v["rho_miou"] > 0 for v in P)),
                   "conf_sig_bonf": int(sum(v["p_conf"] < 0.005 and v["rho_conf"] > 0 for v in P))}
        report["sets"][set_name] = {"pairs": res, "summary": summary}
        print(f"\n== {set_name} ({len(mods)} segmenters)")
        for k, v in res.items():
            print(f"  {k:24s} rho mIoU {v['rho_miou']:+.2f} conf {v['rho_conf']:+.2f} | delta {v['delta']:+.2f} "
                  f"CI img {v['delta_ci95_images']} | CI img+models {v['delta_ci95_images_and_models']} "
                  f"| p mIoU {v['p_miou']:.4f} conf {v['p_conf']:.4f} | ceiling {v['noise_ceiling_conf']}")
            print("      LOFO:", {f: (x["rho_miou"], x["rho_conf"]) for f, x in v["lofo"].items()})
        print("  summary:", summary)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
