"""E37: analysis of the relabel pass (pilot.py --mode relabel; review of 2026-11-22, critical issue 1, major 4 and 5).

A. Definition check. The paper's target requires (1) at most 5% of the object's pixels correct, (2) at least 50% of
   them wrong with confidence >= 0.9, and (3) a stuff class as the most frequent prediction. (2) and (3) may hold on
   different pixels. The joint criterion requires at least 50% of the pixels to be BOTH predicted as stuff AND
   confident; the strict variant requires one single stuff class (the most frequent stuff class). Transition table
   per domain and segmenter, ties of the majority class, and the confident share under the joint criterion.
B. Re-scoring with FIXED scores (trained on the original labels): recall at matched strict false alarms for the
   verifier and the baselines when the targets are the joint-criterion objects; then the verifier retrained on them.
C. Maximum softmax probability (MSP) inside each box as a separate baseline (best of mean/max and sign).
D. Connected-component units on ACDC and WildDash2 (the object definition used for BDD100K): targets and box matches
   recomputed on components, scores fixed; does the comparison between monitors change?
E. Method comparison (R@1, paired difference) under the 15 threshold variants of Appendix A, ACDC, fixed scores.

usage: python e37_relabel.py <v11> <v13> <v15> <rl acdc dir> <rl bdd dir> <rl wd dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e10_e12 import paired_ci  # noqa: E402
from e18_comparison import best_raw, found_at, gbm, score_row  # noqa: E402
from e34_review_offline import L3_COLS, logit, loco, predict  # noqa: E402
from e_suite_v8 import ADVERSE, MODELS, STUFF, feature_sets, load  # noqa: E402

DET = ["rtdetr", "rtdetr_gamma"]


def read_rl(d, prefix):
    fr = []
    for tag in MODELS:
        f = os.path.join(d, f"{prefix}_{tag}.csv")
        if os.path.exists(f):
            x = pd.read_csv(f)
            x["model"] = tag
            fr.append(x)
    X = pd.concat(fr, ignore_index=True)
    if "stem" in X:
        X["stem"] = X.stem.astype(str)
    return X


def labels(o):
    o = o.copy()
    o["old"] = o.confident_omission & o.predicted_as.isin(STUFF)
    o["joint"] = o.omitted & (o.frac_conf_stuff >= 0.5)
    o["joint_same"] = o.omitted & (o.frac_conf_top_stuff >= 0.5)
    return o


def transition(o):
    r = {"old": int(o.old.sum()), "joint": int(o.joint.sum()), "both": int((o.old & o.joint).sum()),
         "old_only": int((o.old & ~o.joint).sum()), "joint_only": int((~o.old & o.joint).sum()),
         "joint_same": int(o.joint_same.sum()), "old_ties": int((o.old & o.maj_tie).sum())}
    old = o[o.old]
    if len(old):
        r["old_frac_conf_stuff_median"] = round(float(old.frac_conf_stuff.median()), 3)
        r["old_share_frac_conf_stuff_below_0.5"] = round(float((old.frac_conf_stuff < 0.5).mean()), 4)
        r["old_share_frac_conf_stuff_below_0.25"] = round(float((old.frac_conf_stuff < 0.25).mean()), 4)
        r["old_only_mean_frac_conf_thing_err"] = round(float(o[o.old & ~o.joint].frac_conf_thing_err.mean()), 3) if r["old_only"] else None
    r["agreement_old_joint"] = round(r["both"] / max(1, r["old"] + r["joint_only"]), 4)   # Jaccard
    return r


def with_targets(T, O, col):
    """Copy of proposals T whose positives are the objects of O flagged by `col` (cover >= 0.5)."""
    t = O[O[col]][["model", "stem", "condition", "instance_id"]].rename(columns={"instance_id": "gt_instance_id"})
    t["_tgt"] = True
    X = T.drop(columns=[c for c in ("_tgt",) if c in T]).merge(t, on=["model", "stem", "condition", "gt_instance_id"], how="left")
    X.index = T.index
    X["pos_co_stuff"] = X._tgt.fillna(False).astype(bool) & (X.gt_cover >= 0.5)
    keys = list(zip(O[O[col]].model, O[O[col]].stem, O[O[col]].instance_id))
    return X, keys, O[O[col]].stem.to_numpy()


def table(T, S, keys, stems):
    if not keys:
        return {"n_targets": 0, "rows": {k: {"R@0.5": None, "R@1.0": None, "R@2.0": None} for k in S}, "paired": None}
    rows = {k: {kk: round(v, 4) for kk, v in score_row(T, s, keys).items()} for k, s in S.items()}
    base = [k for k in S if k != "ours"]
    best = max(base, key=lambda k: rows[k]["R@1.0"])
    d, ci, _ = paired_ci(found_at(T, S["ours"], keys, 1.0).astype(float), found_at(T, S[best], keys, 1.0).astype(float), stems)
    return {"n_targets": len(keys), "rows": rows, "paired": {"vs": best, "diff": round(d, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]}}


def main():
    v11, v13, v15, rla, rlb, rlw, out = sys.argv[1:8]
    res = {"A_definition": {}, "B_rescoring": {}, "C_msp": {}, "D_components": {}}
    P, _ = load(v11)
    fs = feature_sets(P)
    cheap = [c for c in fs["cheap_single_model"] if not c.startswith(("p_energy", "p_rba", "p_fgbg"))]
    meta = [c for c in P.columns if c.startswith(("p_ent", "p_margin", "p_thing_prob"))] + [
        "p_self_stuff_frac", "p_area_frac", "p_box_h", "p_box_w", "p_center_y", "p_fill"] + [f"m_{t}" for t in MODELS]
    Dd = P[P.det_src.isin(DET)].copy()
    Dd["stem"] = Dd.stem.astype(str)
    fit = {"ours": (gbm, cheap), "meta": (gbm, meta), "logit3": (logit, L3_COLS)}

    domains = {}
    OA = labels(read_rl(rla, "objects_rl"))
    sA = {k: loco(Dd, lambda tr, f=f, c=c: (lambda m: (lambda D: predict(m, D, c)))(f(tr, c))) for k, (f, c) in fit.items()}
    TA = Dd[Dd.condition.isin(ADVERSE)]
    domains["ACDC"] = (TA, {k: v[TA.index].to_numpy() for k, v in sA.items()}, OA, read_rl(rla, "props_rl"),
                       read_rl(rla, "objects_cc") if os.path.exists(os.path.join(rla, "objects_cc_segformer-b0.csv")) else None)
    models_all = {k: f(Dd, c) for k, (f, c) in fit.items()}
    # verifier and meta-classifier retrained on the joint-criterion targets (ACDC objects incl. reference frames)
    DdJ, _, _ = with_targets(Dd, OA, "joint")
    sJ = {k: loco(DdJ, lambda tr, f=fit[k][0], c=fit[k][1]: (lambda m: (lambda D: predict(m, D, c)))(f(tr, c))) for k in ("ours", "meta")}
    refit = {"ACDC": {k: v[TA.index].to_numpy() for k, v in sJ.items()}}
    models_J = {k: fit[k][0](DdJ, fit[k][1]) for k in ("ours", "meta")}
    for name, d, rl in (("BDD100K", v13, rlb), ("WildDash2", v15, rlw)):
        if not os.path.exists(os.path.join(rl, "objects_rl_segformer-b0.csv")):
            print("skip", name)
            continue
        PX, _ = load(d)
        T = PX[PX.det_src.isin(DET)].copy()
        T["stem"] = T.stem.astype(str)
        S = {k: predict(models_all[k], T, fit[k][1]) for k in fit}
        cc = os.path.join(rl, "objects_cc_segformer-b0.csv")
        refit[name] = {k: predict(models_J[k], T, fit[k][1]) for k in ("ours", "meta")}
        domains[name] = (T, S, labels(read_rl(rl, "objects_rl")), read_rl(rl, "props_rl"),
                         read_rl(rl, "objects_cc") if os.path.exists(cc) else None)

    for name, (T, S, O, PR, OC) in domains.items():
        T = T.copy()
        S = dict(S, rule=T.rule.to_numpy())
        Ot = O[O.condition.isin(ADVERSE)] if name == "ACDC" else O
        # A. definition
        A = {"all": transition(Ot)}
        for m in MODELS:
            A[m] = transition(Ot[Ot.model == m])
        share = {}
        for m in MODELS:
            for scope, oo in ((("adv", Ot), ("ref", O[O.condition.str.endswith("_ref")])) if name == "ACDC" else (("all", Ot),)):
                x = oo[oo.model == m]
                om = max(1, int(x.omitted.sum()))
                share[f"{m}|{scope}"] = {"old": round(100 * x.old.sum() / om, 2), "joint": round(100 * x.joint.sum() / om, 2),
                                         "joint_same": round(100 * x.joint_same.sum() / om, 2)}
        A["confident_share_pct"] = share
        res["A_definition"][name] = A
        print(f"\n== {name} A:", A["all"])
        print("   shares:", share)
        # B. re-scoring with fixed scores
        B = {}
        for col in ("old", "joint", "joint_same"):
            X, keys, stems = with_targets(T, Ot, col)
            B[f"{col}_fixed_scores"] = table(X, S, keys, stems)
            print(f"   B {col:10s} n={len(keys)}", {k: v["R@1.0"] for k, v in B[f"{col}_fixed_scores"]["rows"].items()},
                  B[f"{col}_fixed_scores"]["paired"])
        X, keys, stems = with_targets(T, Ot, "joint")
        SJ = dict(S, ours=refit[name]["ours"], meta=refit[name]["meta"])
        B["joint_refit"] = table(X, SJ, keys, stems)
        print("   B joint refit", {k: v["R@1.0"] for k, v in B["joint_refit"]["rows"].items()}, B["joint_refit"]["paired"])
        res["B_rescoring"][name] = B
        # C. MSP on the stored boxes (original targets)
        M = PR[["model", "row", "p_msp_mean", "p_msp_max"]]
        X = T.merge(M, on=["model", "row"], how="left")
        X.index = T.index
        Xo, keys, stems = with_targets(X, Ot, "old")
        C = {}
        for stem in ("msp", "margin", "ent"):
            b = best_raw(Xo, stem, keys) if keys else None
            if b:
                C[stem] = dict(b[0], variant=b[2])
        C["coverage_msp_rows"] = round(float(X.p_msp_mean.notna().mean()), 4)
        res["C_msp"][name] = C
        print("   C:", {k: (v["R@0.5"], v["R@1.0"], v["R@2.0"], v["variant"]) for k, v in C.items() if isinstance(v, dict)})
        # D. connected components (true instances available)
        if OC is not None and "cc_id" in PR:
            oc = OC[OC.condition.isin(ADVERSE)] if name == "ACDC" else OC
            oc = oc.assign(old=oc.confident_omission & oc.predicted_as.isin(STUFF))
            X = T.merge(PR[["model", "row", "cc_id", "cc_cover"]], on=["model", "row"], how="left")
            X.index = T.index
            X["gt_instance_id"], X["gt_cover"] = X.cc_id, X.cc_cover
            Xc, keys, stems = with_targets(X, oc, "old")
            Dc = table(Xc, S, keys, stems)
            Dc["n_targets_instances"] = int(Ot.old.sum())
            res["D_components"][name] = Dc
            print("   D components:", Dc["n_targets"], "targets;", {k: v["R@1.0"] for k, v in Dc["rows"].items()}, Dc["paired"])
    # E. method comparison under the 15 threshold variants of Appendix A (ACDC, fixed scores, stored objects)
    T, S = domains["ACDC"][0].copy(), dict(domains["ACDC"][1])
    S["rule"] = T.rule.to_numpy()
    st = []
    for tag in MODELS:
        o = pd.read_csv(os.path.join(v11, f"objects_{tag}.csv"),
                        usecols=["stem", "condition", "instance_id", "frac_true", "frac_conf_err", "mean_conf", "predicted_as"])
        o["model"] = tag
        st.append(o[o.condition.isin(ADVERSE)])
    st = pd.concat(st, ignore_index=True)
    st["stem"] = st.stem.astype(str)
    E = {}
    for om in (0.02, 0.05, 0.10):
        for crit in ("px50@0.9", 0.80, 0.85, 0.90, 0.95):
            conf = st.frac_conf_err >= 0.5 if crit == "px50@0.9" else st.mean_conf >= crit
            col = f"v_{om}_{crit}"
            st[col] = (st.frac_true <= om) & conf & st.predicted_as.isin(STUFF)
            X, keys, stems = with_targets(T, st, col)
            tb = table(X, {k: S[k] for k in ("ours", "rule", "meta", "logit3")}, keys, stems)
            E[f"omit<={om}|{crit}"] = {"n": len(keys), "R@1": {k: v["R@1.0"] for k, v in tb["rows"].items()}, "paired": tb["paired"]}
            print(f"   E {om} {crit}: n={len(keys)}", E[f"omit<={om}|{crit}"]["R@1"], tb["paired"])
    # refit ours and the logistic baseline on the variants where the fixed-score comparison is not clear
    allst = []
    for tag in MODELS:
        o = pd.read_csv(os.path.join(v11, f"objects_{tag}.csv"),
                        usecols=["stem", "condition", "instance_id", "frac_true", "frac_conf_err", "mean_conf", "predicted_as"])
        o["model"] = tag
        allst.append(o)
    allst = pd.concat(allst, ignore_index=True)
    allst["stem"] = allst.stem.astype(str)
    for crit in ("px50@0.9", 0.90, 0.95):
        conf = allst.frac_conf_err >= 0.5 if crit == "px50@0.9" else allst.mean_conf >= crit
        allst["vv"] = (allst.frac_true <= 0.05) & conf & allst.predicted_as.isin(STUFF)
        DdV, _, _ = with_targets(Dd, allst, "vv")
        sV = {k: loco(DdV, lambda tr, f=fit[k][0], c=fit[k][1]: (lambda m: (lambda D: predict(m, D, c)))(f(tr, c)))[T.index].to_numpy()
              for k in ("ours", "logit3")}
        st["vv"] = (st.frac_true <= 0.05) & (st.frac_conf_err >= 0.5 if crit == "px50@0.9" else st.mean_conf >= crit) & st.predicted_as.isin(STUFF)
        X, keys, stems = with_targets(T, st, "vv")
        tb = table(X, dict(sV, rule=S["rule"]), keys, stems)
        E[f"refit omit<=0.05|{crit}"] = {"n": len(keys), "R@1": {k: v["R@1.0"] for k, v in tb["rows"].items()}, "paired": tb["paired"]}
        print(f"   E refit 0.05 {crit}: n={len(keys)}", E[f"refit omit<=0.05|{crit}"]["R@1"], tb["paired"])
    res["E_definition_variants_ACDC"] = E
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
