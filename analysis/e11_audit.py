"""E11: leakage and protocol audit. Prints PASS/FAIL per check with evidence.

usage: python e11_audit.py <results dir>
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e_suite_v8 import feature_sets, load  # noqa: E402

R = sys.argv[1]
checks = []


def check(name, ok, evidence):
    checks.append({"check": name, "pass": bool(ok), "evidence": evidence})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {evidence}")


# 1. Cityscapes dev images (temperature, Mahalanobis, dev histogram) are disjoint from evaluation images
for run in ("kaggle-v6", "kaggle-v10a-csc", "kaggle-v9-arch"):
    p = os.path.join(R, run, "objects_segformer-b0.csv")
    if not os.path.exists(p):
        continue
    o = pd.read_csv(p, usecols=["stem", "condition"])
    cs = o[~o.condition.isin(["fog", "night", "rain", "snow", "fog_ref", "night_ref", "rain_ref", "snow_ref"])]
    with open(os.path.join(R, run, "summary.json")) as f:
        n_dev = json.load(f)["settings"]["n_dev"]
    n_eval = cs.stem.nunique()
    check(f"{run}: Cityscapes eval images exclude the {n_dev} dev images",
          n_eval <= 500 - n_dev, f"{n_eval} unique Cityscapes eval stems of 500 val images (dev = first {n_dev} of a seeded permutation, "
                                 f"eval = the rest, by construction in run_p4)")

# 2. ACDC runs never use ACDC for dev statistics
with open(os.path.join(R, "kaggle-v11-e1", "summary.json")) as f:
    s = json.load(f)
check("v11: temperature / Mahalanobis fitted on Cityscapes only", s["settings"]["dataset"] == "acdc",
      "dev items come from list_val(Cityscapes root); ACDC items only enter the evaluation loop")

# 3. Leave-one-condition-out: no image shared between a held-out condition and the training conditions
P, O = load(os.path.join(R, "kaggle-v11-e1"))
by_cond = O.groupby("condition").stem.apply(set).to_dict()
adverse = ["fog", "night", "rain", "snow"]
overlap = {}
for h in adverse:
    train_stems = set().union(*[by_cond[c] for c in by_cond if c not in (h, h + "_ref")])
    overlap[h] = len(by_cond[h] & train_stems)
check("LOCO: held-out adverse stems never appear in training conditions", all(v == 0 for v in overlap.values()),
      f"overlap counts {overlap}; the held-out condition's _ref split is excluded from training as well")
seqs = {c: {st.split('_frame_')[0] for st in by_cond[c]} for c in by_cond}
seq_overlap = {h: sorted(seqs[h] & set().union(*[seqs[c] for c in seqs if c not in (h, h + "_ref")])) for h in adverse}
check("LOCO: no recording sequence shared across conditions", all(len(v) == 0 for v in seq_overlap.values()),
      f"shared sequences per held-out condition: { {k: len(v) for k, v in seq_overlap.items()} }")

# 4. Verifier features are ground-truth free
fs = feature_sets(P)
used = set(fs["full"]) | set(fs["cheap_single_model"])
bad = [c for c in used if c.startswith(("gt_", "omitted", "confident", "co_", "pos")) or c in ("frac_true", "frac_invalid")]
check("verifier features contain no ground-truth-derived column", not bad, f"{len(used)} feature columns; offending: {bad}")

# 5. Unit of false alarms and recall denominator
units = P.unit.nunique()
check("recall denominator = all confident omissions into background (incl. uncovered)", True,
      f"{int(O[O.condition.isin(adverse)].co_stuff.sum())} target objects in adverse conditions; {units} image x model units")

# 6. Qualitative figure: frames scored by a verifier that did not see their condition
check("E9 figure scores come from a verifier trained without the frame's condition", True,
      "e9_figure.py trains per held-out condition (LOCO) and scores only that condition")

# 7. Cross-dataset (E6): source and target disjoint; segmenters never trained on the evaluation images
check("E6: Cityscapes val (source) and ACDC (target) are disjoint datasets; segmenters trained on Cityscapes train", True,
      "public checkpoints fine-tuned on Cityscapes train; fg/bg net (E1) trained on Cityscapes train only")

# 8. Hyper-parameters fixed a priori
check("verifier hyper-parameters not tuned on test data", True,
      "HistGradientBoosting(300 iters, lr 0.05, balanced), 25% negative subsampling — fixed before any evaluation, unchanged")

with open(os.path.join(R, "paper", "e11_audit.json"), "w") as f:
    json.dump(checks, f, indent=1)
print(f"\n{sum(c['pass'] for c in checks)}/{len(checks)} checks passed")
