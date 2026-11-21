"""Build the input of the 15-model rank analysis (E13 + DeepLabV3+ R101/R18), WRITING_KIT section 0 (CNN part (a)).

Creates <out>/merged: the 13 objects files of e13-merged-13models plus the two CNNs (Cityscapes(-C) objects from
v19a concatenated with ACDC objects from v19c), and a summary.json with all 15 models' mIoU per condition; and
<out>/common/images_segformer-b0.csv: the (image, condition) pairs of the original E13 common set that the CNN runs
also contain, so that all 15 models are compared on identical images.

usage: python e32_build_15models.py <e13 merged dir> <v9 dir> <v19a dir> <v19c dir> <out dir>
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pandas as pd

CNNS = ["deeplabv3plus-r101", "deeplabv3plus-r18"]


def main():
    merged13, v9, v19a, v19c, out = sys.argv[1:6]
    md, cd = os.path.join(out, "merged"), os.path.join(out, "common")
    os.makedirs(md, exist_ok=True)
    os.makedirs(cd, exist_ok=True)
    for f in os.listdir(merged13):
        if f.startswith("objects_"):
            shutil.copy(os.path.join(merged13, f), os.path.join(md, f))
    summ = json.load(open(os.path.join(merged13, "summary.json"), encoding="utf-8"))
    sa = json.load(open(os.path.join(v19a, "summary.json"), encoding="utf-8"))
    sc = json.load(open(os.path.join(v19c, "summary.json"), encoding="utf-8"))
    have = None
    for m in CNNS:
        o = pd.concat([pd.read_csv(os.path.join(v19a, f"objects_{m}.csv")), pd.read_csv(os.path.join(v19c, f"objects_{m}.csv"))],
                      ignore_index=True)
        o.to_csv(os.path.join(md, f"objects_{m}.csv"), index=False)
        mi = dict(sa["models"][m]["miou_by_condition"])
        mi.update(sc["models"][m]["miou_by_condition"])
        summ["models"][m] = {"miou_by_condition": mi}
        pairs = set(zip(o.stem, o.condition))
        have = pairs if have is None else have & pairs
    json.dump(summ, open(os.path.join(md, "summary.json"), "w", encoding="utf-8"), indent=1)
    common = pd.read_csv(os.path.join(v9, "images_segformer-b0.csv"), usecols=["stem", "condition"]).drop_duplicates()
    keep = common[[(s, c) in have for s, c in zip(common.stem, common.condition)]]
    keep.to_csv(os.path.join(cd, "images_segformer-b0.csv"), index=False)
    print("common pairs:", len(common), "->", len(keep), "| per condition:", keep.condition.value_counts().to_dict())


if __name__ == "__main__":
    main()
