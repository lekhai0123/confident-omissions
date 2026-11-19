"""E29: how much does defining objects as connected components of the semantic labels (as we must on BDD100K) change
the targets? Measured on ACDC, where true instances exist.

For every adverse ACDC image (train + val): true instances from the panoptic ground truth (as in the main pipeline);
connected-component objects from the semantic labels derived from the same ground truth (pilot.semantic_to_instances,
exactly the BDD100K procedure). A true thing instance (non-crowd, >= 64 px) is "merged" when its connected component
also contains pixels of another instance or of a crowd region of the same class.
Reported: share of all thing instances merged, and share of the confident-omission-into-background targets of the four
main segmenters (v11) that are merged (these are the targets whose definition would change under connected components).

usage: python e29_cc_merge.py <acdc root> <v11 dir> <out json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebook"))
import pilot  # noqa: E402

STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky", "pole", "traffic light", "traffic sign"}
ADVERSE = ["fog", "night", "rain", "snow"]


def main():
    root, v11, out = sys.argv[1:4]
    items = [it for it in pilot.list_acdc(root, ("train", "val"), False) if it["cond"] in ADVERSE]
    targets = []
    for m in ["segformer-b0", "segformer-b2", "segformer-b5", "mask2former-swin-small"]:
        o = pd.read_csv(os.path.join(v11, f"objects_{m}.csv"), usecols=["stem", "condition", "instance_id", "confident_omission", "predicted_as"])
        o = o[o.condition.isin(ADVERSE) & o.confident_omission & o.predicted_as.isin(STUFF)]
        targets += list(zip(o.stem, o.instance_id))
    tset = {}
    for s, i in targets:
        tset.setdefault(s, []).append(i)
    lut = np.full(256, 255, np.uint8)
    for lid, tid in pilot.LABELID_TO_TRAINID.items():
        lut[lid] = tid
    rows, merged_ids = [], {}
    for k, it in enumerate(items):
        inst = pilot.panoptic_to_instances(np.asarray(Image.open(it["pan"]).convert("RGB")), it["segments"])
        label = np.where(inst >= 1000, inst // 1000, inst).astype(np.int64)
        tid = lut[np.clip(label, 0, 255)]
        cc = pilot.semantic_to_instances(tid)
        merged = set()
        for iid in np.unique(inst[inst >= 1000]):
            m = inst == iid
            if m.sum() < 64:
                continue
            comp = np.unique(cc[m])
            comp = comp[comp >= 1000]
            if len(comp) == 0:
                continue
            region = np.isin(cc, comp)
            others = np.unique(inst[region & ~m])
            lab = iid // 1000
            # another instance of the same class, or a crowd region (labelId without instance) of the same class
            is_merged = any((o >= 1000 and o != iid and o // 1000 == lab) or o == lab for o in others)
            rows.append({"stem": it["stem"], "instance_id": int(iid), "label": int(lab), "area": int(m.sum()), "merged": bool(is_merged)})
            if is_merged:
                merged.add(int(iid))
        merged_ids[it["stem"]] = merged
        if k % 200 == 0:
            print(k, len(items), flush=True)
    R = pd.DataFrame(rows)
    tm = [i in merged_ids.get(s, set()) for s, i in targets]
    names = {24: "person", 25: "rider", 26: "car", 27: "truck", 28: "bus", 31: "train", 32: "motorcycle", 33: "bicycle"}
    res = {"images": len(items), "thing_instances": int(len(R)), "merged_share_all": round(float(R.merged.mean()), 4),
           "merged_share_by_class": {names.get(k, str(k)): round(float(v), 3) for k, v in R.groupby("label").merged.mean().items()},
           "targets": len(targets), "targets_merged_share": round(float(np.mean(tm)), 4), "targets_merged": int(np.sum(tm))}
    print(json.dumps(res, indent=1))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
