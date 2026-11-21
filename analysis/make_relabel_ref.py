"""Reference inputs for the relabel pass (pilot.py --mode relabel, review of 2026-11-22).

From a finished p4 run it keeps only what the relabel pass needs: summary.json (settings, temperatures), the list of
evaluated (stem, condition) pairs, and the stored detector boxes (RT-DETR normal + gamma rows, the proposals scored by
all monitors) with their row index in the original proposals CSV, so new box features join back exactly.

usage: python make_relabel_ref.py <p4 run dir> <out dir>
"""

from __future__ import annotations

import glob
import os
import shutil
import sys

import pandas as pd

DET = ["rtdetr", "rtdetr_gamma"]


def main():
    src, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(src, "summary.json"), os.path.join(out, "summary.json"))
    imgs = None
    for f in sorted(glob.glob(os.path.join(src, "proposals_*.csv"))):
        tag = os.path.basename(f)[len("proposals_"):-4]
        d = pd.read_csv(f, usecols=["det_src", "stem", "condition", "box_x0", "box_y0", "box_x1", "box_y1"])
        d["row"] = d.index
        d = d[d.det_src.isin(DET)][["row", "stem", "condition", "box_x0", "box_y0", "box_x1", "box_y1"]]
        d.to_csv(os.path.join(out, f"props_{tag}.csv.gz"), index=False, compression="gzip")
        print(tag, len(d))
        im = pd.read_csv(os.path.join(src, f"images_{tag}.csv"), usecols=["stem", "condition"]).drop_duplicates()
        imgs = im if imgs is None else imgs.merge(im, how="inner")
    imgs.to_csv(os.path.join(out, "images.csv"), index=False)
    print("images", len(imgs))


if __name__ == "__main__":
    main()
