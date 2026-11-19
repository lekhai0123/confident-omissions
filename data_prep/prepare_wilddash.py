"""E17 data prep: WildDash2 public GT (wd_public_v2p0.zip) -> Cityscapes-style instanceIds PNGs for pilot.py.

WildDash2 category ids 0-33 equal the Cityscapes labelIds. Extra WD2 categories are mapped to the Cityscapes
class whose official definition covers them, or to 0 (unlabelled = ignored):
  pickup -> truck (Cityscapes 'truck' includes pickup trucks), van -> car (Cityscapes 'car' includes vans),
  motorcyclist / bicyclist / otherrider -> rider, road marking / crosswalks / tram track / bike & service lane -> road,
  curb / curb cut / sidewalk bike lane / pedestrian area -> sidewalk, utility pole / sign frame -> pole,
  everything else (billboard, street light, animals, other vehicles, boats, mountain, snow, ...) -> 0.
Non-crowd things of the 8 Cityscapes instance classes get labelId * 1000 + k; crowd segments get their labelId.
Images: the anonymized version where available (licence: personal data must be protected), else the original.

Licence: WildDash2 may not be shared; the output goes only into a PRIVATE Kaggle dataset on the owner's account.

usage: python -I prepare_wilddash.py <wd_public_v2p0.zip> <out dir>
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
from PIL import Image

INSTANCE_LABELIDS = {24, 25, 26, 27, 28, 31, 32, 33}
MAP = {i: i for i in range(34)}
MAP.update({34: 27, 35: 26, 47: 25, 48: 25, 49: 25,
            38: 7, 61: 7, 64: 7, 75: 7, 76: 7, 79: 7,
            52: 8, 65: 8, 70: 8, 80: 8,
            45: 17, 46: 17})


def convert(pan_rgb: np.ndarray, segments: list[dict]) -> np.ndarray:
    a = pan_rgb.astype(np.int64)
    ids = a[..., 0] + 256 * a[..., 1] + 65536 * a[..., 2]
    inst = np.zeros(ids.shape, dtype=np.uint16)
    counters: dict[int, int] = {}
    for s in segments:
        lab = MAP.get(int(s["category_id"]), 0)
        m = ids == s["id"]
        if lab in INSTANCE_LABELIDS and not s.get("iscrowd", 0):
            k = counters.get(lab, 0)
            counters[lab] = k + 1
            inst[m] = lab * 1000 + k
        else:
            inst[m] = lab
    return inst


def main():
    zpath, out = sys.argv[1:3]
    z = zipfile.ZipFile(zpath)
    names = set(z.namelist())
    pan = json.loads(z.read("panoptic.json"))
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    for f in ("license.txt", "authors.txt", "readme.txt", "random_split/training.txt", "random_split/validation.txt"):
        with open(os.path.join(out, os.path.basename(f)), "wb") as fh:
            fh.write(z.read(f))
    n_anon = n_obj = 0
    for i, a in enumerate(pan["annotations"]):
        stem = a["file_name"][:-4]
        lab_path = os.path.join(out, "labels", stem + "_instanceIds.png")
        img_path = os.path.join(out, "images", stem + ".jpg")
        if not os.path.exists(lab_path):
            inst = convert(np.asarray(Image.open(io.BytesIO(z.read("panoptic/" + a["file_name"]))).convert("RGB")),
                           a["segments_info"])
            Image.fromarray(inst).save(lab_path)
            n_obj += int(sum(1 for v in np.unique(inst) if v >= 1000))
        src = f"anonymized/{stem}.jpg" if f"anonymized/{stem}.jpg" in names else f"images/{stem}.jpg"
        n_anon += src.startswith("anonymized")
        if not os.path.exists(img_path):
            with open(img_path, "wb") as fh:
                fh.write(z.read(src))
        if i % 500 == 0:
            print(f"{i}/{len(pan['annotations'])}", flush=True)
    print(f"done: {len(pan['annotations'])} frames, {n_anon} anonymized images, {n_obj} thing instances (newly converted)")


if __name__ == "__main__":
    main()
