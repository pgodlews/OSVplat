#!/usr/bin/env python3
"""Build the training view of a fisheye rig dataset, then become the trainer.

LichtFeld reads masks from <-d>/masks/. The sfm stage's dataset carries the
valid-circle masks only, because that stage is shared between masked and
unmasked jobs. When the job masks people, the training mask is that circle AND
the person mask, and it has to be written under the TRAIN stage's key -- the
same reason the stitched path builds a private dataset view there.

Combining ~200 full-size masks needs numpy and OpenCV, which the queue service
itself does not have, so this runs as the stage's process and then os.execv's
LichtFeld: same PID, same process group, same log, so cancellation, the stall
watchdog and progress parsing see the trainer exactly as before.

usage (venv): 89_fisheye_train_view.py --dataset SFM/dataset --view TRAIN/dataset \
              [--person-masks MASK/fisheye_masks] -- <LichtFeld-Studio argv...>
"""
import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: 89_fisheye_train_view.py [options] -- <trainer argv>")
    k = sys.argv.index("--")
    own, trainer = sys.argv[1:k], sys.argv[k + 1:]
    if not trainer:
        raise SystemExit("no trainer command after --")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--person-masks", default="")
    a = ap.parse_args(own)
    ds, view = Path(a.dataset), Path(a.view)

    (view / "sparse").mkdir(parents=True, exist_ok=True)
    link = view / "sparse" / "0"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((ds / "sparse" / "0").resolve(), target_is_directory=True)

    masks = view / "masks"
    if masks.is_symlink() or masks.is_file():
        masks.unlink()
    elif masks.is_dir():
        shutil.rmtree(masks)
    stems = sorted(p.stem for p in (ds / "masks").glob("*.png"))
    if not stems:
        raise SystemExit(f"{ds}/masks is empty; the fisheye dataset needs its valid-circle masks")

    if not a.person_masks:
        masks.symlink_to((ds / "masks").resolve(), target_is_directory=True)
        print(f"training masks: valid circle only ({len(stems)})", flush=True)
    else:
        masks.mkdir()
        person = Path(a.person_masks)

        def one(stem):
            lens, frame = stem.split("_", 1)          # lens0_frame_0001
            circle = cv2.imread(str(ds / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            pm = cv2.imread(str(person / lens / f"{frame}.jpg.png"), cv2.IMREAD_GRAYSCALE)
            if circle is None or pm is None or circle.shape != pm.shape:
                return stem
            ok = cv2.imwrite(str(masks / f"{stem}.png"), np.where(pm >= 128, circle, 0).astype(np.uint8))
            return None if ok else stem
        with ThreadPoolExecutor(8) as ex:
            bad = [s for s in ex.map(one, stems) if s]
        if bad:
            raise SystemExit(f"could not combine {len(bad)} training masks, e.g. {bad[:3]}")
        print(f"training masks: valid circle AND person ({len(stems)})", flush=True)

    sys.stdout.flush()
    os.execv(trainer[0], trainer)


if __name__ == "__main__":
    main()
