#!/usr/bin/env python3
"""Person masks for a fisheye rig, made where the detector works: on a stitch.

This is the queue's mask stage for a raw .OSV. 70_person_masks.py segments
equirect panoramas -- its detectors and its 13-view scaffolding are built for
that projection -- so rather than teach it fisheye:

  1. stitch every rig frame with the camera's own calibration (81 stitch)
  2. run 70_person_masks.py on those panoramas, unchanged: same backends, same
     overlays, contact sheet and summary.json, so the review gate and
     mask_finalize work exactly as they do for stitched input
  3. carry the masks back into each fisheye through the SAME geometry
     (81 masks, no valid circle). The round trip is exact by construction, so
     masks land on the right fisheye pixels even where the stored lens model is
     not quite right -- and the circle is left to the SfM, which is the stage
     that knows which radius each pass can trust.

usage (venv_gs):
  87_fisheye_masks.py --images SELECT/images --calib FRAMES/calibration.json --out MASK_DIR \
                      --masker-python PY [-- <70_person_masks.py options>]
Writes MASK_DIR/{erp,masks,masks_colmap,overlay,review_sheet.jpg,summary.json,fisheye_masks/lens{0,1}}.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(argv, what):
    argv = [str(x) for x in argv]
    print(f"$ {' '.join(argv)}", flush=True)
    t = time.time()
    if subprocess.run(argv).returncode != 0:
        raise SystemExit(f"{what} failed")
    print(f"{what}: {time.time() - t:.1f}s", flush=True)


def main():
    k = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    own, masker_args = sys.argv[1:k], sys.argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--masker-python", required=True)
    a = ap.parse_args(own)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    run([sys.executable, HERE / "81_fisheye_stitch.py", "stitch", "--calib", a.calib,
         "--images", a.images, "--out", out / "erp", "--fscale", "1.0"], "stitch for masking")
    run([a.masker_python, HERE / "70_person_masks.py", "--panos", out / "erp",
         "--out", out / "masks", "--colmap-out", out / "masks_colmap",
         "--overlay", out / "overlay", "--sheet", out / "review_sheet.jpg",
         "--summary", out / "summary.json", *masker_args], "person masks")
    run([sys.executable, HERE / "81_fisheye_stitch.py", "masks", "--calib", a.calib,
         "--images", a.images, "--erp-masks", out / "masks", "--out", out / "fisheye_masks",
         "--fscale", "1.0", "--valid-radius", "0"], "masks back to fisheye")


if __name__ == "__main__":
    main()
