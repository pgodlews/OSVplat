#!/usr/bin/env python3
"""The fisheye rig SfM as one step: what the queue's sfm stage runs for a raw .OSV.

One pass of 82_fisheye_sfm.py, seeded entirely from the camera's own
calibration: intrinsics refit to COLMAP's four-coefficient model (DJI stores k5
as well), and the rig rotation from the stored module extrinsic. Features come
from inside an 88 deg circle computed from each lens's own polynomial, so the
same code serves the Osmo 360's 3840 px and the Avata 360's 3000 px lenses.

One pass is enough. On the Osmo 360 garden clip (docs/how-it-works.md, "Fisheye rig") the stored
calibration held FIXED reconstructed as well as a two-pass run that refined the
intrinsics from scratch: 109,137 vs 108,625 points, 1.044 vs 1.019 px, on the
same masks. Intrinsics are still refined here, seeded from the stored values,
which costs nothing and absorbs a unit or a temperature that drifts.

Then the usual model pick and outlier drop (32_pick_model.py), the refined
calibration (84), reprojection error in degrees (83), and a LichtFeld dataset
with flattened names and valid-circle masks (85). Person masks, when the job
has them, go into the SfM masks here; training masks are combined later by
89_fisheye_train_view.py under the train stage's own key.

usage (venv): 88_fisheye_sfm.py --images SELECT/images --calib FRAMES/calibration.json \
              --out SFM_DIR [--person-masks MASK/fisheye_masks] [--overlap 10] [--max-deg 88]
Writes SFM_DIR/summary.json and prints it as the final line.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from osmo_fisheye import lenses, theta_d  # noqa: E402

PY = sys.executable
MIN_REGISTERED = 0.9


def run(argv, what):
    argv = [str(x) for x in argv]
    print(f"$ {' '.join(argv)}", flush=True)
    t = time.time()
    if subprocess.run(argv).returncode != 0:
        raise SystemExit(f"{what} failed")
    return time.time() - t


def run_capture(argv, what):
    argv = [str(x) for x in argv]
    print(f"$ {' '.join(argv)}", flush=True)
    p = subprocess.run(argv, capture_output=True, text=True)
    sys.stdout.write(p.stderr[-6000:])
    if p.returncode != 0:
        sys.stdout.write(p.stdout[-6000:])
        raise SystemExit(f"{what} failed")
    return p.stdout


def usable_radius(lens, max_deg):
    """Pixel radius the lens model can be trusted to: inside its polynomial's fold, and within max_deg.

    COLMAP's OPENCV_FISHEYE cannot unproject a keypoint past the fold (or past
    90 deg: it works in tan(theta)), and a missing mask means "extract everywhere".
    """
    th = np.linspace(0.0, math.radians(110), 11001)
    td = theta_d(th, lens)
    falling = np.diff(td) <= 0
    fold = int(np.argmax(falling)) if falling.any() else len(th) - 1
    r_fold = lens["fx"] * td[fold]
    lim = math.radians(max_deg)
    r_deg = lens["fx"] * theta_d(lim, lens) if lim < th[fold] else r_fold
    return float(min(0.98 * r_fold, r_deg))


def build_masks(images, out, L, radii, person=None):
    """COLMAP-layout masks lens{i}/frame_NNNN.jpg.png: the valid circle, AND the person mask when given."""
    for i, lens in enumerate(L):
        d = out / f"lens{i}"
        d.mkdir(parents=True, exist_ok=True)
        w = int(round(lens["width"])); h = int(round(lens.get("height", lens["width"])))
        circle = np.zeros((h, w), np.uint8)
        cv2.circle(circle, (int(round(lens["cx"] * 16)), int(round(lens["cy"] * 16))),
                   int(round(radii[i] * 16)), 255, -1, cv2.LINE_8, 4)
        names = sorted(p.name for p in (images / f"lens{i}").glob("*.jpg"))
        if person is None:
            base = d / ".circle.png"
            cv2.imwrite(str(base), circle)
            for n in names:
                dst = d / f"{n}.png"
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.link(base, dst)
            continue

        def one(n, i=i, d=d, circle=circle):
            pm = cv2.imread(str(person / f"lens{i}" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
            if pm is None or pm.shape != circle.shape:
                return n
            cv2.imwrite(str(d / f"{n}.png"), np.where(pm >= 128, circle, 0).astype(np.uint8))
            return None
        with ThreadPoolExecutor(8) as ex:
            bad = [n for n in ex.map(one, names) if n]
        if bad:
            raise SystemExit(f"lens{i}: {len(bad)} person masks missing or the wrong size, e.g. {bad[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--person-masks", default="")
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--min-registered", type=float, default=0.85)
    ap.add_argument("--max-deg", type=float, default=88.0)
    a = ap.parse_args()
    images, out = Path(a.images), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    person = Path(a.person_masks) if a.person_masks else None
    n_frames = len(list((images / "lens0").glob("*.jpg")))
    t_all = time.time()

    L = lenses(a.calib)
    radii = [usable_radius(l, a.max_deg) for l in L]
    print(f"FISHEYE PASS 1/1 frames={n_frames} radius_px={[round(r) for r in radii]}", flush=True)
    build_masks(images, out / "masks", L, radii, person)
    t_sfm = run([PY, HERE / "82_fisheye_sfm.py", "--calib", a.calib, "--images", images,
                 "--masks", out / "masks", "--out", out / "rig", "--refine-intrinsics",
                 "--overlap", a.overlap], "rig SfM")

    picked = json.loads(run_capture([PY, HERE / "32_pick_model.py", out / "rig", out / "picked"],
                                    "model pick").strip().splitlines()[-1])
    if "error" in picked:
        raise SystemExit(picked["error"])
    model = out / "picked" / "sparse" / "0"

    import pycolmap
    rec = pycolmap.Reconstruction(str(model))
    reg_frames = rec.num_reg_frames()
    if reg_frames < a.min_registered * n_frames:
        raise SystemExit(f"the rig registered only {reg_frames} of {n_frames} frames; "
                         f"the fisheye rig did not reconstruct this clip")

    run([PY, HERE / "84_refined_calib.py", model, a.calib, out / "calibration_refined.json"], "refined calibration")
    angular = next(iter(json.loads(run_capture([PY, HERE / "83_sfm_angular_error.py", model],
                                               "angular error")).values()))
    # Training sees the same circle the features came from; people are added by
    # the train stage when the job masks them.
    build_masks(images, out / "valid_masks", L, radii, None)
    run([PY, HERE / "85_fisheye_dataset.py", "fisheye", model, images, out / "valid_masks",
         out / "dataset", "--holdout-every", "0"], "dataset build")

    # 32_pick_model walks images, and a rig registers two per frame: count
    # frames, and measure the path along one lens so it does not zig-zag.
    centres = sorted((rec.images[i].name, np.asarray(rec.images[i].projection_center()))
                     for i in rec.reg_image_ids() if rec.images[i].name.startswith("lens0/"))
    path = float(sum(np.linalg.norm(b[1] - c[1]) for c, b in zip(centres, centres[1:])))
    refined = lenses(out / "calibration_refined.json")
    summary = dict(picked)
    summary.update({
        "pipeline": "fisheye_rig",
        "num_reg_frames": reg_frames,
        "num_reg_images": rec.num_reg_images(),
        "n_panos": n_frames,
        "registration_pct": 100.0 * reg_frames / n_frames if n_frames else 0.0,
        "path_length": path,
        "radius_px": [round(r, 1) for r in radii],
        "angular_median_deg": angular["all"].get("median_deg"),
        "angular_p90_deg": angular["all"].get("p90_deg"),
        "calib_fx": [round(l["fx"], 2) for l in L],
        "refined_fx": [round(l["fx"], 2) for l in refined],
        "needs_gut": True,
        "dataset": str(out / "dataset"),
        "images_dir": str(out / "dataset" / "images"),
        "seconds_sfm": round(t_sfm, 1),
        "seconds": round(time.time() - t_all, 1),
    })
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
