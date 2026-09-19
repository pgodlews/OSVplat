#!/usr/bin/env python3
"""Pick the largest COLMAP model from <sfm_dir>/sparse/, drop stray cameras, put the
result at <dataset>/sparse/0, and print a JSON summary including the run-3 geometry
diagnostic.

COLMAP can return several models; sparse/0 is not necessarily the real one (run 3's
sparse/0 was a 2-frame fragment while sparse/1 held all 265 frames, and LichtFeld's
-d reads sparse/0 without complaining). This makes that choice explicit and checked.

It can also register a frame in completely the wrong place. osmo360 registered
110/110 at 1.16 px, but one of them (`pano_0008`, 3 observations) landed 4.8e6 units
from a trajectory 10 units long. One camera, and every statistic computed from the
*mean* camera centre moves with it: the trainer read a scene scale of 44042 for a
10-unit scene, which multiplies the position learning rate and every densification
threshold. The diagnostic below is therefore computed around the *median* centre,
which a single outlier cannot move, and the outlier is removed before training.

usage: 32_pick_model.py <sfm_dir> <dataset_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

# A camera this many times the median camera radius from the median centre is a
# failed registration, not a wide shot. Real trajectories are far tighter: the
# widest run here (river, 471 panoramas) has a max/median radius of about 3.
OUTLIER_FACTOR = 20.0
# If "outliers" are this common they are not outliers -- the reconstruction is
# split or the scene really is that spread out. Drop nothing and say so, rather
# than silently deleting a third of the run.
MAX_OUTLIER_FRACTION = 0.10

sfm_dir = Path(sys.argv[1]).expanduser()
dataset = Path(sys.argv[2]).expanduser()

sparse = sfm_dir / "sparse"
models = []
for p in sorted(sparse.iterdir()) if sparse.is_dir() else []:
    if not p.is_dir():
        continue
    if not ((p / "images.bin").exists() or (p / "images.txt").exists()):
        continue
    try:
        rec = pycolmap.Reconstruction(str(p))
    except Exception:
        continue
    models.append((len(rec.images), len(rec.points3D), p.name, rec))

if not models:
    print(json.dumps({"error": "no valid reconstruction models"}))
    sys.exit(2)

models.sort(key=lambda m: (m[0], m[1]), reverse=True)
n_img, n_pts, name, rec = models[0]

# Mean reprojection error over all 3D points.
errs = [pt.error for pt in rec.points3D.values() if pt.error >= 0]
mean_reproj = float(np.mean(errs)) if errs else None

# ------------------------------------------------------------------ outliers
# Sort by image NAME, not image id. COLMAP assigns ids in feature-extraction
# completion order, which is multithreaded and therefore not capture order: the
# masked osmo360 run diverged from name order at index 60, and walking the
# trajectory by id turned a 12.4-unit path into a 17.0-unit scrambled tour. The
# outlier and depth statistics do not care about ordering, but path_length and
# median_step are meaningless without it -- and they are exactly the numbers
# someone compares between two runs.
order = sorted(rec.images.keys(), key=lambda k: rec.images[k].name)
centres = np.array([rec.images[k].projection_center() for k in order])
median_centre = np.median(centres, axis=0)
radii = np.linalg.norm(centres - median_centre, axis=1)
median_radius = float(np.median(radii))

drop_ids: list[int] = []
outlier_note = None
if median_radius > 0:
    flagged = [k for k, r in zip(order, radii) if r > OUTLIER_FACTOR * median_radius]
    if len(flagged) > MAX_OUTLIER_FRACTION * len(order):
        outlier_note = (
            f"{len(flagged)} of {len(order)} cameras sit beyond "
            f"{OUTLIER_FACTOR:g}x the median radius; that is a spread-out or split "
            f"reconstruction, not stray registrations, so none were dropped")
    else:
        drop_ids = flagged
# Dropping is per rig FRAME (deregister_frame below), which removes every lens
# of that instant, so count the partner images as dropped too. Otherwise the
# summary and the statistics kept a lens the written model no longer has.
if drop_ids:
    frames = {rec.images[k].frame_id for k in drop_ids}
    drop_ids = [k for k in order if rec.images[k].frame_id in frames]

dropped = [{"image_id": int(k),
            "name": rec.images[k].name,
            "radius": float(r),
            "num_points3D": int(rec.images[k].num_points3D)}
           for k, r in zip(order, radii) if k in set(drop_ids)]

# Every statistic below describes what the trainer will actually see, so measure
# it on the kept cameras only.
keep = np.array([k not in set(drop_ids) for k in order])
centres = centres[keep]
kept_ids = [k for k, ok in zip(order, keep) if ok]

# ---------------------------------------------------------------- diagnostic
# Parallax available to the trainer. baseline/median depth below ~0.5 means the
# scene is mostly at infinity and will reconstruct as haze.
path = centres
steps = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.array([0.0])
path_len = float(steps.sum())
pts = np.array([p.xyz for p in rec.points3D.values()]) if rec.points3D else np.zeros((0, 3))
centroid = np.median(centres, axis=0) if len(centres) else median_centre
depths = np.linalg.norm(pts - centroid, axis=1) if len(pts) else np.array([0.0])
median_depth = float(np.median(depths))
bbox = (centres.max(axis=0) - centres.min(axis=0)) if len(centres) else np.zeros(3)
baseline = float(np.linalg.norm(bbox))

summary = {
    "models": [{"name": m[2], "images": m[0], "points": m[1]} for m in models],
    "chosen": name,
    "num_reg_frames": len(kept_ids),
    "num_reg_frames_raw": n_img,
    "num_points3D": n_pts,
    "mean_reproj": mean_reproj,
    "dropped_cameras": dropped,
    "path_length": path_len,
    "median_step": float(np.median(steps)),
    "bbox_extent": [round(float(x), 3) for x in bbox],
    "baseline": baseline,
    "median_depth": median_depth,
    "baseline_over_depth": (baseline / median_depth) if median_depth > 0 else None,
    "depth_p50": float(np.percentile(depths, 50)) if len(depths) else None,
    "depth_p90": float(np.percentile(depths, 90)) if len(depths) else None,
    "depth_p99": float(np.percentile(depths, 99)) if len(depths) else None,
}
if outlier_note:
    summary["outlier_note"] = outlier_note

# Camera model decides whether LichtFeld needs the 3DGUT path. Read it before the
# model is mutated below.
cam_models = sorted({cam.model.name for cam in rec.cameras.values()})
summary["camera_models"] = cam_models
summary["needs_gut"] = any("EQUIRECTANGULAR" in m for m in cam_models)

# ------------------------------------------------- publish as <dataset>/sparse/0
# With nothing to drop this stays a symlink to the chosen model, which is what it
# has always been. With outliers, a filtered copy is written instead -- the source
# reconstruction is never modified in place, so re-running this is idempotent and
# the raw SfM output stays available for diagnosis.
(dataset / "sparse").mkdir(parents=True, exist_ok=True)
link = dataset / "sparse" / "0"
if link.is_symlink() or link.is_file():
    link.unlink()
elif link.is_dir():
    import shutil
    shutil.rmtree(link)

if drop_ids:
    for k in drop_ids:
        # deregister_frame, not the image: a rig frame without a pose fails
        # pycolmap's own pose check the moment anything reads it back.
        rec.deregister_frame(rec.images[k].frame_id)
    link.mkdir(parents=True)
    rec.write(str(link))
else:
    link.symlink_to((sparse / name).resolve(), target_is_directory=True)

print(json.dumps(summary))
