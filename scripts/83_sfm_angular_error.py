#!/usr/bin/env python3
"""Reprojection error in DEGREES, so a fisheye rig and an equirect model share one scale.

COLMAP's mean reprojection error is in pixels of each model's own images, and a
pixel is not the same angle in the two: a 7680-wide equirect pixel is 0.047 deg
everywhere, a fisheye pixel is ~0.055 deg on the axis and several times that
near the rim. So each observation is unprojected to a unit ray
(Camera.cam_ray_from_img) and compared with the ray towards its 3D point.

For fisheye cameras the error is also split by angle off the optical axis. Wrong
intrinsics show up as error that grows towards the rim; wrong poses do not care
where in the image a point sits.

usage (venv): 83_sfm_angular_error.py <model_dir> [<model_dir> ...]   -> JSON on stdout
"""
import json
import math
import sys

import numpy as np
import pycolmap

BINS = [(0, 30), (30, 60), (60, 75), (75, 90)]


def call(x):
    return x() if callable(x) else x


def summarise(a):
    a = np.asarray(a)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median_deg": round(float(np.median(a)), 5),
            "mean_deg": round(float(a.mean()), 5), "p90_deg": round(float(np.percentile(a, 90)), 5)}


out = {}
for path in sys.argv[1:]:
    rec = pycolmap.Reconstruction(path)
    T = {iid: np.asarray(call(rec.images[iid].cam_from_world).matrix()) for iid in rec.reg_image_ids()}
    per_cam, per_bin, allv, skipped = {}, {}, [], 0
    for pt in rec.points3D.values():
        X = np.append(pt.xyz, 1.0)
        for el in pt.track.elements:
            img = rec.images[el.image_id]
            cam = rec.cameras[img.camera_id]
            ray = cam.cam_ray_from_img(img.points2D[el.point2D_idx].xy)
            if ray is None or el.image_id not in T:
                skipped += 1
                continue
            ray = np.asarray(ray).ravel()
            v = T[el.image_id] @ X
            v /= np.linalg.norm(v)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(ray @ v)))))
            allv.append(ang)
            per_cam.setdefault(str(img.camera_id), []).append(ang)
            if "FISHEYE" in cam.model.name:
                off_axis = math.degrees(math.acos(max(-1.0, min(1.0, float(ray[2])))))
                for lo, hi in BINS:
                    if lo <= off_axis < hi:
                        per_bin.setdefault(f"{lo}-{hi}", []).append(ang)
    out[path] = {
        "models": sorted({c.model.name for c in rec.cameras.values()}),
        "reg_images": rec.num_reg_images(), "points3D": rec.num_points3D(),
        "mean_reproj_px": round(rec.compute_mean_reprojection_error(), 4),
        "all": summarise(allv),
        "per_camera": {k: summarise(v) for k, v in per_cam.items()},
        "fisheye_by_off_axis_deg": {k: summarise(per_bin.get(k, [])) for k in (f"{lo}-{hi}" for lo, hi in BINS)},
        "skipped_observations": skipped,
    }
print(json.dumps(out, indent=1))
