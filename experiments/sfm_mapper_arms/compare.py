#!/usr/bin/env python3
"""All-common-camera Sim(3) ATE: compare.py BASELINE ARM=MODEL [ARM=MODEL ...]."""
import argparse
import re

import numpy as np
import pycolmap
from common import model_path, write_json


def natural_key(name):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def centres(rec):
    return {rec.images[i].name: np.asarray(rec.images[i].projection_center())
            for i in rec.reg_image_ids()}


def align_sim3(source, target):
    """Least-squares Umeyama alignment, positive scale, proper rotation, no RANSAC."""
    source, target = np.asarray(source), np.asarray(target)
    if len(source) < 3 or source.shape != target.shape:
        raise ValueError("Need at least three corresponding camera centres")
    x, y = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError("Camera centres are coincident/collinear; Sim(3) is underdetermined")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = (singular * sign).sum() / np.mean(np.sum(x * x, axis=1))
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    return scale * source @ rotation.T + translation, {
        "scale": float(scale), "rotation": rotation.tolist(), "translation": translation.tolist()}


def path_length(rec, camera_id=None):
    """One lens, natural filename order, full baseline (never lens-to-lens zigzags)."""
    images = [rec.images[i] for i in rec.reg_image_ids()]
    if camera_id is None:
        camera_id = min({im.camera_id for im in images},
                        key=lambda c: (-sum(im.camera_id == c for im in images), c))
    trajectory = sorted((im for im in images if im.camera_id == camera_id),
                        key=lambda im: natural_key(im.name))
    if len(trajectory) < 2:
        raise ValueError("Path camera must have at least two registered images")
    xyz = np.array([im.projection_center() for im in trajectory])
    length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    if length <= 0:
        raise ValueError("Baseline path has zero length")
    return length, camera_id, len(trajectory)


def compare(base, arm, camera_id=None):
    b, a = centres(base), centres(arm)
    common = sorted(b.keys() & a.keys(), key=natural_key)
    aligned, transform = align_sim3([a[n] for n in common], [b[n] for n in common])
    errors = np.linalg.norm(aligned - np.array([b[n] for n in common]), axis=1)
    length, camera_id, path_images = path_length(base, camera_id)
    return {"common_images": len(common), "baseline_images": len(b), "arm_images": len(a),
            "missing_baseline_images": len(b.keys() - a.keys()),
            "extra_arm_images": len(a.keys() - b.keys()),
            "path_length": length, "path_camera_id": camera_id, "path_images": path_images,
            "median": float(np.median(errors)), "max": float(errors.max()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "median_percent_path": float(np.median(errors) / length * 100),
            "max_percent_path": float(errors.max() / length * 100),
            "sim3_arm_to_baseline": transform}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("arms", nargs="+", help="NAME=MODEL or NAME=SPARSE_DIR (one model)")
    ap.add_argument("--path-camera-id", type=int)
    ap.add_argument("--out", help="Optional comparison JSON")
    a = ap.parse_args()
    baseline = pycolmap.Reconstruction(model_path(a.baseline))
    results = {}
    for spec in a.arms:
        name, path = spec.split("=", 1)
        if not name or name in results:
            raise ValueError("Arm names must be nonempty and unique")
        result = compare(baseline, pycolmap.Reconstruction(model_path(path)), a.path_camera_id)
        results[name] = result
        print(f"{name}: {result['common_images']}/{result['baseline_images']} images; "
              f"ATE median={result['median_percent_path']:.8f}% "
              f"max={result['max_percent_path']:.8f}% of baseline path")
    if a.out:
        write_json(a.out, results)


if __name__ == "__main__":
    main()
