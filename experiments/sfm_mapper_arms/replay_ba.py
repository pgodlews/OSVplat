#!/usr/bin/env python3
"""Load a sparse rig model, optionally perturb, run exactly one global Ceres BA."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pycolmap
from common import (mirror, model_path, model_stats, new_output, options_for,
                    threads_default, write_json)
from instrumentation import NativeLog


def perturb(rec, seed, points, translation, rotation):
    rng = np.random.default_rng(seed)
    if points:
        for pid in sorted(rec.points3D):
            rec.points3D[pid].xyz += rng.normal(0, points, 3)
    if not translation and not rotation:
        return
    for fid in sorted(rec.reg_frame_ids()):
        pose = rec.frames[fid].rig_from_world
        rot = pycolmap.Rotation3d(rng.normal(0, np.deg2rad(rotation), 3))
        rec.frames[fid].rig_from_world = pycolmap.Rigid3d(
            rot * pose.rotation, pose.translation + rng.normal(0, translation, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="New directory; existing output is refused")
    ap.add_argument("--threads", type=int, default=threads_default())
    ap.add_argument("--ceres-options", type=Path,
                    help="JSON object merged into ba_options.ceres, including solver_options")
    ap.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--perturb-points", type=float, default=0, help="XYZ Gaussian sigma, model units")
    ap.add_argument("--perturb-translation", type=float, default=0, help="Pose translation sigma, model units")
    ap.add_argument("--perturb-rotation", type=float, default=0, help="Pose rotation-vector sigma, degrees")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if min(a.perturb_points, a.perturb_translation, a.perturb_rotation) < 0:
        ap.error("Perturbation sigmas must be nonnegative")
    opts = mirror.global_bundle_adjustment_options(options_for("baseline", a.threads),
                                                  mirror.DIRECT_SOLVER_MAX_IMAGES)
    if a.ceres_options:
        opts.ceres.mergedict(json.loads(a.ceres_options.read_text()))
    if a.use_gpu is not None:
        opts.ceres.use_gpu = a.use_gpu
    rec = pycolmap.Reconstruction(model_path(a.model))
    if rec.num_reg_frames() < 3 or rec.num_points3D() == 0:
        raise ValueError("Need a nonempty model with at least three registered rig frames")
    out = new_output(a.out)
    report = {"status": "running", "build": pycolmap.COLMAP_build,
              "pycolmap_version": pycolmap.__version__,
              "options": json.loads(json.dumps(opts.todict(), default=str)),
              "perturbation": {"seed": a.seed, "points_sigma": a.perturb_points,
                               "translation_sigma": a.perturb_translation,
                               "rotation_sigma_deg": a.perturb_rotation}}
    try:
        perturb(rec, a.seed, a.perturb_points, a.perturb_translation, a.perturb_rotation)
        rec.update_point_3d_errors()
        report["before"] = model_stats(rec)
        # One native BA, with the mapper's usual two-camera gauge. No filtering,
        # normalization, redundant-point phase, or iterative refinement here.
        config = pycolmap.BundleAdjustmentConfig()
        for iid in rec.reg_image_ids():
            config.add_image(iid)
        config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        with NativeLog(out / "ba.log"):
            start = perf_counter()
            adjuster = pycolmap.create_default_bundle_adjuster(opts, config, rec)
            report["setup_s"] = perf_counter() - start
            start = perf_counter()
            summary = adjuster.solve()
            report["solve_wall_s"] = perf_counter() - start
        report["ba_wall_s"] = report["setup_s"] + report["solve_wall_s"]
        report["summary"] = json.loads(json.dumps(summary.todict(), default=str))
        report["brief_report"] = summary.brief_report()
        report["solution_usable"] = summary.is_solution_usable()
        if not report["solution_usable"]:
            raise RuntimeError("Single global BA failed; inspect ba.log")
        rec.update_point_3d_errors()
        report["after"] = model_stats(rec)
        sparse = out / "sparse"
        sparse.mkdir()
        rec.write(sparse)
        report["status"] = "ok"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(out / "report.json", report)
    print(f"BA {report['ba_wall_s']:.3f}s; {report['brief_report']}")


if __name__ == "__main__":
    main()
