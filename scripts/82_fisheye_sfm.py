#!/usr/bin/env python3
"""Rig SfM straight on Osmo 360 fisheye frames -- no stitch, known intrinsics.

Each instant is one COLMAP frame of a two-camera rig. lens0 (rear, calibration
slot 1) is the reference sensor; lens1 (front, slot 2) hangs off it at the
rotation the camd calibration gives (osmo_fisheye.rig_rotation). Both cameras are OPENCV_FISHEYE initialised from DJI's per-unit values:
COLMAP's model is the same Kannala-Brandt polynomial the calibration stores, up to k4; DJI's k5 is folded into a refit of k1..k4.

The variable this script exposes is whether bundle adjustment may move the
intrinsics:
  (default)            focal and k1-k4 held at DJI's values (x --fscale)
  --refine-intrinsics  focal and k1-k4 refined; principal point still fixed
The rig's relative pose is refined in both, because the calibration gives the
lenses' rotation but not their ~cm separation.

Masks are required (fmasks_colmap/ from 81_fisheye_stitch.py). Outside the
valid circle a keypoint is black corner or past the fold of DJI's polynomial,
where COLMAP's undistortion stops converging -- and COLMAP treats a missing
mask as "extract everywhere" without saying so.

Mapping goes through colmap_incremental.incremental_mapping, which lifts COLMAP's
1000-image limit on the direct bundle-adjustment solver. Past it (500 rig
frames) COLMAP's own mapper makes every global pass iterative: a 7-minute clip
spent about two hours per pass. With the limit lifted the same 1404 frames
mapped in 4 h 57 min and converged to a slightly lower cost
(colmap_incremental.py has the measurements).

usage (venv): 82_fisheye_sfm.py --calib calibration.json --images DIR/images
              --masks DIR/fmasks_colmap --out DIR/sfm_fish_fixed
              [--refine-intrinsics] [--fscale 1.0] [--overlap 10]
              [--direct-solver-max-images N]
"""
import argparse
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap

from colmap_incremental import DIRECT_SOLVER_MAX_IMAGES, incremental_mapping
from osmo_fisheye import colmap_params, lenses, rig_rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# COLMAP threads: the queue sets SPLAT_THREADS to the container's CPU allowance
# (queue/app/resources.py); -1, all visible cores, when run by hand.
THREADS = int(os.environ.get("SPLAT_THREADS") or -1)


def call(x):
    """pycolmap 4 turned several pose attributes into methods; accept either."""
    return x() if callable(x) else x


def rot_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refine-intrinsics", action="store_true")
    ap.add_argument("--fscale", type=float, default=1.0)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--direct-solver-max-images", type=int, default=DIRECT_SOLVER_MAX_IMAGES,
                    help="global BA uses the direct sparse solver up to this many images "
                         "(COLMAP's own default is 1000)")
    a = ap.parse_args()

    L = lenses(a.calib)
    images, masks, out = (Path(p).expanduser() for p in (a.images, a.masks, a.out))
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "database.db"
    if db_path.exists():
        db_path.unlink()
    for i in (0, 1):
        n_img = len(list((images / f"lens{i}").glob("*.jpg")))
        n_mask = len(list((masks / f"lens{i}").glob("*.jpg.png")))
        if n_img == 0 or n_mask < n_img:
            raise SystemExit(f"lens{i}: {n_mask} masks for {n_img} images under {masks}; "
                             "refusing to extract features outside the valid circle")

    logging.info(f"START refine_intrinsics={a.refine_intrinsics} fscale={a.fscale} "
                 f"frames={n_img} overlap={a.overlap} cuda={pycolmap.has_cuda} "
                 f"direct_solver_max_images={a.direct_solver_max_images}")
    t_all = t = time.time()
    extraction = pycolmap.FeatureExtractionOptions(use_gpu=True, num_threads=THREADS)
    for i, l in enumerate(L):
        # k1..k4 refitted to the lens's full polynomial: COLMAP has no k5.
        params = colmap_params(l, a.fscale)
        pycolmap.extract_features(
            db_path, images,
            image_names=sorted(f"lens{i}/{p.name}" for p in (images / f"lens{i}").glob("*.jpg")),
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=pycolmap.ImageReaderOptions(
                camera_model="OPENCV_FISHEYE",
                camera_params=",".join(f"{p:.9g}" for p in params),
                mask_path=str(masks)),
            extraction_options=extraction)
    logging.info(f"STAGE extract {time.time() - t:.0f}s")

    R10 = rig_rotation(L)  # lens1_from_lens0: stored module extrinsic when present
    rig = pycolmap.RigConfig(cameras=[
        pycolmap.RigConfigCamera(ref_sensor=True, image_prefix="lens0/"),
        pycolmap.RigConfigCamera(ref_sensor=False, image_prefix="lens1/",
                                 cam_from_rig=pycolmap.Rigid3d(pycolmap.Rotation3d(R10),
                                                               np.zeros((3, 1)))),
    ])
    with pycolmap.Database.open(db_path) as db:
        pycolmap.apply_rig_config([rig], db)
        logging.info(f"rig: {db.num_rigs()} rigs, {db.num_frames()} frames, {db.num_images()} images, "
                     f"{db.num_cameras()} cameras; lens1_from_lens0 {rot_deg(R10):.3f} deg")

    t = time.time()
    pycolmap.match_sequential(
        db_path,
        matching_options=pycolmap.FeatureMatchingOptions(use_gpu=True, num_threads=THREADS,
                                                         skip_image_pairs_in_same_frame=True),
        pairing_options=pycolmap.SequentialPairingOptions(overlap=a.overlap, expand_rig_images=True))
    logging.info(f"STAGE match {time.time() - t:.0f}s")

    t = time.time()
    opts = pycolmap.IncrementalPipelineOptions(num_threads=THREADS, random_seed=0)
    opts.ba_refine_focal_length = a.refine_intrinsics
    opts.ba_refine_extra_params = a.refine_intrinsics
    opts.ba_refine_principal_point = False
    opts.ba_refine_sensor_from_rig = True
    opts.mapper.abs_pose_refine_focal_length = a.refine_intrinsics
    opts.mapper.abs_pose_refine_extra_params = a.refine_intrinsics
    sparse = out / "sparse"
    if sparse.is_symlink():
        sparse.unlink()
    elif sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(exist_ok=True)
    recs = incremental_mapping(db_path, images, sparse, opts,
                               direct_solver_max_images=a.direct_solver_max_images)
    if not recs:
        raise SystemExit("rig SfM produced no reconstruction models")
    logging.info(f"STAGE map {time.time() - t:.0f}s")
    logging.info(f"TOTAL {time.time() - t_all:.0f}s")

    report = {"refine_intrinsics": a.refine_intrinsics, "fscale": a.fscale,
              "dji": [{"fx": l["fx"], "fy": l["fy"], "k": [l["k1"], l["k2"], l["k3"], l["k4"]]} for l in L],
              "models": []}
    for idx, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
        poses = {}
        for iid in rec.reg_image_ids():
            img = rec.images[iid]
            poses.setdefault(img.frame_id, {})[img.name.split("/")[0]] = \
                (img.name, np.asarray(call(img.cam_from_world).matrix()))
        both = [f for f in poses.values() if "lens0" in f and "lens1" in f]
        rig_rot, rig_t, step = None, None, None
        if both:
            (_, T0), (_, T1) = both[0]["lens0"], both[0]["lens1"]
            Rr = T1[:, :3] @ T0[:, :3].T
            rig_rot = rot_deg(Rr @ R10.T)
            rig_t = float(np.linalg.norm(T1[:, 3] - Rr @ T0[:, 3]))
            c0 = [(-(T[:, :3].T @ T[:, 3])) for _, T in sorted(f["lens0"] for f in poses.values() if "lens0" in f)]
            if len(c0) > 1:
                step = float(np.median(np.linalg.norm(np.diff(np.array(c0), axis=0), axis=1)))
        cams = {}
        for cid, cam in rec.cameras.items():
            p = [float(x) for x in cam.params]
            cams[str(cid)] = {"model": cam.model.name, "fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3],
                              "k": p[4:8]}
        m = {"name": str(idx), "reg_frames": rec.num_reg_frames(), "reg_images": rec.num_reg_images(),
             "points3D": rec.num_points3D(), "mean_reproj_px": rec.compute_mean_reprojection_error(),
             "mean_track_length": rec.compute_mean_track_length(),
             "cameras": cams,
             "rig_rotation_change_deg": rig_rot,
             "rig_baseline_over_median_step": (rig_t / step) if (rig_t is not None and step) else None}
        report["models"].append(m)
        logging.info(f"model {idx}: frames {m['reg_frames']}, images {m['reg_images']}, "
                     f"points {m['points3D']}, reproj {m['mean_reproj_px']:.3f} px")
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
