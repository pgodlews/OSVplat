#!/usr/bin/env python3
"""CPU synthetic integration checks; all data/output live in a fresh temp directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import pycolmap
from common import ARMS, model_path, require_version, write_json
from compare import align_sim3, compare

HERE = Path(__file__).resolve().parent


def write_png(path, width, height):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                     chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress((b"\0" + b"\x80" * width * 3) * height)) +
                     chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="New temp-style output directory; defaults to /tmp/sfm-arms-smoke-*")
    a = ap.parse_args()
    root = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="sfm-arms-smoke-", dir="/tmp"))
    if a.out:
        root.mkdir(parents=True, exist_ok=False)
    os.environ.update(SPLAT_ROOT=str(root / "splat"), QUEUE_ROOT=str(root / "queue"),
                      QUEUE_GPUS="", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    require_version()
    pycolmap.set_random_seed(0)
    opts = pycolmap.SyntheticDatasetOptions(
        num_rigs=1, num_cameras_per_rig=2, num_frames_per_rig=14, num_points3D=400,
        camera_model_id=pycolmap.CameraModelId.OPENCV_FISHEYE,
        camera_params=[512., 512., 512., 384., 0., 0., 0., 0.],
        camera_has_prior_focal_length=True)
    db = root / "database.db"
    with pycolmap.Database.open(db) as database:
        truth = pycolmap.synthesize_dataset(opts, database)
    # Add reproducible subpixel noise to synthetic keypoints only, so the BA
    # paths exercise real refinement instead of starting at a zero-cost model.
    import sqlite3
    rng = np.random.default_rng(42)
    with sqlite3.connect(db) as conn:
        for iid, rows, cols, data in conn.execute("SELECT image_id, rows, cols, data FROM keypoints").fetchall():
            keypoints = np.frombuffer(data, dtype=np.float32).copy().reshape(rows, cols)
            keypoints[:, :2] += rng.normal(0, 0.25, (rows, 2))
            conn.execute("UPDATE keypoints SET data=? WHERE image_id=?", (keypoints.tobytes(), iid))
    before_hash = hashlib.sha256(db.read_bytes()).hexdigest()
    images = root / "images"
    images.mkdir()
    for im in truth.images.values():
        write_png(images / im.name, opts.camera_width, opts.camera_height)
    runs = root / "runs"
    runs.mkdir()

    def execute(*args):
        subprocess.run([sys.executable, *map(str, args)], check=True)

    # Separate fresh processes: native and instrumented must see identical RNG state.
    native = runs / "native"
    execute("-c", "import sys; sys.path.insert(0, sys.argv[1]); from ab_map import run; "
            "run(sys.argv[2],sys.argv[3],sys.argv[4],'baseline',1,instrumented=False)",
            HERE, db, images, native)
    reports = {}
    for arm in ARMS:
        execute(HERE / "ab_map.py", "--db", db, "--images", images,
                "--out", runs / arm, "--arm", arm, "--threads", "1")
        report = json.loads((runs / arm / "report.json").read_text())
        reports[arm] = report
        assert report["status"] == "ok"
        assert report["models"][0]["reg_frames"] == 14
        assert report["models"][0]["reg_images"] == 28
        assert report["timings"]["local_call_count"] >= 12
        assert report["timings"]["global_call_count"] > 0
        assert all(p["solves"] for p in report["passes"]["global"])
        assert all(p["solves"] for p in report["passes"]["local"])
        assert sum(b["calls"] for b in report["timings"]["local_residual_histogram"]) == report["timings"]["local_call_count"]
        threshold = 5000 if arm == "localmt" else 50000
        assert all(p["threading_threshold"] == threshold for p in report["passes"]["local"])
        assert report["global_ba_options"]["ceres"]["min_num_residuals_for_cpu_multi_threading"] == 50000
        if "redundant" in arm:
            assert any(p["ignored_redundant_points"] > 0 for p in report["passes"]["global"])
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before_hash
    baseline = pycolmap.Reconstruction(model_path(runs / "baseline" / "sparse"))
    reference = pycolmap.Reconstruction(model_path(native / "sparse"))
    parity = compare(reference, baseline)
    assert parity["max_percent_path"] < 1e-6, parity
    assert baseline.num_points3D() == reference.num_points3D()
    assert abs(baseline.compute_mean_reprojection_error() - reference.compute_mean_reprojection_error()) < 1e-7
    for pid in reference.points3D:
        np.testing.assert_array_equal(baseline.points3D[pid].xyz, reference.points3D[pid].xyz)
        assert sorted((e.image_id, e.point2D_idx) for e in baseline.points3D[pid].track.elements) == sorted(
            (e.image_id, e.point2D_idx) for e in reference.points3D[pid].track.elements)
    for fid in reference.reg_frame_ids():
        np.testing.assert_array_equal(baseline.frames[fid].rig_from_world.matrix(),
                                      reference.frames[fid].rig_from_world.matrix())
    for cid in reference.cameras:
        np.testing.assert_array_equal(baseline.cameras[cid].params, reference.cameras[cid].params)
    # Exercise the multithreaded local BA path, too (parity above stays single-threaded).
    execute(HERE / "ab_map.py", "--db", db, "--images", images,
            "--out", runs / "localmt_two_threads", "--arm", "localmt", "--threads", "2")
    mt_report = json.loads((runs / "localmt_two_threads" / "report.json").read_text())
    assert mt_report["models"][0]["reg_frames"] == 14
    assert any(p["residuals_estimate"] >= 5000 for p in mt_report["passes"]["local"])
    # Verify Sim(3) convention independently with a known scale/rotation/translation.
    points = rng.normal(size=(20, 3))
    rotation = pycolmap.Rotation3d([0.3, -0.2, 0.1]).matrix()
    target = 2.7 * points @ rotation.T + [5, -4, 2]
    aligned, transform = align_sim3(points, target)
    np.testing.assert_allclose(aligned, target, atol=1e-12)
    assert abs(transform["scale"] - 2.7) < 1e-12
    execute(HERE / "compare.py", runs / "baseline" / "sparse",
            *[f"{arm}={runs / arm / 'sparse'}" for arm in ARMS[1:]],
            "--out", root / "comparison.json")
    settings = root / "ceres.json"
    write_json(settings, {"use_gpu": False, "auto_select_solver_type": False,
                          "solver_options": {"max_num_iterations": 50,
                                             "linear_solver_type": "SPARSE_SCHUR"}})
    for name, perturb_args in [("replay", []), ("replay_perturbed", ["--perturb-points", "0.001",
                                                    "--perturb-translation", "0.001",
                                                    "--perturb-rotation", "0.01"])]:
        execute(HERE / "replay_ba.py", "--model", runs / "baseline" / "sparse",
                "--out", root / name, "--threads", "1", "--no-use-gpu",
                "--ceres-options", settings, *perturb_args)
        replay = json.loads((root / name / "report.json").read_text())
        assert replay["summary"]["ceres_summary"]["linear_solver_type_used"] == "LinearSolverType.SPARSE_SCHUR"
        assert replay["status"] == "ok" and replay["solution_usable"]
        assert replay["after"]["mean_reproj_px"] <= replay["before"]["mean_reproj_px"] + 1e-6
    # Failure paths must not silently reuse a successful report or accept no model.
    from ab_map import run
    saved_report = (runs / "baseline" / "report.json").read_bytes()
    try:
        run(db, images, runs / "baseline", "baseline", 1)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing output was not refused")
    assert (runs / "baseline" / "report.json").read_bytes() == saved_report
    empty_db = root / "empty.db"
    with pycolmap.Database.open(empty_db):
        pass
    try:
        run(empty_db, images, runs / "empty", "baseline", 1)
    except RuntimeError as exc:
        assert "no useful model" in str(exc)
    else:
        raise AssertionError("Empty mapping was accepted")
    assert json.loads((runs / "empty" / "report.json").read_text())["status"] == "failed"
    try:
        align_sim3([[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                   [[0, 0, 0], [2, 0, 0], [4, 0, 0]])
    except ValueError:
        pass
    else:
        raise AssertionError("Degenerate alignment was accepted")
    summary = {"build": pycolmap.COLMAP_build, "native_parity": parity, "native_geometry_exact": True,
               "localmt_two_threads": mt_report["models"][0],
               "synthetic": {"rigs": 1, "frames": 14, "cameras_per_frame": 2, "points": 400,
                             "keypoint_noise_sigma_px": 0.25},
               "arms": {arm: {"model": r["models"][0], "timings": r["timings"],
                              "ignored_per_global_pass": [p["ignored_redundant_points"]
                                                          for p in r["passes"]["global"]]}
                        for arm, r in reports.items()}}
    write_json(root / "smoke_summary.json", summary)
    print(f"PASS: CPU synthetic rig, five arms, native parity, Sim(3), two BA replays. {root}")


if __name__ == "__main__":
    main()
