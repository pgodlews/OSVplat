"""Shared helpers for the isolated COLMAP 4.2 mapping experiments."""
import json
import os
from pathlib import Path
import sys

import pycolmap

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import colmap_incremental as mirror  # noqa: E402

ARMS = ("baseline", "redundant", "ratio2", "redundant_ratio2", "localmt")


def require_version():
    if pycolmap.__version__ != "4.2.0":
        raise RuntimeError(f"Requires pycolmap 4.2.0, got {pycolmap.__version__}; no fallback")


def options_for(arm, threads):
    """82_fisheye_sfm.py's default (fixed intrinsics), including default model policy."""
    require_version()
    opts = pycolmap.IncrementalPipelineOptions(num_threads=threads, random_seed=0)
    opts.ba_refine_focal_length = False
    opts.ba_refine_extra_params = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_sensor_from_rig = True
    opts.mapper.abs_pose_refine_focal_length = False
    opts.mapper.abs_pose_refine_extra_params = False
    opts.ba_global_frames_ratio = 1.4
    opts.ba_global_points_ratio = 1.4
    opts.ba_global_max_refinements = 2
    if "ratio2" in arm:
        opts.ba_global_frames_ratio = opts.ba_global_points_ratio = 2.0
    if "redundant" in arm:
        opts.mapper.ba_global_ignore_redundant_points3D = True
        assert opts.mapper.ba_global_prune_points_min_coverage_gain == 0.05
    return opts


def threads_default():
    return int(os.environ.get("SPLAT_THREADS") or -1)


def model_stats(rec):
    return {
        "reg_frames": rec.num_reg_frames(), "reg_images": rec.num_reg_images(),
        "points3D": rec.num_points3D(),
        "mean_reproj_px": rec.compute_mean_reprojection_error(),
        "mean_track_length": rec.compute_mean_track_length(),
        "observations": rec.compute_num_observations(),
    }


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def new_output(path):
    path = Path(path).resolve()
    # No accidental replacement of either inputs or previous measurements.
    path.mkdir(parents=True, exist_ok=False)
    return path


def model_path(path):
    path = Path(path)
    if (path / "cameras.bin").exists() or (path / "cameras.txt").exists():
        return path
    children = sorted(p for p in path.iterdir() if p.is_dir() and
                      ((p / "cameras.bin").exists() or (p / "cameras.txt").exists()))
    if len(children) != 1:
        raise ValueError(f"Select an explicit model directory: found {len(children)} in {path}")
    return children[0]
