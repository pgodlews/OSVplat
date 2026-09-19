"""Wall-time estimator.

Constants are seeded from the measured runs 2-5 in the project notes, then refined
from this queue's own history as jobs complete, so the estimate sharpens with
use instead of rotting.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from . import db
from .jobs import JobConfig

# Seed constants (README runs 2-5).
SEED = {
    "frames_realtime": 0.8,       # x clip duration, NVDEC dump
    "select_per_cand": 0.0067,    # s per candidate frame
    "sfm_per_pano": 2.4,          # s per panorama (observed 1.8-3.3)
    "mask_per_pano": 0.7,         # 13 tangent views + Mask R-CNN (osmo360: 0.68)
    "train_it_per_s": 5.2,        # at max_width 3840, cap 3M, --gut
    "export_s": 60.0,
    # Fisheye rig pipeline (docs/how-it-works.md, "Fisheye rig", garden clip, 104 rig frames on a 3090).
    "fisheye_frames_realtime": 1.75,    # both 3840^2 lenses decoded + JPEG
    "fisheye_select_per_cand": 0.008,
    "fisheye_mask_per_frame": 1.05,     # stitch 17 s + masker 78 s + warp 13 s
    # Fisheye rig SfM is superlinear in rig frames. Extraction and matching are
    # per frame, but each global bundle adjustment covers the whole model and
    # recurs at every ~10% of growth. Mapping is fitted over the queue's
    # fisheye runs, 67-1404 frames, as 0.0794 * n^1.7 s: exact at 1404 frames
    # (4 h 57 min), about 2x high at 330. Clips of equal length still differ by
    # more than 3x. Assumes 82_fisheye_sfm.py's lifted direct-solver limit.
    "fisheye_sfm_extract_match_per_frame": 0.7,
    "fisheye_sfm_map_coeff": 0.0794,
    "fisheye_sfm_map_exponent": 1.7,
    "fisheye_sfm_post_per_frame": 0.15,  # pick, refined calib, angular error, dataset
    "fisheye_train_it_per_s": 6.2,      # 30k in 80.5 min, 3M cap, --gut, masks
}


# Only fit from runs long enough that per-run overhead (dataset load, eval,
# export) is not most of the wall time. A 3k-iteration smoke job is not evidence
# about a 30k one.
MIN_FIT_ITERS = 5000
MAX_FIT_JOBS = 20


def _train_scaling(max_width: int, max_cap: int) -> float:
    """Cost multiplier of a config relative to the width-3840 / 3M baseline.

    Kept in one place because it is used twice and in opposite directions: to
    predict a rate from the baseline, and to normalise an observed rate back to
    the baseline when fitting.
    """
    width = max_width or 3840
    px_scale = (width / 3840.0) ** 2
    cap_scale = max(0.5, min(1.5, max_cap / 3_000_000))
    return max(0.05, px_scale * (0.6 + 0.4 * cap_scale))


def _is_fisheye_cfg(cfg: dict) -> bool:
    return str((cfg.get("input") or {}).get("file", "")).lower().endswith(".osv")


def _fitted_rate(fisheye: bool = False) -> Optional[float]:
    """Median baseline it/s implied by completed jobs.

    Each historical job is normalised back to the baseline config before
    averaging -- otherwise a batch of small, fast, low-resolution jobs drags the
    constant up and the predictor then divides by the resolution factor a second
    time, badly underestimating full-size runs.
    """
    try:
        rows = db.conn().execute(
            """SELECT j.config AS config,
                      MAX(CASE WHEN m.name='train_seconds' THEN m.value END) AS secs,
                      MAX(CASE WHEN m.name='final_step'    THEN m.value END) AS steps
               FROM jobs j JOIN metrics m ON m.job_id = j.id
               WHERE j.state='done'
                 AND (lower(substr(COALESCE(json_extract(j.config, '$.input.file'), ''), -4)) = '.osv') = ?
               GROUP BY j.id
               HAVING secs > 0 AND steps >= ?
               ORDER BY j.id DESC LIMIT ?""",
            (int(fisheye), MIN_FIT_ITERS, MAX_FIT_JOBS)).fetchall()
    except Exception:                                          # noqa: BLE001
        return None

    rates = []
    for r in rows:
        try:
            full = json.loads(r["config"])
            # A fisheye rig trains on a different image set at a different
            # rate; fitting the two together would skew both estimates.
            if _is_fisheye_cfg(full) != fisheye:
                continue
            cfg = full["train"]
            observed = float(r["steps"]) / float(r["secs"])
            rates.append(observed * _train_scaling(
                int(cfg.get("max_width") or 3840),
                int(cfg.get("max_cap") or 3_000_000)))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
    if not rates:
        return None
    rates.sort()
    median = rates[len(rates) // 2]
    return median if 0.2 <= median <= 200 else None


# The fit is a GROUP BY over every completed job, and the queue view asks for
# an ETA per job every four seconds. Uncached, a page of 200 jobs was 200
# identical scans of the metrics table per poll. The fitted rate moves as jobs
# complete, which is to say roughly hourly, so a minute of staleness costs
# nothing.
_CONST_TTL = 60.0
_const_cache: tuple[float, dict] = (0.0, {})


def constants() -> dict:
    """Seed constants overridden by the median of comparable completed jobs."""
    global _const_cache
    at, cached = _const_cache
    now = time.monotonic()
    if cached and now - at < _CONST_TTL:
        return dict(cached)
    c = dict(SEED)
    fitted = _fitted_rate()
    if fitted:
        c["train_it_per_s"] = fitted
        c["_train_rate_source"] = "fitted"
    else:
        c["_train_rate_source"] = "seed"
    fitted_fish = _fitted_rate(fisheye=True)
    if fitted_fish:
        c["fisheye_train_it_per_s"] = fitted_fish
        c["_fisheye_train_rate_source"] = "fitted"
    else:
        c["_fisheye_train_rate_source"] = "seed"
    _const_cache = (now, dict(c))
    return c


def train_rate(cfg: JobConfig) -> float:
    """Iterations per second this config should average over a whole run.

    Deliberately a whole-run average and not an instantaneous rate: iterations
    slow steadily as densification grows the splat count, so an instantaneous
    reading extrapolates a run to be 2-4x shorter than it turns out to be.
    """
    c = constants()
    base = c["fisheye_train_it_per_s"] if cfg.is_fisheye else c["train_it_per_s"]
    return max(0.01, base / _train_scaling(cfg.train.max_width, cfg.train.max_cap))


def fisheye_sfm_seconds(frames: int, c: dict) -> float:
    """Wall time of the fisheye rig SfM stage for `frames` rig frames."""
    per_frame = c["fisheye_sfm_extract_match_per_frame"] + c["fisheye_sfm_post_per_frame"]
    return frames * per_frame + c["fisheye_sfm_map_coeff"] * frames ** c["fisheye_sfm_map_exponent"]


def estimate(cfg: JobConfig, clip_seconds: float,
             cached: Optional[dict] = None) -> dict:
    """Per-stage seconds. `cached` maps stage -> True for cache hits (0 s)."""
    c = constants()
    cached = cached or {}

    dur = clip_seconds
    if cfg.input.trim_end is not None:
        dur = cfg.input.trim_end - (cfg.input.trim_start or 0.0)
    elif cfg.input.trim_start:
        dur = max(0.0, clip_seconds - cfg.input.trim_start)

    candidates = max(1, int(dur * cfg.frames.fps))
    if cfg.select.mode == "target":
        panos = cfg.select.target_panos
    elif cfg.select.mode == "distance":
        from avata_motion import plan
        from .config import SPLAT_ROOT
        path, groups = plan(SPLAT_ROOT / cfg.input.file, candidates, cfg.frames.fps,
                            cfg.input.trim_start or 0., cfg.select.distance_m, cfg.select.max_gap_s)
        panos = len(groups)
    else:
        panos = max(1, candidates // max(1, cfg.select.window))

    iters = cfg.train.effective_iters
    rate = train_rate(cfg)

    if cfg.is_fisheye:
        # "panos" are rig frames here: two fisheye images each.
        est = {
            "frames": dur * c["fisheye_frames_realtime"],
            "select": candidates * c["fisheye_select_per_cand"],
            "mask": panos * c["fisheye_mask_per_frame"] if cfg.mask.enabled else 0.0,
            "sfm": fisheye_sfm_seconds(panos, c),
            "train": iters / max(0.01, rate),
            "export": c["export_s"],
        }
    else:
        est = {
            "frames": dur * c["frames_realtime"],
            "select": candidates * c["select_per_cand"],
            "mask": panos * c["mask_per_pano"] if cfg.mask.enabled else 0.0,
            "sfm": panos * c["sfm_per_pano"],
            "train": iters / max(0.01, rate),
            "export": c["export_s"],
        }
    for k in est:
        if cached.get(k):
            est[k] = 0.0
    est = {k: round(v, 1) for k, v in est.items()}
    est["total"] = round(sum(est.values()), 1)
    est["_assumptions"] = {
        "clip_seconds": round(dur, 1), "candidates": candidates,
        "panos": panos, "iterations": iters, "it_per_s": round(rate, 2),
        "baseline_it_per_s": round(c["train_it_per_s"], 2),
        "rate_source": c.get("_fisheye_train_rate_source" if cfg.is_fisheye
                             else "_train_rate_source", "seed"),
        "pipeline": "fisheye_rig" if cfg.is_fisheye else "equirect",
    }
    if cfg.select.mode == "distance":
        est["_assumptions"].update(selection_mode="distance", distance_m=cfg.select.distance_m,
                                    path_length_m=round(path["distance_m"][-1], 1))
        if any(b - a > cfg.select.distance_m for a, b in zip(path["distance_m"], path["distance_m"][1:])):
            est["_assumptions"]["warning"] = "Candidate spacing exceeds the requested distance; increase candidate fps."
    return est
