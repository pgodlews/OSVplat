#!/usr/bin/env python3
"""Raw DJI dual fisheye (.OSV) -> rig frames for SfM, with no stitch in between.

Decodes both lens tracks (streams 0 and 1: 3840x3840 on the Osmo 360, 3000x3000
on the Avata 360) at --fps, then keeps the sharpest *instant* out of every
--window candidates. An instant is scored on both lenses and ranked by the LOWER
of the two: a rig frame is only as sharp as its blurrier half, and picking each
lens's sharpest frame on its own would pair two images taken at different moments.

With --imu, the camera's own orientation stream votes first. Each candidate's
rotational smear is predicted as fx * mean |omega| over its exposure; within a
window, candidates predicted to smear more than --imu-tolerance-px beyond the
stillest one are out, and the Laplacian ranks the rest. The Laplacian at quarter
scale does not see smear below ~8 px, which is most of what a 10 ms indoor
exposure produces. In daylight every candidate is well under a pixel, and the
selection comes out exactly as without --imu (docs/how-it-works.md, "Gyro blur veto").

With --distance-m, use Avata NED velocity to select near equally spaced path
targets, with --max-gap-s for temporal coverage. In --select-only mode pass the
original recording as --motion-src. See queue/README.md for the algorithm and
limitations. The distance option does not yet perform SfM feedback/reselection.

Writes <out>/images/lens0/frame_NNNN.jpg and <out>/images/lens1/frame_NNNN.jpg as
hardlinks to the candidates, plus <out>/selection.json. The names are identical
across the two folders on purpose: COLMAP's rig config groups images into one
frame by the name that is left after stripping the image_prefix ("lens0/").

usage:
  80_fisheye_frames.py <in.OSV> <out_dir> [--fps 10] [--window 3] [--start 0] [--duration S]
                       [--imu <in.OSV> [--calib calibration.json] [--imu-tolerance-px 0.5]]
      decode into <out_dir>/cand/lens{0,1}/, then select
  80_fisheye_frames.py --select-only <cand_root> <out_dir> [--window 3] [--fps 10] [--start 0] [--imu ...]
      <cand_root> already holds lens0/ and lens1/ (the queue's frames stage decodes)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import avata_motion

# Nominal video frame rates. The camera's frame clock (3-1-2) runs up to ~150 ppm
# off nominal, but ffmpeg picks frames by container timestamps, which are exact.
VIDEO_RATES = (24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 48000 / 1001, 48.0, 50.0,
               60000 / 1001, 60.0, 100.0, 120000 / 1001, 120.0)


def score(f):
    img = cv2.imread(str(f), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    # A truncated last JPEG from an interrupted dump decodes to None; -1 sorts
    # below every real score so a readable neighbour always wins its window.
    if img is None:
        return -1.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def decode(src, cand, fps, start, duration):
    for d in cand:
        d.mkdir(parents=True, exist_ok=True)
    t = time.time()
    trim = (["-ss", str(start)] if start else []) + (["-t", str(duration)] if duration else [])
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "cuda",
         *trim, "-i", src,
         "-filter_complex", f"[0:v:0]fps={fps}[a];[0:v:1]fps={fps}[b]",
         "-map", "[a]", "-q:v", "2", str(cand[0] / "%05d.jpg"),
         "-map", "[b]", "-q:v", "2", str(cand[1] / "%05d.jpg")],
        check=True)
    return time.time() - t


def video_frames(n, fps, video_fps, start=0.0):
    """Index in the clip of the video frame behind each of the first n fps-filter candidates.

    ffmpeg's fps filter emits, for output tick k, the LAST input frame whose
    timestamp rounds to tick k or earlier -- not the frame nearest the tick. At
    50 -> 10 fps candidate k is frame 5k+2, 40 ms after k/10 s. With -ss the
    ticks count from the trim while frames keep their place in the clip.
    Matched pixel-exactly against full-rate decodes at 50 and 29.97 fps, trimmed
    and not, and against the queue's own CUDA dump (docs/how-it-works.md, "Gyro blur veto").
    """
    k = np.arange(n)
    return (np.ceil(video_fps * (start + (k + 0.5) / fps) - 1e-9) - 1).astype(int)


def predicted_blur(src, calib, n, fps, start=0.0):
    """Rotational smear in px over each candidate's exposure, from the clip's orientation stream.

    blur = fx * mean |omega| * exposure: the arc a ray sweeps across the sensor
    while the shutter is open. Translation is left out. At walking pace it can
    reach pixels on a wall 3 m away in 10 ms, but it barely changes across a
    window of candidates, so it moves them all alike, while rotation swings with
    every step. Returns (blur per candidate, dict of the per-candidate inputs).
    """
    import osv_imu
    from osmo_fisheye import lenses

    s = osv_imu.decode(str(src))
    ft, expo = s["frame_t"], s["exposure"].astype(float)
    measured = 1e6 / np.median(np.diff(ft))
    video_fps = min(VIDEO_RATES, key=lambda r: abs(r - measured))
    rec = video_frames(n, fps, video_fps, start)
    if rec[-1] >= len(ft):
        sys.exit(f"--imu: candidate {n - 1} is video frame {rec[-1]}, but {src} carries orientation "
                 f"records for {len(ft)} frames; were the candidates decoded from this clip at {fps:g} fps?")
    missing = np.isnan(expo)
    if missing.all():
        sys.exit(f"--imu: {src} records no exposure time (3-2-4-1)")
    if missing.any():
        i = np.arange(len(expo))
        expo[missing] = np.interp(i[missing], i[~missing], expo[~missing])
    rate = osv_imu.exposure_rates(s["t"], s["w"], ft[rec], expo[rec])
    fx = float(np.mean([l["fx"] for l in lenses(calib)]))
    return fx * rate * expo[rec] / 1e6, dict(video_frame=rec, video_fps=video_fps, fx=fx,
                                             exposure_ms=expo[rec] / 1e3, rate_dps=np.degrees(rate))


def pick(rig, window, blur=None, tolerance_px=0.5):
    """The candidate kept from each window: the highest rig score, after the IMU's veto.

    A candidate scored -1 (unreadable) never wins, and a window of nothing but
    those yields no frame. With blur, candidates predicted to smear more than
    tolerance_px beyond the stillest readable one in their window are out
    before the Laplacian ranks the rest.
    """
    sel = []
    for i in range(0, len(rig), window):
        idx = [k for k in range(i, min(i + window, len(rig))) if rig[k] >= 0]
        if not idx:
            continue
        if blur is not None:
            floor = min(blur[k] for k in idx)
            idx = [k for k in idx if blur[k] <= floor + tolerance_px]
        sel.append(max(idx, key=lambda k: rig[k]))
    return sel


def blur_stats(b):
    return {"median_px": round(float(np.median(b)), 3), "p90_px": round(float(np.percentile(b, 90)), 3),
            "max_px": round(float(np.max(b)), 2), "over_1px": int((b > 1).sum()),
            "over_2px": int((b > 2).sum()), "over_4px": int((b > 4).sum())}


def pick_distance(rig, path, groups, spacing_m, blur=None, tolerance_px=0.5):
    """Prefer near-bin-centre candidates, then apply the existing rig/blur ranking.

    The shortlist is every readable candidate within 20% of the spacing of the
    closest one to the bin centre. A group holding the centre gets the central
    40% band; a slow-flight time split away from the centre still ranks all of
    its (nearly co-located) candidates instead of keeping only the nearest.
    Fast flight with sparse candidates keeps ~1 choice by design: widening
    further trades away the spacing, so raise candidate fps instead.
    """
    selected = []
    distance = np.asarray(path["distance_m"])
    for group in groups:
        idx = [i for i in group if rig[i] >= 0]
        if not idx:
            continue
        centre = (np.floor(distance[group[0]] / spacing_m + 1e-9) + .5) * spacing_m
        offset = np.abs(distance[idx] - centre)
        # While hovering all candidates can be at the same distance: score them all.
        allowed = float(offset.min()) + .2 * spacing_m + 1e-9
        idx = [i for i, delta in zip(idx, offset) if delta <= allowed]
        if blur is not None:
            floor = min(blur[i] for i in idx)
            idx = [i for i in idx if blur[i] <= floor + tolerance_px]
        selected.append(max(idx, key=lambda i: rig[i]))
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--select-only", action="store_true",
                    help="src is a candidate root holding lens0/ and lens1/; skip decoding")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--distance-m", type=float, default=None,
                    help="Avata only: select near targets this many metres apart")
    ap.add_argument("--motion-src", default="", help="original Avata .OSV for --select-only distance mode")
    ap.add_argument("--max-gap-s", type=float, default=2.,
                    help="distance mode: maximum time between readable selected instants")
    ap.add_argument("--imu", default="",
                    help="the .OSV the candidates were decoded from; its orientation stream "
                         "vetoes candidates predicted to be motion-blurred")
    ap.add_argument("--calib", default="",
                    help="calibration.json for --imu (default: beside the candidates, then in out_dir)")
    ap.add_argument("--imu-tolerance-px", type=float, default=0.5,
                    help="predicted smear a candidate may carry beyond the stillest in its window")
    a = ap.parse_args()

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.select_only:
        cand = [Path(a.src).expanduser() / f"lens{i}" for i in (0, 1)]
    else:
        cand = [out / "cand" / f"lens{i}" for i in (0, 1)]
        secs = decode(a.src, cand, a.fps, a.start, a.duration)
    f0 = sorted(cand[0].glob("*.jpg"))
    f1 = sorted(cand[1].glob("*.jpg"))
    if not f0 or len(f0) != len(f1):
        sys.exit(f"lens candidate counts are empty or differ: {len(f0)} vs {len(f1)}")
    if not a.select_only:
        print(f"decode: {len(f0)} instants x 2 lenses in {secs:.1f}s", flush=True)

    t = time.time()
    with ProcessPoolExecutor(16) as ex:
        s0 = np.array(list(ex.map(score, f0, chunksize=8)))
        s1 = np.array(list(ex.map(score, f1, chunksize=8)))
    rig = np.minimum(s0, s1)
    blur = imu = None
    if a.imu:
        calib = Path(a.calib) if a.calib else next(
            (p for p in (cand[0].parent / "calibration.json", out / "calibration.json") if p.is_file()), None)
        if calib is None:
            sys.exit("--imu needs --calib: no calibration.json beside the candidates or in out_dir")
        blur, imu = predicted_blur(a.imu, calib, len(f0), a.fps, a.start)
    path = distance_groups = None
    if a.distance_m is not None:
        motion_src = a.motion_src or (a.src if not a.select_only else "")
        if not motion_src:
            ap.error("--select-only with --distance-m needs --motion-src <original Avata.OSV>")
        try:
            path, distance_groups = avata_motion.plan(
                motion_src, len(f0), a.fps, a.start, a.distance_m, a.max_gap_s)
        except ValueError as exc:
            ap.error(str(exc))
        sel = pick_distance(rig, path, distance_groups, a.distance_m, blur, a.imu_tolerance_px)
        if any(path["t_sec"][j] - path["t_sec"][i] > a.max_gap_s + 1e-6
               for i, j in zip(sel, sel[1:])):
            ap.error("Unreadable candidates prevent the requested maximum time gap")
    else:
        sel = pick(rig, a.window, blur, a.imu_tolerance_px)
    if not sel:
        sys.exit("every candidate instant was unreadable")

    img = [out / "images" / f"lens{i}" for i in (0, 1)]
    for d in img:
        d.mkdir(parents=True, exist_ok=True)
        # Clear, never overwrite piecemeal: a frame_0004.jpg left from a run with
        # a different window is a different instant.
        for p in d.glob("*.jpg"):
            p.unlink()
    frames = []
    for k, j in enumerate(sel):
        name = f"frame_{k:04d}.jpg"
        os.link(f0[j], img[0] / name)
        os.link(f1[j], img[1] / name)
        fr = {"name": name, "candidate": int(j), "t_sec": round(a.start + j / a.fps, 3)}
        if path is not None:
            fr.update(video_frame=path["video_frame"][j], t_sec=round(path["t_sec"][j], 6),
                      distance_m=round(path["distance_m"][j], 4),
                      position_ned_m=[round(x, 4) for x in path["position_ned_m"][j]],
                      speed_mps=round(path["speed_mps"][j], 4))
        if imu:
            fr.update(video_frame=int(imu["video_frame"][j]), exposure_ms=round(float(imu["exposure_ms"][j]), 3),
                      rate_dps=round(float(imu["rate_dps"][j]), 1), blur_px=round(float(blur[j]), 2))
        frames.append(fr)
    doc = {"src": a.src, "fps": a.fps, "window": a.window, "frames": frames}
    note = ""
    if path is not None:
        gaps = np.diff(np.asarray(path["distance_m"])[sel])
        undersampled = int(np.sum(np.diff(path["distance_m"]) > a.distance_m))
        doc["mode"] = "distance"
        doc["window"] = None
        doc["distance"] = {"src": motion_src, "spacing_m": a.distance_m, "max_gap_s": a.max_gap_s,
                           "path_length_m": round(path["distance_m"][-1], 3),
                           "speed_floor_mps": avata_motion.SPEED_FLOOR_MPS,
                           "groups": len(distance_groups),
                           "gap_median_m": round(float(np.median(gaps)), 3) if len(gaps) else 0.,
                           "gap_max_m": round(float(max(gaps)), 3) if len(gaps) else 0.,
                           "undersampled_intervals": undersampled}
        if undersampled:
            doc["distance"]["warning"] = (
                f"{undersampled} candidate intervals exceed the requested distance; increase candidate fps")
        # Keep every candidate's provisional path for inspection / future SfM feedback.
        with open(out / "motion_path.json", "w") as f:
            json.dump(path, f)
    if imu:
        plain = (pick_distance(rig, path, distance_groups, a.distance_m)
                 if path is not None else pick(rig, a.window))
        overruled = int(sum(p != q for p, q in zip(sel, plain)))
        doc["imu"] = {"src": a.imu, "calib": str(calib), "tolerance_px": a.imu_tolerance_px,
                      "fx": round(imu["fx"], 2), "video_fps": round(imu["video_fps"], 3),
                      "windows": len(sel), "overruled": overruled,
                      "selected": blur_stats(blur[sel]), "laplacian_only": blur_stats(blur[plain])}
        got, alone = doc["imu"]["selected"], doc["imu"]["laplacian_only"]
        note = (f"; imu overruled the Laplacian in {overruled} of {len(sel)} windows, predicted smear "
                f"median {got['median_px']:.2f} px with {got['over_4px']} over 4 px "
                f"(Laplacian alone {alone['median_px']:.2f} px, {alone['over_4px']})")
    json.dump(doc, open(out / "selection.json", "w"), indent=1)
    method = f"distance {a.distance_m:g} m" if path is not None else f"window {a.window}"
    print(f"select: {len(f0)} instants -> {len(sel)} rig frames ({method}) "
          f"in {time.time() - t:.1f}s; rig score median {np.median(rig):.1f}, "
          f"selected median {np.median(rig[sel]):.1f}{note}")
    if path is not None:
        print("distance: " + json.dumps(doc["distance"]))


if __name__ == "__main__":
    main()
