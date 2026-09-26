#!/usr/bin/env python3
"""Put a fisheye rig reconstruction upright, and in metres when the clip has GPS.

SfM leaves a model in whatever frame and scale its first image pair implied, so
the splat comes out tilted and in arbitrary units. A raw .OSV carries what is
needed to fix both:

  gravity  the fused orientation stream, 1 kHz (Osmo 360) or 4 kHz (Avata 360):
           body -> world, world z down (docs/osmo360-telemetry.md §6)
  metres   GpsBasic fixes at ~10 Hz (docs/avata360-telemetry.md §6); the Osmo
           360 writes the same message with no fix yet

How, per selected rig frame (lens 0's pose from SfM, the stream interpolated at
the frame's mid-exposure):

1. Lens-from-body rotation X, hand-eye style. Two frames' relative rotation is
   the same turn seen by two sensors, A = X B X^T, so the rotation vectors of
   those turns satisfy log(A) = X log(B): Kabsch over many frame pairs. Not
   taken from the calibration: whether DJI's lens angles are in the stream's
   body frame is unverified (docs/osmo360-telemetry.md §9). Needs turns about
   two axes, which a handheld walk or a flight has.
2. World-from-SfM rotation R, averaged over frames, then X and R refined
   together. The per-frame residual is the check: a stream that does not
   describe this camera, or is timed wrong, cannot fit every frame.
3. With GPS (any clip with valid fixes, whichever camera): yaw about gravity,
   scale and offset from the camera path against the fixes in local
   north-east-down metres. Rotation stays gravity's; GPS only turns it about
   the vertical, so a straight flight line is enough.

Output frame: x east, y DOWN, z north; metres when GPS scaled it; origin at the
median camera centre. y-down is COLMAP's and OpenCV's convention, and what
SuperSplat expects (it turns a loaded splat 180 deg about z, so -y shows up).
Without GPS, "north" is the orientation stream's own heading reference, which
is not verified to be north, and the units stay SfM's.

Fails safe: when a step cannot be trusted, the model is left as it was (no
gravity) or unscaled (no usable GPS), and the report says why. It never fails
the reconstruction.

usage (venv): upright.py <model_dir> <out_model_dir> --osv CLIP.OSV \
              --selection SELECT/selection.json [--start S]
Writes <out_model_dir> (the transformed model) and alignment.json beside it.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from avata_motion import VIDEO_RATES  # noqa: E402
from osv_meta import camd_payload, f64, fields, find_camd, get, records  # noqa: E402

# Gates. A fused orientation stream against a bundle-adjusted pose disagrees by
# timing (~20 ms at the p90 of a handheld walk's 80 deg/s is 1.6 deg) and fusion
# error; beyond these something is wrong, not noisy.
MAX_RESIDUAL_MEDIAN_DEG = 2.0
MAX_RESIDUAL_P90_DEG = 5.0
# Second over first singular value of the rotation-vector correlation: 0 means
# every turn was about one axis, which leaves X free about that axis.
MIN_EXCITATION = 0.02
MIN_FRAMES = 8
MIN_PAIRS = 10
# GPS: a scale from a path shorter than a few GPS errors is noise.
MIN_GPS_EXTENT_M = 20.0
MAX_GPS_RMS_FRACTION = 0.1

# out = NED_TO_OUT @ ned: x east, y down, z north.
NED_TO_OUT = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
WGS84_A, WGS84_E2 = 6378137.0, 6.69437999014e-3


# ------------------------------------------------------------------ rotations

def quat_matrix(q):
    """w, x, y, z (Hamilton) -> 3x3, for one quaternion or an (n, 4) array."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def project_rotation(M):
    """The rotation nearest M (Frobenius): the chordal mean of a sum of rotations."""
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1., 1., np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def rot_log(R):
    """Rotation vector (axis x angle, rad) of each 3x3 in R (..., 3, 3); angles below ~179 deg."""
    ang = np.radians(angle_deg(R))
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(ang)
    k = np.where(s > 1e-9, ang / (2 * np.where(s > 1e-9, s, 1)), 0.5)
    return v * k[..., None]


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


# ---------------------------------------------------------------- timing

def video_frames(candidates, fps, video_fps, start=0.0):
    """Video frame behind fps-filter candidate k: 80_fisheye_frames.video_frames, per index."""
    k = np.asarray(candidates, dtype=float)
    return (np.ceil(video_fps * (start + (k + 0.5) / fps) - 1e-9) - 1).astype(int)


def orientation_at(t_us, q, when_us):
    """The stream's quaternion at each time in when_us: normalised lerp of the two neighbours.

    Neighbours are 0.25-1 ms apart, far below anything the fit resolves.
    """
    when_us = np.asarray(when_us, dtype=float)
    hi = np.clip(np.searchsorted(t_us, when_us), 1, len(t_us) - 1)
    lo = hi - 1
    a, b = q[lo], q[hi]
    b = np.where((np.sum(a * b, axis=1) < 0)[:, None], -b, b)
    f = np.clip((when_us - t_us[lo]) / np.maximum(t_us[hi] - t_us[lo], 1e-9), 0, 1)[:, None]
    out = (1 - f) * a + f * b
    return out / np.linalg.norm(out, axis=1, keepdims=True)


# ------------------------------------------------------------------ gravity

def hand_eye(R_cw, R_wb, max_gap=64):
    """Solve R_cw[i] ~ X @ R_wb[i].T @ W for the constant rotations X (lens from body) and W.

    R_cw: SfM cam_from_world of lens 0 per frame; R_wb: the stream's body -> world
    at the same instants. W is IMU world from SfM world, so W.T @ (0, 0, 1) is
    down in the SfM model. Returns (X, W, report).
    """
    R_cw, R_wb = np.asarray(R_cw, float), np.asarray(R_wb, float)
    n = len(R_cw)
    if n < MIN_FRAMES:
        raise ValueError(f"only {n} frames to level with; need {MIN_FRAMES}")
    a, b = [], []
    gap = 1
    while gap < min(n, max_gap + 1):
        A = R_cw[gap:] @ np.swapaxes(R_cw[:-gap], 1, 2)
        B = np.swapaxes(R_wb[gap:], 1, 2) @ R_wb[:-gap]
        ang_a, ang_b = angle_deg(A), angle_deg(B)
        keep = (ang_a > 2) & (ang_a < 150) & (np.abs(ang_a - ang_b) < 10)
        a.append(rot_log(A[keep]))
        b.append(rot_log(B[keep]))
        gap *= 2
    a, b = np.concatenate(a), np.concatenate(b)
    if len(a) < MIN_PAIRS:
        raise ValueError(f"only {len(a)} frame pairs turn by 2-150 deg with a consistent angle in both "
                         f"(need {MIN_PAIRS}): the camera barely rotated, or the stream is not this clip's")
    H = b.T @ a
    U, S, Vt = np.linalg.svd(H)
    excitation = float(S[1] / S[0]) if S[0] > 0 else 0.0
    if excitation < MIN_EXCITATION:
        raise ValueError(f"the camera turned about one axis only (excitation {excitation:.3f} < "
                         f"{MIN_EXCITATION}), which cannot fix its tilt")
    X = Vt.T @ np.diag([1., 1., np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    W = project_rotation(np.sum(R_wb @ X.T @ R_cw, axis=0))
    for _ in range(20):
        X = project_rotation(np.sum(R_cw @ W.T @ R_wb, axis=0))
        W = project_rotation(np.sum(R_wb @ X.T @ R_cw, axis=0))
    resid = angle_deg(np.swapaxes(R_cw, 1, 2) @ X @ np.swapaxes(R_wb, 1, 2) @ W)
    report = {"frames": n, "pairs": int(len(a)), "excitation": round(excitation, 4),
              "residual_median_deg": round(float(np.median(resid)), 3),
              "residual_p90_deg": round(float(np.percentile(resid, 90)), 3)}
    if report["residual_median_deg"] > MAX_RESIDUAL_MEDIAN_DEG or report["residual_p90_deg"] > MAX_RESIDUAL_P90_DEG:
        raise ValueError(f"the orientation stream does not fit the reconstruction: residual median "
                         f"{report['residual_median_deg']} deg, p90 {report['residual_p90_deg']} deg "
                         f"(limits {MAX_RESIDUAL_MEDIAN_DEG}, {MAX_RESIDUAL_P90_DEG})")
    return X, W, report


# ---------------------------------------------------------------------- GPS

def _gps_basic(msg):
    """(lat, lon, alt_m or nan) from a GpsBasic message, or None when it carries no fix.

    GpsBasic (dvtm_library.proto): 1 gps_coordinates {2 latitude, 3 longitude as
    f64 degrees}, 2 gps_altitude_mm (int32), 3 gps_status (absent = normal).
    """
    coords = alt = status = None
    for fn, wt, v in fields(msg):
        if fn == 1 and wt == 2:
            coords = v
        elif fn == 2 and wt == 0:
            alt = v - (1 << 64) if v >= 1 << 63 else v
        elif fn == 3 and wt == 0:
            status = v
    if coords is None or status:
        return None
    ll = {fn: f64(v) for fn, wt, v in fields(coords) if wt == 1 and fn in (2, 3)}
    if 2 not in ll or 3 not in ll:
        return None
    lat, lon = ll[2], ll[3]
    if not (math.isfinite(lat) and math.isfinite(lon)) or abs(lat) > 90 or abs(lon) > 180 or (lat == 0 and lon == 0):
        return None
    return lat, lon, (alt / 1000.0 if alt is not None else math.nan)


def gps_from_records(recs):
    """Distinct fixes as (t_us, lat, lon, alt_m) arrays, from (field, body) records.

    GpsBasic is found by content under the record's field 4, not by number: the
    Avata 360 keeps it at 3-4-4 and the Osmo 360 at 3-4-2 (status-only so far,
    so the Osmo layout of a real fix is inferred from the shared message type).
    A fix is held over several records (five at 50 fps on the Avata); each is
    timed at the first record that carries it.
    """
    found = []
    for fn, body in recs:
        if fn != 3:
            continue
        ts = get(body, 1, 2)
        dev = get(body, 4)
        if not isinstance(ts, int) or not isinstance(dev, (bytes, bytearray)):
            continue
        fix = None
        for sfn, swt, v in fields(dev):
            if swt == 2 and v:
                try:
                    fix = _gps_basic(v)
                except Exception:  # not a protobuf message at all
                    fix = None
                if fix:
                    break
        if fix:
            found.append((ts, *fix))
    # Sorted first: both of the Avata's metadata tracks carry every fix.
    out, last = [], None
    for row in sorted(found):
        if row[1:] != last:
            out.append(row)
            last = row[1:]
    if not out:
        return None
    a = np.array(out, dtype=float)
    return {"t_us": a[:, 0], "lat": a[:, 1], "lon": a[:, 2], "alt_m": a[:, 3]}


def gps_track(path):
    return gps_from_records(records(camd_payload(find_camd(str(path)))))


def ecef(lat, lon, alt):
    la, lo = np.radians(lat), np.radians(lon)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(la) ** 2)
    return np.stack([(n + alt) * np.cos(la) * np.cos(lo), (n + alt) * np.cos(la) * np.sin(lo),
                     (n * (1 - WGS84_E2) + alt) * np.sin(la)], -1)


def geodetic_to_ned(lat, lon, alt, ref):
    """Local tangent north-east-down metres about ref = (lat, lon, alt)."""
    d = ecef(lat, lon, alt) - ecef(*ref)
    la, lo = math.radians(ref[0]), math.radians(ref[1])
    R = np.array([[-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                  [-math.sin(lo), math.cos(lo), 0.],
                  [-math.cos(la) * math.cos(lo), -math.cos(la) * math.sin(lo), -math.sin(la)]])
    return d @ R.T


def fit_yaw_scale(p, q, vertical=True):
    """q ~ s Rz(yaw) p + t for p, q in north-east-down; yaw about down only.

    Gravity already levelled p, so only the heading, the scale and the offset
    are free. With vertical=False (no GPS altitude) the down axis is left out
    of scale and residual. Returns (s, yaw, t, rms_m).
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    pc, qc = p - p.mean(0), q - q.mean(0)
    yaw = math.atan2(np.sum(pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0]),
                     np.sum(pc[:, 0] * qc[:, 0] + pc[:, 1] * qc[:, 1]))
    rp = pc @ rot_z(yaw).T
    axes = slice(0, 3 if vertical else 2)
    s = float(np.sum(rp[:, axes] * qc[:, axes]) / np.sum(rp[:, axes] ** 2))
    t = q.mean(0) - s * rot_z(yaw) @ p.mean(0)
    err = (s * p @ rot_z(yaw).T + t - q)[:, axes]
    return s, yaw, t, float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def umeyama_rotation(p, q):
    """Rotation of the least-squares similarity q ~ s R p + t (for a cross-check only)."""
    pc, qc = p - p.mean(0), q - q.mean(0)
    U, S, Vt = np.linalg.svd(qc.T @ pc)
    D = np.diag([1., 1., np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt, S


def fit_gps(centres_ned, frame_t_us, gps):
    """Scale, yaw and offset of the levelled camera path against the fixes; raises when unusable."""
    t = np.asarray(frame_t_us, float)
    inside = (t >= gps["t_us"][0]) & (t <= gps["t_us"][-1])
    if inside.sum() < MIN_FRAMES:
        raise ValueError(f"only {int(inside.sum())} frames fall within the GPS track")
    vertical = bool(np.all(np.isfinite(gps["alt_m"])))
    alt = gps["alt_m"] if vertical else np.zeros_like(gps["lat"])
    ref = (float(gps["lat"][0]), float(gps["lon"][0]), float(alt[0]))
    ned = geodetic_to_ned(gps["lat"], gps["lon"], alt, ref)
    q = np.stack([np.interp(t[inside], gps["t_us"], ned[:, k]) for k in range(3)], 1)
    p = np.asarray(centres_ned, float)[inside]
    extent = float(np.linalg.norm(np.ptp(q[:, :2], axis=0)))
    if extent < MIN_GPS_EXTENT_M:
        raise ValueError(f"the GPS path spans {extent:.1f} m, under {MIN_GPS_EXTENT_M:g} m: too short to scale by")
    s, yaw, t_off, rms = fit_yaw_scale(p, q, vertical)
    if not s > 0:
        raise ValueError("the GPS path runs against the reconstruction (negative scale)")
    if rms > MAX_GPS_RMS_FRACTION * extent:
        raise ValueError(f"the camera path does not fit the GPS track: {rms:.1f} m RMS over a "
                         f"{extent:.0f} m path (limit {MAX_GPS_RMS_FRACTION:.0%})")
    report = {"fixes": int(len(gps["t_us"])), "frames": int(inside.sum()), "extent_m": round(extent, 1),
              "rms_m": round(rms, 2), "yaw_correction_deg": round(math.degrees(yaw), 2),
              "vertical": vertical}
    # Independent check of gravity: a path that is not a straight line fixes a
    # full rotation by itself. Report, do not gate: GPS altitude is the weak axis.
    R_full, S = umeyama_rotation(p, q)
    if S[1] > 0.1 * S[0] and vertical:
        down = R_full.T @ np.array([0., 0., 1.])
        report["gravity_vs_gps_deg"] = round(math.degrees(math.acos(np.clip(down[2], -1, 1))), 2)
    return s, yaw, t_off, ref, report


# ----------------------------------------------------------------- the solve

def solve(R_cw, centres, R_wb, frame_t_us, gps=None):
    """new_from_old similarity (s, R, t) into the output frame, and a report.

    Raises ValueError when gravity cannot be trusted (the caller keeps the model
    as it is). A GPS track that cannot be used leaves the scale alone and is
    reported under "gps_skipped".
    """
    X, W, grav = hand_eye(R_cw, R_wb)
    centres = np.asarray(centres, float)
    c_ned = centres @ W.T                         # levelled, SfM units, stream heading
    s, yaw, t_ned, ref = 1.0, 0.0, np.zeros(3), None
    report = {"upright": True, "metric": False, "gravity": grav}
    if gps is None:
        report["gps_skipped"] = "the clip carries no GPS fix"
    else:
        try:
            s, yaw, t_ned, ref, report["gps"] = fit_gps(c_ned, frame_t_us, gps)
            report["metric"] = True
        except ValueError as exc:
            report["gps_skipped"] = str(exc)
    R = NED_TO_OUT @ rot_z(yaw) @ W
    moved = s * centres @ R.T
    t = -np.median(moved, axis=0)
    report["scale"] = s
    report["lens_from_body"] = X.round(6).tolist()
    if ref is not None:
        # Where the new origin sits: ned = s Rz W c + t_ned for a camera centre c,
        # and the origin is the camera whose out-frame position is the median.
        origin_ned = NED_TO_OUT.T @ (-t) + t_ned
        report["geo"] = {"reference_lat_lon_alt": [round(ref[0], 9), round(ref[1], 9), round(ref[2], 3)],
                         "origin_ned_m": [round(float(x), 3) for x in origin_ned]}
    return s, R, t, report


# ---------------------------------------------------------- telemetry at frames

def frame_telemetry(osv, selection, start, names):
    """(frame times µs at mid-exposure, body->world 3x3) for each image name frame_NNNN.jpg."""
    import osv_imu
    doc = json.loads(Path(selection).read_text())
    by_name = {f["name"]: f for f in doc["frames"]}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"selection.json does not list {missing[:3]}")
    s = osv_imu.decode(str(osv))
    ft, expo = s["frame_t"], np.nan_to_num(s["exposure"].astype(float))
    measured = 1e6 / np.median(np.diff(ft))
    video_fps = min(VIDEO_RATES, key=lambda r: abs(r - measured))
    rec = video_frames([by_name[n]["candidate"] for n in names], float(doc["fps"]), video_fps, start)
    listed = [by_name[n].get("video_frame") for n in names]
    if any(v is not None and v != r for v, r in zip(listed, rec)):
        raise ValueError("selection.json names different video frames than its candidates imply; "
                         "was --start the trim the frames were decoded with?")
    if rec.max() >= len(ft) or rec.min() < 0:
        raise ValueError(f"frame {int(rec.max())} is past the {len(ft)} orientation records in {osv}")
    when = ft[rec] + expo[rec] / 2
    return when, quat_matrix(orientation_at(s["t"], s["q"], when))


def align_model(model, out, osv, selection, start=0.0):
    """Write the upright model to out and return the report; on failure return it unaligned.

    The report always carries "upright"; when False, "reason" says why and out is
    not written.
    """
    import pycolmap

    rec = pycolmap.Reconstruction(str(model))
    lens0 = sorted((rec.images[i].name, i) for i in rec.reg_image_ids() if rec.images[i].name.startswith("lens0/"))
    names = [n.split("/", 1)[1] for n, _ in lens0]
    try:
        when, R_wb = frame_telemetry(osv, selection, start, names)
        poses = [np.asarray(_call(rec.images[i].cam_from_world).matrix()) for _, i in lens0]
        R_cw = np.array([P[:, :3] for P in poses])
        centres = np.array([-P[:, :3].T @ P[:, 3] for P in poses])
        try:
            gps = gps_track(osv)
        except (Exception, SystemExit) as exc:
            gps = None
            print(f"upright: no GPS read ({exc})", flush=True)
        s, R, t, report = solve(R_cw, centres, R_wb, when, gps)
    except (Exception, SystemExit) as exc:  # fail safe: an unreadable stream leaves the model as it is
        reason = str(exc) if isinstance(exc, (ValueError, SystemExit)) else f"{type(exc).__name__}: {exc}"
        return {"upright": False, "metric": False, "reason": reason}

    before = {i: np.asarray(_call(rec.images[i].projection_center)) for i in rec.reg_image_ids()}
    rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
    # Every image, lens 1 too: a rig's baseline has to scale with the world.
    extent = max(1e-9, float(np.ptp(np.array([s * R @ c for c in before.values()]), axis=0).max()))
    worst = max(float(np.linalg.norm(np.asarray(_call(rec.images[i].projection_center)) - (s * R @ c + t)))
                for i, c in before.items())
    if worst > 1e-6 * extent + 1e-9:
        raise SystemExit(f"upright: an image moved {worst:.3g} off the similarity (extent {extent:.3g})")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rec.write_binary(str(out))
    report["transform"] = {"scale": s, "rotation": R.round(9).tolist(), "translation": t.round(6).tolist()}
    return report


def _call(x):
    return x() if callable(x) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--osv", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    a = ap.parse_args()
    report = align_model(a.model, a.out, a.osv, a.selection, a.start)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    (Path(a.out).parent / "alignment.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("geo", "transform", "lens_from_body")}))


if __name__ == "__main__":
    main()
