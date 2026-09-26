#!/usr/bin/env python3
"""CPU checks for upright.py: levelling from the orientation stream, GPS scale (numpy only).

Runs anywhere with numpy:  python3 scripts/test_upright.py
Synthetic flights with a known answer; the pycolmap half (the model transform)
is in test_fisheye.py.
"""
import math
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import upright  # noqa: E402


def euler(yaw, pitch, roll):
    """Body -> NED world from aircraft angles (rad)."""
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch),
                              math.cos(roll), math.sin(roll))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def small_rotation(rng, deg):
    v = rng.normal(size=3)
    v *= math.radians(deg) / np.linalg.norm(v)
    th = np.linalg.norm(v)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


REF = (47.3, 8.5, 450.0)


def flight(n=80, seed=0, imu_noise_deg=0.2, single_axis=False, heading_offset_deg=0.0):
    """A curved, climbing flight: truth in NED metres, SfM in its own similarity frame."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 60, n)
    c_ned = np.stack([80 * np.sin(t / 20), 120 * (1 - np.cos(t / 25)), -10 - 0.2 * t], 1)
    if single_axis:
        R_wb = np.array([euler(0.05 * k, 0, 0) for k in range(n)])
    else:
        R_wb = np.array([euler(t_ / 12, 0.25 * math.sin(t_ / 3), 0.2 * math.cos(t_ / 4)) for t_ in t])
    X = euler(0.7, -1.2, 2.1)                             # lens from body
    Q = euler(-2.0, 0.4, 1.1)                             # SfM world from NED world
    s0, t0 = 0.037, np.array([3.0, -1.0, 0.5])
    R_cw = np.array([X @ R.T @ Q.T for R in R_wb])        # lens from SfM world
    centres = s0 * c_ned @ Q.T + t0
    H = euler(math.radians(heading_offset_deg), 0, 0)     # stream world = NED turned about down
    R_imu = np.array([H @ R @ small_rotation(rng, imu_noise_deg) for R in R_wb])
    lat = REF[0] + c_ned[:, 0] / 111_200.0
    lon = REF[1] + c_ned[:, 1] / (111_200.0 * math.cos(math.radians(REF[0])))
    alt = REF[2] - c_ned[:, 2]
    truth = upright.geodetic_to_ned(lat, lon, alt, REF)   # exact metres of those fixes
    times = t * 1e6
    gps = {"t_us": times, "lat": lat, "lon": lon, "alt_m": alt}
    return dict(R_cw=R_cw, centres=centres, R_wb=R_imu, t_us=times, gps=gps, truth=truth,
                X=X, Q=Q, s0=s0, H=H)


class Gravity(unittest.TestCase):
    def test_hand_eye_recovers_both_rotations(self):
        f = flight()
        X, W, rep = upright.hand_eye(f["R_cw"], f["R_wb"])
        self.assertLess(upright.angle_deg(X @ f["X"].T), 0.3)
        self.assertLess(upright.angle_deg(W @ f["Q"]), 0.3)
        self.assertLess(rep["residual_median_deg"], 0.5)

    def test_down_becomes_plus_y_without_gps(self):
        f = flight()
        s, R, t, rep = upright.solve(f["R_cw"], f["centres"], f["R_wb"], f["t_us"], gps=None)
        self.assertTrue(rep["upright"])
        self.assertFalse(rep["metric"])
        self.assertEqual(s, 1.0)
        down_sfm = f["Q"] @ np.array([0., 0., 1.])       # NED down, seen in the SfM frame
        self.assertGreater((R @ down_sfm)[1], math.cos(math.radians(0.3)))
        out = f["centres"] @ R.T + t
        self.assertTrue(np.allclose(np.median(out, axis=0), 0, atol=1e-9))

    def test_one_axis_of_rotation_is_refused(self):
        f = flight(single_axis=True)
        with self.assertRaisesRegex(ValueError, "one axis"):
            upright.hand_eye(f["R_cw"], f["R_wb"])

    def test_a_stream_from_another_clip_is_refused(self):
        f = flight()
        other = flight(seed=3)["R_wb"][::-1]
        with self.assertRaises(ValueError):
            upright.hand_eye(f["R_cw"], other)

    def test_world_to_body_quaternions_are_refused(self):
        # The convention read backwards (world -> body) cannot fit every frame.
        f = flight()
        with self.assertRaises(ValueError):
            upright.hand_eye(f["R_cw"], np.swapaxes(f["R_wb"], 1, 2))


class Metres(unittest.TestCase):
    def check_metric(self, f, rep, s, R, t):
        self.assertTrue(rep["metric"], rep.get("gps_skipped"))
        self.assertAlmostEqual(s * f["s0"], 1.0, delta=0.01)
        out = f["centres"] @ (s * R).T + t
        want = f["truth"] @ upright.NED_TO_OUT.T
        want -= np.median(want, axis=0)
        self.assertLess(np.abs(out - want).max(), 2.0)     # metres, over a ~200 m flight
        # The origin is the median camera, stated in NED about the reported reference fix.
        g = f["gps"]
        about_ref = upright.geodetic_to_ned(g["lat"], g["lon"], g["alt_m"], rep["geo"]["reference_lat_lon_alt"])
        origin = np.array(rep["geo"]["origin_ned_m"])
        self.assertLess(np.abs(origin - np.median(about_ref, axis=0)).max(), 2.0)

    def test_gps_scales_to_metres(self):
        f = flight()
        s, R, t, rep = upright.solve(f["R_cw"], f["centres"], f["R_wb"], f["t_us"], f["gps"])
        self.check_metric(f, rep, s, R, t)
        self.assertLess(abs(rep["gps"]["yaw_correction_deg"]), 0.5)
        self.assertLess(rep["gps"]["gravity_vs_gps_deg"], 1.0)

    def test_gps_finds_north_when_the_stream_heading_is_off(self):
        f = flight(heading_offset_deg=30)
        s, R, t, rep = upright.solve(f["R_cw"], f["centres"], f["R_wb"], f["t_us"], f["gps"])
        self.check_metric(f, rep, s, R, t)
        self.assertAlmostEqual(rep["gps"]["yaw_correction_deg"], -30, delta=0.5)

    def test_noisy_gps_still_scales(self):
        f = flight()
        rng = np.random.default_rng(1)
        g = dict(f["gps"])
        g["lat"] = g["lat"] + rng.normal(0, 1.5 / 111_200, len(g["lat"]))
        g["lon"] = g["lon"] + rng.normal(0, 1.5 / 75_000, len(g["lon"]))
        s, R, t, rep = upright.solve(f["R_cw"], f["centres"], f["R_wb"], f["t_us"], g)
        self.assertTrue(rep["metric"])
        self.assertAlmostEqual(s * f["s0"], 1.0, delta=0.02)

    def test_a_hover_is_not_scaled(self):
        f = flight()
        g = dict(f["gps"])
        g["lat"] = REF[0] + (g["lat"] - REF[0]) * 0.02    # ~5 m of GPS path
        g["lon"] = REF[1] + (g["lon"] - REF[1]) * 0.02
        s, R, t, rep = upright.solve(f["R_cw"], f["centres"], f["R_wb"], f["t_us"], g)
        self.assertTrue(rep["upright"])
        self.assertFalse(rep["metric"])
        self.assertIn("too short", rep["gps_skipped"])
        self.assertEqual(s, 1.0)

    def test_gps_from_another_flight_is_not_used(self):
        f = flight()
        g = dict(f["gps"])
        g["lat"], g["lon"] = g["lat"][::-1].copy(), g["lon"][::-1].copy()
        g["lat"] = REF[0] + (g["lat"] - REF[0]) * np.linspace(1, -1, len(g["lat"])) ** 3
        _, _, _, rep = upright.solve(f["R_cw"], f["centres"], f["R_wb"], f["t_us"], g)
        self.assertFalse(rep["metric"])
        self.assertIn("gps_skipped", rep)

    def test_no_altitude_scales_horizontally(self):
        f = flight()
        g = dict(f["gps"], alt_m=np.full(len(f["gps"]["lat"]), np.nan))
        s, R, t, rep = upright.solve(f["R_cw"], f["centres"], f["R_wb"], f["t_us"], g)
        self.assertTrue(rep["metric"])
        self.assertFalse(rep["gps"]["vertical"])
        self.assertAlmostEqual(s * f["s0"], 1.0, delta=0.01)


# ------------------------------------------------------------ protobuf GPS

def varint(v):
    v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7f
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def fld(fn, value):
    if isinstance(value, float):
        return varint(fn << 3 | 1) + struct.pack("<d", value)
    if isinstance(value, int):
        return varint(fn << 3) + varint(value)
    return varint(fn << 3 | 2) + varint(len(value)) + value


def gps_basic(lat, lon, alt_mm=None, status=None):
    body = fld(1, fld(2, lat) + fld(3, lon))
    if alt_mm is not None:
        body += fld(2, alt_mm)
    if status is not None:
        body += fld(3, status)
    return body


def record(ts, device):
    return 3, fld(1, fld(1, 7) + fld(2, ts)) + fld(4, device)


class GpsDecode(unittest.TestCase):
    def test_avata_layout_two_tracks_held_fixes(self):
        velocity = fld(1, b"\x00\x00\x80\x3f")             # 3-4-2: f32s, not a fix
        recs = []
        for k in range(10):
            fix = gps_basic(47.0 + (k // 5) * 1e-4, 8.0, alt_mm=-12_345)
            dev = fld(2, velocity) + fld(4, fix)
            recs += [record(20_000 * k, dev), record(20_000 * k + 17, dev)]
        g = upright.gps_from_records(recs)
        self.assertEqual(list(g["t_us"]), [0, 100_000])     # each fix timed at its first record
        self.assertAlmostEqual(g["alt_m"][0], -12.345)
        self.assertAlmostEqual(g["lat"][1], 47.0001)

    def test_osmo_layout_status_only_is_no_fix(self):
        recs = [record(20_000 * k, fld(1, fld(1, 1)) + fld(2, fld(3, 1))) for k in range(5)]
        self.assertIsNone(upright.gps_from_records(recs))

    def test_osmo_layout_with_a_fix(self):
        recs = [record(20_000 * k, fld(2, gps_basic(51.5, -0.1 - k * 1e-5, alt_mm=30_000)))
                for k in range(5)]
        g = upright.gps_from_records(recs)
        self.assertEqual(len(g["t_us"]), 5)
        self.assertAlmostEqual(g["lon"][4], -0.10004)

    def test_zero_position_and_bad_status_are_not_fixes(self):
        recs = [record(0, fld(4, gps_basic(0.0, 0.0))), record(1, fld(4, gps_basic(47.0, 8.0, status=2)))]
        self.assertIsNone(upright.gps_from_records(recs))

    def test_ned_of_a_known_offset(self):
        lat0, lon0 = 47.3, 8.5
        ned = upright.geodetic_to_ned(np.array([lat0 + 0.001]), np.array([lon0]), np.array([400.0]),
                                      (lat0, lon0, 410.0))[0]
        self.assertAlmostEqual(ned[0], 111.2, delta=0.3)  # 0.001 deg of latitude
        self.assertAlmostEqual(ned[1], 0.0, delta=1e-6)
        self.assertAlmostEqual(ned[2], 10.0, delta=0.01)


class Timing(unittest.TestCase):
    def test_same_frames_as_the_selection(self):
        self.assertEqual(list(upright.video_frames(range(6), 10, 50)), [2, 7, 12, 17, 22, 27])
        self.assertEqual(list(upright.video_frames([0, 1, 4], 10, 50, start=1.3)), [67, 72, 87])

    def test_interpolation_takes_the_short_way(self):
        t = np.array([0., 1000.])
        q = np.array([[1., 0, 0, 0], [-math.cos(0.1), 0, 0, -math.sin(0.1)]])   # same as +(cos, 0, 0, sin)
        mid = upright.orientation_at(t, q, [500.])[0]
        self.assertAlmostEqual(abs(mid[3]), math.sin(0.05), places=3)


class FrameTelemetry(unittest.TestCase):
    """selection.json -> video frame -> the stream at that frame's mid-exposure."""

    def stream(self, n=400):
        # 50 fps frames, a 1 kHz stream turning about z at 1 rad/s, 2 ms exposures.
        t = np.arange(n * 20) * 1000.0
        half = t / 1e6 / 2
        q = np.stack([np.cos(half), 0 * half, 0 * half, np.sin(half)], 1)
        return {"frame_t": np.arange(n) * 20_000.0, "exposure": np.full(n, 2000.0), "t": t, "q": q}

    def run_with(self, doc, start=0.0, names=("frame_0000.jpg", "frame_0001.jpg")):
        import json
        import tempfile
        from unittest.mock import patch
        import osv_imu
        with tempfile.TemporaryDirectory() as td:
            sel = Path(td) / "selection.json"
            sel.write_text(json.dumps(doc))
            with patch.object(osv_imu, "decode", return_value=self.stream()):
                return upright.frame_telemetry("clip.OSV", sel, start, list(names))

    def test_candidates_map_to_their_video_frames(self):
        doc = {"fps": 10, "frames": [{"name": "frame_0000.jpg", "candidate": 0},
                                     {"name": "frame_0001.jpg", "candidate": 3}]}
        when, R = self.run_with(doc, start=1.0)
        # Candidates 0 and 3 after a 1 s trim are video frames 52 and 67 at 50 fps.
        self.assertEqual(list(when), [52 * 20_000 + 1000, 67 * 20_000 + 1000])
        yaw = math.atan2(R[1][1, 0], R[1][0, 0])
        self.assertAlmostEqual(yaw, when[1] / 1e6, places=4)

    def test_a_selection_decoded_with_another_trim_is_refused(self):
        doc = {"fps": 10, "frames": [{"name": "frame_0000.jpg", "candidate": 0, "video_frame": 2},
                                     {"name": "frame_0001.jpg", "candidate": 3, "video_frame": 17}]}
        self.run_with(doc)                                  # untrimmed: 2 and 17, as listed
        with self.assertRaisesRegex(ValueError, "different video frames"):
            self.run_with(doc, start=1.0)

    def test_frames_past_the_stream_are_refused(self):
        doc = {"fps": 10, "frames": [{"name": "frame_0000.jpg", "candidate": 0},
                                     {"name": "frame_0001.jpg", "candidate": 500}]}
        with self.assertRaisesRegex(ValueError, "past the"):
            self.run_with(doc)


if __name__ == "__main__":
    unittest.main()
