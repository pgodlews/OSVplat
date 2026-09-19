"""CPU tests for Avata distance selection, including real CLI output on a tiny fixture."""
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import avata_motion as motion

spec = importlib.util.spec_from_file_location("fisheye_frames", HERE / "80_fisheye_frames.py")
frames = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frames)


def varint(n):
    b = bytearray()
    while n > 127:
        b.append((n & 127) | 128)
        n >>= 7
    return bytes(b) + bytes([n])


def field(n, value):
    if isinstance(value, int):
        return varint(n << 3) + varint(value)
    if isinstance(value, float):
        return varint((n << 3) | 5) + struct.pack("<f", value)
    return varint((n << 3) | 2) + varint(len(value)) + value


def fixture(velocities, duplicate=True, camera=b"dvtm_AVATA360.proto", missing=None, second=None):
    raw = field(1, field(1, field(1, camera)))
    for i, v in enumerate(velocities):
        for offset in ([0, 17] if duplicate else [0]):
            header = (field(1, i) if i else b"") + field(2, 314000000 + i * 20000 + offset)
            if offset and second and i in second:
                v = second[i]
            vel = b"".join(field(a + 1, float(x)) for a, x in enumerate(v) if x != 0)
            raw += field(3, field(1, header) + (field(4, field(2, vel)) if i != missing else b""))
    return raw


class Telemetry(unittest.TestCase):
    def test_duplicate_tracks_and_omitted_zero_scalars(self):
        m = motion.decode_records(fixture([(0, 0, 0), (3, 4, 0), (3, 4, 0)]))
        self.assertEqual(m["fps"], 50)
        self.assertEqual(m["velocity"], [(0, 0, 0), (3, 4, 0), (3, 4, 0)])
        np.testing.assert_allclose(m["t"], [0, .02, .04])

    def test_duplicate_track_velocity_update_is_averaged_but_large_conflict_refused(self):
        m = motion.decode_records(fixture([(1, 0, 0)] * 3, second={1: (1.25, 0, 0)}))
        self.assertAlmostEqual(m["velocity"][1][0], 1.125, places=6)
        self.assertEqual(m["velocity"][2], (1, 0, 0))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            motion.decode_records(fixture([(1, 0, 0)] * 3, second={1: (2, 0, 0)}))

    def test_reject_wrong_camera_missing_and_nan_velocity(self):
        for raw in (fixture([(1, 0, 0)] * 5, camera=b"dvtm_oq101.proto"),
                    fixture([(1, 0, 0)] * 5, missing=2),
                    fixture([(float("nan"), 0, 0)] * 5)):
            with self.assertRaises(ValueError):
                motion.decode_records(raw)

    def test_reject_missing_frame_without_silently_shifting_alignment(self):
        raw = fixture([(1, 0, 0)] * 5, duplicate=False)
        from osv_meta import records
        parts = list(records(raw))
        with self.assertRaisesRegex(ValueError, "missing or duplicated"):
            motion.decode_records(b"".join(field(fn, b) for i, (fn, b) in enumerate(parts) if i != 3))

    def test_integrates_speed_not_net_displacement_and_filters_hover_noise(self):
        d, p, speed = motion.integrate([0, 1, 2, 3], [(1, 0, 0), (1, 0, 0), (-1, 0, 0), (-1, 0, 0)])
        self.assertEqual(d[-1], 3)
        self.assertEqual(p[-1], (0, 0, 0))
        self.assertEqual(motion.integrate([0, 10], [(.05, .05, .05)] * 2)[0][-1], 0)

    def test_trim_timing_matches_existing_ffmpeg_mapping(self):
        m = motion.decode_records(fixture([(3, 4, 0)] * 200))
        p = motion.candidates(m, 10, 10, start=1.3)
        self.assertEqual(p["video_frame"], frames.video_frames(10, 10, 50, 1.3).tolist())
        self.assertEqual(p["video_frame"][0], 67)
        self.assertAlmostEqual(p["t_sec"][0], 1.34)
        self.assertAlmostEqual(p["distance_m"][-1], 4.5)
        self.assertEqual(p["position_ned_m"][0], (0., 0., 0.))
        with self.assertRaises(ValueError):
            motion.candidates(m, 100, 10, 3)
        with self.assertRaises(ValueError):
            motion.candidates(m, 100, 100, 0)


class Selection(unittest.TestCase):
    def path(self, speed):
        t = np.arange(len(speed)) / 10
        d, p, s = motion.integrate(t, [(v, 0, 0) for v in speed])
        return {"t_sec": list(t), "distance_m": d}

    def test_regular_distance_despite_speed_change(self):
        p = self.path([2.] * 100 + [10.] * 100)
        groups = motion.groups(p, 5, 10)
        sel = frames.pick_distance(np.ones(200), p, groups, 5)
        # Faster movement gets roughly five times as many shots per second.
        slow = sum(i < 100 for i in sel)
        fast = sum(i >= 100 for i in sel)
        self.assertGreater(fast, slow * 4)
        gaps = np.diff(np.asarray(p["distance_m"])[sel])
        self.assertLess(float(gaps.max()), 7)

    def test_quality_and_blur_within_distance_band(self):
        p = {"distance_m": [0, 3, 4, 5, 6, 7, 9], "t_sec": [i / 10 for i in range(7)]}
        scores = np.array([100., 2., 3., 8., 4., 2., 100.])
        self.assertEqual(frames.pick_distance(scores, p, [list(range(7))], 10), [3])
        blur = np.array([0., 0., 0., 10., 0., 0., 0.])
        self.assertEqual(frames.pick_distance(scores, p, [list(range(7))], 10, blur), [4])
        self.assertEqual(frames.pick_distance(np.full(7, -1.), p, [list(range(7))], 10), [])

    def test_slow_time_splits_rank_sharpness_and_fast_flight_keeps_spacing(self):
        scores = np.random.default_rng(3).uniform(0, 100, 600)
        p = self.path([.5] * 600)
        groups = motion.groups(p, 5, 2)
        sel = frames.pick_distance(scores, p, groups, 5)
        # Every chunk ranks all its candidates (<0.5 m apart): the sharpest wins.
        self.assertEqual(sel, [max(g, key=lambda i: scores[i]) for g in groups])
        p = self.path([18.] * 600)
        sel = frames.pick_distance(scores, p, motion.groups(p, 5, 2), 5)
        d = np.asarray(p["distance_m"])[sel]
        centre = (np.floor(d / 5 + 1e-9) + .5) * 5
        # Nearest candidate is <= 0.9 m off; the tolerance adds at most 1 m.
        self.assertLessEqual(float(np.abs(d - centre).max()), 1.9 + 1e-6)

    def test_hover_retains_time_coverage_and_candidates_are_unique(self):
        rng = np.random.default_rng(11)
        for speed in ([0.] * 300, list(rng.uniform(0, 20, 300))):
            p = self.path(speed)
            groups = motion.groups(p, 5, 2)
            scores = rng.uniform(0, 100, 300)
            sel = frames.pick_distance(scores, p, groups, 5)
            self.assertEqual(sel, sorted(set(sel)))
            self.assertLessEqual(max(np.diff(np.asarray(p["t_sec"])[sel])), 2 + 1e-6)
            self.assertEqual([i for g in groups for i in g], list(range(300)))
        self.assertLess(len(motion.groups(self.path([0.] * 300), 5, 2)), 40)

    def test_cli_writes_synchronized_pairs_and_motion_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = fixture([(5, 0, 0)] * 300)
            mdat = struct.pack(">I4s", len(raw) + 8, b"mdat") + raw
            src = root / "flight.OSV"
            src.write_bytes(struct.pack(">I4s", len(mdat) + 8, b"camd") + mdat)
            for lens in (0, 1):
                d = root / "cand" / f"lens{lens}"
                d.mkdir(parents=True)
                for i in range(50):
                    img = np.random.default_rng(i + lens).integers(0, 255, (64, 64), dtype=np.uint8)
                    cv2.imwrite(str(d / f"{i + 1:05d}.jpg"), img)
            subprocess.run([sys.executable, str(HERE / "80_fisheye_frames.py"), "--select-only",
                            str(root / "cand"), str(root / "selected"), "--distance-m", "2",
                            "--motion-src", str(src), "--start", "0.5"], check=True, capture_output=True)
            report = json.loads((root / "selected/selection.json").read_text())
            self.assertEqual(report["mode"], "distance")
            self.assertGreater(len(report["frames"]), 8)
            self.assertTrue((root / "selected/motion_path.json").is_file())
            for fr in report["frames"]:
                self.assertAlmostEqual(fr["t_sec"], (27 + 5 * fr["candidate"]) / 50)
                for lens in (0, 1):
                    selected = root / "selected/images" / f"lens{lens}" / fr["name"]
                    original = root / "cand" / f"lens{lens}" / f"{fr['candidate'] + 1:05d}.jpg"
                    self.assertTrue(os.path.samefile(selected, original))


if __name__ == "__main__":
    unittest.main()
