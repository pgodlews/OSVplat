#!/usr/bin/env python3
"""CPU checks for IMU-aware rig frame selection (numpy and OpenCV, no pycolmap).

Runs anywhere with numpy and OpenCV:  python3 scripts/test_imu_select.py
The last class decodes CAM_20260813121712_0040_D.OSV from the repo root and is
skipped where that clip is not present.
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import osv_imu  # noqa: E402
import osv_meta  # noqa: E402

spec = importlib.util.spec_from_file_location("fisheye_frames", HERE / "80_fisheye_frames.py")
frames = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frames)

OSMO = HERE.parent / "CAM_20260813121712_0040_D.OSV"


class CandidateTiming(unittest.TestCase):
    def test_fps_filter_keeps_the_last_frame_of_each_tick(self):
        # What the box's ffmpeg produced, matched pixel-exactly against a full-rate decode.
        self.assertEqual(list(frames.video_frames(6, 10, 50)), [2, 7, 12, 17, 22, 27])
        self.assertEqual(list(frames.video_frames(5, 10, 50, start=1.3)), [67, 72, 77, 82, 87])
        ntsc = 30000 / 1001
        self.assertEqual(list(frames.video_frames(10, 10, ntsc)), [1, 4, 7, 10, 13, 16, 19, 22, 25, 28])
        self.assertEqual(list(frames.video_frames(6, 10, ntsc, start=1.25)), [38, 41, 44, 47, 50, 53])

    def test_a_twenty_minute_clip_does_not_drift(self):
        self.assertEqual(frames.video_frames(12000, 10, 50)[-1], 5 * 11999 + 2)


class Rates(unittest.TestCase):
    def test_constant_spin_reads_back_its_rate(self):
        t = np.arange(0, 2e6, 1000.0)
        w = np.tile([0.0, 0.0, 1.0], (len(t), 1))
        r = osv_imu.exposure_rates(t, w, np.array([5e5, 1e6]), np.array([10000.0, 200.0]))
        np.testing.assert_allclose(r, 1.0)

    def test_the_window_is_the_exposure_not_its_neighbours(self):
        t = np.arange(0, 1e6, 1000.0)
        w = np.zeros((len(t), 3))
        w[(t >= 400e3) & (t < 410e3), 0] = 2.0
        inside, before = osv_imu.exposure_rates(t, w, np.array([400e3, 380e3]), np.array([10000.0, 10000.0]))
        self.assertAlmostEqual(inside, 2.0)
        self.assertEqual(before, 0.0)

    def test_body_rates_of_a_steady_spin(self):
        t = np.arange(0, 1e6, 997.0)
        ang = 0.5 * t / 1e6
        q = np.stack([np.cos(ang / 2), 0 * ang, np.sin(ang / 2), 0 * ang], 1)
        np.testing.assert_allclose(osv_imu.body_rates(t, q)[1:-1], [[0, 0.5, 0]] * (len(t) - 2), atol=1e-6)


class Pick(unittest.TestCase):
    rig = np.array([5., 9., 7., 3., 8., 6., -1., -1., -1., 4., 2.])

    def test_without_blur_it_is_the_laplacian_argmax(self):
        self.assertEqual(frames.pick(self.rig, 3), [1, 4, 9])

    def test_blur_within_tolerance_changes_nothing(self):
        blur = np.array([.1, .6, .2, .9, .05, .3, 0, 0, 0, .4, .1])
        self.assertEqual(frames.pick(self.rig, 3, blur, 1.0), [1, 4, 9])

    def test_a_blur_gap_past_tolerance_overrules_the_laplacian(self):
        blur = np.array([.8, 4.5, 1.2, .9, 3.0, 2.5, 0, 0, 0, 6.0, 1.0])
        self.assertEqual(frames.pick(self.rig, 3, blur, 1.0), [2, 3, 10])


@unittest.skipUnless(OSMO.is_file(), f"{OSMO.name} is not in the repo root")
class OsmoClip(unittest.TestCase):
    def test_decode_times_every_frame(self):
        s = osv_imu.decode(str(OSMO))
        self.assertEqual((len(s["frame_t"]), len(s["t"])), (3154, 63160))
        self.assertTrue(np.isfinite(s["exposure"]).all())
        self.assertAlmostEqual(float(np.degrees(np.median(np.linalg.norm(s["w"], axis=1)))), 21.2, delta=0.5)

    def test_the_command_line_still_writes_both_csvs(self):
        with tempfile.TemporaryDirectory() as td:
            osv_imu.main(str(OSMO), td)
            with open(Path(td) / "frames.csv") as fh:
                self.assertEqual(sum(1 for _ in fh), 1 + 3154)
            with open(Path(td) / "imu.csv") as fh:
                self.assertEqual(sum(1 for _ in fh), 1 + 63160)

    def test_daylight_garden_walk_predicts_no_smear_worth_a_pixel(self):
        with tempfile.TemporaryDirectory() as td:
            osv_meta.main(str(OSMO), td)
            blur, info = frames.predicted_blur(OSMO, Path(td) / "calibration.json", 310, 10)
        self.assertEqual(len(blur), 310)
        self.assertEqual(info["video_frame"][-1], 5 * 309 + 2)
        self.assertLess(float(blur.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
