#!/usr/bin/env python3
"""CPU regressions for the standalone rig pipeline (numpy and pycolmap)."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pycolmap
import upright
from osmo_fisheye import colmap_params, theta_d

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fisheye_sfm", HERE / "82_fisheye_sfm.py")
sfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sfm)


class FisheyeRegressions(unittest.TestCase):
    def test_failed_rerun_cannot_reuse_previous_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out, images, masks = root / "sfm", root / "images", root / "masks"
            stale = out / "sparse" / "1"
            stale.mkdir(parents=True)
            pycolmap.synthesize_dataset(pycolmap.SyntheticDatasetOptions()).write_binary(stale)
            for i in (0, 1):
                (images / f"lens{i}").mkdir(parents=True)
                (masks / f"lens{i}").mkdir(parents=True)
                (images / f"lens{i}" / "frame_0000.jpg").touch()
                (masks / f"lens{i}" / "frame_0000.jpg.png").touch()
            lens = dict(fx=1000, fy=1000, cx=1920, cy=1920, k1=0, k2=0, k3=0, k4=0)
            args = ["82_fisheye_sfm.py", "--calib", "unused.json", "--images", str(images),
                    "--masks", str(masks), "--out", str(out)]
            # An empty database exercises a real unsuccessful COLMAP mapping.
            with patch.object(sys, "argv", args), patch.object(sfm, "lenses", return_value=[lens, lens]), \
                    patch.object(sfm, "rig_rotation", return_value=np.eye(3)), \
                    patch.object(pycolmap, "extract_features"), patch.object(pycolmap, "match_sequential"), \
                    patch.object(pycolmap, "apply_rig_config"):
                with self.assertRaisesRegex(SystemExit, "no reconstruction models"):
                    sfm.main()
            self.assertFalse(stale.exists())
            self.assertEqual(list((out / "sparse").iterdir()), [])

    def test_renderer_coefficients_preserve_higher_order_distortion(self):
        lens = dict(fx=1030, fy=1030, cx=1920, cy=1920,
                    k1=0, k2=0, k3=0, k4=0, k5=.0008)
        params = colmap_params(lens)
        refit = dict(lens)
        refit.pop("k5")
        refit.update(zip(("k1", "k2", "k3", "k4"), params[4:]))
        angles = np.linspace(0, np.deg2rad(90), 1000)
        error = np.abs(theta_d(angles, lens) - theta_d(angles, refit)) * lens["fx"]
        self.assertLess(error.max(), .4)
        unchanged = dict(lens)
        unchanged.pop("k5")
        self.assertEqual(colmap_params(unchanged)[4:], [0, 0, 0, 0])


def rig_model(root, frames=30):
    """A synthetic two-lens rig model named the way 85/upright.py expect."""
    o = pycolmap.SyntheticDatasetOptions()
    o.num_rigs, o.num_cameras_per_rig, o.num_frames_per_rig, o.num_points3D = 1, 2, frames, 200
    rec = pycolmap.synthesize_dataset(o)
    order = sorted(rec.images, key=lambda i: rec.images[i].name)
    for i in order:
        img = rec.images[i]
        k = int(img.name.split("frame")[1].split(".")[0])
        img.name = f"lens{img.camera_id - 1}/frame_{k:04d}.jpg"
    root.mkdir(parents=True)
    rec.write_binary(str(root))
    return rec


class UprightModel(unittest.TestCase):
    """upright.align_model on a real pycolmap rig: the transform, the rig baseline, the fail-safe."""

    def telemetry(self, rec, W):
        lens0 = sorted((rec.images[i].name, i) for i in rec.reg_image_ids()
                       if rec.images[i].name.startswith("lens0/"))
        X = upright.quat_matrix([0.3, -0.5, 0.7, 0.2])
        R_wb = np.array([W @ np.asarray(rec.images[i].cam_from_world().rotation.matrix()).T @ X
                         for _, i in lens0])
        centres = np.array([rec.images[i].projection_center() for _, i in lens0])
        return np.arange(len(lens0)) * 5e5, R_wb, centres

    def test_levels_the_model_and_scales_the_rig(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rec = rig_model(root / "model")
            W = upright.quat_matrix([0.9, 0.1, -0.3, 0.2])        # stream world from SfM world
            when, R_wb, centres = self.telemetry(rec, W)
            ned = 40.0 * centres @ W.T                              # GPS: 40 m per SfM unit
            gps = {"t_us": when, "lat": 47.3 + ned[:, 0] / 111_200, "alt_m": 400 - ned[:, 2],
                   "lon": 8.5 + ned[:, 1] / (111_200 * np.cos(np.radians(47.3)))}
            with patch.object(upright, "frame_telemetry", return_value=(when, R_wb)), \
                    patch.object(upright, "gps_track", return_value=gps):
                rep = upright.align_model(root / "model", root / "aligned", "clip.OSV", "selection.json")
            self.assertTrue(rep["upright"], rep.get("reason"))
            self.assertTrue(rep["metric"], rep.get("gps_skipped"))
            s = rep["transform"]["scale"]
            self.assertAlmostEqual(s, 40.0, delta=0.2)
            out = pycolmap.Reconstruction(str(root / "aligned"))
            down_sfm = W.T @ np.array([0., 0., 1.])
            R = np.asarray(rep["transform"]["rotation"])
            self.assertGreater((R @ down_sfm)[1], 0.9999)             # down is +y
            for i in out.reg_image_ids():                             # lens 1 too: the baseline scaled
                c_old = np.asarray(rec.images[i].projection_center())
                c_new = np.asarray(out.images[i].projection_center())
                self.assertLess(np.linalg.norm(s * R @ c_old + rep["transform"]["translation"] - c_new), 1e-5)

    def test_an_unreadable_stream_leaves_the_model_alone(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rig_model(root / "model")
            with patch.object(upright, "frame_telemetry", side_effect=SystemExit("no orientation block")):
                rep = upright.align_model(root / "model", root / "aligned", "clip.OSV", "selection.json")
            self.assertFalse(rep["upright"])
            self.assertIn("no orientation block", rep["reason"])
            self.assertFalse((root / "aligned").exists())


if __name__ == "__main__":
    unittest.main()
