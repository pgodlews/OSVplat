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


if __name__ == "__main__":
    unittest.main()
