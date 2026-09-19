"""Distance-option API, cache and stage tests; no GPU or running queue required."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "queue"))
TMP = tempfile.TemporaryDirectory(prefix="avata_queue_test_")
os.environ.update(SPLAT_ROOT=TMP.name, QUEUE_ROOT=str(Path(TMP.name) / "queue"), QUEUE_GPUS="")

import avata_motion
from app import estimate, main, stages
from app.jobs import JobConfig, key_of
from fastapi import HTTPException


def config(**select):
    return JobConfig.model_validate({"name": "test", "input": {"file": "samples/flight.OSV"},
                                     "select": {"mode": "distance", **select}})


class DistanceOption(unittest.TestCase):
    def test_old_cache_keys_unchanged_and_distance_forks_downstream(self):
        old = config(mode="window")
        self.assertEqual(old.k_select(), key_of("select", old.k_frames(),
                                               {"mode": "window", "window": 5, "target_panos": 300}))
        new = config()
        self.assertEqual(old.k_frames(), new.k_frames())
        for stage in ("select", "mask", "sfm", "train", "export"):
            self.assertNotEqual(old.keys()[stage], new.keys()[stage])
        self.assertNotEqual(new.k_select(), config(distance_m=2).k_select())
        self.assertNotEqual(new.k_select(), config(max_gap_s=3).k_select())
        self.assertNotEqual(new.k_select(), config(imu=True).k_select())
        self.assertEqual(new.k_select(), config(window=20, target_panos=999).k_select())
        self.assertEqual(old.k_select(), config(mode="window", distance_m=20).k_select())

    def test_config_validation(self):
        for kw in ({"distance_m": 0}, {"distance_m": float("nan")},
                   {"max_gap_s": float("inf")}, {"max_gap_s": .01}):
            with self.assertRaises(ValueError):
                config(**kw)
        with self.assertRaisesRegex(ValueError, "Avata"):
            JobConfig.model_validate({"name": "bad", "input": {"file": "samples/x.mp4"},
                                      "select": {"mode": "distance"}})

    def test_stage_cli_and_report_contract(self):
        cfg = config(distance_m=3, max_gap_s=1, imu=True)
        cfg.input.trim_start = 2.5
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ctx = stages.Ctx(job_id=1, cfg=cfg, gpu=0, keys=cfg.keys())
            ctx.dir = lambda stage: root / stage
            ctx.derived["window"] = 3
            argv = stages.fisheye_select_argv(ctx)
            for arg, val in (("--distance-m", "3.0"), ("--max-gap-s", "1.0"), ("--start", "2.5")):
                self.assertEqual(argv[argv.index(arg) + 1], val)
            self.assertEqual(argv[argv.index("--motion-src") + 1], argv[argv.index("--imu") + 1])
            for lens in (0, 1):
                d = root / "select/images" / f"lens{lens}"
                d.mkdir(parents=True)
                for i in range(8):
                    (d / f"frame_{i:04d}.jpg").touch()
            report = root / "select/selection.json"
            report.write_text('{}')
            with self.assertRaisesRegex(RuntimeError, "distance summary"):
                stages.fisheye_select_finalize(ctx)
            report.write_text(json.dumps({"mode": "distance", "distance": {"spacing_m": 3},
                                          "imu": {"overruled": 0}}))
            info = stages.fisheye_select_finalize(ctx)
            self.assertIsNone(info["window"])
            with self.assertRaisesRegex(RuntimeError, "motion report"):
                stages.fisheye_select_verify(ctx, info)
            (root / "select/motion_path.json").write_text('{}')
            stages.fisheye_select_verify(ctx, info)

    def test_prepare_refuses_missing_motion_even_if_mask_explicit(self):
        samples = Path(TMP.name) / "samples"
        samples.mkdir(exist_ok=True)
        (samples / "flight.OSV").write_bytes(b"fixture")
        cfg = config().model_dump()
        with patch.object(avata_motion, "load", side_effect=ValueError("No Avata velocity")):
            with self.assertRaises(HTTPException) as cm:
                main._prepare(cfg)
            self.assertEqual(cm.exception.status_code, 400)
        with patch.object(avata_motion, "load", return_value={"fps": 50}):
            self.assertEqual(main._prepare(cfg).select.mode, "distance")
            cfg["frames"]["fps"] = 60
            with self.assertRaises(HTTPException):
                main._prepare(cfg)

    def test_estimate_uses_motion_and_respects_trim(self):
        data = {"fps": 50., "t": [i / 50 for i in range(1000)], "velocity": [(10., 0., 0.)] * 1000}
        cfg = config(distance_m=5)
        cfg.input.trim_start, cfg.input.trim_end = 2., 12.
        with patch.object(avata_motion, "load", return_value=data), patch.object(estimate, "constants", return_value=estimate.SEED.copy()):
            result = estimate.estimate(cfg, 20)
            self.assertEqual(result["_assumptions"]["panos"], 20)
            self.assertAlmostEqual(result["_assumptions"]["path_length_m"], 99.)
            cfg.select.distance_m = .1
            self.assertIn("warning", estimate.estimate(cfg, 20)["_assumptions"])

    def test_estimate_api_returns_useful_error_for_invalid_motion(self):
        with patch.object(main, "_prepare", return_value=config()), \
             patch.object(main, "input_path", return_value=Path(TMP.name)), \
             patch.object(main, "ffprobe", return_value={"duration": 10}), \
             patch.object(main, "quick_hash", return_value="hash"), \
             patch.object(main, "_cache_state", return_value={}), \
             patch.object(estimate, "estimate", side_effect=ValueError("Candidate frames extend beyond telemetry")):
            with self.assertRaises(HTTPException) as cm:
                main.api_estimate(main.EstimateReq(config=config().model_dump()))
            self.assertEqual(cm.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
