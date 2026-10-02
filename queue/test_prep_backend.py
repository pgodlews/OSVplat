#!/usr/bin/env python3
"""The queue on a host that preps without CUDA (an Apple silicon Mac).

Runs anywhere, no GPU:  python3 queue/test_prep_backend.py
QUEUE_PREP_BACKEND=apple makes any machine behave as the Mac does.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TMP = tempfile.TemporaryDirectory(prefix="queue_prep_backend_")
os.environ["SPLAT_ROOT"] = TMP.name
os.environ["QUEUE_ROOT"] = str(Path(TMP.name) / "queue")
os.environ["QUEUE_GPUS"] = "all"
os.environ["QUEUE_PREP_BACKEND"] = "apple"
os.environ.pop("OSVPLAT_VARIANT", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import HTTPException  # noqa: E402
from app import config, db, gpu, handoff, main, worker  # noqa: E402
from app.jobs import JobConfig  # noqa: E402

OSV = {"name": "c", "input": {"file": "samples/clip.OSV", "quick_hash": "abc"}}


class Device(unittest.TestCase):
    def test_one_device_without_nvidia_smi(self):
        self.assertEqual(config.GPUS, [0])
        with patch.object(gpu, "_nvidia_smi", side_effect=AssertionError("queried")):
            self.assertEqual(gpu.free_gpus(), [0])
            self.assertEqual(gpu.free_gpus(held=[0]), [])
            row = gpu.status()[0]
            self.assertEqual(gpu.compute_caps(), {})
        self.assertTrue(row["schedulable"] and row["probe_ok"] and row["available"])
        self.assertFalse(row["busy_foreign"] or row["unsupported"])

    def test_it_is_a_prep_install(self):
        self.assertEqual(config.IMAGE_VARIANT, "prep")


class Keys(unittest.TestCase):
    def test_every_key_differs_from_the_cuda_one(self):
        cuda = JobConfig.model_validate(OSV)
        apple = JobConfig.model_validate({**OSV, "prep_backend": "apple"})
        self.assertEqual(cuda.prep_backend, "cuda")
        for stage, key in cuda.keys().items():
            self.assertNotEqual(key, apple.keys()[stage], stage)

    def test_a_bundle_carries_the_backend_to_the_train_box(self):
        # The train box computes keys from the bundle's config alone, so the
        # term has to live there and not in handoff.version_terms().
        apple = JobConfig.model_validate({**OSV, "prep_backend": "apple"})
        again = JobConfig.model_validate(apple.model_dump())
        self.assertEqual(again.keys(), apple.keys())
        self.assertNotIn("PREP_APPLE", handoff.version_terms())


class Submission(unittest.TestCase):
    def setUp(self):
        samples = Path(TMP.name) / "samples"
        samples.mkdir(exist_ok=True)
        for name in ("clip.OSV", "clip.mp4"):
            (samples / name).write_bytes(b"\0" * 4096)

    def prepare(self, cfg):
        with patch.object(main, "detect_camera", lambda p: {"recommended_mask": False}):
            return main._prepare(cfg)

    def test_the_host_sets_the_backend(self):
        base = {"name": "c", "input": {"file": "samples/clip.OSV"}}
        self.assertEqual(self.prepare(base).prep_backend, "apple")
        self.assertEqual(self.prepare({**base, "prep_backend": "cuda"}).prep_backend, "apple")

    def test_stitched_input_is_refused(self):
        with self.assertRaises(HTTPException) as cm:
            self.prepare({"name": "c", "input": {"file": "samples/clip.mp4"}})
        self.assertIn(".OSV clips only", cm.exception.detail)

    def test_a_job_that_would_train_is_refused(self):
        cfg = self.prepare({"name": "c", "input": {"file": "samples/clip.OSV"}})
        with self.assertRaises(HTTPException) as cm:
            main._refuse_for_image(cfg)
        self.assertIn("a Mac has no trainer", cm.exception.detail)
        cfg.run_until = "sfm"
        main._refuse_for_image(cfg)


class WrongBackend(unittest.TestCase):
    def test_a_cuda_job_is_not_built_here(self):
        db.init()
        db.conn().execute("DELETE FROM jobs")
        samples = Path(TMP.name) / "samples"
        samples.mkdir(exist_ok=True)
        (samples / "clip.OSV").write_bytes(b"\0" * 4096)
        cfg = JobConfig.model_validate({"name": "c", "input": {"file": "samples/clip.OSV"},
                                        "run_until": "frames"})
        cfg.input.quick_hash = worker.quick_hash(samples / "clip.OSV")
        jid = db.create_job(cfg.name, cfg.model_dump())
        with patch.object(worker, "_build_stage", side_effect=AssertionError("built")):
            worker.run_job(jid, 0)
        row = db.get_job(jid)
        self.assertEqual(row["state"], "failed")
        self.assertIn("prep_backend is 'cuda'", row["error"])


if __name__ == "__main__":
    unittest.main()
