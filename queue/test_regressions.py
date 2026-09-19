#!/usr/bin/env python3
"""Review regressions. Run with the service venv plus ffmpeg; no GPU needed."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

TMP = tempfile.TemporaryDirectory(prefix="queue_regressions_")
os.environ["SPLAT_ROOT"] = TMP.name
os.environ["QUEUE_ROOT"] = str(Path(TMP.name) / "queue")
os.environ["QUEUE_GPUS"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import db, estimate, main, progress, worker
from app.jobs import JobConfig
from app.stages import Ctx, fisheye_frames_argv


def job(state, seconds):
    return {"state": state, "eta": {"remaining": seconds, "plan_total": seconds}}


class ReviewRegressions(unittest.TestCase):
    def setUp(self):
        db.init()
        db.conn().execute("DELETE FROM jobs")
        self.cfg = JobConfig.model_validate({"name": "review", "input": {"file": "clip.mp4"}})

    def test_trim_bounds_both_lenses(self):
        with tempfile.TemporaryDirectory(dir=TMP.name) as td:
            root = Path(td)
            src = root / "dual.mkv"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=10:d=3",
                "-f", "lavfi", "-i", "testsrc2=s=64x64:r=10:d=3",
                "-map", "0:v", "-map", "1:v", "-c:v", "ffv1", str(src)], check=True)
            src = src.rename(src.with_suffix(".OSV"))
            cfg = self.cfg.model_copy(deep=True)
            cfg.input.file = str(src.relative_to(TMP.name))
            cfg.input.trim_start, cfg.input.trim_end = .5, 1.5
            ctx = Ctx(job_id=1, cfg=cfg, gpu=0, keys=cfg.keys())
            ctx.dir = lambda stage: root / "frames"
            argv = fisheye_frames_argv(ctx)
            i = argv.index("-hwaccel")
            del argv[i:i + 2]  # Exercise the production command with CPU decoding.
            subprocess.run(argv, check=True, capture_output=True)
            self.assertEqual([len(list((ctx.dir("frames") / f"lens{i}").glob("*.jpg")))
                              for i in (0, 1)], [10, 10])

    def test_concurrent_thumbnail_publication(self):
        src = Path(TMP.name) / "clip.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=s=64x64:r=10:d=1", "-c:v", "mpeg4", str(src)], check=True)
        cfg = main._prepare(self.cfg.model_dump())
        jid = db.create_job(cfg.name, cfg.model_dump())
        run = subprocess.run
        barrier = threading.Barrier(2)

        def concurrent_run(*args, **kwargs):
            barrier.wait(timeout=10)
            result = run(*args, **kwargs)
            barrier.wait(timeout=10)  # Both writers finish before either publishes.
            return result

        with tempfile.TemporaryDirectory(dir=TMP.name) as thumbs:
            with patch.object(main, "THUMB_ROOT", Path(thumbs)), \
                    patch.object(main.subprocess, "run", side_effect=concurrent_run):
                with ThreadPoolExecutor(2) as ex:
                    calls = [ex.submit(main.api_thumb, jid, w=480) for _ in range(2)]
                    responses = [c.result(timeout=15) for c in calls]
            self.assertEqual([r.status_code for r in responses], [200, 200])
            files = list(Path(thumbs).iterdir())
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].read_bytes().startswith(b"\xff\xd8"))

    def test_concurrency_reduction_retires_slots(self):
        jobs = [job("running", 100), job("running", 200),
                job("queued", 50), job("queued", 50)]
        result = progress.queue_eta(jobs, 1, now=1000)
        self.assertEqual([j["eta"]["starts_in"] for j in jobs[2:]], [200, 250])
        self.assertEqual(result["drains_in"], 300)

    def test_blocked_scheduling_has_no_deadline(self):
        for capacity, paused in [(0, False), (2, True)]:
            with self.subTest(capacity=capacity, paused=paused):
                jobs = [job("running", 100), job("queued", 50)]
                result = progress.queue_eta(jobs, capacity, now=1000, paused=paused)
                self.assertIsNone(result["drains_in"])
                for field in ("starts_in", "remaining", "finish_at"):
                    self.assertIsNone(jobs[1]["eta"][field])
                self.assertEqual(jobs[0]["eta"]["remaining"], 100)
        self.assertEqual(progress.queue_eta([job("running", 100)], 0, now=1000)["drains_in"], 100)
        self.assertEqual(progress.queue_eta([], 0, now=1000)["drains_in"], 0)

    def test_zero_capacity_is_cached(self):
        busy = [{"schedulable": True, "probe_ok": True, "busy_foreign": True}]
        with patch.object(worker, "_capacity_cache", (0.0, 0)), \
                patch.object(worker.gpu, "status", return_value=busy) as probe:
            self.assertEqual(worker.schedulable_capacity(), 0)
            self.assertEqual(worker.schedulable_capacity(), 0)
            self.assertEqual(probe.call_count, 1)

    def test_pagination_keeps_preceding_work(self):
        for state, seconds in [("running", 100), ("queued", 50), ("queued", 50)]:
            jid = db.create_job(self.cfg.name, self.cfg.model_dump())
            db.set_job_state(jid, state)
            db.set_plan(jid, {"train": seconds, "total": seconds})
            db.upsert_stage(jid, "train", self.cfg.k_train(), "pending")
        with patch.object(worker, "schedulable_capacity", return_value=1), \
                patch.object(worker, "paused", return_value=False):
            full = main.api_jobs(limit=10, offset=0)
            page = main.api_jobs(limit=1, offset=2)
        self.assertEqual(page[0]["id"], full[2]["id"])
        self.assertEqual(page[0]["eta"]["starts_in"], 150)
        self.assertEqual(page[0]["eta"]["starts_in"], full[2]["eta"]["starts_in"])
        with patch.object(worker, "paused", return_value=True):
            self.assertIsNone(main.api_jobs(limit=1, offset=2)[0]["eta"]["starts_in"])

    def test_history_limit_is_per_pipeline(self):
        for ext, seconds, count in [("mp4", 2000, 1), ("OSV", 3000, 20)]:
            cfg = self.cfg.model_copy(deep=True)
            cfg.input.file = f"clip.{ext}"
            for _ in range(count):
                jid = db.create_job(cfg.name, cfg.model_dump())
                db.set_job_state(jid, "done")
                db.put_metric(jid, "train_seconds", seconds)
                db.put_metric(jid, "final_step", 30000)
        self.assertEqual(estimate._fitted_rate(False), 15)
        self.assertEqual(estimate._fitted_rate(True), 10)


class CacheGcPreview(unittest.TestCase):
    def test_dry_run_stops_once_enough_would_be_freed(self):
        from app import retention
        db.init()
        db.conn().execute("DELETE FROM cache")
        for i in range(5):
            d = Path(TMP.name) / "gc" / f"e{i}"
            d.mkdir(parents=True, exist_ok=True)
            db.cache_put(f"gc{i}", "frames", str(d), 10 * 2**30)   # 10 GiB each
        free = 100 * 2**30
        with patch.object(retention, "free_bytes", lambda: free), \
             patch.object(retention, "protected_keys", lambda: set()):
            out = retention.gc_cache(target_free=free + 15 * 2**30, budget=None,
                                     dry_run=True)
        # 15 GiB short, 10 GiB entries: two deletions suffice, not all five.
        self.assertEqual(out["n_evicted"], 2, out)


class SubmitValidation(unittest.TestCase):
    """Requests that must be refused at submit, not an hour into the run."""

    def test_camera_default_masking_checks_backend_availability(self):
        from fastapi import HTTPException
        samples = Path(TMP.name) / "samples"
        samples.mkdir(exist_ok=True)
        (samples / "osmo.OSV").write_bytes(b"\0" * 4096)
        with patch.object(main, "detect_camera",
                          lambda p: {"camera": "osmo360", "recommended_mask": True}):
            with self.assertRaises(HTTPException) as cm:
                main._prepare({"name": "m", "input": {"file": "samples/osmo.OSV"},
                               "mask": {"backend": "sam3"}})
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("not available", cm.exception.detail)

    def test_perspective_render_refuses_masks_during_sfm(self):
        base = {"name": "p", "input": {"file": "clip.mp4"},
                "sfm": {"render": "perspective_overlapping"}}
        with self.assertRaises(ValueError):
            JobConfig.model_validate({**base, "mask": {"enabled": True}})
        JobConfig.model_validate({**base, "mask": {"enabled": True, "use_for_sfm": False}})

    def test_sweep_axis_through_a_null_is_a_400(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as cm:
            main._set_dotted({"train": None}, "train.sh_degree", 3)
        self.assertEqual(cm.exception.status_code, 400)


class FinalizerRefusals(unittest.TestCase):
    """Stage results that must fail rather than be cached (review findings)."""

    def ctx(self, **cfg):
        c = JobConfig.model_validate({"name": "f", "input": {"file": "clip.mp4"}, **cfg})
        return Ctx(job_id=1, cfg=c, gpu=0, keys=c.keys())

    def test_stitched_sfm_fragment_is_refused(self):
        import json as _json
        from app import stages
        ctx = self.ctx()
        ctx.derived["n_panos"] = 100
        summary = {"num_reg_frames": 2, "num_points3D": 50, "mean_reproj": 1.0,
                   "models": [{"name": "0"}], "chosen": "0"}

        class Proc:
            returncode = 0
            stdout = _json.dumps(summary) + "\n"
            stderr = ""
        with patch.object(stages.subprocess, "run", return_value=Proc()):
            with self.assertRaisesRegex(RuntimeError, "registered only 2 of 100"):
                stages.sfm_finalize(ctx)

    def test_exports_from_different_checkpoints_are_refused(self):
        from app import stages
        ctx = self.ctx(export={"formats": ["ply", "sog"]})
        out = ctx.dir("train")
        out.mkdir(parents=True, exist_ok=True)
        (out / "splat_30000.ply").write_bytes(b"x" * 10)
        (out / "splat_1000.sog").write_bytes(b"x" * 10)
        with self.assertRaisesRegex(RuntimeError, "different checkpoints"):
            stages.train_finalize(ctx)


class MaskBackendRegistry(unittest.TestCase):
    """Backends come from app/mask_backends.py; weights gate availability."""

    def cfg(self, **mask):
        return JobConfig.model_validate({"name": "m", "input": {"file": "clip.mp4"},
                                         "mask": {"enabled": True, **mask}})

    def test_unknown_backend_refused(self):
        with self.assertRaises(ValueError):
            self.cfg(backend="sam9")

    def test_prompts_need_a_prompt_backend(self):
        with self.assertRaises(ValueError):
            self.cfg(backend="maskrcnn", prompts=["dog"])
        self.cfg(backend="sam3", prompts=["person", "dog"])

    def test_availability_follows_weights(self):
        from app import mask_backends
        with tempfile.TemporaryDirectory(dir=TMP.name) as td:
            root = Path(td)
            av = {b["name"]: b for b in mask_backends.availability(root)}
            self.assertTrue(av["maskrcnn"]["available"])
            self.assertFalse(av["sam3"]["available"])
            self.assertIn("huggingface.co/facebook/sam3", av["sam3"]["reason"])
            (root / "sam3").mkdir()
            (root / "sam3" / "config.json").write_text("{}")
            av = {b["name"]: b for b in mask_backends.availability(root)}
            self.assertTrue(av["sam3"]["available"])

    def test_submit_refuses_missing_weights(self):
        from fastapi import HTTPException
        samples = Path(TMP.name) / "samples"
        samples.mkdir(exist_ok=True)
        (samples / "clip.mp4").write_bytes(b"\0" * 4096)
        with self.assertRaises(HTTPException) as cm:
            main._prepare({"name": "m", "input": {"file": "samples/clip.mp4"},
                           "mask": {"enabled": True, "backend": "sam3"}})
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("not available", cm.exception.detail)

    def test_sam3_argv_points_at_models_dir(self):
        from app.config import MODELS_ROOT
        from app.stages import _masker_options
        opts = _masker_options(self.cfg(backend="sam3", prompts=["person"]).mask)
        self.assertEqual(opts[opts.index("--model") + 1], str(MODELS_ROOT / "sam3"))
        self.assertIn("--prompts", opts)
        opts = _masker_options(self.cfg(backend="maskrcnn").mask)
        self.assertNotIn("--model", opts)
        self.assertNotIn("--prompts", opts)


class TrainerFlagConflicts(unittest.TestCase):
    def test_exposure_correction_excludes_bilateral_grid(self):
        base = {"name": "t", "input": {"file": "clip.mp4"}}
        with self.assertRaises(ValueError):
            JobConfig.model_validate({**base, "train": {
                "exposure_correction": True, "bilateral_grid": True}})
        JobConfig.model_validate({**base, "train": {"exposure_correction": True,
                                  "enable_mip": True}})
        with self.assertRaises(ValueError):     # crashes the pinned LichtFeld
            JobConfig.model_validate({**base, "train": {"background_improvements": True}})
        JobConfig.model_validate({**base, "train": {"bilateral_grid": True}})

    def test_igs_plus_needs_a_pinhole_reconstruction(self):
        for inp, sfm in [("clip.OSV", {}), ("clip.mp4", {}),
                         ("clip.mp4", {"render": "spherical"})]:
            with self.assertRaises(ValueError, msg=(inp, sfm)):
                JobConfig.model_validate({"name": "t", "input": {"file": inp},
                                          "sfm": sfm, "train": {"strategy": "igs+"}})
        JobConfig.model_validate({"name": "t", "input": {"file": "clip.mp4"},
                                  "sfm": {"render": "perspective_overlapping"},
                                  "train": {"strategy": "igs+"}})


class WebUI(unittest.TestCase):
    def test_ui_script_parses(self):
        """A syntax error anywhere in the inline script kills the whole UI while
        every API test still passes; one shipped that way for a day."""
        import re
        import shutil
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        html = (Path(__file__).parent / "app/static/index.html").read_text()
        js = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", html, re.S))
        with tempfile.NamedTemporaryFile("w", suffix=".js", dir=TMP.name) as fh:
            fh.write(js)
            fh.flush()
            r = subprocess.run([node, "--check", fh.name], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
