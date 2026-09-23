#!/usr/bin/env python3
"""Host benchmark service side: the machine hold, the run record. No GPU.

The workload itself (scripts/benchmark.py) is tested by
scripts/test_benchmark.py, which needs venv_gs.

    python3 queue/test_benchmark.py
"""
import json
import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

TMP = tempfile.TemporaryDirectory(prefix="queue_benchmark_")
# Assigned before any app import: config.py reads these at import time.
os.environ["SPLAT_ROOT"] = TMP.name
os.environ["QUEUE_ROOT"] = str(Path(TMP.name) / "queue")
os.environ["QUEUE_GPUS"] = ""
os.environ.pop("QUEUE_BENCHMARK", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import benchmark, db, main, worker                   # noqa: E402

FAKE = Path(TMP.name) / "fake_benchmark.py"


def fake_script(body: str) -> None:
    """A stand-in for scripts/benchmark.py: same arguments, instant."""
    FAKE.write_text(
        "import argparse, json, sys, time\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--out'); ap.add_argument('--scratch')\n"
        "ap.add_argument('--disk-gb'); ap.add_argument('--cpus')\n"
        "ap.add_argument('--no-gpu', action='store_true')\n"
        "a = ap.parse_args()\n" + body)


def wait_idle(timeout=10.0) -> dict:
    deadline = time.time() + timeout
    while benchmark.running() and time.time() < deadline:
        time.sleep(0.05)
    return benchmark.status()


class Run(unittest.TestCase):
    def setUp(self):
        db.init()
        benchmark.RESULT.unlink(missing_ok=True)
        self.patches = [patch.object(benchmark, "BENCHMARK", FAKE),
                        patch.object(benchmark, "GS_PY", Path(sys.executable)),
                        # One "free" GPU; nvidia-smi is absent, so health is null.
                        patch.object(worker.gpu, "free_gpus", lambda **k: [0])]
        for p in self.patches:
            p.start()

    def tearDown(self):
        wait_idle()
        for p in self.patches:
            p.stop()

    def test_done_record(self):
        fake_script(
            "time.sleep(0.5)\n"
            "json.dump({'cpu': {'jpeg_decode_1t_per_s': 100.0}, 'gpu': {'gsplat_it_s': 9.0},\n"
            "           'errors': {}, 'durations_s': {'cpu': 0.1}}, open(a.out, 'w'))\n")
        self.assertEqual(benchmark.status(), {"state": "idle"})
        body = main.api_benchmark_run()
        self.assertEqual((body["state"], body["gpu_index"]), ("running", 0))
        # While it runs: a second run, a render and the dispatcher all wait.
        with self.assertRaises(main.HTTPException) as e:
            main.api_benchmark_run()
        self.assertEqual(e.exception.status_code, 409)
        with worker.reserve_gpu() as g:
            self.assertIsNone(g)
        self.assertTrue(worker.running_state()["benchmark_running"])
        rec = wait_idle()
        self.assertEqual(rec["state"], "done", rec)
        self.assertEqual(rec["cpu"], {"jpeg_decode_1t_per_s": 100.0})
        self.assertEqual(rec["gpu"], {"gsplat_it_s": 9.0})
        self.assertEqual(rec["errors"], {})
        self.assertIn("cpu", rec["host"])
        self.assertIn("samples", rec["resources"])
        self.assertNotIn(socket.gethostname(), json.dumps(rec))
        self.assertFalse(worker.exclusive_held())
        self.assertEqual(main.api_benchmark()["state"], "done")

    def test_a_failed_part_fails_the_run(self):
        fake_script(
            "json.dump({'cpu': {'sift_1t_per_s': 1.0}, 'gpu': None,\n"
            "           'errors': {'gpu': 'RuntimeError: torch sees no CUDA device'}},\n"
            "          open(a.out, 'w'))\n"
            "sys.exit(1)\n")
        self.assertTrue(benchmark.start()[0])
        rec = wait_idle()
        self.assertEqual(rec["state"], "failed")
        self.assertEqual(rec["errors"], {"gpu": "RuntimeError: torch sees no CUDA device"})
        self.assertEqual(rec["cpu"], {"sift_1t_per_s": 1.0})
        self.assertFalse(worker.exclusive_held())

    def test_crash_without_result(self):
        fake_script("raise SystemExit(3)\n")
        self.assertTrue(benchmark.start()[0])
        rec = wait_idle()
        self.assertEqual(rec["state"], "failed")
        self.assertIn("exited 3", rec["errors"]["run"])

    def test_no_free_gpu_runs_cpu_part_and_says_so(self):
        fake_script("assert a.no_gpu\njson.dump({'errors': {}}, open(a.out, 'w'))\n")
        with patch.object(worker.gpu, "free_gpus", lambda **k: []):
            self.assertTrue(benchmark.start()[0])
            rec = wait_idle()
        self.assertEqual(rec["state"], "failed")
        self.assertIn("no free GPU", rec["errors"]["gpu"])
        self.assertNotIn("run", rec["errors"])

    def test_hold_released_when_the_record_cannot_be_written(self):
        fake_script("json.dump({'errors': {}}, open(a.out, 'w'))\n")
        with patch.object(benchmark, "_write", side_effect=OSError("disk full")):
            self.assertTrue(benchmark.start()[0])
            wait_idle()
        self.assertFalse(worker.exclusive_held())
        self.assertFalse(benchmark.running())

    def test_refused_while_a_job_runs(self):
        fake_script("json.dump({'errors': {}}, open(a.out, 'w'))\n")
        with worker._lock:
            worker._held[99] = 0
        try:
            with self.assertRaises(main.HTTPException) as e:
                main.api_benchmark_run()
            self.assertEqual(e.exception.status_code, 409)
            self.assertFalse(worker.exclusive_held())
        finally:
            with worker._lock:
                worker._held.pop(99, None)


class Dispatcher(unittest.TestCase):
    def test_no_dispatch_while_held(self):
        db.init()
        db.conn().execute("DELETE FROM jobs")
        jid = db.create_job("x", {"name": "x", "input": {"file": "samples/x.mp4"}})
        started = []
        with patch.object(worker.gpu, "free_gpus", lambda **k: [0]), \
                patch.object(worker, "paused", lambda: False), \
                patch.object(worker, "run_job", lambda j, g: started.append(j)), \
                patch.object(worker._stop, "wait", lambda t: worker._stop.set()):
            with worker.exclusive() as gpus:
                self.assertEqual(gpus, [0])
                worker._stop.clear()
                worker.dispatcher()             # one pass, then _stop
                time.sleep(0.1)
                self.assertEqual(started, [])
            worker._stop.clear()
            worker.dispatcher()
            time.sleep(0.2)
        worker._stop.clear()
        with worker._lock:                  # the fake run_job never releases it
            worker._held.pop(jid, None)
        self.assertEqual(started, [jid])


if __name__ == "__main__":
    unittest.main(verbosity=1)
