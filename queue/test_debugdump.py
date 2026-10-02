#!/usr/bin/env python3
"""Failure debug bundles (queue/app/debugdump.py). No GPU, no server.

    python3 queue/test_debugdump.py
"""
import hashlib
import http.server
import json
import os
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

TMP = tempfile.TemporaryDirectory(prefix="queue_debug_")
# Assigned before any app import: config.py reads these at import time.
os.environ["SPLAT_ROOT"] = TMP.name
os.environ["QUEUE_ROOT"] = str(Path(TMP.name) / "queue")
os.environ["QUEUE_GPUS"] = ""
os.environ["QUEUE_PREP_BACKEND"] = "cuda"      # a CUDA host, also when run on a Mac
os.environ["QUEUE_TOKEN"] = "tok-5f2b9c1e7a"
os.environ["QUEUE_DEBUG"] = "basic"
os.environ["INPUT_URL"] = "https://s3.example/in/clip.OSV?X-Amz-Signature=abcdef123456"
os.environ.pop("DEBUG_UPLOAD_URL", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config, db, debugdump, worker                 # noqa: E402

UUID = "GPU-c3588589-1247-4661-ec4b-367dfff6928f"
SIGNED = "https://s3.example/out/result.tar?X-Amz-Credential=AKIA&X-Amz-Signature=deadbeef"


def members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path) as tar:
        return {m.name.split("/", 1)[1]: tar.extractfile(m).read()
                for m in tar.getmembers() if m.isfile()}


def failed_job(train_files: dict = None, log_text: str = "") -> int:
    """A job whose training failed, with a log and a train cache dir."""
    jid = db.create_job("j", {"name": "j", "input": {"file": "samples/x.OSV",
                                                     "quick_hash": "deadbeef"}})
    db.set_job_state(jid, "running", started=time.time() - 30)
    log = config.LOG_ROOT / f"job{jid:05d}_train.log"
    log.write_text("$ /opt/splat/LichtFeld-Studio --headless -d /data/x --max-cap 6000000\n"
                   + log_text)
    train = Path(TMP.name) / "cache" / "train" / f"t{jid}"
    train.mkdir(parents=True)
    for name, data in (train_files or {}).items():
        (train / name).parent.mkdir(parents=True, exist_ok=True)
        (train / name).write_bytes(data)
    db.upsert_stage(jid, "train", "k", "failed", path=str(train), log_path=str(log))
    db.set_job_state(jid, "failed", ended=time.time(), error="train exited -11")
    return jid


class Bundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init()

    def setUp(self):
        self.crash = Path(tempfile.mkdtemp(dir=TMP.name))
        p = patch.object(debugdump, "CRASH_GLOBS", (str(self.crash / "lichtfeld-studio-crash-*.log"),))
        p.start()
        self.addCleanup(p.stop)

    def test_basic_holds_what_the_5090_diagnosis_needed(self):
        (self.crash / "lichtfeld-studio-crash-72688.log").write_text(
            "LichtFeld Studio fatal signal SIGSEGV\ncudaFreeAsync+0x20b\n")
        jid = failed_job(log_text=f"VRAM: free=3 MiB, used=32105 MiB\n{UUID}\n")
        st = debugdump.build(jid, "basic")
        self.assertEqual(st["state"], "built")
        got = members(debugdump.bundle_path(jid))
        for name in ("MANIFEST.json", "job.json", "repro.sh", "logs/train.log",
                     "crash/lichtfeld-studio-crash-72688.log", "gpu/nvidia-smi-q.txt",
                     "host/env.txt"):
            self.assertIn(name, got)
        self.assertIn(b"used=32105 MiB", got["logs/train.log"])
        job = json.loads(got["job.json"])
        self.assertEqual(job["stages"][0]["state"], "failed")
        # The failed stage's command line, ready to rerun by hand.
        self.assertIn(b"--max-cap 6000000", got["repro.sh"])
        # sha256 in debug.json is the file's.
        self.assertEqual(st["sha256"], hashlib.sha256(
            debugdump.bundle_path(jid).read_bytes()).hexdigest())

    def test_crash_logs_from_before_the_job_are_not_its_own(self):
        old = self.crash / "lichtfeld-studio-crash-1.log"
        old.write_text("an earlier run's crash")
        os.utime(old, (time.time() - 7200, time.time() - 7200))
        jid = failed_job()
        got = members(Path(debugdump.build(jid, "basic")["path"]))
        self.assertNotIn("crash/lichtfeld-studio-crash-1.log", got)

    def test_secrets_are_stripped(self):
        jid = failed_job(log_text=f"output upload: {SIGNED}\ntoken tok-5f2b9c1e7a\n{UUID}\n")
        got = members(Path(debugdump.build(jid, "basic")["path"]))
        blob = b"".join(got.values())
        self.assertNotIn(b"tok-5f2b9c1e7a", blob)
        self.assertNotIn(b"X-Amz-Signature", blob)
        self.assertNotIn(b"deadbeef&", blob)
        self.assertNotIn(UUID.encode(), blob)
        self.assertIn(b"https://s3.example/out/result.tar?<signature removed>", got["logs/train.log"])
        env = got["host/env.txt"].decode()
        self.assertIn("QUEUE_TOKEN=<redacted>", env)
        self.assertIn("INPUT_URL=<redacted>", env)

    def test_serials_in_nvidia_smi_q(self):
        q = ("    Serial Number                         : 1652322011234\n"
             f"    GPU UUID                              : {UUID}\n"
             "    Product Name                          : NVIDIA GeForce RTX 5090\n")
        out = debugdump.scrub(q, [])
        self.assertNotIn("1652322011234", out)
        self.assertNotIn("c3588589", out)
        self.assertIn("RTX 5090", out)

    def test_artifacts_level_keeps_what_training_exported(self):
        files = {"splat_50000.spz": b"s" * 100, "splat_50000.ply": b"p" * 300,
                 "project.licht": b"l" * 50, "metrics.csv": b"iteration,psnr\n30000,26.9\n"}
        jid = failed_job(files)
        basic = members(Path(debugdump.build(jid, "basic")["path"]))
        self.assertFalse(any(n.startswith("train/") for n in basic))
        got = members(Path(debugdump.build(jid, "artifacts")["path"]))
        for name in files:
            self.assertEqual(got[f"train/{name}"], files[name])

    def test_cap_leaves_out_the_lowest_priority_and_says_so(self):
        files = {"splat_50000.spz": b"s" * 1000, "splat_50000.ply": b"p" * 200_000}
        jid = failed_job(files)
        with patch.object(debugdump, "DEBUG_MAX_BYTES", 150_000):
            st = debugdump.build(jid, "artifacts")
        got = members(Path(st["path"]))
        self.assertIn("train/splat_50000.spz", got)
        self.assertNotIn("train/splat_50000.ply", got)
        self.assertEqual([s["name"] for s in st["skipped"]], ["train/splat_50000.ply"])
        manifest = json.loads(got["MANIFEST.json"])
        self.assertEqual(manifest["skipped"][0]["name"], "train/splat_50000.ply")
        # Level basic goes in whatever the cap.
        with patch.object(debugdump, "DEBUG_MAX_BYTES", 1):
            got = members(Path(debugdump.build(jid, "basic", reuse=False)["path"]))
        self.assertIn("logs/train.log", got)

    def test_heavy_adds_cores_and_the_dataset_view_but_not_frames(self):
        files = {"dataset/sparse/0/points3D.bin": b"pts", "dataset/masks/lens0_0001.png": b"m",
                 "dataset/images/lens0_0001.jpg": b"frame", "core.4242": b"core"}
        jid = failed_job(files)
        got = members(Path(debugdump.build(jid, "heavy")["path"]))
        self.assertIn("train/dataset/sparse/0/points3D.bin", got)
        self.assertIn("train/dataset/masks/lens0_0001.png", got)
        self.assertNotIn("train/dataset/images/lens0_0001.jpg", got)
        self.assertIn("cores/core.4242", got)

    def test_reuse_returns_a_bundle_already_as_deep(self):
        jid = failed_job()
        a = debugdump.build(jid, "artifacts")
        self.assertEqual(debugdump.build(jid, "basic")["created"], a["created"])
        time.sleep(0.2)
        b = debugdump.build(jid, "heavy")
        self.assertNotEqual(b["created"], a["created"])
        self.assertEqual(b["level"], "heavy")

    def test_a_build_that_breaks_is_recorded_not_raised(self):
        jid = failed_job()
        with patch.object(debugdump, "_collect", side_effect=OSError("disk full")):
            st = debugdump.build(jid, "basic", reuse=False)
        self.assertEqual(st["state"], "failed")
        self.assertIn("disk full", st["error"])

    def test_bad_level(self):
        with self.assertRaises(ValueError):
            debugdump.build(failed_job(), "everything")


class _Sink(http.server.BaseHTTPRequestHandler):
    """S3-like presigned PUT: returns the body's MD5 as ETag."""
    got: list = []

    def do_PUT(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).got.append({"path": self.path, "body": body})
        self.send_response(200)
        self.send_header("ETag", '"%s"' % hashlib.md5(body).hexdigest())
        self.end_headers()

    def log_message(self, *a):
        pass


class Upload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init()
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), _Sink)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}/results/run/debug.tar?X-Amz-Signature=x"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        _Sink.got = []
        for d in config.RUNS_ROOT.glob("job*"):
            (d / "debug.json").unlink(missing_ok=True)

    def test_streams_the_bundle_and_checks_it_arrived_whole(self):
        jid = failed_job({"splat_50000.spz": b"s" * 5000})
        debugdump.build(jid, "artifacts")
        st = debugdump.upload(jid, {"method": "PUT", "url": self.url})
        self.assertEqual(st["state"], "uploaded", st)
        self.assertEqual(_Sink.got[0]["body"], debugdump.bundle_path(jid).read_bytes())
        self.assertEqual(st["upload"]["url"], self.url.split("?")[0])

    def test_one_fixed_object_is_not_overwritten_by_a_later_job(self):
        target = {"method": "PUT", "url": self.url}
        a, b = failed_job(), failed_job()
        debugdump.build(a, "basic")
        debugdump.build(b, "basic")
        self.assertEqual(debugdump.upload(a, target)["state"], "uploaded")
        st = debugdump.upload(b, target)
        self.assertEqual(st["state"], "refused")
        self.assertEqual(len(_Sink.got), 1)
        self.assertTrue(debugdump.bundle_path(b).is_file())      # kept local

    def test_job_placeholder_gives_each_job_its_own_object(self):
        target = {"method": "PUT", "url": self.url.replace("debug.tar", "{job}-debug.tar")}
        a, b = failed_job(), failed_job()
        for j in (a, b):
            debugdump.build(j, "basic")
            self.assertEqual(debugdump.upload(j, target)["state"], "uploaded")
        self.assertEqual(len({g["path"] for g in _Sink.got}), 2)


class WorkerHook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init()

    def test_a_failed_job_gets_a_bundle_a_cancelled_one_does_not(self):
        calls = []
        with patch.object(debugdump, "on_failure", calls.append):
            jid = db.create_job("bad", {"name": "t", "input": {"file": "x.mp4"},
                                        "train": {"sh_degrees": 3}})
            worker.run_job(jid, 0)
            self.assertEqual(db.get_job(jid)["state"], "failed")
            self.assertEqual(calls, [jid])

            (Path(TMP.name) / "samples").mkdir(exist_ok=True)
            (Path(TMP.name) / "samples" / "c.mp4").write_bytes(b"clip")
            jid = db.create_job("cancel", {"name": "t", "input": {"file": "samples/c.mp4"}})
            worker._cancel.add(jid)
            worker.run_job(jid, 0)
            self.assertEqual(db.get_job(jid)["state"], "cancelled")
            self.assertEqual(calls, [calls[0]])

    def test_on_failure_never_raises(self):
        jid = failed_job()
        with patch.object(debugdump, "build", side_effect=RuntimeError("boom")):
            debugdump.on_failure(jid)                   # returns; the thread logs
            time.sleep(0.3)


if __name__ == "__main__":
    unittest.main(verbosity=1)
