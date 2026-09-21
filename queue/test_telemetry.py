#!/usr/bin/env python3
"""Telemetry: the record, the redacted log bundle, uploads. No GPU, no server.

    python3 queue/test_telemetry.py
"""
import http.server
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

TMP = tempfile.TemporaryDirectory(prefix="queue_telemetry_")
# Assigned before any app import: config.py reads these at import time.
os.environ["SPLAT_ROOT"] = TMP.name
os.environ["QUEUE_ROOT"] = str(Path(TMP.name) / "queue")
os.environ["QUEUE_GPUS"] = ""
os.environ.pop("QUEUE_TELEMETRY", None)
os.environ.pop("QUEUE_TELEMETRY_UPLOAD", None)
os.environ["QUEUE_TELEMETRY_PLACEMENT"] = '{"provider": "test", "price_per_hour": 0.5}'
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config, db, main, telemetry, worker           # noqa: E402
from app.stages import ORDER, STAGES                           # noqa: E402
from app.config import LOG_ROOT, SPLAT_ROOT                    # noqa: E402

CLIP = "samples/garden_walk_north.OSV"


def make_job(name="my private job") -> int:
    cfg = {"name": name, "input": {"file": CLIP, "quick_hash": "abc123"},
           "train": {"iter": 30000}}
    jid = db.create_job(name, cfg)
    db.set_plan(jid, {"frames": 10.0, "train": 100.0, "total": 110.0})
    return jid


class Record(unittest.TestCase):
    def setUp(self):
        db.init()
        db.conn().execute("DELETE FROM jobs")

    def test_fields_and_what_is_left_out(self):
        jid = make_job()
        now = time.time()
        db.upsert_stage(jid, "frames", "k1", "done", started=now - 12, ended=now,
                        progress=json.dumps({"candidates": 300, "images_dir":
                                             f"{SPLAT_ROOT}/x", "ok": True}))
        db.upsert_stage(jid, "select", "k2", "cached", ended=now)
        telemetry.record_resources(jid, "frames", {"cpu_seconds": 30.0})
        rec = telemetry.build(jid)
        text = json.dumps(rec)

        self.assertEqual(rec["schema"], telemetry.SCHEMA)
        fr = rec["stages"][0]
        self.assertEqual((fr["stage"], fr["wall_s"], fr["planned_s"]), ("frames", 12.0, 10.0))
        self.assertEqual(fr["info"], {"candidates": 300, "ok": True})
        self.assertEqual(fr["resources"], {"cpu_seconds": 30.0})
        self.assertEqual(rec["stages"][1]["state"], "cached")
        self.assertEqual(rec["job"]["config"]["input"]["file_ext"], ".osv")
        self.assertEqual(rec["job"]["config"]["input"]["quick_hash"], "abc123")
        self.assertEqual(rec["placement"], {"provider": "test", "price_per_hour": 0.5})
        self.assertIn("cpu", rec["host"])
        self.assertIn("queue_root_free_bytes", rec["host"]["disk"])
        # Names, paths and the host are not in the record.
        for leak in ("garden_walk_north", "my private job", str(SPLAT_ROOT),
                     socket.gethostname()):
            self.assertNotIn(leak, text)

    def test_write_is_atomic_and_can_be_switched_off(self):
        jid = make_job()
        telemetry.write(jid)
        p = telemetry.telemetry_path(jid)
        self.assertEqual(json.loads(p.read_text())["job"]["id"], jid)
        self.assertEqual([x.name for x in p.parent.iterdir()], ["telemetry.json"])

        jid2 = make_job()
        with patch.object(telemetry, "TELEMETRY_ENABLED", False):
            telemetry.write(jid2, final=True)
        self.assertFalse(telemetry.run_dir(jid2).exists())

    def test_error_is_redacted(self):
        jid = make_job()
        db.set_job_state(jid, "failed", error=f"input {CLIP} changed; see {SPLAT_ROOT}/x")
        err = telemetry.build(jid)["job"]["error"]
        self.assertEqual(err, "input samples/<clip>.OSV changed; see $SPLAT_ROOT/x")

    def test_never_raises(self):
        jid = make_job()
        with patch.object(telemetry, "build", side_effect=RuntimeError("boom")):
            telemetry.write(jid)                    # printed, not raised


class Logs(unittest.TestCase):
    def setUp(self):
        db.init()
        db.conn().execute("DELETE FROM jobs")

    def bundle(self, jid) -> dict:
        telemetry.write(jid, final=True)
        with tarfile.open(telemetry.logs_path(jid)) as tar:
            return {m.name: tar.extractfile(m).read().decode() for m in tar}

    def test_redaction(self):
        jid = make_job()
        log = LOG_ROOT / f"job{jid:05d}_sfm.log"
        log.write_text(
            f"$ {SPLAT_ROOT}/venv/bin/python run.py {SPLAT_ROOT}/{CLIP}\n"
            f"home {Path.home()}/x on {socket.gethostname()}\n"
            "  device: {'serial': '1ABCD2345EF', 'model': 'Osmo360'}\n"
            "fix lat=51.50731 lon: -0.12763 step 0198\n")
        db.upsert_stage(jid, "sfm", "k", "done", log_path=str(log))
        text = self.bundle(jid)["sfm.log"]
        for leak in (str(SPLAT_ROOT), str(Path.home()), socket.gethostname(),
                     "1ABCD2345EF", "51.50731", "0.12763", "garden_walk_north"):
            self.assertNotIn(leak, text)
        self.assertIn("$SPLAT_ROOT/venv/bin/python", text)
        self.assertIn("'model': 'Osmo360'", text)
        self.assertIn("step 0198", text)            # numbers are not clip names

    def test_cached_stage_logs_belong_to_another_job(self):
        jid = make_job()
        log = LOG_ROOT / "job99999_frames.log"
        log.write_text("someone else's run\n")
        db.upsert_stage(jid, "frames", "k", "cached", log_path=str(log))
        self.assertEqual(self.bundle(jid), {})

    def test_big_logs_are_capped(self):
        jid = make_job()
        log = LOG_ROOT / f"job{jid:05d}_train.log"
        line = b"Iteration 100/30000 loss 0.1\n"
        log.write_bytes(b"HEAD\n" + line * 100_000 + b"TAIL\n")
        db.upsert_stage(jid, "train", "k", "done", log_path=str(log))
        with patch.object(telemetry, "LOG_HEAD", 1000), patch.object(telemetry, "LOG_TAIL", 1000):
            text = self.bundle(jid)["train.log"]
        self.assertTrue(text.startswith("HEAD") and text.endswith("TAIL\n"))
        self.assertIn("bytes cut", text)
        self.assertLess(len(text), 3000)


class _Sink(http.server.BaseHTTPRequestHandler):
    got: list = []

    def _take(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Sink.got.append((self.command, self.path, dict(self.headers), self.rfile.read(n)))
        self.send_response(204)
        self.end_headers()

    do_PUT = do_POST = _take

    def log_message(self, *a):
        pass


class Upload(unittest.TestCase):
    def test_put_template(self):
        req = telemetry.upload_request(
            {"url": "https://hub/t/{job}/{file}?sig=x", "headers": {"Authorization": "Bearer t"}},
            7, "telemetry.json", b"{}")
        self.assertEqual((req.get_method(), req.full_url),
                         ("PUT", "https://hub/t/job00007/telemetry.json?sig=x"))
        self.assertEqual(req.get_header("Authorization"), "Bearer t")

    def test_s3_post_form(self):
        req = telemetry.upload_request(
            {"method": "POST", "url": "https://bucket.s3", "key": "batch1/{job}/{file}",
             "fields": {"policy": "P", "x-amz-signature": "S"}},
            7, "logs.tar.gz", b"DATA")
        body = req.data.decode()
        self.assertIn('name="key"\r\n\r\nbatch1/job00007/logs.tar.gz', body)
        self.assertIn('name="policy"\r\n\r\nP', body)
        # S3 ignores every field after the file, so it has to come last.
        self.assertGreater(body.index('name="file"'), body.index('name="x-amz-signature"'))
        self.assertIn("multipart/form-data; boundary=", req.get_header("Content-type"))

    def test_uploads_after_each_write(self):
        db.init()
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Sink)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        target = {"url": f"http://127.0.0.1:{srv.server_port}/up/{{job}}/{{file}}"}
        _Sink.got.clear()
        jid = make_job()
        try:
            with patch.object(telemetry, "TELEMETRY_UPLOAD", target):
                telemetry.write(jid, final=True)
                deadline = time.time() + 10
                while len(_Sink.got) < 2 and time.time() < deadline:
                    time.sleep(0.05)
        finally:
            srv.shutdown()
            srv.server_close()
        paths = sorted(p for _, p, _, _ in _Sink.got)
        self.assertEqual(paths, [f"/up/job{jid:05d}/logs.tar.gz",
                                 f"/up/job{jid:05d}/telemetry.json"])
        body = next(b for _, p, _, b in _Sink.got if p.endswith(".json"))
        self.assertEqual(json.loads(body)["job"]["id"], jid)

    def test_malformed_target_is_fatal(self):
        with patch.dict(os.environ, {"X": "[1, 2]"}):
            with self.assertRaises(ValueError):
                config._json_env("X")
        with patch.dict(os.environ, {"X": "{nope"}):
            with self.assertRaises(ValueError):
                config._json_env("X")
        self.assertEqual(config._json_env("UNSET_FOR_SURE"), {})


class Webhook(unittest.TestCase):
    def test_off_without_url(self):
        with patch.object(telemetry, "WEBHOOK_URL", ""), \
                patch.object(telemetry.db, "get_job") as get:
            telemetry.notify("stage.started", 1, "frames", "running")
        get.assert_not_called()

    def test_signed_and_in_order(self):
        import hashlib
        import hmac
        db.init()
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Sink)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        _Sink.got.clear()
        jid = make_job()
        url = f"http://127.0.0.1:{srv.server_port}/hook"
        try:
            with patch.object(telemetry, "WEBHOOK_URL", url), \
                    patch.object(telemetry, "WEBHOOK_SECRET", "s3cret"):
                telemetry.notify("stage.started", jid, "sfm", "running")
                telemetry.notify("stage.finished", jid, "sfm", "done")
                telemetry.notify("job.finished", jid, state="done")
                deadline = time.time() + 10
                while len(_Sink.got) < 3 and time.time() < deadline:
                    time.sleep(0.05)
        finally:
            srv.shutdown()
            srv.server_close()
        events = [json.loads(b) for _, _, _, b in _Sink.got]
        self.assertEqual([(e["event"], e["stage"], e["state"]) for e in events],
                         [("stage.started", "sfm", "running"),
                          ("stage.finished", "sfm", "done"),
                          ("job.finished", None, "done")])
        self.assertEqual(events[0]["placement"]["provider"], "test")
        self.assertNotIn("my private job", json.dumps(events))
        _, _, headers, body = _Sink.got[0]
        want = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        # HTTP header names are case-insensitive, and urllib sends this one
        # as X-osvplat-signature.
        got = {k.lower(): v for k, v in headers.items()}
        self.assertEqual(got["x-osvplat-signature"], want)


class Sampler(unittest.TestCase):
    def test_measures_a_busy_child(self):
        p = subprocess.Popen([sys.executable, "-c",
                              "import time\nt=time.time()\nwhile time.time()-t<1.5: pass"])
        s = telemetry.ResourceSampler(None, p.pid, interval=0.2)
        s.start()
        p.wait()
        s.stop()
        out = s.summary(1.5)
        self.assertGreater(out["samples"], 0)
        if Path("/proc").is_dir():
            self.assertGreater(out["cpu_seconds"], 0.5)
            self.assertGreater(out["rss_peak_bytes"], 0)
        else:
            self.assertIsNone(out["cpu_seconds"])   # no /proc: unknown, not zero
        self.assertIsNone(out["gpu_util_p50"])


def _fake_stage(stage):
    return {"argv": lambda ctx: [sys.executable, "-c",
                                 f"import time; print('{stage} on {SPLAT_ROOT}'); time.sleep(0.8)"],
            "finalize": lambda ctx: {"n": 3, "dir": str(SPLAT_ROOT)},
            "parse": None, "prepare": None}


class EndToEnd(unittest.TestCase):
    """run_job through fake stage subprocesses: record, bundle, events."""

    def run_fake_job(self, fail_at=None):
        db.init()
        # Same config in every test, so clear the cache or the second run is
        # all cache hits.
        import shutil
        shutil.rmtree(config.CACHE_ROOT, ignore_errors=True)
        db.conn().execute("DELETE FROM cache")
        (Path(SPLAT_ROOT) / "samples").mkdir(exist_ok=True)
        (Path(SPLAT_ROOT) / CLIP).write_bytes(b"x" * 1000)
        cfg = {"name": "e2e", "input": {"file": CLIP}}
        jid = db.create_job("e2e", cfg)
        fakes = {st: _fake_stage(st) for st in ORDER}
        if fail_at:
            fakes[fail_at]["argv"] = lambda ctx: [sys.executable, "-c", "raise SystemExit(3)"]
        events = []
        with patch.dict(STAGES, fakes), \
                patch.object(worker.retention, "ensure_space", lambda *a, **k: None), \
                patch.object(telemetry, "WEBHOOK_URL", "http://unused"), \
                patch.object(telemetry, "_hook_q", type("Q", (), {"put_nowait": events.append})()):
            worker.run_job(jid, -1)
        return jid, events

    def test_done(self):
        jid, events = self.run_fake_job()
        rec = json.loads(telemetry.telemetry_path(jid).read_text())
        self.assertEqual(rec["job"]["state"], "done")
        self.assertEqual([s["stage"] for s in rec["stages"]], ORDER)
        self.assertTrue(all(s["state"] == "done" and s["wall_s"] >= 0.8 for s in rec["stages"]))
        self.assertEqual(rec["stages"][0]["info"], {"n": 3})
        self.assertGreater(rec["stages"][0]["resources"]["samples"], 0)
        self.assertEqual(rec["job"]["input_bytes"], 1000)
        with tarfile.open(telemetry.logs_path(jid)) as tar:
            logs = {m.name: tar.extractfile(m).read().decode() for m in tar}
        self.assertEqual(sorted(logs), sorted(f"{st}.log" for st in ORDER))
        self.assertIn("sfm on $SPLAT_ROOT", logs["sfm.log"])
        self.assertEqual(events[-1]["event"], "job.finished")
        self.assertEqual(events[-1]["state"], "done")
        self.assertEqual(sum(e["event"] == "stage.started" for e in events), len(ORDER))

    def test_failed_stage_is_recorded(self):
        jid, events = self.run_fake_job(fail_at="sfm")
        rec = json.loads(telemetry.telemetry_path(jid).read_text())
        self.assertEqual(rec["job"]["state"], "failed")
        states = {s["stage"]: s["state"] for s in rec["stages"]}
        self.assertEqual((states["mask"], states["sfm"]), ("done", "failed"))
        self.assertNotIn("train", states)
        self.assertIn("exited 3", rec["job"]["error"])
        self.assertEqual((events[-1]["event"], events[-1]["stage"], events[-1]["state"]),
                         ("job.finished", "sfm", "failed"))
        self.assertTrue(telemetry.logs_path(jid).is_file())


class Purge(unittest.TestCase):
    def test_purge_removes_run_dir(self):
        db.init()
        jid = make_job()
        telemetry.write(jid)
        db.set_job_state(jid, "done", ended=time.time())
        db.hide_finished()
        self.assertEqual(main.api_clear(purge=True)["purged"], 1)
        self.assertFalse(telemetry.run_dir(jid).exists())


if __name__ == "__main__":
    unittest.main(verbosity=1)
