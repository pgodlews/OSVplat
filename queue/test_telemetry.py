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

    def test_uploads_once_at_the_end(self):
        db.init()
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Sink)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        target = {"url": f"http://127.0.0.1:{srv.server_port}/up/{{job}}/{{file}}"}
        _Sink.got.clear()
        jid = make_job()
        try:
            with patch.object(telemetry, "TELEMETRY_UPLOAD", target):
                telemetry.write(jid)                # after a stage: local only
                time.sleep(0.3)
                self.assertEqual(_Sink.got, [])
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

    def test_samplers_can_be_joined(self):
        # Both kept their stop Event in self._stop, which is the name of the
        # method Thread.join() calls: join() raised TypeError.
        for s in (telemetry.ResourceSampler(None, os.getpid(), interval=0.05),
                  worker.VramSampler(-1, lambda: [])):
            s.start()
            s.stop()
            s.join(5)
            self.assertFalse(s.is_alive(), type(s).__name__)


class GpuHealth(unittest.TestCase):
    """nvidia-smi answers faked: the fallbacks and what the summary makes of them."""

    def setUp(self):
        telemetry._sample_query = None

    def tearDown(self):
        telemetry._sample_query = None

    def fake_smi(self, rows, known):
        """nvidia-smi that fails any query naming a field outside `known`."""
        asked = []

        def smi(query, extra=None):
            asked.append(query)
            fields = query.split("=", 1)[1].split(",")
            return None if any(f not in known for f in fields) else rows(fields)
        return smi, asked

    def test_old_driver_falls_back_to_throttle_reasons(self):
        known = set(telemetry._SAMPLE_QUERIES[1].split("=", 1)[1].split(","))
        seq = iter([
            # util, mem, power, sm, mem clk, temp, gen, width, ecc, reasons
            ["0", "90", "4000", "300.5", "1900", "9500", "70", "4", "16", "[N/A]", "0x0000000000000004"],
            ["0", "95", "4100", "320.0", "1800", "9500", "83", "4", "16", "[N/A]", "0x0000000000000024"],
            ["0", "10", "4100", "30.0", "210", "405", "60", "1", "16", "[N/A]", "0x0000000000000001"],
        ])
        smi, asked = self.fake_smi(lambda fields: [next(seq)], known)
        s = telemetry.ResourceSampler(0, os.getpid())
        with patch.object(telemetry.gpu, "_nvidia_smi", smi):
            for _ in range(3):
                s.sample()
        self.assertIn("clocks_throttle_reasons.active", telemetry._sample_query)
        # The event-reasons query was tried once, not on every sample.
        self.assertEqual(sum("clocks_event_reasons" in q for q in asked), 1)
        out = s.summary(15)
        self.assertEqual(out["gpu_busy_samples"], 2)
        self.assertEqual(out["gpu_power_w_max"], 320.0)
        self.assertEqual(out["gpu_sm_mhz_busy_p50"], 1900.0)   # idle 210 left out
        self.assertEqual(out["gpu_temp_c_max"], 83.0)
        self.assertEqual((out["gpu_pcie_gen_max"], out["gpu_pcie_width_max"]), (4, 16))
        self.assertIsNone(out["gpu_ecc_uncorrected"])            # [N/A]: unknown
        self.assertEqual(out["gpu_clock_reasons"],
                         {"gpu_idle": 0.33, "sw_power_cap": 0.67, "sw_thermal": 0.33})
        self.assertEqual(out["gpu_mem_peak_mib"], 4100)

    def test_driver_without_health_fields_still_samples_utilisation(self):
        known = {"index", "utilization.gpu", "memory.used"}
        smi, _ = self.fake_smi(lambda fields: [["0", "80", "1234"]], known)
        s = telemetry.ResourceSampler(0, os.getpid())
        with patch.object(telemetry.gpu, "_nvidia_smi", smi):
            s.sample()
        out = s.summary(5)
        self.assertEqual((out["gpu_util_p50"], out["gpu_mem_peak_mib"]), (80.0, 1234))
        self.assertIsNone(out["gpu_clock_reasons"])
        self.assertIsNone(out["gpu_busy_samples"])
        self.assertIsNone(out["gpu_power_w_busy_p50"])

    def test_no_nvidia_smi_is_not_remembered(self):
        with patch.object(telemetry.gpu, "_nvidia_smi", lambda q, extra=None: None):
            telemetry.ResourceSampler(0, os.getpid()).sample()
        self.assertIsNone(telemetry._sample_query)

    def test_host_limits_fall_back(self):
        base = telemetry._GPU_HOST.split("=", 1)[1].split(",")
        row = ["0", "NVIDIA GeForce RTX 3090", "24576", "8.6", "595.84", "4", "16", "350.00"]
        smi, _ = self.fake_smi(lambda fields: [row], set(base))
        with patch.object(telemetry.gpu, "_nvidia_smi", smi):
            g = telemetry._gpus()[0]
        self.assertEqual((g["name"], g["power_limit_w"]), ("NVIDIA GeForce RTX 3090", 350.0))
        self.assertIsNone(g["power_default_w"])
        smi, _ = self.fake_smi(lambda fields: [row + ["350.00", "400.00", "2100", "9751"]],
                               set(base) | set(telemetry._GPU_LIMITS[1:].split(",")))
        with patch.object(telemetry.gpu, "_nvidia_smi", smi):
            g = telemetry._gpus()[0]
        self.assertEqual((g["power_default_w"], g["power_max_w"], g["sm_clock_max_mhz"],
                          g["mem_clock_max_mhz"]), (350.0, 400.0, 2100, 9751))


class Transfers(unittest.TestCase):
    def setUp(self):
        db.init()
        db.conn().execute("DELETE FROM jobs")
        telemetry.INPUT_FETCH.unlink(missing_ok=True)

    def tearDown(self):
        telemetry.INPUT_FETCH.unlink(missing_ok=True)

    def test_input_download_matches_the_job_clip_only(self):
        jid = make_job()
        self.assertIsNone(telemetry.build(jid)["transfers"]["input"])
        telemetry.INPUT_FETCH.parent.mkdir(parents=True, exist_ok=True)
        telemetry.INPUT_FETCH.write_text(json.dumps(
            {"file": CLIP, "bytes": 50_000_000, "seconds": 4.0,
             "first_byte_s": 0.12, "ended": 1790000000}))
        rec = telemetry.build(jid)
        self.assertEqual(rec["transfers"]["input"],
                         {"bytes": 50_000_000, "seconds": 4.0, "mb_s": 12.5,
                          "first_byte_s": 0.12, "ended": 1790000000})
        self.assertNotIn("garden_walk_north", json.dumps(rec))
        telemetry.INPUT_FETCH.write_text(json.dumps({"file": "samples/other.OSV", "bytes": 1}))
        self.assertIsNone(telemetry.build(jid)["transfers"]["input"])

    def test_output_upload_without_its_destination(self):
        jid = make_job()
        d = telemetry.run_dir(jid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "upload.json").write_text(json.dumps(
            {"state": "done", "url": "https://bucket.example/private-name/job.tar",
             "bytes": 30_000_000, "transfer_s": 2.0, "pack_s": 0.4, "attempts": 1,
             "sha256": "ab" * 32, "ended": 1790000100}))
        rec = telemetry.build(jid)
        self.assertEqual(rec["transfers"]["output"],
                         {"state": "done", "bytes": 30_000_000, "seconds": 2.0,
                          "mb_s": 15.0, "attempts": 1, "pack_s": 0.4, "ended": 1790000100})
        self.assertNotIn("private-name", json.dumps(rec))

    def test_resources_survive_the_final_write(self):
        # The final write drops resources from memory; the rewrite after an
        # output upload (or a service restart) must keep them.
        jid = make_job()
        db.upsert_stage(jid, "sfm", "k", "done", started=1, ended=2)
        telemetry.record_resources(jid, "sfm", {"cpu_seconds": 7.0})
        telemetry.write(jid, final=True)
        telemetry.write(jid)
        rec = json.loads(telemetry.telemetry_path(jid).read_text())
        self.assertEqual(rec["stages"][0]["resources"], {"cpu_seconds": 7.0})


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


class UploadAtTheEnd(unittest.TestCase):
    """One telemetry upload per job, after the result upload when there is one."""

    run_fake_job = EndToEnd.run_fake_job

    def run_uploading(self, output: bool):
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Sink)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_port}"
        _Sink.got.clear()
        try:
            with patch.object(telemetry, "TELEMETRY_UPLOAD", {"url": base + "/t/{job}/{file}"}), \
                    patch.object(worker.outputs, "OUTPUT_UPLOAD",
                                 {"url": base + "/out/{job}.tar"} if output else {}):
                jid, _ = self.run_fake_job()
                deadline = time.time() + 15
                while len([g for g in _Sink.got if g[1].startswith("/t/")]) < 2 \
                        and time.time() < deadline:
                    time.sleep(0.05)
                time.sleep(0.3)                     # nothing more may follow
        finally:
            srv.shutdown()
            srv.server_close()
        sent = [(p, b) for _, p, _, b in _Sink.got if p.startswith("/t/")]
        self.assertEqual(sorted(p for p, _ in sent),
                         [f"/t/job{jid:05d}/logs.tar.gz", f"/t/job{jid:05d}/telemetry.json"])
        return json.loads(next(b for p, b in sent if p.endswith(".json")))

    def test_without_output_upload(self):
        rec = self.run_uploading(output=False)
        self.assertEqual(rec["job"]["state"], "done")
        self.assertIsNone(rec["transfers"]["output"])

    def test_after_output_upload(self):
        # The fake stages export nothing, so the result upload fails loudly;
        # what matters is that the record sent is the one written after it.
        rec = self.run_uploading(output=True)
        self.assertEqual(rec["transfers"]["output"]["state"], "failed")
        self.assertTrue(all(s["resources"] for s in rec["stages"]))


class FailuresNeverReachTheJob(unittest.TestCase):
    """A broken telemetry probe is one line in the log; the job runs on."""

    run_fake_job = EndToEnd.run_fake_job

    def assert_done(self):
        jid, events = self.run_fake_job()
        self.assertEqual(db.get_job(jid)["state"], "done")
        self.assertEqual(events[-1]["state"], "done")
        return jid

    def test_sampler_that_cannot_start(self):
        with patch.object(telemetry, "ResourceSampler", side_effect=RuntimeError("no threads")):
            jid = self.assert_done()
        rec = json.loads(telemetry.telemetry_path(jid).read_text())
        self.assertTrue(all(s["resources"] is None for s in rec["stages"]))

    def test_summary_that_raises(self):
        with patch.object(telemetry.ResourceSampler, "summary", side_effect=ZeroDivisionError):
            self.assert_done()

    def test_samples_that_raise_are_logged_once(self):
        s = telemetry.ResourceSampler(None, os.getpid(), interval=0.05)
        out = io.StringIO()
        with patch.object(s, "sample", side_effect=OSError("proc gone")), \
                patch("sys.stdout", out):
            s.start()
            time.sleep(0.8)
            s.stop()
            s.join(2)
        self.assertEqual(out.getvalue().count("resource sample failed"), 1)

    def test_record_that_cannot_be_built(self):
        with patch.object(telemetry, "build", side_effect=RuntimeError("boom")):
            self.assert_done()

    def test_host_probe_that_raises(self):
        with patch.object(telemetry, "_host", None), \
                patch.object(telemetry, "_cpu", side_effect=IndexError("odd cpuinfo")), \
                patch.object(telemetry, "_gpus", side_effect=OSError("nvidia-smi hung")):
            h = telemetry.host()
        self.assertEqual((h["cpu"], h["gpus"]), ({}, None))
        self.assertIn("total_bytes", h["memory"])

    def test_result_archive_without_readable_telemetry(self):
        from app import outputs
        jid = make_job()
        d = telemetry.run_dir(jid)
        d.mkdir(parents=True, exist_ok=True)
        telemetry.telemetry_path(jid).mkdir()          # unreadable as a file
        f = Path(TMP.name) / "splat.ply"
        f.write_bytes(b"ply")
        meta = outputs.build_archive(jid, [f], d / "out.tar")
        with tarfile.open(d / "out.tar") as tar:
            self.assertEqual(tar.getnames(), [f"job{jid:05d}/splat.ply"])
        self.assertGreater(meta["bytes"], 0)


class EndedInReview(unittest.TestCase):
    """A job that ends while parked for mask review never returns to run_job."""

    def setUp(self):
        db.init()
        db.conn().execute("DELETE FROM jobs")

    def parked(self) -> int:
        jid = make_job()
        db.upsert_stage(jid, "mask", "k", "done", started=1, ended=2)
        db.set_job_state(jid, "awaiting_review")
        telemetry.write(jid)
        return jid

    def check_final(self, jid, events):
        rec = json.loads(telemetry.telemetry_path(jid).read_text())
        self.assertEqual(rec["job"]["state"], "cancelled")
        self.assertTrue(telemetry.logs_path(jid).is_file())
        self.assertEqual((events[-1]["event"], events[-1]["state"]),
                         ("job.finished", "cancelled"))

    def hooks(self, events):
        return (patch.object(telemetry, "WEBHOOK_URL", "http://unused"),
                patch.object(telemetry, "_hook_q",
                             type("Q", (), {"put_nowait": events.append})()))

    def test_stopped(self):
        jid, events = self.parked(), []
        a, b = self.hooks(events)
        with a, b:
            self.assertTrue(worker.cancel(jid))
        self.check_final(jid, events)

    def test_rejected(self):
        jid, events = self.parked(), []
        a, b = self.hooks(events)
        with a, b:
            main.api_review_post(jid, main.ReviewReq(approved=False, note="bad"))
        self.check_final(jid, events)


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
