#!/usr/bin/env python3
"""Remote-run support: CPU/GPU discovery and the output upload. No GPU, no server.

    python3 queue/test_remote.py
"""
import base64
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

TMP = tempfile.TemporaryDirectory(prefix="queue_remote_")
# Assigned before any app import: config.py reads these at import time.
os.environ["SPLAT_ROOT"] = TMP.name
os.environ["QUEUE_ROOT"] = str(Path(TMP.name) / "queue")
os.environ["QUEUE_GPUS"] = ""
os.environ.pop("OUTPUT_UPLOAD_URL", None)
os.environ.pop("QUEUE_CPUS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config, db, gpu, outputs, resources, worker   # noqa: E402


def cgroup(tmp: Path, **files) -> Path:
    for name, text in files.items():
        p = tmp / name.replace("__", "/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp


class Cpus(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(dir=TMP.name))

    def test_v2_quota_below_visible(self):
        # The RunPod pod: 112 visible, 23.8 CPUs of quota.
        root = cgroup(self.d, **{"cpu.max": "2380000 100000\n"})
        self.assertAlmostEqual(resources.cgroup_cpu_quota(root), 23.8)
        with patch.object(resources, "affinity_cpus", return_value=112):
            self.assertEqual(resources.effective_cpus(root), 23)
            self.assertEqual(resources.stage_threads(1, root), 23)
            self.assertEqual(resources.stage_threads(2, root), 11)

    def test_v2_unlimited(self):
        root = cgroup(self.d, **{"cpu.max": "max 100000"})
        self.assertIsNone(resources.cgroup_cpu_quota(root))
        with patch.object(resources, "affinity_cpus", return_value=16):
            self.assertEqual(resources.effective_cpus(root), 16)

    def test_v1_quota(self):
        root = cgroup(self.d, **{"cpu__cpu.cfs_quota_us": "400000",
                                 "cpu__cpu.cfs_period_us": "100000"})
        self.assertEqual(resources.cgroup_cpu_quota(root), 4.0)
        root2 = cgroup(Path(tempfile.mkdtemp(dir=TMP.name)),
                       **{"cpu__cpu.cfs_quota_us": "-1",
                          "cpu__cpu.cfs_period_us": "100000"})
        self.assertIsNone(resources.cgroup_cpu_quota(root2))

    def test_affinity_smaller_than_quota(self):
        root = cgroup(self.d, **{"cpu.max": "3200000 100000"})
        with patch.object(resources, "affinity_cpus", return_value=8):
            self.assertEqual(resources.effective_cpus(root), 8)

    def test_fraction_and_floor(self):
        root = cgroup(self.d, **{"cpu.max": "50000 100000"})     # half a CPU
        with patch.object(resources, "affinity_cpus", return_value=4):
            self.assertEqual(resources.effective_cpus(root), 1)
            self.assertEqual(resources.stage_threads(8, root), 1)

    def test_override(self):
        root = cgroup(self.d, **{"cpu.max": "2380000 100000"})
        with patch.dict(os.environ, {"QUEUE_CPUS": "6"}):
            self.assertEqual(resources.effective_cpus(root), 6)

    def test_thread_env_reaches_stages(self):
        env = resources.thread_env(11)
        self.assertEqual(env["SPLAT_THREADS"], "11")
        self.assertEqual(env["OMP_NUM_THREADS"], "11")

    def test_summary_line(self):
        gpus = [{"index": 0, "name": "NVIDIA L4", "memory_mib": 23034, "compute_cap": "8.9"},
                {"index": 1, "name": "Quadro P2000", "memory_mib": 5120, "compute_cap": "6.1"}]
        line = resources.summary(gpus, 1)
        self.assertIn("gpu0 NVIDIA L4", line)
        self.assertIn("UNSUPPORTED (compute capability 6.1 < 7.5)", line)
        self.assertIn("no GPU visible", resources.summary([], 1))


class Gpus(unittest.TestCase):
    def test_unsupported_reason(self):
        self.assertIsNone(resources.unsupported_reason("7.5"))
        self.assertIsNone(resources.unsupported_reason("12.0"))
        self.assertIsNone(resources.unsupported_reason(None))
        self.assertIn("6.1", resources.unsupported_reason("6.1"))

    def test_old_card_is_listed_but_not_schedulable(self):
        rows = {"gpu=index,uuid": [["0", "GPU-a"], ["1", "GPU-b"]],
                "compute-apps=gpu_uuid,pid,process_name,used_memory": [],
                "gpu=index,memory.used,memory.total,utilization.gpu":
                    [["0", "0", "24576", "0"], ["1", "0", "5120", "0"]],
                "gpu=index,compute_cap": [["0", "8.6"], ["1", "6.1"]]}
        with patch.object(gpu, "_nvidia_smi", lambda q, extra=None: rows[q]), \
                patch.object(gpu, "GPUS", [0, 1]), patch.object(gpu, "_caps", None):
            st = {r["index"]: r for r in gpu.status()}
        self.assertTrue(st[0]["available"])
        self.assertIsNone(st[0]["unsupported"])
        self.assertFalse(st[1]["schedulable"])
        self.assertFalse(st[1]["available"])
        self.assertIn("6.1", st[1]["unsupported"])

    def test_build_architectures(self):
        full = "7.5;8.0;8.6;8.9;9.0;12.0"
        for cc in ("7.5", "8.6", "8.9", "9.0", "12.0"):
            self.assertIsNone(resources.unsupported_reason(cc, full), cc)
        # A build trimmed to one card, as docs/docker.md suggests for local builds.
        self.assertIsNone(resources.unsupported_reason("8.6", "8.6"))
        self.assertIsNone(resources.unsupported_reason("8.9", "8.6"))     # same major, newer minor
        self.assertIn("not in this build", resources.unsupported_reason("12.0", "8.6"))
        self.assertIn("not in this build", resources.unsupported_reason("8.0", "8.6"))
        self.assertIn("not in this build", resources.unsupported_reason("7.5", "8.6"))
        self.assertIsNone(resources.unsupported_reason("12.0", "86;120"))  # CMake spelling
        self.assertIsNone(resources.unsupported_reason("12.0", ""))        # native: floor only
        self.assertIn("< 7.5", resources.unsupported_reason("6.1", full))

    def test_no_capable_gpu_refuses_jobs(self):
        from fastapi import HTTPException
        from app import main
        with patch.object(gpu, "compute_caps", lambda: {0: "12.0"}), \
                patch.object(main, "GPUS", [0]), \
                patch.object(resources, "BUILT_ARCH", "8.6"):
            with self.assertRaises(HTTPException) as cm:
                main._refuse_if_cannot_deliver()
        self.assertIn("gpu0", cm.exception.detail)
        with patch.object(gpu, "compute_caps", lambda: {0: "12.0", 1: "8.6"}), \
                patch.object(main, "GPUS", [0, 1]), \
                patch.object(resources, "BUILT_ARCH", "8.6"):
            main._refuse_if_cannot_deliver()                      # one card can: accepted
        with patch.object(gpu, "compute_caps", lambda: {}), patch.object(main, "GPUS", [0]):
            main._refuse_if_cannot_deliver()                      # unknown: not refused

    def test_cuda_probe(self):
        import subprocess as sp
        from fastapi import HTTPException
        from app import main
        ok = sp.CompletedProcess([], 0, stdout="", stderr="")
        bad = sp.CompletedProcess([], 0, stdout="CUDA_ERROR_UNKNOWN 999\n", stderr="")
        with patch.object(resources.subprocess, "run", lambda *a, **k: ok):
            self.assertIsNone(resources.probe_cuda())
        with patch.object(resources.subprocess, "run", lambda *a, **k: bad):
            self.assertIn("CUDA_ERROR_UNKNOWN", resources.probe_cuda())
        try:
            # A broken host: every GPU unsupported, jobs refused.
            rows = {"gpu=index,uuid": [["0", "GPU-a"]],
                    "compute-apps=gpu_uuid,pid,process_name,used_memory": [],
                    "gpu=index,memory.used,memory.total,utilization.gpu": [["0", "1", "24576", "0"]],
                    "gpu=index,compute_cap": [["0", "8.6"]]}
            with patch.object(gpu, "_nvidia_smi", lambda q, extra=None: rows[q]), \
                    patch.object(gpu, "GPUS", [0]), patch.object(gpu, "_caps", None):
                st = gpu.status()[0]
            self.assertFalse(st["available"])
            self.assertIn("CUDA_ERROR_UNKNOWN", st["unsupported"])
            with patch.object(main, "GPUS", [0]):
                with self.assertRaises(HTTPException) as cm:
                    main._refuse_if_cannot_deliver()
            self.assertIn("faulty", cm.exception.detail)
        finally:
            resources.CUDA_ERROR = None
        def hang(*a, **k):
            raise sp.TimeoutExpired("x", 60)
        with patch.object(resources.subprocess, "run", hang):
            self.assertIn("did not return", resources.probe_cuda())
        resources.CUDA_ERROR = None

    def test_probe_failure_refuses_nothing(self):
        with patch.object(gpu, "_nvidia_smi", lambda q, extra=None: None), \
                patch.object(gpu, "_caps", None):
            self.assertEqual(gpu.compute_caps(), {})


class Target(unittest.TestCase):
    def test_plain_url_is_put(self):
        with patch.dict(os.environ, {"OUTPUT_UPLOAD_URL": "https://b.s3.amazonaws.com/x.tar?X-Amz-Signature=s"}):
            t = config._upload_target("OUTPUT_UPLOAD_URL")
        self.assertEqual(t["method"], "PUT")

    def test_json_and_bad_values(self):
        with patch.dict(os.environ, {"OUTPUT_UPLOAD_URL": '{"method": "POST", "url": "https://b", "fields": {}}'}):
            self.assertEqual(config._upload_target("OUTPUT_UPLOAD_URL")["method"], "POST")
        for bad in ('{"method": "PUT"}', "s3://bucket/key", "[1]"):
            with patch.dict(os.environ, {"OUTPUT_UPLOAD_URL": bad}):
                with self.assertRaises(ValueError):
                    config._upload_target("OUTPUT_UPLOAD_URL")

    def test_safe_url_drops_the_signature(self):
        u = outputs.safe_url("https://b.s3.amazonaws.com/osv/run.tar?X-Amz-Signature=secret&X-Amz-Credential=AKIA")
        self.assertEqual(u, "https://b.s3.amazonaws.com/osv/run.tar")


class _Sink(http.server.BaseHTTPRequestHandler):
    got: list = []
    status = 200

    def do_PUT(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        md5 = self.headers.get("Content-MD5")
        ok = md5 == base64.b64encode(hashlib.md5(body).digest()).decode()
        type(self).got.append({"path": self.path, "body": body, "md5_ok": ok,
                               "ctype": self.headers.get("Content-Type")})
        code = type(self).status if ok else 400       # what S3 does on a bad MD5
        self.send_response(code)
        self.end_headers()

    def log_message(self, *a):
        pass


def make_done_job(files: dict) -> int:
    jid = db.create_job("j", {"name": "j", "input": {"file": "samples/x.OSV"}})
    train = Path(TMP.name) / "cache" / "train" / f"t{jid}"
    export = Path(TMP.name) / "cache" / "export" / f"e{jid}"
    train.mkdir(parents=True)
    export.mkdir(parents=True)
    for name, data in files.items():
        (train / name).write_bytes(data)
        (export / name).symlink_to(train / name)      # as the export stage does
    (export / ".done").write_text("{}")
    db.upsert_stage(jid, "export", "k", "done", path=str(export))
    return jid


class Upload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init()
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), _Sink)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        _Sink.got, _Sink.status = [], 200
        for d in config.RUNS_ROOT.glob("job*"):
            (d / "upload.json").unlink(missing_ok=True)

    def test_archive_follows_links_and_checksums(self):
        jid = make_done_job({"splat_3000.ply": b"p" * 1000, "splat_3000.sog": b"s" * 10})
        st = outputs.upload(jid, {"method": "PUT", "url": f"{self.base}/osv/{{job}}.tar?sig=x"})
        self.assertEqual(st["state"], "done", st)
        got = _Sink.got[0]
        self.assertEqual(got["path"], f"/osv/job{jid:05d}.tar?sig=x")
        self.assertTrue(got["md5_ok"])
        self.assertEqual(got["ctype"], "application/x-tar")
        self.assertEqual(st["sha256"], hashlib.sha256(got["body"]).hexdigest())
        self.assertNotIn("sig=x", json.dumps(st))                 # no credential in status
        with tarfile.open(fileobj=__import__("io").BytesIO(got["body"])) as tar:
            names = {m.name: m for m in tar.getmembers()}
        self.assertIn(f"job{jid:05d}/splat_3000.ply", names)
        self.assertTrue(names[f"job{jid:05d}/splat_3000.ply"].isfile())   # not a link
        self.assertEqual(names[f"job{jid:05d}/splat_3000.ply"].size, 1000)
        self.assertEqual(outputs.status(jid)["state"], "done")
        self.assertFalse((config.RUNS_ROOT / f"job{jid:05d}" / f"job{jid:05d}.tar").exists())

    def test_rejected_upload_fails_loudly_and_keeps_archive(self):
        jid = make_done_job({"splat.ply": b"x" * 10})
        _Sink.status = 403                                        # expired URL
        with patch.object(time, "sleep"):
            st = outputs.upload(jid, {"method": "PUT", "url": f"{self.base}/{{job}}.tar"})
        self.assertEqual(st["state"], "failed")
        self.assertIn("403", st["error"])
        self.assertEqual(len(_Sink.got), 3)                       # retried
        self.assertTrue((config.RUNS_ROOT / f"job{jid:05d}" / f"job{jid:05d}.tar").exists())

    def test_single_object_url_is_used_once(self):
        a = make_done_job({"a.ply": b"a"})
        b = make_done_job({"b.ply": b"b"})
        target = {"method": "PUT", "url": f"{self.base}/fixed.tar"}
        self.assertEqual(outputs.upload(a, target)["state"], "done")
        st = outputs.upload(b, target)
        self.assertEqual(st["state"], "refused")
        self.assertEqual(len(_Sink.got), 1)
        # A new URL for another object is fine: only the same object is refused.
        self.assertEqual(outputs.upload(b, {"method": "PUT", "url": f"{self.base}/other.tar"})["state"], "done")

    def test_no_export_is_a_failure(self):
        jid = db.create_job("empty", {"name": "empty", "input": {"file": "x"}})
        st = outputs.upload(jid, {"method": "PUT", "url": f"{self.base}/{{job}}.tar"})
        self.assertEqual(st["state"], "failed")
        self.assertEqual(_Sink.got, [])

    def test_off_without_url(self):
        with patch.object(outputs, "OUTPUT_UPLOAD", {}), \
                patch.object(threading, "Thread") as t:
            outputs.upload_async(1)
        t.assert_not_called()


class Preflight(unittest.TestCase):
    """Know a target is unusable before the job, without using it up."""

    def test_sigv4_expiry(self):
        t = {"url": "https://s3.example.com/b/k?X-Amz-Algorithm=AWS4-HMAC-SHA256"
                    "&X-Amz-Date=20260921T210000Z&X-Amz-Expires=14400&X-Amz-Signature=x"}
        self.assertEqual(outputs.expires_at(t),
                         __import__("calendar").timegm((2026, 9, 22, 1, 0, 0)))

    def test_sigv2_google_and_none(self):
        self.assertEqual(outputs.expires_at({"url": "https://b.s3.amazonaws.com/k?Expires=1790000000&Signature=x"}),
                         1790000000)
        g = {"url": "https://storage.googleapis.com/b/k?X-Goog-Date=20260921T000000Z&X-Goog-Expires=60"}
        self.assertEqual(outputs.expires_at(g), __import__("calendar").timegm((2026, 9, 21, 0, 1, 0)))
        self.assertIsNone(outputs.expires_at({"url": "https://my.server/upload/{job}"}))

    def test_post_policy_expiry(self):
        pol = base64.b64encode(json.dumps({"expiration": "2026-09-22T09:30:00.000Z",
                                           "conditions": []}).encode()).decode()
        t = {"method": "POST", "url": "https://s3.example.com/b", "fields": {"policy": pol}}
        self.assertEqual(outputs.expires_at(t), __import__("calendar").timegm((2026, 9, 22, 9, 30, 0)))

    def test_expired_target_refuses_jobs(self):
        from fastapi import HTTPException
        from app import main
        old = {"method": "PUT", "url": "https://s3.example.com/b/k?X-Amz-Date=20200101T000000Z&X-Amz-Expires=60"}
        with patch.object(outputs, "OUTPUT_UPLOAD", old):
            self.assertIn("expired", outputs.problems()[0])
            with self.assertRaises(HTTPException) as cm:
                main._refuse_if_cannot_deliver()
            self.assertEqual(cm.exception.status_code, 400)
        with patch.object(outputs, "OUTPUT_UPLOAD", {}), \
                patch.object(outputs, "OUTPUT_UPLOAD_ERROR", "OUTPUT_UPLOAD_URL: bad"):
            with self.assertRaises(HTTPException):
                main._refuse_if_cannot_deliver()
        with patch.object(outputs, "OUTPUT_UPLOAD", {}):
            main._refuse_if_cannot_deliver()                       # off: nothing to refuse
        fresh = {"method": "PUT", "url": "https://s3.example.com/b/k?X-Amz-Date=20990101T000000Z&X-Amz-Expires=60"}
        with patch.object(outputs, "OUTPUT_UPLOAD", fresh):
            self.assertEqual(outputs.problems(), [])

    def test_reachable(self):
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Sink)     # answers HEAD with 501
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            self.assertIsNone(outputs.reachable(f"http://127.0.0.1:{srv.server_port}/b/k?sig=x"))
        finally:
            srv.shutdown()
        self.assertIn("TCP connect", outputs.reachable("http://127.0.0.1:9/b/k", timeout=2))
        self.assertIn("DNS lookup", outputs.reachable("https://no-such-host.invalid/b/k"))


class SelfSigned(unittest.TestCase):
    """QUEUE_TLS_INSECURE: off by default, and then a self-signed host is refused."""

    @classmethod
    def setUpClass(cls):
        import shutil
        import ssl
        import subprocess
        if not shutil.which("openssl"):
            raise unittest.SkipTest("no openssl")
        db.init()
        d = Path(tempfile.mkdtemp(dir=TMP.name))
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                        "-subj", "/CN=localhost", "-keyout", str(d / "k.pem"), "-out", str(d / "c.pem")],
                       check=True, capture_output=True)
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), _Sink)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(d / "c.pem", d / "k.pem")
        cls.srv.socket = ctx.wrap_socket(cls.srv.socket, server_side=True)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"https://127.0.0.1:{cls.srv.server_port}/{{job}}.tar"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_refused_by_default(self):
        self.assertIsNone(config.ssl_context())
        jid = make_done_job({"s.ply": b"s" * 10})
        with patch.object(time, "sleep"):
            st = outputs.upload(jid, {"method": "PUT", "url": self.url})
        self.assertEqual(st["state"], "failed")
        self.assertIn("CERTIFICATE_VERIFY_FAILED", st["error"])

    def test_accepted_when_insecure(self):
        import ssl
        with patch.object(config, "TLS_INSECURE", True):
            ctx = config.ssl_context()
        self.assertEqual(ctx.verify_mode, ssl.CERT_NONE)
        jid = make_done_job({"s.ply": b"s" * 10})
        with patch.object(outputs, "ssl_context", lambda: ctx):
            st = outputs.upload(jid, {"method": "PUT", "url": self.url})
        self.assertEqual(st["state"], "done", st.get("error"))


class BadSettingsDoNotStopTheService(unittest.TestCase):
    """A typo in an optional setting must not stop jobs (restart loop, billing)."""

    def _import_config(self, **env):
        import subprocess
        e = dict(os.environ, **env)
        code = ("import sys; sys.path.insert(0, %r); from app import config; "
                "print('UP', config.TELEMETRY_UPLOAD, config.TELEMETRY_PLACEMENT, "
                "config.OUTPUT_UPLOAD, config.OUTPUT_UPLOAD_ERROR)" % str(Path(__file__).resolve().parent))
        return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True)

    def test_malformed_telemetry_settings(self):
        r = self._import_config(QUEUE_TELEMETRY_UPLOAD="{not json", QUEUE_TELEMETRY_PLACEMENT='["x"]')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("UP {} {}", r.stdout)
        self.assertIn("WARNING: QUEUE_TELEMETRY_UPLOAD ignored", r.stdout)
        self.assertIn("WARNING: QUEUE_TELEMETRY_PLACEMENT ignored", r.stdout)
        r = self._import_config(QUEUE_TELEMETRY_UPLOAD='{"method": "PUT"}')
        self.assertIn('needs a "url"', r.stdout)

    def test_malformed_output_url_is_recorded_not_raised(self):
        r = self._import_config(OUTPUT_UPLOAD_URL="s3://bucket/key")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("must be an http(s) URL", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=1)
