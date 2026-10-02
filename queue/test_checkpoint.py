#!/usr/bin/env python3
"""Restore points and resume for spot GPUs (#10).

CPU only, no GPU, no trainer: LichtFeld's output lines come from a small fake
trainer, and its `lichtfeld.io` module from a stub that "cleans" by copying, so
scripts/licht_restore_point.py runs for real around it. What the real module
does to a project was measured on 3090s (issue #10); this checks everything
the queue does around it.

    python3 queue/test_checkpoint.py
"""
import hashlib
import http.server
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

TMP = tempfile.mkdtemp(prefix="queue_checkpoint_test_")
# Assigned, never setdefault (AGENTS.md, "Tests").
os.environ["SPLAT_ROOT"] = TMP
os.environ["QUEUE_ROOT"] = str(Path(TMP) / "q")
os.environ["QUEUE_GPUS"] = ""
os.environ["QUEUE_PREP_BACKEND"] = "cuda"      # a CUDA host, also when run on a Mac
os.environ["QUEUE_TELEMETRY"] = "1"
os.environ["QUEUE_RESTORE_POINTS"] = "1"          # local restore points, no upload target
os.environ["QUEUE_LFS_PYTHON"] = sys.executable
os.environ.pop("CHECKPOINT_UPLOAD_URL", None)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The deployed layout: scripts/ copied under SPLAT_ROOT, the trainer's build
# directory with its Python module (here a stub) under src/python.
shutil.copytree(HERE.parent / "scripts", Path(TMP) / "scripts",
                ignore=shutil.ignore_patterns("__pycache__", "test_*"))
STUB = Path(TMP) / "LichtFeld-Studio" / "build" / "src" / "python" / "lichtfeld"
STUB.mkdir(parents=True)
(STUB / "__init__.py").write_text("")
(STUB / "io.py").write_text('''
import os
class _V:
    def __init__(self, ok):
        self.status = "ProjectVerificationStatus." + ("VERIFIED" if ok else "CHUNK_MISMATCH")
        self.verified_chunks = 12
        self.first_mismatch = None if ok else 3
def clean_project_file(src, dst="", expected_commit="", progress=None, cancel=None):
    # Keep the "latest checkpoint": the last 1000 bytes the fake trainer appended.
    data = open(src, "rb").read()
    open(dst, "wb").write(b"CLEAN" + data[-1000:])
def verify_project_file(path):
    return _V(os.environ.get("FAKE_VERIFY", "ok") == "ok")
''')

from app import checkpoints, config, db, handoff, preempt, stages, telemetry, worker  # noqa: E402
checkpoints.RETRY_S = 0.0
from app.jobs import TRAINER, JobConfig, key_of                                     # noqa: E402
from app.stages import ORDER, STAGES, Ctx                                           # noqa: E402

db.init()
CACHE = config.CACHE_ROOT
MP4 = {"name": "pano", "input": {"file": "samples/c.mp4", "quick_hash": "q2"},
       "train": {"checkpoint_every": 1000, "iter": 30000, "steps_scaler": 0.1}}
OSV = {"name": "rig", "input": {"file": "samples/c.OSV", "quick_hash": "q1"}}

# LichtFeld's lines, as 3067e9e0 with scripts/lichtfeld-patches/ prints them.
FAKE_TRAINER = Path(TMP) / "fake_trainer.py"
FAKE_TRAINER.write_text('''
import os, pathlib, signal, sys, time
out, total = pathlib.Path(sys.argv[1]), int(sys.argv[2])
snaps = {int(x) for x in sys.argv[3].split(",") if x}
out.mkdir(parents=True, exist_ok=True)
gen, pending = [0], [False]
def snapshot(it):
    gen[0] += 1
    u = "c3d3dfa1-%04d" % gen[0]
    with open(out / "project.licht", "ab") as f:
        f.write(os.urandom(1000))
    print(f"[08:32:31.783] [info] trainer.cpp:4247  Prepared .licht snapshot {u} for "
          f"iteration {it} (547129741 checkpoint bytes)", flush=True)
    print(f"[08:32:32.233] [info] trainer.cpp:5301  Background .licht append complete: "
          f"{out}/project.licht generation={gen[0]} snapshot={u} rewritten=5 reused=7", flush=True)
print("Loading dataset from: x", flush=True)
time.sleep(float(os.environ.get("FAKE_LOAD_S", "0")))
signal.signal(signal.SIGUSR1, lambda *a: pending.__setitem__(0, True))
print("[08:32:30.745] [info] trainer.cpp:3168  Trainer initialization complete", flush=True)
for it in range(1, total + 1):
    if it in snaps:
        snapshot(it)
    if pending[0]:
        pending[0] = False
        snapshot(it)
    if it % 100 == 0:
        print(f"Training [##] {it}/{total} | Loss: 0.1 | Splats: 10", flush=True)
    time.sleep(float(os.environ.get("FAKE_STEP_S", "0")))
snapshot(total)
''')


class _NoVram:
    peak = None

    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def _ctx(cfg_d: dict, job_id=None) -> Ctx:
    cfg = JobConfig.model_validate(cfg_d)
    jid = job_id or db.create_job(cfg.name, cfg.model_dump())
    return Ctx(job_id=jid, cfg=cfg, gpu=0, keys=cfg.keys())


PACED = {"FAKE_STEP_S": "0.0005"}          # ~1.5 s for 3000 steps: time to make each


def _run_fake_trainer(ctx: Ctx, total: int, snaps: str, env: dict = None,
                      during=None) -> dict:
    """run_stage("train") around the fake trainer; `during(ctx)` runs alongside."""
    env = {**PACED, **(env or {})}
    argv = [sys.executable, str(FAKE_TRAINER), str(ctx.dir("train")), str(total), snaps]
    th = threading.Thread(target=during, args=(ctx,)) if during else None
    with patch.object(worker, "VramSampler", _NoVram), \
            patch.dict(os.environ, env or {}):
        if th:
            th.start()
        out = worker.run_stage(ctx, "train", argv, Path(TMP) / f"train{ctx.job_id}.log")
        if th:
            th.join(30)
    return out


def _wait_for(pred, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _restore_points(job_id):
    return (checkpoints.status(job_id) or {}).get("restore_points") or []


# ------------------------------------------------------------------ config

class Config(unittest.TestCase):
    def test_checkpoint_every_bounds(self):
        JobConfig.model_validate({**MP4, "train": {"checkpoint_every": 0}})
        with self.assertRaisesRegex(ValueError, "at least 500"):
            JobConfig.model_validate({**MP4, "train": {"checkpoint_every": 499}})

    def test_snapshot_steps_are_the_jobs_own(self):
        t = JobConfig.model_validate({**OSV, "train": {"checkpoint_every": 5000}}).train
        self.assertEqual(t.checkpoint_steps(), [5000, 10000, 15000, 20000, 25000])
        t = JobConfig.model_validate(MP4).train                   # 3000 steps
        self.assertEqual(t.checkpoint_steps(), [1000, 2000])
        self.assertEqual(JobConfig.model_validate(OSV).train.checkpoint_steps(), [])

    def test_not_a_cache_key_term(self):
        a = JobConfig.model_validate(OSV)
        b = JobConfig.model_validate({**OSV, "train": {"checkpoint_every": 5000}})
        self.assertEqual(a.keys(), b.keys())
        # And the key is what it was before the field existed: the train
        # section hashed without it.
        d = a.train.model_dump()
        d.pop("checkpoint_every")
        self.assertEqual(a.k_train(), key_of("train", TRAINER, a.k_sfm(), a.k_mask(), d))

    def test_save_steps_is_managed(self):
        with self.assertRaisesRegex(ValueError, "--save-steps"):
            JobConfig.model_validate({**OSV, "train": {"extra_args": "--save-steps 100"}})
        with self.assertRaisesRegex(ValueError, "--resume"):
            JobConfig.model_validate({**OSV, "train": {"extra_args": "--resume x.licht"}})


class Argv(unittest.TestCase):
    def argv(self, cfg_d, resume=None):
        ctx = _ctx(cfg_d)
        ctx.derived.update(dataset=str(CACHE / "sfm" / "x" / "dataset"),
                           images=str(CACHE / "select" / "x"), needs_gut=True)
        if resume:
            ctx.derived["resume"] = resume
        return ctx, STAGES["train"]["argv"](ctx)

    def save_steps(self, argv):
        return [a for a in argv if a.startswith("--save-steps")]

    def test_unscaled_save_steps(self):
        # LichtFeld scales --save-steps by --steps-scaler, like --eval-steps.
        _, a = self.argv({**OSV, "train": {"checkpoint_every": 5000}})
        self.assertEqual(self.save_steps(a), ["--save-steps=5000,10000,15000,20000,25000"])
        _, a = self.argv(MP4)                                     # scaler 0.1: steps 1000, 2000
        self.assertEqual(self.save_steps(a), ["--save-steps=10000,20000"])
        _, a = self.argv({**OSV, "train": {"iter": 50000, "checkpoint_every": 10000}})
        self.assertIn("--steps-scaler", a)                        # bare iter -> scaler 5/3
        self.assertEqual(self.save_steps(a), ["--save-steps=6000,12000,18000,24000"])

    def test_off_leaves_lichtfelds_defaults(self):
        _, a = self.argv(OSV)
        self.assertEqual(self.save_steps(a), [])

    def test_resume_passes_nothing_but_the_restore_point_and_export(self):
        ctx, a = self.argv(MP4, resume={"licht": "/r/x.licht"})
        self.assertEqual(a, [str(config.LFS_BIN), "--headless", "--resume", "/r/x.licht",
                             "-o", str(ctx.dir("train")), "--export=ply,sog,spz"])
        ctx, a = self.argv(OSV, resume={"licht": "/r/y.licht"})
        # The fisheye view is rebuilt first, then script 89 execs the resume.
        self.assertEqual(a[1], str(config.FISHEYE_TRAIN))
        self.assertEqual(a[a.index("--") + 1:], [str(config.LFS_BIN), "--headless", "--resume",
                                                  "/r/y.licht", "-o", str(ctx.dir("train")),
                                                  "--export=ply,sog,spz"])

    def test_space_estimate_counts_snapshots(self):
        off = stages.space_estimate(_ctx(OSV), "train")
        on = stages.space_estimate(_ctx({**OSV, "train": {"checkpoint_every": 5000}}), "train")
        self.assertGreater(on, off * 2)


# ---------------------------------------------------------------- tracker

class Tracker(unittest.TestCase):
    def test_lines(self):
        t = checkpoints.Tracker()
        self.assertFalse(t.ready)
        self.assertIsNone(t.feed("[..] trainer.cpp:3168  Trainer initialization complete"))
        self.assertTrue(t.ready)
        self.assertIsNone(t.feed("Prepared .licht snapshot abc-1 for iteration 700 (5 checkpoint bytes)"))
        self.assertEqual(t.feed("Background .licht append complete: /d/project.licht "
                                "generation=2 snapshot=abc-1 rewritten=5 reused=7"), (2, 700))
        # A commit whose preparation was not seen says nothing about its step.
        self.assertIsNone(t.feed("Background .licht append complete: /d/project.licht "
                                 "generation=3 snapshot=zzz rewritten=5"))


# ----------------------------------------------------------- making them

class Making(unittest.TestCase):
    def test_restore_point_per_snapshot_but_not_the_final_save(self):
        ctx = _ctx(MP4)
        # ~1 s between snapshots and after the last: each is made before training
        # ends (a restore point still being made then is dropped, by design).
        out = _run_fake_trainer(ctx, 3000, "1000,2000", env={"FAKE_STEP_S": "0.001"})
        self.assertEqual(out.get("step"), 3000)
        self.assertTrue(_wait_for(lambda: len(_restore_points(ctx.job_id)) == 2))
        time.sleep(0.5)
        pts = _restore_points(ctx.job_id)
        self.assertEqual([p["iteration"] for p in pts], [1000, 2000])   # not 3000
        tar = checkpoints.local_path(ctx.job_id)
        meta = checkpoints.read_meta(tar)
        self.assertEqual((meta["iteration"], meta["generation"]), (2000, 2))
        self.assertEqual(meta["trainer"], TRAINER)
        self.assertEqual(meta["keys"], ctx.keys)
        self.assertEqual(meta["cache_root"], str(CACHE))
        with tarfile.open(tar) as tf:
            licht = tf.extractfile(checkpoints.LICHT).read()
        self.assertTrue(licht.startswith(b"CLEAN"))
        self.assertEqual(hashlib.sha256(licht).hexdigest(), meta["licht"]["sha256"])

    def test_one_that_does_not_verify_is_not_kept(self):
        ctx = _ctx(MP4)
        _run_fake_trainer(ctx, 3000, "1000", env={"FAKE_VERIFY": "bad"})
        self.assertTrue(_wait_for(lambda: _restore_points(ctx.job_id)))
        time.sleep(0.3)
        (rec,) = _restore_points(ctx.job_id)
        self.assertIn("does not verify", rec["error"])
        self.assertFalse(checkpoints.local_path(ctx.job_id).exists())

    def test_sigusr1_only_once_the_trainer_is_up(self):
        ctx = _ctx(MP4)
        seen = {}

        def during(c):
            # While the "dataset loads", nobody is asked: SIGUSR1 would kill it.
            _wait_for(lambda: c.job_id in worker._trainers, 10)
            seen["early"] = worker.request_snapshots()
            _wait_for(lambda: worker._trainers.get(c.job_id, (None, checkpoints.Tracker()))[1].ready, 10)
            time.sleep(0.2)
            seen["ready"] = worker.request_snapshots()

        _run_fake_trainer(ctx, 3000, "", env={"FAKE_LOAD_S": "1", "FAKE_STEP_S": "0.001"},
                          during=during)
        self.assertEqual(seen["early"], [])
        self.assertEqual(seen["ready"], [ctx.job_id])
        self.assertTrue(_wait_for(lambda: _restore_points(ctx.job_id)))
        self.assertLess(_restore_points(ctx.job_id)[0]["iteration"], 3000)
        self.assertNotIn(ctx.job_id, worker._trainers)


# ----------------------------------------------------------- uploading

class _Sink(http.server.BaseHTTPRequestHandler):
    got, etag = [], None

    def do_PUT(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        _Sink.got.append((self.path, body))
        self.send_response(200)
        self.send_header("ETag", '"%s"' % (_Sink.etag or hashlib.md5(body).hexdigest()))
        self.end_headers()

    def log_message(self, *a):
        pass


class Uploading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), _Sink)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        _Sink.got, _Sink.etag = [], None

    def made(self):
        ctx = _ctx(MP4)
        project = ctx.dir("train") / "project.licht"
        project.parent.mkdir(parents=True, exist_ok=True)
        project.write_bytes(os.urandom(5000))
        tar, rec = checkpoints.make(ctx, 1, 1000, dict(os.environ))
        return ctx, tar, rec

    def test_put_replaces_one_object_per_job(self):
        ctx, tar, rec = self.made()
        target = {"method": "PUT", "url": self.base + "/cp/{job}/{file}"}
        up = checkpoints.upload(ctx.job_id, tar, rec, target)
        self.assertEqual(up["state"], "done", up)
        path, body = _Sink.got[-1]
        self.assertEqual(path, f"/cp/job{ctx.job_id:05d}/job{ctx.job_id:05d}-restore.tar")
        self.assertEqual(body, tar.read_bytes())

    def test_damaged_upload_fails(self):
        ctx, tar, rec = self.made()
        _Sink.etag = "0" * 32
        up = checkpoints.upload(ctx.job_id, tar, rec, {"url": self.base + "/x/{job}"})
        self.assertEqual(up["state"], "failed")
        self.assertIn("ETag", up["error"])

    def test_fixed_url_belongs_to_the_first_job(self):
        target = {"url": self.base + "/fixed.tar"}
        a, tar_a, rec_a = self.made()
        m = checkpoints.Maker(a, dict(os.environ), upload_target=target)
        self.assertEqual(checkpoints.upload(a.job_id, tar_a, rec_a, target)["state"], "done")
        checkpoints._update(a.job_id, lambda st: checkpoints._append(
            st, {**rec_a, "upload": {"state": "done"}}))
        b, tar_b, rec_b = self.made()
        checkpoints.Maker(b, dict(os.environ), upload_target=target)
        up = checkpoints.upload(b.job_id, tar_b, rec_b, target)
        self.assertEqual(up["state"], "refused")
        self.assertIn(f"job {a.job_id}", up["error"])
        del m

    def test_maker_sends_each_and_skips_after_completion(self):
        ctx = _ctx(MP4)
        target = {"url": self.base + "/m/{job}.tar"}
        with patch.object(checkpoints, "CHECKPOINT_UPLOAD", target):
            _run_fake_trainer(ctx, 3000, "1000,2000")
        self.assertTrue(_wait_for(lambda: len(_restore_points(ctx.job_id)) >= 1))
        time.sleep(0.5)
        states = [(p.get("upload") or {}).get("state") for p in _restore_points(ctx.job_id)]
        # Sent while training ran; one still being made at the end is not.
        self.assertIn("done", states)
        self.assertTrue(set(states) <= {"done", "skipped"}, states)


# ----------------------------------------------------------------- resume

def build_cache(cfg: JobConfig) -> None:
    """Stitched cache entries for frames..sfm, the shape test_handoff.py uses."""
    k = cfg.keys()
    fr, se, sf = (CACHE / s / k[s] for s in ("frames", "select", "sfm"))
    for d in (fr, se, sf):
        shutil.rmtree(d, ignore_errors=True)
    for n in range(1, 21):
        p = fr / f"{n:06d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(os.urandom(64))
    (fr / ".done").write_text(json.dumps({"candidates": 20}))
    se.mkdir(parents=True)
    for n in range(1, 11):
        os.link(fr / f"{2 * n:06d}.jpg", se / f"pano_{n:04d}.jpg")
    (se / ".done").write_text(json.dumps({"panos": 10, "window": 2}))
    run = sf / "sfm_spherical_incremental"
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        (run / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (run / "sparse" / "0" / f).write_bytes(os.urandom(64))
    (run / "database.db").write_bytes(os.urandom(256))
    (sf / "dataset" / "sparse").mkdir(parents=True)
    (sf / "dataset" / "sparse" / "0").symlink_to((run / "sparse" / "0").resolve(),
                                                  target_is_directory=True)
    (sf / ".done").write_text(json.dumps({"num_reg_frames": 10, "registration_pct": 100.0,
                                          "needs_gut": False, "images_dir": str(se),
                                          "warnings": []}))


def restore_point(cfg_d=MP4, iteration=1000, **meta_over) -> tuple[str, dict]:
    """A restore point as a training run of cfg_d at `iteration` makes it."""
    ctx = _ctx(cfg_d)
    project = ctx.dir("train") / "project.licht"
    project.parent.mkdir(parents=True, exist_ok=True)
    project.write_bytes(os.urandom(3000))
    tar, rec = checkpoints.make(ctx, 1, iteration, dict(os.environ))
    if meta_over:
        with tarfile.open(tar) as tf:
            meta = json.loads(tf.extractfile(checkpoints.META).read())
            licht = tf.extractfile(checkpoints.LICHT).read()
        meta.update(meta_over)
        with tarfile.open(tar, "w") as tf:
            for name, data in ((checkpoints.META, json.dumps(meta).encode()),
                               (checkpoints.LICHT, licht)):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    name = f"job{ctx.job_id:05d}-restore.tar"
    shutil.copy(tar, config.RESUME_ROOT / name)
    return name, checkpoints.read_meta(config.RESUME_ROOT / name)


class Resume(unittest.TestCase):
    def test_refused_unless_it_resumes_faithfully(self):
        cases = [({"trainer": "lichtfeld-other"}, "trainer"),
                 ({"cache_root": "/elsewhere/cache"}, "absolute paths"),
                 ({"version_terms": {"config_version": 1}}, "version terms"),
                 ({"iteration": 3000}, "nothing to resume")]
        for over, why in cases:
            with self.subTest(why=why):
                name, _ = restore_point(**over)
                with self.assertRaisesRegex(checkpoints.ResumeError, why):
                    checkpoints.import_restore(name)
        name, meta = restore_point()
        keys = dict(meta["keys"])
        keys["train"] = "0" * 16
        name, _ = restore_point(keys=keys)
        with self.assertRaisesRegex(checkpoints.ResumeError, "train key"):
            checkpoints.import_restore(name)

    def test_damaged_checkpoint_is_refused(self):
        name, meta = restore_point(licht={"bytes": 1, "sha256": "0" * 64})
        with self.assertRaisesRegex(checkpoints.ResumeError, "sha256"):
            checkpoints.import_restore(name)
        with self.assertRaisesRegex(checkpoints.ResumeError, "sha256"):
            checkpoints.import_restore(restore_point()[0], sha256="1" * 64)

    def test_same_machine_needs_the_cache(self):
        cfg = JobConfig.model_validate(MP4)
        for st in ("frames", "select", "sfm"):
            shutil.rmtree(CACHE / st / cfg.keys()[st], ignore_errors=True)
        name, _ = restore_point()
        with self.assertRaisesRegex(checkpoints.ResumeError, "handoff bundle"):
            checkpoints.import_restore(name)
        build_cache(cfg)
        out = checkpoints.import_restore(name, job_name="resumed")
        row = db.get_job(out["id"])
        r = db.job_resume(row)                     # set with the row, not after it
        self.assertEqual((row["name"], row["state"]), ("resumed", "queued"))
        self.assertEqual((r["from_step"], r["effective_iters"]), (1000, 3000))
        self.assertTrue(Path(r["licht"]).is_file())
        self.assertEqual(json.loads(row["config"])["train"]["checkpoint_every"], 1000)

    def test_worker_resumes_with_the_restore_point(self):
        cfg = JobConfig.model_validate(MP4)
        build_cache(cfg)
        out = checkpoints.import_restore(restore_point()[0])
        seen = {}

        def argv(ctx):
            seen["argv"] = stages._lichtfeld_argv(ctx, "d", "i", True, masked=False)
            return None

        fakes = {"train": {**STAGES["train"], "argv": argv,
                           "finalize": lambda ctx: {"final_step": 3000}},
                 "export": {**STAGES["export"], "finalize": lambda ctx: {}}}
        with patch.dict(STAGES, fakes), \
                patch.object(worker.retention, "ensure_space", lambda *a, **k: None), \
                patch.object(worker, "_check_input_unchanged", lambda cfg: None):
            worker.run_job(out["id"], -1)
        row = db.get_job(out["id"])
        self.assertEqual(row["state"], "done", row["error"])
        self.assertEqual(seen["argv"][2:4], ["--resume", out["resume"]["licht"]])
        # Done: the extracted checkpoint is not kept.
        self.assertFalse(Path(out["resume"]["licht"]).exists())

    def test_with_a_handoff_bundle(self):
        cfg = JobConfig.model_validate({**MP4, "run_until": "sfm"})
        build_cache(cfg)
        prep = db.create_job("prep", cfg.model_dump())
        for st, key in cfg.keys().items():
            state = "done" if st in ("frames", "select", "sfm") else "skipped"
            db.upsert_stage(prep, st, key, state, path=str(CACHE / st / key),
                            progress=(CACHE / st / key / ".done").read_text()
                            if state == "done" else None)
        handoff.export(prep)
        bundle = handoff.bundle_path(prep)
        shutil.copy(bundle, config.HANDOFF_ROOT / bundle.name)
        name, _ = restore_point()
        out = checkpoints.import_restore(name, bundle=bundle.name)
        row = db.get_job(out["id"])
        self.assertEqual(db.job_resume(row)["id"], out["resume"]["id"])
        self.assertIsNotNone(db.job_handoff(row))
        # The bundle brings only the upstream stages, so a restore point with
        # other training settings for the same clip resumes from it...
        tweaked, _ = restore_point({**MP4, "train": {**MP4["train"], "sh_degree": 1}})
        shutil.copy(bundle, config.HANDOFF_ROOT / bundle.name)
        self.assertIn("id", checkpoints.import_restore(tweaked, bundle=bundle.name))
        # ...while one for another clip is refused before anything is queued.
        other, _ = restore_point({**MP4, "input": {"file": "samples/d.mp4", "quick_hash": "q9"}})
        shutil.copy(bundle, config.HANDOFF_ROOT / bundle.name)
        before = db.conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        with self.assertRaisesRegex(handoff.HandoffError, "not for the restore point"):
            checkpoints.import_restore(other, bundle=bundle.name)
        self.assertEqual(db.conn().execute("SELECT COUNT(*) FROM jobs").fetchone()[0], before)


# ------------------------------------------------------------- notices

class _Meta(http.server.BaseHTTPRequestHandler):
    action = rebalance = preempted = None

    def _send(self, body):
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body.encode())

    def do_PUT(self):
        self._send("token" if self.path == "/latest/api/token" else None)

    def do_GET(self):
        if self.path.startswith("/latest/") and self.headers.get("X-aws-ec2-metadata-token") != "token":
            self.send_response(401)
            self.end_headers()
            return
        self._send({"/latest/meta-data/spot/instance-action": _Meta.action,
                    "/latest/meta-data/events/recommendations/rebalance": _Meta.rebalance,
                    "/computeMetadata/v1/instance/preempted":
                        _Meta.preempted if self.headers.get("Metadata-Flavor") == "Google" else None,
                    }.get(self.path))

    def log_message(self, *a):
        pass


class Notices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), _Meta)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        _Meta.action = _Meta.rebalance = _Meta.preempted = None

    def test_aws(self):
        src = preempt.Aws(self.base)
        self.assertIsNone(src.poll())
        _Meta.rebalance = '{"noticeTime": "2026-09-30T08:22:00Z"}'
        self.assertEqual(src.poll()["kind"], "rebalance")
        _Meta.action = '{"action": "terminate", "time": "2026-09-30T08:24:00Z"}'
        n = src.poll()
        self.assertEqual((n["kind"], n["action"]), ("reclaim", "terminate"))

    def test_gcp(self):
        src = preempt.Gcp(self.base)
        _Meta.preempted = "FALSE"
        self.assertIsNone(src.poll())
        _Meta.preempted = "TRUE"
        self.assertEqual(src.poll()["kind"], "reclaim")

    def test_reclaim_snapshots_and_pauses_once(self):
        calls = []
        was = worker.paused()
        try:
            with patch.object(worker, "request_snapshots", lambda: calls.append(1) or [7]), \
                    patch.object(checkpoints, "record_notice", lambda *a: None):
                stop = threading.Event()
                _Meta.action = '{"action": "stop", "time": "x"}'
                th = threading.Thread(target=preempt.watch,
                                      args=(preempt.Aws(self.base), 0.05, stop))
                th.start()
                self.assertTrue(_wait_for(lambda: calls, 5))
                time.sleep(0.3)
                stop.set()
                th.join(5)
            self.assertEqual(len(calls), 1)            # acted on once, however often polled
            self.assertTrue(worker.paused())
            self.assertEqual(json.loads(preempt.NOTICE_RECORD.read_text())["jobs"], [7])
        finally:
            worker.set_paused(was)

    def test_rebalance_snapshots_without_pausing(self):
        was = worker.paused()
        worker.set_paused(False)
        try:
            with patch.object(worker, "request_snapshots", lambda: []):
                preempt.handle({"kind": "rebalance", "provider": "aws"})
            self.assertFalse(worker.paused())
        finally:
            worker.set_paused(was)


# ------------------------------------------------------------ telemetry

class Telemetry(unittest.TestCase):
    def test_records_counts_not_urls(self):
        ctx = _ctx(MP4)
        target = {"url": "http://127.0.0.1:9/secret-bucket/{job}.tar"}
        with patch.object(checkpoints, "CHECKPOINT_UPLOAD", target), \
                patch.object(checkpoints, "RETRY_S", 0.0):
            _run_fake_trainer(ctx, 3000, "1000")
        self.assertTrue(_wait_for(lambda: _restore_points(ctx.job_id)))
        time.sleep(0.3)
        rec = telemetry.build(ctx.job_id)
        cp = rec["checkpoints"]
        self.assertEqual((cp["every"], cp["restore_points"], cp["failed"]), (1000, 1, 1))
        self.assertEqual(cp["last"]["iteration"], 1000)
        self.assertNotIn("secret-bucket", json.dumps(rec))
        self.assertIsNone(rec["resumed"])


if __name__ == "__main__":
    try:
        unittest.main(verbosity=1)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
