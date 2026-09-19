#!/usr/bin/env python3
"""Lifecycle tests: cancellation races, kill escalation, restart recovery.

Run on the box with the service venv (needs pydantic; no server, no GPU):
    ~/splat/queue_app/venv/bin/python ~/splat/queue_app/test_worker.py

These are the paths that used to be reasoned about and never exercised. Each
one below corresponds to a specific way a job could be left holding a GPU, or a
cache directory left with two writers:

  * a cancel that races the dispatcher and marks a RUNNING job cancelled
    without ever signalling its process;
  * a cancel that lands before the subprocess is registered and signals
    nothing;
  * a process that ignores SIGTERM and is never escalated;
  * restart recovery releasing a lock before the orphan has actually exited;
  * restart recovery signalling a pid that has been reused by something else.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="queue_worker_test_")
# Assigned, never setdefault: run from a shell that has the service's env
# loaded, setdefault kept the LIVE QUEUE_ROOT, and these tests (reconcile()
# included, which signals running jobs) then worked on the real queue.
os.environ["QUEUE_ROOT"] = TMP
os.environ["SPLAT_ROOT"] = TMP
os.environ["QUEUE_GPUS"] = ""
os.environ["QUEUE_CANCEL_GRACE"] = "3"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import db, worker                                    # noqa: E402
from app.stages import (Ctx, lock_path, read_lock,            # noqa: E402
                        release_lock, take_lock)
from app.jobs import JobConfig                                # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def spawn(script, wait_for=None):
    """A child in its own session, like every stage subprocess."""
    p = subprocess.Popen([sys.executable, "-u", "-c", script],
                         stdout=subprocess.PIPE, start_new_session=True)
    if wait_for:
        assert p.stdout.readline().strip() == wait_for.encode()
    return p


IGNORES_TERM = ("import signal, time\n"
                "signal.signal(signal.SIGTERM, lambda *a: None)\n"
                "print('armed', flush=True)\n"
                "while True: time.sleep(0.2)\n")
SLEEPS = "import time\nprint('armed', flush=True)\ntime.sleep(120)\n"

db.init()
CFG = {"name": "t", "input": {"file": "x.mp4", "quick_hash": "deadbeef"}}


# 1. Cancelling a QUEUED job settles it in the database, conditionally.
jid = db.create_job("queued one", CFG)
check("cancel of a queued job returns True", worker.cancel(jid) is True)
check("queued job ends up cancelled", db.get_job(jid)["state"] == "cancelled")
check("cancel leaves no stale intent", jid not in worker._cancel)

# 2. The dispatcher winning the race must NOT produce a job recorded as
# cancelled whose process nobody ever signalled. The conditional UPDATE only
# fires while the row still says queued.
jid = db.create_job("dispatched underneath us", CFG)
db.set_job_state(jid, "running")            # the dispatcher got there first
check("cancel of a running job returns True", worker.cancel(jid) is True)
check("a running job is NOT silently marked cancelled",
      db.get_job(jid)["state"] == "running", db.get_job(jid)["state"])
check("the cancel intent is recorded for the stage loop", jid in worker._cancel)
worker._cancel.discard(jid)

# 3. A cancel that arrives before the subprocess exists still kills it.
# run_stage re-checks _cancel the moment the pid is known.
jid = db.create_job("cancel before popen", CFG)
cfg = JobConfig.model_validate(CFG)
ctx = Ctx(job_id=jid, cfg=cfg, gpu=0, keys=cfg.keys())
worker._cancel.add(jid)
t0 = time.time()
try:
    worker.run_stage(ctx, "frames", [sys.executable, "-c", "import time; time.sleep(120)"],
                     Path(TMP) / "precancel.log")
    check("a pre-registration cancel kills the stage", False, "run_stage returned")
except RuntimeError as e:
    took = time.time() - t0
    check("a pre-registration cancel kills the stage", took < 30,
          f"stage ended in {took:.1f}s: {str(e)[:50]}")
worker._cancel.discard(jid)

# 4. SIGTERM is escalated. A trainer that will not exit used to hold a GPU
# indefinitely with its job already recorded as cancelled.
p = spawn(IGNORES_TERM, wait_for="armed")
worker._terminate(p.pid, waiter=p.wait, grace=3)
time.sleep(1.5)
check("a process ignoring SIGTERM is still alive during the grace period",
      worker._pid_running(p.pid))
check("it is SIGKILLed once the grace period expires", p.wait(timeout=20) == -9,
      f"exit status {p.returncode}")

# 5. A cooperative process is never escalated.
p = spawn(SLEEPS, wait_for="armed")
worker._terminate(p.pid, waiter=p.wait, grace=30)
check("a cooperative process exits on SIGTERM alone", p.wait(timeout=10) == -15,
      f"exit status {p.returncode}")

# 6. Restart recovery: kill the orphan, and only THEN release its lock.
# Releasing first hands the cache directory to a second writer while the first
# is still writing it.
d = Path(TMP) / "orphan_stage"
d.mkdir(parents=True, exist_ok=True)
orphan = spawn(SLEEPS, wait_for="armed")
release_lock(d)
take_lock(d, orphan.pid)
jid = db.create_job("interrupted by a restart", CFG)
db.set_job_state(jid, "running")
db.upsert_stage(jid, "frames", "orphankey", "running", path=str(d))
n = worker.reconcile()
check("reconcile reports the interrupted job", n >= 1, f"n={n}")
check("the orphaned stage process is gone", not worker._pid_running(orphan.pid)
      or orphan.poll() is not None)
check("its lock is released only after it exits", not lock_path(d).exists())
check("the job is marked failed with a resumable explanation",
      db.get_job(jid)["state"] == "failed"
      and "re-queueing" in (db.get_job(jid)["error"] or ""))
orphan.wait(timeout=10)

# 7. PID reuse: a lock naming a live pid that is NOT the process that wrote it
# must never be signalled. The lock is stale, so it is released; the innocent
# process is left alone.
d2 = Path(TMP) / "reused_pid_stage"
d2.mkdir(parents=True, exist_ok=True)
innocent = spawn(SLEEPS, wait_for="armed")
release_lock(d2)
take_lock(d2, innocent.pid)
# Rewrite the identity as if the lock had been written by a long-dead process
# whose pid the kernel later handed to this one.
import json                                                   # noqa: E402
lock_path(d2).write_text(json.dumps(
    {"pid": innocent.pid, "ident": "written-by-a-process-that-died", "at": 0}))
jid = db.create_job("stale lock, reused pid", CFG)
db.set_job_state(jid, "running")
db.upsert_stage(jid, "frames", "reusedkey", "running", path=str(d2))
worker.reconcile()
time.sleep(1.0)
check("a pid-reuse victim is not signalled", innocent.poll() is None,
      f"exit status {innocent.poll()}")
check("the stale lock is still cleared", not lock_path(d2).exists())
innocent.kill()
innocent.wait(timeout=10)

# 8. The dispatcher records _held[job] = gpu BEFORE the job thread starts, so
# every way out of run_job has to give the slot back. Both early exits used to
# sit above the try/finally and leaked the GPU permanently -- with the job left
# queued, so the dispatcher picked it up again and leaked the next slot too.
worker._held.clear()

jid = db.create_job("config predating the current schema",
                    {"name": "t", "input": {"file": "x.mp4"},
                     "train": {"sh_degrees": 3}})       # misspelt: extra=forbid
worker._held[jid] = 0
worker.run_job(jid, 0)
check("an unparseable config does not leak its GPU slot",
      jid not in worker._held, f"_held={worker._held}")
check("and the job is failed, not left queued for the dispatcher to retry",
      db.get_job(jid)["state"] == "failed", db.get_job(jid)["state"])

jid = db.create_job("purged between dispatch and start", CFG)
worker._held[jid] = 1
db.conn().execute("DELETE FROM jobs WHERE id=?", (jid,))
worker.run_job(jid, 1)
check("a job whose row is gone does not leak its GPU slot either",
      jid not in worker._held, f"_held={worker._held}")

# 9. The other side of test 2: a cancel that lands in the window between the
# dispatcher reading the queued row and run_job claiming it. cancel() settles
# the row and drops its intent, so run_job's own transition has to be
# conditional or the job runs to completion after a successful cancel.
jid = db.create_job("cancelled in the dispatch window", CFG)
check("cancel of the queued job returns True", worker.cancel(jid) is True)
worker._held[jid] = 0
worker.run_job(jid, 0)                        # the thread the dispatcher started
check("a job cancelled in the dispatch window stays cancelled",
      db.get_job(jid)["state"] == "cancelled", db.get_job(jid)["state"])
check("and it gives the GPU slot back", jid not in worker._held)

# 10. Queue capacity is what the box can actually run, not what the setting
# allows. GPU 1 carrying somebody else's process is skipped by reserve_gpu(),
# so with max_concurrent=2 the box still runs one job at a time -- and the
# queue ETA has to agree, or the second job is told it starts immediately and
# then waits for the whole of the first.
_real_status = worker.gpu.status
worker.gpu.status = lambda *a, **k: [
    {"index": 0, "schedulable": True, "probe_ok": True, "busy_foreign": False},
    {"index": 1, "schedulable": True, "probe_ok": True, "busy_foreign": True}]
worker._capacity_cache = (0.0, 0)
check("a GPU running foreign work is not queue capacity",
      worker.schedulable_capacity() == 1, str(worker.schedulable_capacity()))

worker.gpu.status = lambda *a, **k: [
    {"index": 0, "schedulable": True, "probe_ok": True, "busy_foreign": False},
    {"index": 1, "schedulable": True, "probe_ok": True, "busy_foreign": False}]
worker._capacity_cache = (0.0, 0)
check("two free GPUs are, up to the configured ceiling",
      worker.schedulable_capacity() == min(2, worker.max_concurrent()))

# An unreadable probe cannot supply a scheduling slot.
worker.gpu.status = lambda *a, **k: [
    {"index": 0, "schedulable": True, "probe_ok": False, "busy_foreign": False}]
worker._capacity_cache = (0.0, 0)
check("a failed GPU probe supplies no scheduling slots",
      worker.schedulable_capacity() == 0)
worker.gpu.status = _real_status
worker._capacity_cache = (0.0, 0)

# 10. A stage keeps its cache dir locked through finalize and until .done is
# published. The lock used to name the (exited) stage subprocess while
# finalize ran, and was released before .done existed; either gap let a sweep
# sibling reclaim the dir and empty it under the finalizer.
jid = db.create_job("lock through finalize", CFG)
ctx = Ctx(job_id=jid, cfg=JobConfig.model_validate(CFG), gpu=0, keys={})
d = Path(TMP) / "cache" / "select" / "lockcheck"
d.mkdir(parents=True)
seen = {}


def _fin(_ctx):
    seen["pid"] = read_lock(d).get("pid")
    return {"ok": 1}


_real_release = worker.release_lock


def _spy_release(p):
    seen["done_at_release"] = (Path(p) / ".done").exists()
    _real_release(p)


worker.release_lock = _spy_release
take_lock(d, os.getpid())
worker._build_stage(ctx, jid, "select",
                    {"argv": lambda c: [sys.executable, "-c", "print('hi')"],
                     "finalize": _fin, "prepare": None},
                    d, "lockcheck", Path(TMP) / "logs" / "lockcheck.log")
worker.release_lock = _real_release
check("finalize runs under the service's lock, not the exited child's pid",
      seen.get("pid") == os.getpid(), str(seen.get("pid")))
check(".done is published before the lock is released",
      seen.get("done_at_release") is True)
check("and the lock is gone afterwards", not lock_path(d).exists())

# 11. A disk-space refusal after the lock is taken releases it again. It used
# to leave the lock held by the live service, so every job sharing the stage
# waited out the full cache-wait timeout even after space was freed.
d2 = Path(TMP) / "cache" / "select" / "nospace"
d2.mkdir(parents=True)
take_lock(d2, os.getpid())
_real_space = worker.retention.ensure_space


def _no_space(stage):
    raise RuntimeError("not enough free disk")


worker.retention.ensure_space = _no_space
try:
    worker._build_stage(ctx, jid, "select",
                        {"argv": lambda c: [], "finalize": None, "prepare": None},
                        d2, "nospace", Path(TMP) / "logs" / "nospace.log")
    raised = False
except RuntimeError:
    raised = True
worker.retention.ensure_space = _real_space
check("a disk-space refusal still fails the stage", raised)
check("and releases the stage lock", not lock_path(d2).exists())

print()
print("FAILURES:", fails if fails else "none")
raise SystemExit(1 if fails else 0)
