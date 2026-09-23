"""Scheduler: one dispatcher thread, one worker thread per running job.

A job holds a single GPU for its whole lifetime. That is coarser than the
resource model in the design doc (frames/select do not really need a GPU) but it
keeps the scheduler honest and still gives two concurrent jobs on this box.
30_run_sfm.py hardcodes gpu_index='0', so the GPU is selected by setting
CUDA_VISIBLE_DEVICES, which remaps the chosen device to index 0.
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator, Optional

from . import db, gpu, outputs, retention, telemetry
from .config import (DEFAULT_MAX_CONCURRENT, FIRST_PROGRESS_GRACE, LOG_ROOT,
                     SPLAT_ROOT, STALL_TIMEOUT, START_PAUSED)
from .jobs import JobConfig, quick_hash
from .resources import stage_threads, thread_env
from .stages import (ORDER, STAGES, Ctx, dir_bytes, done_marker, images_dir,
                     is_cached, lock_holder_alive, mark_done, pid_alive,
                     read_done, read_lock, release_lock, reset_stage_dir,
                     restamp_lock, take_lock)

class ReviewRequired(Exception):
    """The mask stage is done and a human has not signed it off yet.

    Not a failure. The job stops before the expensive stages, gives back its
    GPU, and waits in `awaiting_review` until approved -- at which point it is
    re-queued and every completed stage is a cache hit, so approval costs
    nothing to resume. That is why this is an early exit rather than a block
    inside the stage loop: blocking would hold a GPU idle for however long the
    human takes.
    """


_procs: dict[int, subprocess.Popen] = {}      # job_id -> running subprocess
_cancel: set[int] = set()
_held: dict[int, int] = {}                    # job_id -> gpu index
_reserved: dict[str, int] = {}                # token -> gpu index (renders)
_lock = threading.Lock()
# Serialises "look at the GPUs, then take one". Held across the nvidia-smi call,
# which _lock is not, because two allocators that both look before either takes
# will both pick the same GPU: a compare render and a dispatch can land on one
# card before either process shows up in GPU telemetry.
_alloc_lock = threading.Lock()
_stop = threading.Event()
# Set while the host benchmark runs: it measures the whole machine (every
# core, the disk, a GPU), so nothing is dispatched or reserved beside it.
_exclusive = threading.Event()


def own_pids() -> list[int]:
    with _lock:
        pids = []
        for p in _procs.values():
            if p.poll() is None:
                pids.append(p.pid)
                pids.extend(_child_pids(p.pid))
        return pids


def _child_pids(pid: int) -> list[int]:
    """Direct children, so a wrapper process does not hide the real GPU user."""
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True,
                             text=True, timeout=5)
        return [int(x) for x in out.stdout.split()]
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []


def paused() -> bool:
    return db.get_setting("paused", "1" if START_PAUSED else "0") == "1"


def enforce_start_paused() -> bool:
    """Make "the queue starts paused" true rather than merely usually true.

    paused is persisted, so a queue resumed before a restart came back
    scheduling: the documented promise that deploying cannot disturb a
    hand-launched run held only until the first time anyone pressed resume.
    Set QUEUE_START_PAUSED=0 to keep the stored state across restarts instead.
    """
    if not START_PAUSED:
        return False
    was = paused()
    set_paused(True)
    return not was


def set_paused(v: bool) -> None:
    db.set_setting("paused", "1" if v else "0")


def max_concurrent() -> int:
    try:
        return max(1, int(db.get_setting("max_concurrent",
                                         str(DEFAULT_MAX_CONCURRENT))))
    except ValueError:
        return DEFAULT_MAX_CONCURRENT


def set_max_concurrent(n: int) -> None:
    global _capacity_cache
    db.set_setting("max_concurrent", str(max(1, int(n))))
    _capacity_cache = (0.0, 0)


# Two subprocess probes per call, and both the queue view and the status poll
# want it every four seconds. A GPU does not change hands that fast.
CAPACITY_TTL = 5.0
_capacity_cache: tuple[float, int] = (0.0, 0)


def schedulable_capacity() -> int:
    """How many jobs can actually be in flight, not how many are allowed to be.

    max_concurrent is a ceiling, not a capacity: reserve_gpu() skips any card
    carrying somebody else's process, so a box with one of two GPUs busy runs
    one job at a time whatever the setting says. A queue ETA that took the
    setting at face value told the second job it would start immediately, and
    then left it queued for the length of the first one.
    """
    global _capacity_cache
    at, val = _capacity_cache
    now = time.monotonic()
    if at and now - at < CAPACITY_TTL:
        return val
    rows = gpu.status(own_pids(), held_gpus())
    # Match dispatch: unavailable or unprobed GPUs cannot accept new jobs.
    val = min(max_concurrent(),
              sum(1 for r in rows if r["schedulable"]
                  and r["probe_ok"] and not r["busy_foreign"]))
    _capacity_cache = (now, val)
    return val


# ----------------------------------------------------------- GPU allocation

def held_gpus() -> list[int]:
    """Every GPU this service has handed out: running jobs plus reservations."""
    with _lock:
        return list(_held.values()) + list(_reserved.values())


@contextlib.contextmanager
def reserve_gpu() -> Iterator[Optional[int]]:
    """Take one free GPU for the duration of the block, or yield None.

    Used by work that runs inside a request (compare renders) rather than as a
    queued job, so the dispatcher cannot hand the same card to a training run
    while the render is still starting up.
    """
    token = uuid.uuid4().hex
    with _alloc_lock:
        free = [] if _exclusive.is_set() else gpu.free_gpus(
            own_pids=own_pids(), held=held_gpus())
        if not free:
            chosen = None
        else:
            chosen = free[0]
            with _lock:
                _reserved[token] = chosen
    try:
        yield chosen
    finally:
        with _lock:
            _reserved.pop(token, None)


@contextlib.contextmanager
def exclusive() -> Iterator[Optional[list[int]]]:
    """Hold the whole machine for the block: yields the free GPUs, or None.

    None when anything of this service's is running (a job, a render, another
    holder); the caller refuses rather than waits. While held, the dispatcher
    starts no job and reserve_gpu() hands out nothing.
    """
    with _alloc_lock:
        with _lock:
            busy = bool(_held or _reserved) or _exclusive.is_set()
        if busy:
            got = None
        else:
            _exclusive.set()
            got = gpu.free_gpus(own_pids=own_pids(), held=[])
    if got is None:
        yield None
        return
    try:
        yield got
    finally:
        _exclusive.clear()


def exclusive_held() -> bool:
    return _exclusive.is_set()


# --------------------------------------------------------------- termination

# How long a cancelled or orphaned process gets to exit on SIGTERM before it is
# killed. LichtFeld exports what it has on SIGTERM, which takes seconds on a 3M
# splat model, so the grace period has to allow for that.
CANCEL_GRACE = float(os.environ.get("QUEUE_CANCEL_GRACE", 60))


def _signal_group(pid: int, sig: int) -> None:
    """Signal a process group, falling back to the single process."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _terminate(pid: int, waiter=None, grace: float = CANCEL_GRACE) -> None:
    """SIGTERM now, SIGKILL later if it is still there.

    Cancellation used to send SIGTERM and hope. A process that ignores it, or
    wedges during its own shutdown, then held a GPU indefinitely with the job
    already recorded as cancelled --- and the stage lock could not be reclaimed
    while its pid stayed alive.
    """
    _signal_group(pid, signal.SIGTERM)

    def escalate() -> None:
        deadline = time.time() + grace
        while time.time() < deadline:
            if waiter is not None:
                try:
                    waiter(timeout=max(0.1, deadline - time.time()))
                    return
                except subprocess.TimeoutExpired:
                    break
                except Exception:                              # noqa: BLE001
                    break
            if not _pid_running(pid):
                return
            time.sleep(1.0)
        if _pid_running(pid):
            print(f"pid {pid} ignored SIGTERM for {grace:.0f}s; sending SIGKILL")
            _signal_group(pid, signal.SIGKILL)

    threading.Thread(target=escalate, daemon=True,
                     name=f"terminate{pid}").start()


def _pid_running(pid: int) -> bool:
    """Zombie-aware liveness -- see stages.pid_alive."""
    return pid_alive(pid)


# ------------------------------------------------------------ line handling

def _iter_lines(stream):
    """Yield logical lines, splitting on \\n and \\r (progress bars use \\r).

    Uses read1(), NOT read(). BufferedReader.read(n) blocks until it has all n
    bytes, so with ~100-byte progress lines a 4096-byte read only surfaces
    output in ~40-line batches. That made healthy jobs look frozen for minutes
    at a time -- both here and in the log file, which is written from this loop
    -- and cost two cancelled-for-nothing training runs. read1() returns
    whatever is already available.
    """
    buf = b""
    while True:
        chunk = stream.read1(65536)
        if not chunk:
            break
        buf += chunk
        parts = buf.replace(b"\r", b"\n").split(b"\n")
        buf = parts.pop()
        for p in parts:
            yield p.decode("utf-8", "replace")
    if buf:
        yield buf.decode("utf-8", "replace")


class VramSampler(threading.Thread):
    """Track peak GPU memory used by our own process tree during a stage.

    There is no reliable a-priori VRAM model for LichtFeld here -- run_lub sat at
    ~4.7 GB with 1.5M splats at width 3840, which is nothing like gsplat's
    24 GB blowup at 3M. So measure it, accumulate history, and let the pre-check
    be built from data rather than from a guessed formula.
    """

    def __init__(self, gpu_index: int, pid_getter):
        super().__init__(daemon=True, name="vram")
        self.gpu = gpu_index
        self._pids = pid_getter
        self.peak = 0
        self._halt = threading.Event()   # not _stop: Thread.join() calls _stop()

    def run(self) -> None:
        while not self._halt.wait(10.0):
            try:
                pids = set(self._pids())
                procs = gpu.compute_procs() or {}
                for p in procs.get(self.gpu, []):
                    if p["pid"] in pids:
                        self.peak = max(self.peak, p["mib"])
            except Exception:                                  # noqa: BLE001
                pass

    def stop(self) -> None:
        self._halt.set()


def run_stage(ctx: Ctx, stage: str, argv: list[str], log_path: Path,
              cache_dir: Path | None = None) -> dict:
    """Run one stage subprocess, streaming its output to log_path."""
    spec = STAGES[stage]
    parse = spec["parse"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(ctx.gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Size CPU-bound work to what the container may use, shared between the
    # jobs that can run side by side (resources.py). A value already in the
    # service's environment wins, so it stays overridable per install.
    for k, v in thread_env(stage_threads(max_concurrent())).items():
        env.setdefault(k, v)

    progress: dict = {}
    last_push = 0.0
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Shared with the watchdog thread below.
    state = {"last_progress": None, "started": time.time(),
             "stalled": False, "done_iterating": False}

    with log_path.open("wb") as log:
        log.write(f"$ {' '.join(argv)}\n".encode())
        log.flush()
        proc = subprocess.Popen(
            argv, cwd=str(SPLAT_ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True)
        with _lock:
            _procs[ctx.job_id] = proc
        # A cancel that arrived while this process was being created found no
        # entry in _procs and signalled nothing, so the stage ran to completion
        # after the job was already marked cancelled. Catch it here, now that
        # the pid exists.
        if ctx.job_id in _cancel:
            _terminate(proc.pid, waiter=proc.wait)
        # Re-stamp the lock with the CHILD pid. The subprocess runs in its own
        # session, so it survives a service restart; recording it here is what
        # lets reconcile() actually kill the orphan rather than just noting it.
        if cache_dir is not None:
            restamp_lock(cache_dir, proc.pid)
        sampler = None
        watchdog = None
        resources = telemetry.start_sampler(ctx.gpu, proc.pid,
                                            job_id=ctx.job_id, stage=stage)
        if stage == "train":
            sampler = VramSampler(
                ctx.gpu, lambda: [proc.pid] + _child_pids(proc.pid))
            sampler.start()
            watchdog = threading.Thread(
                target=_stall_watchdog, args=(proc, state), daemon=True,
                name=f"watchdog{ctx.job_id}")
            watchdog.start()
        try:
            for line in _iter_lines(proc.stdout):
                log.write(line.encode("utf-8", "replace") + b"\n")
                log.flush()
                if parse:
                    # The accumulated blob is passed back in: a phase counter
                    # only means something relative to the phase before it, and
                    # a restart is only visible as a count going backwards.
                    got = parse(line, progress)
                    if got:
                        progress.update(got)
                        now = time.time()
                        if got.get("step") is not None:
                            state["last_progress"] = now
                            # Once the last iteration lands, eval and export run
                            # with no step lines and can take minutes. Disarm,
                            # or the watchdog kills a healthy job mid-export.
                            total = got.get("total") or progress.get("total")
                            if total and got["step"] >= total:
                                state["done_iterating"] = True
                        if now - last_push > 2.0:
                            last_push = now
                            db.stage_progress(ctx.job_id, stage, progress)
            proc.wait()
        finally:
            telemetry.finish_sampler(ctx.job_id, stage, resources,
                                     time.time() - state["started"])
            if sampler:
                sampler.stop()
                if sampler.peak:
                    progress["peak_vram_mib"] = sampler.peak
                    ctx.derived["peak_vram_mib"] = sampler.peak
            with _lock:
                _procs.pop(ctx.job_id, None)

    if progress:
        db.stage_progress(ctx.job_id, stage, progress)
    if state["stalled"]:
        step = progress.get("step")
        raise RuntimeError(
            f"{stage} stalled: no iteration progress for "
            f"{STALL_TIMEOUT/60:.0f} min"
            + (f" (stuck at step {step})" if step is not None else
               f" and never reached its first iteration within "
               f"{FIRST_PROGRESS_GRACE/60:.0f} min")
            + "; killed to free the GPU")
    if proc.returncode != 0:
        tail = _tail(log_path, 25)
        raise RuntimeError(
            f"{stage} exited {proc.returncode}\n{tail}")
    return progress


def _stall_watchdog(proc: subprocess.Popen, state: dict) -> None:
    """Kill a training process that stops reporting iteration progress.

    Two clocks: a long grace period until the first iteration (the dataset load
    is legitimately slow at high resolution), then a shorter one between
    progress reports.
    """
    while proc.poll() is None:
        if state["done_iterating"]:
            return
        now = time.time()
        last = state["last_progress"]
        limit = STALL_TIMEOUT if last else FIRST_PROGRESS_GRACE
        if now - (last or state["started"]) > limit:
            state["stalled"] = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass
            return
        time.sleep(20)


def _tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


# --------------------------------------------------------------- job runner

# How long a job waits for another job to finish a stage they share. Generous
# on purpose: the thing being waited on is usually a 95-minute training run or a
# 40-minute SfM that this job would otherwise have to repeat.
CACHE_WAIT_TIMEOUT = float(os.environ.get("QUEUE_CACHE_WAIT", 6 * 3600))
CACHE_WAIT_POLL = 5.0


def _cache_ok(ctx: Ctx, job_id: int, stage: str, spec: dict, d: Path,
              key: str) -> bool:
    """Is the existing cache entry for this stage usable?

    Re-derive rather than trust the stored blob: a .done written by an older
    build can lack fields the current code extracts, and -- more importantly --
    the finalizer validates that the run actually completed. If validation fails
    the entry is poisoned (e.g. a partial export from a cancelled run), so drop
    the marker and rebuild rather than serving it.
    """
    if not is_cached(d):
        return False
    try:
        if stage in ("train", "export") and spec["finalize"]:
            fresh = spec["finalize"](ctx)
            if fresh != read_done(d):
                mark_done(d, fresh)
        elif spec.get("verify"):
            # Cheap consistency check: does the directory still hold what the
            # marker claims? sfm_finalize is deliberately not re-run here --- it
            # shells out to the model picker, which is not free.
            spec["verify"](ctx, read_done(d))
    except Exception as exc:                                  # noqa: BLE001
        try:
            done_marker(d).unlink()
        except OSError:
            pass
        db.conn().execute("DELETE FROM cache WHERE cache_key=?", (key,))
        print(f"job {job_id}: invalidated poisoned {stage} cache {key}: {exc}")
        return False
    return True


def _try_acquire(d: Path) -> bool:
    """One attempt at owning the cache dir, reclaiming a dead holder's lock."""
    if take_lock(d, os.getpid()):
        return True
    if not lock_holder_alive(read_lock(d)):
        # The holder is gone (crash, kill -9, a reboot), or its pid has been
        # reused by an unrelated process. It never wrote .done, so the
        # half-built directory is ours to rebuild.
        release_lock(d)
        return take_lock(d, os.getpid())
    return False


def _claim_stage(ctx: Ctx, job_id: int, stage: str, spec: dict, d: Path,
                 key: str) -> bool:
    """Settle who builds this stage. True = use the cache, False = we run it.

    Sweep variants share their upstream stages by construction, so two jobs
    reaching the same uncached frames/select/sfm key within milliseconds is the
    normal case, not an error. The loser used to fail the whole job with "cache
    dir is locked by live pid"; it now waits for the winner's result, which is
    the entire point of sharing the key.
    """
    waited = 0.0
    announced = False
    while True:
        if _cache_ok(ctx, job_id, stage, spec, d, key):
            return True
        if _try_acquire(d):
            # Re-check now that we hold it. The previous holder can finish and
            # release between our cache check and our claim, and the caller
            # empties the directory before rebuilding -- which would throw away
            # the result that just landed.
            if _cache_ok(ctx, job_id, stage, spec, d, key):
                release_lock(d)
                return True
            return False
        if job_id in _cancel:
            raise RuntimeError("cancelled")
        if _stop.is_set():
            raise RuntimeError(f"service is shutting down while waiting for "
                               f"{stage} {key}")
        if not announced:
            announced = True
            db.upsert_stage(job_id, stage, key, "waiting", path=str(d))
            print(f"job {job_id}: {stage} {key} is already being built by pid "
                  f"{read_lock(d).get('pid')}; waiting for it")
        if waited >= CACHE_WAIT_TIMEOUT:
            raise RuntimeError(
                f"waited {waited/3600:.1f} h for another job to finish {stage} "
                f"{key} and it is still running; giving up")
        _stop.wait(CACHE_WAIT_POLL)
        waited += CACHE_WAIT_POLL


def _build_stage(ctx: Ctx, job_id: int, stage: str, spec: dict, d: Path,
                 key: str, log_path: Path) -> dict:
    """Build one stage into its cache dir, whose lock the caller holds.

    Owns the lock from here on: it is held (under this service's pid) through
    finalize until .done is published, and released on every exit.
    """
    # From here until .done is published this job holds the lock, and
    # every exit -- a failure, a cancel, a full disk -- releases it.
    try:
        # Check the disk BEFORE committing to the stage. A frames dump
        # of an 8K clip is tens of GB and a training run adds its
        # exports on top; running out at 80% of a 95-minute job costs
        # the job. Inside the try: a raise here used to leave the lock
        # held by this (live) service, so every job sharing the stage
        # waited out CACHE_WAIT_TIMEOUT even after space was freed.
        retention.ensure_space(stage)
        db.upsert_stage(job_id, stage, key, "running", path=str(d),
                        log_path=str(log_path), started=time.time())
        # We hold the lock and there is no valid .done, so whatever is
        # in here is debris from a failed or cancelled attempt. Clear
        # it, or this run's output gets mixed with the previous one's.
        reset_stage_dir(d)
        argv = spec["argv"](ctx)
        if argv:
            run_stage(ctx, stage, argv, log_path, cache_dir=d)
        # run_stage stamped the lock with the stage subprocess's pid,
        # which has exited by now. Take it back before finalizing:
        # finalize can run for minutes (SfM model selection), and a
        # lock naming a dead pid is exactly what _try_acquire reclaims,
        # so a sweep sibling could take the dir and empty it mid-way.
        restamp_lock(d, os.getpid())
        # LichtFeld handles SIGTERM gracefully: it exports whatever it
        # has and exits 0. Without this check a cancelled run looks
        # successful and its PARTIAL output gets a .done marker,
        # poisoning the cache for every later job with the same config.
        if job_id in _cancel:
            raise RuntimeError("cancelled")
        info = spec["finalize"](ctx) if spec["finalize"] else {}
        # Publish .done BEFORE letting go: released first, there was a
        # moment with neither lock nor marker, in which a waiting job
        # could claim the dir, find no .done and empty it.
        mark_done(d, info)
    finally:
        release_lock(d)
    return info


def _check_input_unchanged(cfg: JobConfig) -> None:
    """The clip is hashed at submit time and can be replaced before dispatch.

    A job may sit queued for hours. If samples/drone.mp4 is overwritten in the
    meantime, every cache key still describes the OLD file, so the run would
    quietly reuse the previous clip's frames, panoramas and SfM.
    """
    src = SPLAT_ROOT / cfg.input.file
    if not src.is_file():
        raise RuntimeError(f"input {cfg.input.file} is gone since this job was "
                           f"queued")
    now = quick_hash(src)
    if cfg.input.quick_hash and now != cfg.input.quick_hash:
        raise RuntimeError(
            f"input {cfg.input.file} changed since this job was queued "
            f"({cfg.input.quick_hash} -> {now}); its cache keys describe the "
            f"old file, so re-queue rather than run this one")


def _start_running(job_id: int, gpu_index: int) -> bool:
    """Claim the job for this thread. False if it is no longer ours to run.

    Conditional on the row still being queued, because cancel() settles a
    still-queued job with the same conditional write and then drops its
    cancellation intent. An unconditional write here put a job that had been
    cancelled in that window straight back to 'running' with nothing left to
    signal it: the caller was told the cancel succeeded and the job ran to
    completion anyway.
    """
    cur = db.conn().execute(
        "UPDATE jobs SET state='running', gpu=?, started=?, error=NULL "
        "WHERE id=? AND state='queued'", (gpu_index, time.time(), job_id))
    return bool(cur.rowcount)


def run_job(job_id: int, gpu_index: int) -> None:
    # EVERYTHING is inside this try, including the row read and the config
    # parse. Both used to sit above it while the dispatcher had already recorded
    # _held[job_id] = gpu, so a purged row (an early return) or a stored config
    # that no longer validates against the current schema (an exception) left
    # the GPU held forever -- with the job still 'queued', so the dispatcher
    # picked it up again five seconds later and leaked the next slot too. The
    # exception died in this thread, so nothing surfaced it either.
    try:
        row = db.get_job(job_id)
        if row is None:
            print(f"job {job_id}: row is gone; not starting it")
            return
        cfg = JobConfig.model_validate(json.loads(row["config"]))
        ctx = Ctx(job_id=job_id, cfg=cfg, gpu=gpu_index, keys=cfg.keys())
        if not _start_running(job_id, gpu_index):
            cur = db.get_job(job_id)
            print(f"job {job_id}: no longer queued at dispatch "
                  f"({cur['state'] if cur else 'deleted'}); not starting it")
            return

        _check_input_unchanged(cfg)
        review_ok = (db.get_job(job_id)["review_state"] or "") == "approved"
        for stage in ORDER:
            if job_id in _cancel:
                raise RuntimeError("cancelled")
            # Gate everything downstream of the masks, checked before the stage
            # rather than after the mask stage, so a job whose masks came from
            # the cache is held too -- reused masks still have not been looked at.
            if (cfg.mask.enabled and cfg.mask.review and not review_ok
                    and ORDER.index(stage) > ORDER.index("mask")):
                raise ReviewRequired(
                    "masks are ready and waiting for review; approve with "
                    f"POST /api/jobs/{job_id}/review")
            spec = STAGES[stage]
            d = ctx.dir(stage)
            key = ctx.keys[stage]
            log_path = LOG_ROOT / f"job{job_id:05d}_{stage}.log"

            # An optional stage that is switched off still gets a row, so the
            # UI shows why it did not run rather than leaving a silent gap.
            if spec.get("skip") and spec["skip"](ctx):
                db.upsert_stage(job_id, stage, key, "skipped", path=str(d))
                telemetry.write(job_id)
                telemetry.notify("stage.finished", job_id, stage, "skipped")
                continue

            if spec["prepare"]:
                spec["prepare"](ctx)

            # A stage owns its cache dir exclusively while it runs, so a second
            # job with the same key waits for the result instead of writing the
            # same directory concurrently.
            if _claim_stage(ctx, job_id, stage, spec, d, key):
                info = read_done(d)
                _rehydrate(ctx, stage, info)
                db.cache_touch(key)
                db.upsert_stage(job_id, stage, key, "cached", path=str(d),
                                progress=json.dumps(info), ended=time.time())
                _record_metrics(job_id, stage, info)
                telemetry.write(job_id)
                telemetry.notify("stage.finished", job_id, stage, "cached")
                continue

            telemetry.notify("stage.started", job_id, stage, "running")
            info = _build_stage(ctx, job_id, stage, spec, d, key, log_path)
            db.cache_put(key, stage, str(d), dir_bytes(d))
            db.upsert_stage(job_id, stage, key, "done", path=str(d),
                            log_path=str(log_path),
                            progress=json.dumps(info), ended=time.time())
            _record_metrics(job_id, stage, info)
            telemetry.write(job_id)
            telemetry.notify("stage.finished", job_id, stage, "done")

        db.set_job_state(job_id, "done", ended=time.time())
        # With a result to upload, the record goes after it (outputs.py), so
        # its one upload includes how that transfer went.
        telemetry.write(job_id, final=True, upload=not outputs.OUTPUT_UPLOAD)
        telemetry.notify("job.finished", job_id, state="done")
        outputs.upload_async(job_id)
    except ReviewRequired as exc:
        # Deliberately leaves the pending stages pending: this job is going to
        # run them, just not yet.
        db.set_review(job_id, "pending", str(exc))
        db.set_job_state(job_id, "awaiting_review", error=None)
        telemetry.write(job_id)
        telemetry.notify("job.waiting", job_id, state="awaiting_review")
    except Exception as exc:                                  # noqa: BLE001
        state = "cancelled" if job_id in _cancel else "failed"
        db.set_job_state(job_id, state, ended=time.time(), error=str(exc)[:4000])
        failed_stage = None
        for st in db.job_stages(job_id):
            if st["state"] in ("running", "waiting"):
                failed_stage = failed_stage or st["stage"]
                db.upsert_stage(job_id, st["stage"], st["cache_key"], state,
                                ended=time.time())
        telemetry.write(job_id, final=True)
        telemetry.notify("job.finished", job_id, failed_stage, state)
    finally:
        _cancel.discard(job_id)
        with _lock:
            _held.pop(job_id, None)


def _rehydrate(ctx: Ctx, stage: str, info: dict) -> None:
    """Restore the derived values a cached stage would otherwise have set."""
    if stage == "frames":
        ctx.derived["n_candidates"] = info.get("candidates")
    elif stage == "select":
        ctx.derived["n_panos"] = info.get("panos")
        ctx.derived["window"] = info.get("window")
    elif stage == "sfm":
        ctx.derived["dataset"] = str(ctx.dir("sfm") / "dataset")
        ctx.derived["images"] = info.get("images_dir") or str(
            images_dir(ctx.cfg, ctx.dir("sfm"), ctx.dir("select")))
        ctx.derived["needs_gut"] = info.get("needs_gut", False)


def _record_metrics(job_id: int, stage: str, info: dict) -> None:
    flat = {
        "frames": ["candidates"],
        "select": ["panos", "window"],
        "sfm": ["num_reg_frames", "num_points3D", "mean_reproj",
                "registration_pct", "baseline_over_depth", "median_depth",
                "path_length", "angular_median_deg"],
        "train": ["psnr", "ssim", "splats", "final_step", "peak_vram_mib",
                  "eval_s_per_image"],
        "export": [],
    }.get(stage, [])
    for k in flat:
        if info.get(k) is not None:
            db.put_metric(job_id, k, info[k])
    if stage == "train":
        arts = info.get("artifacts", {})
        for ext, meta in arts.items():
            db.put_metric(job_id, f"{ext}_bytes", meta["bytes"])
        st = db.conn().execute(
            "SELECT progress FROM stages WHERE job_id=? AND stage='train'",
            (job_id,)).fetchone()
        if st and st["progress"]:
            p = json.loads(st["progress"])
            p = {**info, **{k: v for k, v in p.items() if v is not None}}
            for k in ("splats", "loss", "psnr", "ssim", "lpips", "step",
                      "peak_vram_mib"):
                if p.get(k) is not None:
                    db.put_metric(
                        job_id, "final_step" if k == "step" else k, p[k])
        stg = db.conn().execute(
            "SELECT started,ended FROM stages WHERE job_id=? AND stage='train'",
            (job_id,)).fetchone()
        if stg and stg["started"] and stg["ended"]:
            db.put_metric(job_id, "train_seconds",
                          round(stg["ended"] - stg["started"], 1))


# -------------------------------------------------------------- dispatcher

def cancel(job_id: int) -> bool:
    row = db.get_job(job_id)
    if row is None:
        return False
    # A job parked awaiting review holds nothing -- no subprocess, no GPU -- so
    # it is settled right here rather than through the signal path below.
    # Without this the ONLY way out of the state was POST .../review with
    # approved:false, and the stop button on a parked job answered "not queued
    # or running", which left it inert in the queue table.
    if row["state"] == "awaiting_review":
        db.set_review(job_id, "rejected", "stopped from the queue")
        db.set_job_state(job_id, "cancelled", ended=time.time(),
                         error="stopped while awaiting mask review")
        # It ends here, not in run_job: its final record and one upload too.
        telemetry.write(job_id, final=True)
        telemetry.notify("job.finished", job_id, state="cancelled")
        return True
    if row["state"] not in ("queued", "running"):
        return False

    # Mark the intent FIRST. Read-state-then-act raced the dispatcher: a job
    # seen as queued could be picked up and started between the read and the
    # write, after which the cancel wrote "cancelled" over a job that was
    # actually running and nothing ever signalled its process.
    _cancel.add(job_id)

    # Conditional transition: only the still-queued case is settled here, and
    # the database decides whether it is still queued.
    cur = db.conn().execute(
        "UPDATE jobs SET state='cancelled', ended=? "
        "WHERE id=? AND state='queued'", (time.time(), job_id))
    if cur.rowcount:
        _cancel.discard(job_id)
        return True

    # It is running (or just started). Kill by PID/process group only --
    # `pkill -f` on this box has a history of matching the shell that runs it.
    with _lock:
        proc = _procs.get(job_id)
    if proc and proc.poll() is None:
        _terminate(proc.pid, waiter=proc.wait)
    # If there is no process yet, run_stage checks _cancel right after Popen.
    return True


SCHED_ERROR_KEY = "scheduler_error"


def dispatcher() -> None:
    while not _stop.is_set():
        try:
            if not paused():
                with _alloc_lock:
                    with _lock:
                        in_flight = len(_held)
                    # Under _alloc_lock, which exclusive() sets it under: a
                    # look before taking the lock could race a benchmark start.
                    free = [] if _exclusive.is_set() else gpu.free_gpus(
                        own_pids=own_pids(), held=held_gpus())
                    if free and in_flight < max_concurrent():
                        row = db.next_queued()
                        if row is not None:
                            g = free[0]
                            with _lock:
                                _held[int(row["id"])] = g
                            t = threading.Thread(
                                target=run_job, args=(int(row["id"]), g),
                                daemon=True, name=f"job{row['id']}")
                            t.start()
            if db.get_setting(SCHED_ERROR_KEY, ""):
                db.set_setting(SCHED_ERROR_KEY, "")
        except Exception as exc:                              # noqa: BLE001
            # Swallowing this made a broken scheduler look like an idle one:
            # jobs sat queued, the UI said "paused: false", and nothing
            # anywhere said why. Record it for /api/status and print it.
            msg = f"{type(exc).__name__}: {exc}"
            print(f"dispatcher error: {msg}")
            try:
                db.set_setting(SCHED_ERROR_KEY, msg[:500])
            except Exception:                                 # noqa: BLE001
                pass
        _stop.wait(5.0)


_dispatcher_thread: Optional[threading.Thread] = None


def reconcile() -> int:
    """Fix up state left behind by a service restart.

    Stage subprocesses are started with start_new_session=True, so a restart of
    the service orphans them rather than killing them; the DB would otherwise
    show those jobs as 'running' forever. Mark them interrupted. Re-queueing is
    cheap because every completed stage is still in the cache -- only the stage
    that was in flight has to run again.
    """
    rows = db.conn().execute(
        "SELECT id FROM jobs WHERE state='running'").fetchall()
    for r in rows:
        jid = int(r["id"])
        # Kill any stage subprocess this service orphaned and clear the locks it
        # left behind, so the cache dir has a single writer again.
        for st in db.job_stages(jid):
            if st["state"] not in ("running", "waiting") or not st["path"]:
                continue
            d = Path(st["path"])
            lk = read_lock(d)
            pid = lk.get("pid")
            # lock_holder_alive, not pid_alive: a stale lock from a run days ago
            # may name a pid the kernel has since handed to something else, and
            # signalling that would kill an unrelated process.
            if pid and int(pid) != os.getpid() and lock_holder_alive(lk):
                _signal_group(int(pid), signal.SIGTERM)
                # Releasing the lock while the orphan is still writing hands the
                # directory to a second writer. Wait for it to actually go, then
                # SIGKILL, and only release once nothing is running.
                deadline = time.time() + CANCEL_GRACE
                while time.time() < deadline and _pid_running(int(pid)):
                    time.sleep(0.5)
                if _pid_running(int(pid)):
                    print(f"reconcile: orphan pid {pid} ignored SIGTERM; "
                          f"sending SIGKILL")
                    _signal_group(int(pid), signal.SIGKILL)
                    time.sleep(1.0)
                if _pid_running(int(pid)):
                    print(f"reconcile: pid {pid} still alive; leaving the lock "
                          f"on {d} in place so nothing else writes it")
                    continue
            release_lock(d)
        db.set_job_state(
            jid, "failed", ended=time.time(),
            error="interrupted by a service restart; completed stages are "
                  "cached, so re-queueing this config resumes from where it "
                  "stopped")
        for st in db.job_stages(jid):
            if st["state"] in ("running", "waiting"):
                db.upsert_stage(jid, st["stage"], st["cache_key"], "failed",
                                ended=time.time())
    return len(rows)


def start() -> None:
    global _dispatcher_thread
    if _dispatcher_thread and _dispatcher_thread.is_alive():
        return
    _stop.clear()
    _dispatcher_thread = threading.Thread(target=dispatcher, daemon=True,
                                          name="dispatcher")
    _dispatcher_thread.start()


def stop() -> None:
    _stop.set()


def running_state() -> dict:
    with _lock:
        held = dict(_held)
        reserved = dict(_reserved)
    all_held = list(held.values()) + list(reserved.values())
    return {"paused": paused(), "held": held,
            # The host benchmark holds the machine: nothing is dispatched.
            "benchmark_running": _exclusive.is_set(),
            "reserved": sorted(set(reserved.values())),
            "max_concurrent": max_concurrent(),
            "scheduler_error": db.get_setting(SCHED_ERROR_KEY, "") or None,
            "gpus": gpu.status(own_pids=own_pids(), held=all_held)}
