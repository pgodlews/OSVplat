"""Restore points for spot and other interruptible GPUs (issue #10, docs/cloud.md "Spot GPUs").

A training run keeps its snapshots in one project.licht, a new generation per
snapshot (train.checkpoint_every, or SIGUSR1 on a reclaim notice: preempt.py).
After each one commits, a restore point is made from it in the background:
scripts/licht_restore_point.py copies the live file and keeps only the latest
checkpoint (LichtFeld's clean_project_file), about half a GB at 3M splats. It
goes into job<id>-restore.tar with restore.json, which says what it is: the job
config and cache keys, TRAINER and the other version terms, the cache root the
dataset paths inside it refer to, the step, and the checkpoint's sha256.

CHECKPOINT_UPLOAD_URL (the OUTPUT_UPLOAD_URL shape) receives it. One object
per job, replaced after each snapshot: S3 (and R2, GCS) replace an object only
when the new upload is complete, so an upload cut short by the reclaim leaves
the previous restore point in place. A URL without {job} names one object, so
it belongs to the first job that uploads to it, as for OUTPUT_UPLOAD_URL.

Resuming (import_restore): the restore point is checked against this build --
TRAINER, the version terms, the cache root -- and its train key against the job
it would resume; then a job is queued whose train stage runs LichtFeld with
--resume. Its upstream stages come from a handoff bundle or, on the machine
that trained it, from the cache. Anything that does not match is refused: a
checkpoint from another trainer or config would resume into a result its cache
key does not describe.

Never in the way of training: a restore point that cannot be made or sent is
recorded (checkpoints.json in the job's run dir, the job API, telemetry) and
logged, and the run goes on.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from . import db, outputs, telemetry
from .config import (CACHE_ROOT, CHECKPOINT_UPLOAD, CHECKPOINT_UPLOAD_ERROR, LFS_BIN,
                     LFS_PYTHON, RESTORE_POINT, RESUME_ROOT, RUNS_ROOT, ssl_context)
from .jobs import TRAINER, JobConfig
from .stages import ORDER, is_cached

SCHEMA = "osvplat.restore/1"
META = "restore.json"
LICHT = "restore.licht"
# Also keep restore points locally without uploading them (a host whose disk
# survives an interruption, like a stopped Vast instance).
LOCAL = os.environ.get("QUEUE_RESTORE_POINTS", "0").strip().lower() in ("1", "true", "yes", "on")
MAX_RECORDS = 50                  # restore points listed in checkpoints.json
RETRY_S = 2.0                     # upload retries back off 2 s, then 4 s

_lock = threading.Lock()          # checkpoints.json writes


def enabled() -> bool:
    """Whether training runs make restore points at all."""
    return bool(CHECKPOINT_UPLOAD) or LOCAL


def restore_name(job_id: int) -> str:
    return f"job{job_id:05d}-restore.tar"


def local_path(job_id: int) -> Path:
    return telemetry.run_dir(job_id) / restore_name(job_id)


# ------------------------------------------------------------ status file

def status_path(job_id: int) -> Path:
    return telemetry.run_dir(job_id) / "checkpoints.json"


def status(job_id: int) -> Optional[dict]:
    try:
        return json.loads(status_path(job_id).read_text())
    except (OSError, ValueError):
        return None


def _update(job_id: int, fn) -> dict:
    """Read-modify-write checkpoints.json under the lock."""
    with _lock:
        st = status(job_id) or {}
        fn(st)
        p = status_path(job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, p)
        return st


def record_notice(job_id: int, notice: dict) -> None:
    _update(job_id, lambda st: st.setdefault("notices", []).append(notice))


# --------------------------------------------------------------- preflight

def problems(now: Optional[float] = None) -> list[str]:
    """Why CHECKPOINT_UPLOAD_URL cannot work; training jobs are refused meanwhile."""
    if CHECKPOINT_UPLOAD_ERROR:
        return [CHECKPOINT_UPLOAD_ERROR]
    if not CHECKPOINT_UPLOAD:
        return []
    exp = outputs.expires_at(CHECKPOINT_UPLOAD)
    now = time.time() if now is None else now
    if exp is not None and exp <= now:
        return [f"CHECKPOINT_UPLOAD_URL expired "
                f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(exp))}"]
    return []


def startup_check() -> None:
    """Log the restore-point setup; never raises."""
    for p in problems():
        print(f"ERROR: {p}. Training jobs are refused until it is fixed.")
    if CHECKPOINT_UPLOAD:
        exp = outputs.expires_at(CHECKPOINT_UPLOAD)
        print(f"restore points: {outputs.safe_url(CHECKPOINT_UPLOAD['url'])}, "
              + ("no expiry in the URL" if exp is None
                 else f"expires in {(exp - time.time()) / 3600:.1f} h"))
    elif LOCAL:
        print("restore points: kept locally only (QUEUE_RESTORE_POINTS=1)")


# ------------------------------------------------------- trainer output

# LichtFeld's snapshot lines (3067e9e0 with scripts/lichtfeld-patches/):
#   Prepared .licht snapshot <uuid> for iteration 700 (547129741 checkpoint bytes)
#   Background .licht append complete: <path> generation=2 snapshot=<uuid> rewritten=5 ...
#   Trainer initialization complete
PREPARED = re.compile(r"Prepared \.licht snapshot (\S+) for iteration (\d+)")
COMMITTED = re.compile(r"\.licht append complete: .*?generation=(\d+) snapshot=(\S+)")
READY = re.compile(r"Trainer initialization complete")


class Tracker:
    """Follows one training run's output: which snapshot is at which step,
    and whether the trainer is far enough along to take SIGUSR1 (before it
    installs its handler, the signal would end the process)."""

    def __init__(self):
        self.ready = False
        self._iters: dict[str, int] = {}

    def feed(self, line: str) -> Optional[tuple[int, int]]:
        """(generation, iteration) when a snapshot has just committed."""
        if not self.ready and READY.search(line):
            self.ready = True
        m = PREPARED.search(line)
        if m:
            self._iters[m.group(1)] = int(m.group(2))
            return None
        m = COMMITTED.search(line)
        if m:
            it = self._iters.pop(m.group(2), None)
            if it is not None:
                return int(m.group(1)), it
        return None


# ---------------------------------------------------------- making one

def _meta(ctx, generation: int, iteration: int, licht: dict) -> dict:
    from . import handoff                    # handoff imports outputs, not this
    row = db.get_job(ctx.job_id)
    imported = db.job_handoff(row) if row is not None else None
    resumed = db.job_resume(row) if row is not None else None
    return {
        "schema": SCHEMA, "id": uuid.uuid4().hex, "created": round(time.time(), 1),
        "source_job": ctx.job_id,
        "config": ctx.cfg.model_dump(), "keys": ctx.keys,
        "trainer": TRAINER, "version_terms": handoff.version_terms(),
        "cache_root": str(CACHE_ROOT),
        "iteration": iteration, "generation": generation,
        "effective_iters": ctx.cfg.train.effective_iters,
        "licht": licht,
        "handoff_id": (imported or {}).get("id"),
        "resumed_from": (resumed or {}).get("id"),
        "image": {"version": os.environ.get("OSVPLAT_VERSION") or None,
                  "revision": os.environ.get("OSVPLAT_REVISION") or None},
    }


def make(ctx, generation: int, iteration: int, gpu_env: dict) -> tuple[Path, dict]:
    """Build job<id>-restore.tar from the run's current project.licht.

    Returns (tar path, record). Raises with the reason when it cannot.
    """
    project = ctx.dir("train") / "project.licht"
    out_dir = telemetry.run_dir(ctx.job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    licht = out_dir / f".restore-{uuid.uuid4().hex[:8]}.licht"
    build = LFS_BIN.parent
    env = {**gpu_env,
           "PYTHONPATH": str(build / "src" / "python"),
           "LD_LIBRARY_PATH": ":".join(x for x in (str(build), gpu_env.get("LD_LIBRARY_PATH", ""))
                                       if x)}
    t0 = time.monotonic()
    try:
        proc = subprocess.run([LFS_PYTHON, str(RESTORE_POINT), str(project), str(licht)],
                              env=env, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout).strip()[-500:]
                               or f"exit {proc.returncode}")
        made = json.loads(proc.stdout.strip().splitlines()[-1])
        make_s = time.monotonic() - t0
        meta = _meta(ctx, generation, iteration,
                     {"bytes": made["bytes"], "sha256": made["sha256"]})
        t0 = time.monotonic()
        tar = local_path(ctx.job_id)
        part = tar.with_suffix(f".{uuid.uuid4().hex[:6]}.part")
        sha, md5 = hashlib.sha256(), hashlib.md5()

        class _Hashing:
            def __init__(self, f):
                self.f = f

            def write(self, b):
                sha.update(b)
                md5.update(b)
                return self.f.write(b)

        with part.open("wb") as raw:
            with tarfile.open(fileobj=_Hashing(raw), mode="w|") as tf:
                data = json.dumps(meta, indent=1).encode()
                info = tarfile.TarInfo(META)
                info.size, info.mtime, info.mode = len(data), int(time.time()), 0o644
                tf.addfile(info, io.BytesIO(data))
                tf.add(licht, arcname=LICHT, recursive=False)
        os.replace(part, tar)
    finally:
        licht.unlink(missing_ok=True)
    rec = {"id": meta["id"], "generation": generation, "iteration": iteration,
           "bytes": tar.stat().st_size, "licht_bytes": made["bytes"],
           "source_bytes": made.get("source_bytes"), "sha256": sha.hexdigest(),
           "md5": md5.hexdigest(), "make_s": round(make_s, 2),
           "pack_s": round(time.monotonic() - t0, 2), "created": meta["created"]}
    return tar, rec


def _single_object_owner(target: dict, dest: str, job_id: int) -> Optional[int]:
    if "{job}" in target["url"] or "{job}" in (target.get("key") or ""):
        return None
    for d in sorted(RUNS_ROOT.glob("job*")):
        jid = d.name[3:]
        if not jid.isdigit() or int(jid) == job_id:
            continue
        st = status(int(jid)) or {}
        if st.get("url") == dest and st.get("uploaded"):
            return int(jid)
    return None


def upload(job_id: int, tar: Path, rec: dict, target: Optional[dict] = None,
           attempts: int = 3) -> dict:
    """Send one restore point. Returns what happened (the record's "upload")."""
    target = target or CHECKPOINT_UPLOAD
    name = restore_name(job_id)
    dest = outputs.safe_url(target["url"].replace("{job}", f"job{job_id:05d}")
                            .replace("{file}", name))
    owner = _single_object_owner(target, dest, job_id)
    if owner is not None:
        return {"state": "refused", "error": f"CHECKPOINT_UPLOAD_URL names one object and "
                                             f"job {owner} already uploaded to it; use a "
                                             f"{{job}} placeholder or a URL per job"}
    size = tar.stat().st_size
    err, tries, transfer_s = None, 0, None
    for attempt in range(attempts):
        tries += 1
        try:
            with tar.open("rb") as f:
                if (target.get("method") or "PUT").upper() == "POST":
                    req = outputs.request(target, job_id, name, f.read())
                else:
                    url = target["url"].replace("{job}", f"job{job_id:05d}").replace("{file}", name)
                    req = urllib.request.Request(
                        url, data=f, method="PUT",
                        headers={"Content-Type": "application/x-tar",
                                 **(target.get("headers") or {}),
                                 "Content-Length": str(size)})
                t0 = time.monotonic()
                with urllib.request.urlopen(req, timeout=3600, context=ssl_context()) as r:
                    r.read()
                    etag = r.headers.get("ETag")
            transfer_s = round(time.monotonic() - t0, 2)
            err = outputs.etag_problem(etag, rec["md5"])
            if err is None:
                break
        except Exception as exc:                              # noqa: BLE001
            err = (f"HTTP {exc.code} {exc.reason}" if hasattr(exc, "code")
                   else f"{type(exc).__name__}: {exc}")
        if attempt < attempts - 1:
            time.sleep(RETRY_S * 2 ** attempt)
    out = {"state": "failed" if err else "done", "url": dest, "bytes": size,
           "attempts": tries, "transfer_s": None if err else transfer_s,
           "error": err, "ended": round(time.time(), 1)}
    return out


class Maker:
    """Restore points for one training run, one at a time in the background.

    A snapshot that commits while the previous restore point is still being
    made or sent replaces any that is waiting: only the newest is worth
    sending. close() at the end of the run drops what is waiting; one already
    being made finishes, and is not sent if training completed meanwhile.
    """

    def __init__(self, ctx, gpu_env: dict, upload_target: Optional[dict] = None):
        self.ctx, self.gpu_env = ctx, gpu_env
        self.target = CHECKPOINT_UPLOAD if upload_target is None else upload_target
        self._pending: Optional[tuple[int, int]] = None
        self._cv = threading.Condition()
        self._closed = self._completed = False
        self._thread: Optional[threading.Thread] = None
        _update(ctx.job_id, lambda st: st.update(
            every=ctx.cfg.train.checkpoint_every, local=LOCAL,
            url=outputs.safe_url(self.target["url"].replace("{job}", f"job{ctx.job_id:05d}")
                                 .replace("{file}", restore_name(ctx.job_id)))
            if self.target else None))

    def request(self, generation: int, iteration: int) -> None:
        with self._cv:
            if self._closed:
                return
            self._pending = (generation, iteration)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True,
                                                name=f"restore{self.ctx.job_id}")
                self._thread.start()
            self._cv.notify_all()

    def close(self, completed: bool) -> None:
        with self._cv:
            self._closed, self._completed = True, completed
            self._pending = None
            self._cv.notify_all()

    def join(self, timeout: Optional[float] = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def _run(self) -> None:
        jid = self.ctx.job_id
        while True:
            with self._cv:
                if self._pending is None:
                    return
                generation, iteration = self._pending
                self._pending = None
            try:
                tar, rec = make(self.ctx, generation, iteration, self.gpu_env)
            except Exception as exc:                          # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"[:500]
                print(f"job {jid}: restore point at step {iteration} FAILED: {msg}")
                _update(jid, lambda st: _append(st, {"generation": generation,
                                                     "iteration": iteration,
                                                     "error": msg,
                                                     "created": round(time.time(), 1)}))
                continue
            print(f"job {jid}: restore point at step {iteration}: {rec['bytes']} bytes "
                  f"in {rec['make_s']} s")
            with self._cv:
                skip = self._closed and self._completed
            if self.target and not skip:
                rec["upload"] = upload(jid, tar, rec, self.target)
                u = rec["upload"]
                print(f"job {jid}: restore point upload to {u.get('url') or 'CHECKPOINT_UPLOAD_URL'} "
                      + (f"done in {u['transfer_s']} s" if u["state"] == "done"
                         else f"{u['state'].upper()}: {u.get('error')}"))
            elif skip:
                rec["upload"] = {"state": "skipped", "error": "training completed"}
            if skip:
                # The job may already have discarded its restore point
                # (worker.run_job) before this one landed; it is no use now.
                tar.unlink(missing_ok=True)
            _update(jid, lambda st: _append(st, rec))
            telemetry.notify("job.restore_point", jid, "train",
                             (rec.get("upload") or {}).get("state") or "local",
                             iteration=iteration)


def _append(st: dict, rec: dict) -> None:
    st.setdefault("restore_points", []).append(rec)
    del st["restore_points"][:-MAX_RECORDS]
    if (rec.get("upload") or {}).get("state") == "done":
        st["uploaded"] = {k: rec[k] for k in ("id", "iteration", "generation", "bytes",
                                              "sha256", "created")}


def discard_local(job_id: int) -> None:
    """The run completed: its restore point is of no further use here."""
    local_path(job_id).unlink(missing_ok=True)


# ------------------------------------------------------------------ resume

class ResumeError(RuntimeError):
    pass


def restore_file(name: str) -> Path:
    if not name or "/" in name or name.startswith(".") or not name.endswith(".tar"):
        raise ResumeError(f"{name!r} is not a restore point name (job<id>-restore.tar)")
    p = RESUME_ROOT / name
    if not p.is_file():
        raise ResumeError(f"no restore point {name} in {RESUME_ROOT.name}/ (RESUME_URL "
                          f"puts one there, or copy it in)")
    return p


def list_restore_points() -> list[dict]:
    out = []
    for p in sorted(RESUME_ROOT.glob("*.tar")):
        try:
            meta = read_meta(p)
            out.append({"file": p.name, "bytes": p.stat().st_size,
                        "iteration": meta.get("iteration"),
                        "effective_iters": meta.get("effective_iters"),
                        "source_job": meta.get("source_job"), "trainer": meta.get("trainer"),
                        "name": (meta.get("config") or {}).get("name")})
        except ResumeError as exc:
            out.append({"file": p.name, "bytes": p.stat().st_size, "error": str(exc)})
    return out


def read_meta(path: Path) -> dict:
    """restore.json from a restore point, checking the tar holds exactly it and
    the checkpoint."""
    try:
        with tarfile.open(path, "r:") as tf:
            names = tf.getnames()
            if sorted(names) != sorted([META, LICHT]):
                raise ResumeError(f"{path.name}: expected {META} and {LICHT}, found {names}")
            meta = json.loads(tf.extractfile(META).read())
    except (tarfile.TarError, OSError, ValueError, AttributeError) as exc:
        raise ResumeError(f"{path.name}: not a readable restore point ({exc})") from None
    if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
        raise ResumeError(f"{path.name}: restore.json is not {SCHEMA}")
    return meta


def check_meta(meta: dict) -> JobConfig:
    """Refuse a restore point this build cannot resume faithfully."""
    from . import handoff
    if meta.get("trainer") != TRAINER:
        raise ResumeError(f"made by trainer {meta.get('trainer')}, this build has {TRAINER}: "
                          f"a checkpoint only resumes in the trainer that wrote it")
    if meta.get("version_terms") != handoff.version_terms():
        raise ResumeError("made by a build with other cache-key version terms "
                          f"({meta.get('version_terms')}); its keys would not match here")
    if meta.get("cache_root") != str(CACHE_ROOT):
        raise ResumeError(f"its dataset paths are under {meta.get('cache_root')}, this cache "
                          f"is {CACHE_ROOT}: the checkpoint's cameras refer to absolute paths, "
                          f"so QUEUE_ROOT has to be the same on both machines")
    cfg_d = dict(meta.get("config") or {})
    cfg_d["run_until"] = None
    try:
        cfg = JobConfig.model_validate(cfg_d)
    except Exception as exc:                                  # noqa: BLE001
        raise ResumeError(f"its job config does not validate here: {exc}") from None
    keys = cfg.keys()
    for st in ORDER:
        if keys[st] != (meta.get("keys") or {}).get(st):
            raise ResumeError(f"{st} key {keys[st]} here is not the restore point's "
                              f"{(meta.get('keys') or {}).get(st)}")
    it, total = meta.get("iteration"), meta.get("effective_iters")
    if not isinstance(it, int) or not isinstance(total, int) or not 0 < it < total:
        raise ResumeError(f"step {it} of {total} is nothing to resume")
    return cfg


def _extract_licht(path: Path, meta: dict) -> Path:
    dest = RESUME_ROOT / f"{meta['id'][:16]}.licht"
    part = dest.with_suffix(".part")
    sha = hashlib.sha256()
    with tarfile.open(path, "r:") as tf:
        src = tf.extractfile(LICHT)
        with part.open("wb") as f:
            for block in iter(lambda: src.read(16 << 20), b""):
                sha.update(block)
                f.write(block)
    want = (meta.get("licht") or {}).get("sha256")
    if sha.hexdigest() != want:
        part.unlink(missing_ok=True)
        raise ResumeError(f"{path.name}: the checkpoint's sha256 {sha.hexdigest()[:16]}... "
                          f"is not the {str(want)[:16]}... restore.json records")
    os.replace(part, dest)
    return dest


def import_restore(name: str, sha256: Optional[str] = None, bundle: Optional[str] = None,
                   bundle_sha256: Optional[str] = None, job_name: Optional[str] = None,
                   priority: int = 0) -> dict:
    """Queue a job that resumes training from a restore point.

    With a handoff bundle, its upstream stages come from it (another machine);
    without, they have to be in this cache already (the machine that trained).
    """
    from . import handoff
    path = restore_file(name)
    if sha256:
        got = handoff.file_sha256(path)
        if got != sha256.strip().lower():
            raise ResumeError(f"{name}: sha256 {got[:16]}... is not the expected "
                              f"{sha256.strip()[:16]}...")
    meta = read_meta(path)
    cfg = check_meta(meta)
    keys = cfg.keys()
    licht = _extract_licht(path, meta)
    resume = {"id": meta["id"], "file": name, "licht": str(licht),
              "from_step": meta["iteration"], "generation": meta.get("generation"),
              "effective_iters": meta["effective_iters"],
              "source_job": meta.get("source_job"), "handoff_id": meta.get("handoff_id"),
              "sha256": (meta.get("licht") or {}).get("sha256")}
    t = cfg.model_dump()
    try:
        if bundle:
            out = handoff.import_bundle(handoff.bundle_file(bundle), sha256=bundle_sha256,
                                        name=job_name or cfg.name, train=t["train"],
                                        export=t["export"], priority=priority,
                                        resume=resume, expect_keys=keys)
            jid = out["id"]
        else:
            # What training reads. Frames are not needed: a missing entry is
            # rebuilt from the clip, which this machine (the one that trained)
            # still has -- the job checks for it like any other.
            missing = [st for st in ("select", "mask", "sfm")
                       if not is_cached(CACHE_ROOT / st / keys[st])
                       and not (st == "mask" and not cfg.mask.enabled)]
            if missing:
                raise ResumeError(
                    f"{', '.join(missing)} not in this cache: resuming on another "
                    f"machine needs the handoff bundle too (bundle=...)")
            d = cfg.model_dump()
            if job_name:
                d["name"] = job_name
            jid = db.create_job(d["name"], d, priority=priority, resume=resume)
    except Exception:
        licht.unlink(missing_ok=True)
        raise
    print(f"job {jid}: resumes from restore point {meta['id'][:12]} (step "
          f"{meta['iteration']} of {meta['effective_iters']}, source job "
          f"{meta.get('source_job')})")
    return {"id": jid, "resume": resume, "keys": keys}


def discard_resume(job_id: int) -> None:
    """The resumed job is done: its extracted checkpoint is not needed again."""
    row = db.get_job(job_id)
    r = db.job_resume(row) if row is not None else None
    if r and r.get("licht"):
        try:
            p = Path(r["licht"])
            if p.parent.resolve() == RESUME_ROOT.resolve():
                p.unlink(missing_ok=True)
        except OSError:
            pass
