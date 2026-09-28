"""Failure debug bundles: what a failed job leaves behind for offline analysis.

When a job fails, everything needed to work out why goes into one tar,
QUEUE_ROOT/runs/job<id>/debug.tar, and to DEBUG_UPLOAD_URL if one is set. On a
rented GPU the machine is usually deleted minutes after a failure, and each
time something was missing afterwards (docs/cloud.md "Debug bundles"): the
trainer's crash log in /tmp, the memory curve at a finer grain than 5 s, the
state of the card, or the splat a finished training had exported before the
job was marked failed.

Levels (QUEUE_DEBUG), each including the one before:

  basic      job record and config, cache keys, every stage log in full,
             telemetry and samples, LichtFeld crash logs, nvidia-smi -q, host
             and cgroup state, a process list, repro.sh. A few MB.
  artifacts  what the failed stages left in their cache dirs: exported splats
             (a finalizer refusing to cache a training run must not also lose
             it), metrics, the trainer's emergency .licht snapshot, the sparse
             model and SfM summaries.
  heavy      core dumps from the job's time window, and the training dataset
             view (its sparse model and masks). Never the frames themselves:
             the clip, the config and the image reproduce them exactly.

Items go in by priority until QUEUE_DEBUG_MAX_GB; what did not fit is listed
in MANIFEST.json, never silently dropped. Everything at level basic always
goes in.

Not anonymous, unlike telemetry: it is for whoever runs this machine, and
paths, clip and job names are part of what makes it useful. It is stripped of
secrets: the queue token, presigned URL signatures, secret-named environment
variables, and GPU serial numbers and UUIDs.

Never raises from on_failure(), and never changes the job's state.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import subprocess
import tarfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import db, gpu, outputs, telemetry
from .config import (DEBUG_LEVEL, DEBUG_LEVELS, DEBUG_MAX_BYTES, DEBUG_UPLOAD,
                     DEBUG_UPLOAD_ERROR, LOG_ROOT, QUEUE_ROOT, QUEUE_TOKEN,
                     RUNS_ROOT, SPLAT_ROOT, ssl_context)

_lock = threading.Lock()          # one build or upload at a time

# LichtFeld writes "Crash diagnostics: /tmp/lichtfeld-studio-crash-<pid>.log".
CRASH_GLOBS = ("/tmp/lichtfeld-studio-crash-*.log",)
# core_pattern decides the name; these cover the kernel default ("core",
# "core.<pid>") in the directories stages run in.
CORE_NAMES = ("core", "core.*")
LOG_CAP = 64 * 2**20              # per stage log: head and tail beyond this
TAR_OVERHEAD = 1024               # header + padding, per member, roughly


def rank(level: str) -> int:
    return DEBUG_LEVELS.index(level)


def bundle_path(job_id: int) -> Path:
    return telemetry.run_dir(job_id) / "debug.tar"


def status_path(job_id: int) -> Path:
    return telemetry.run_dir(job_id) / "debug.json"


def status(job_id: int) -> Optional[dict]:
    try:
        return json.loads(status_path(job_id).read_text())
    except (OSError, ValueError):
        return None


def _set_status(job_id: int, **st) -> dict:
    p = status_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, p)
    return st


# ---------------------------------------------------------------- secrets

_SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSW|KEY|CREDENTIAL|AUTH|_URL$|UPLOAD|WEBHOOK|SSH",
                          re.I)
# A presigned URL's query is its signature: keep the path, drop the query.
_URL_QUERY = re.compile(r"(https?://[^\s?#\"'<>]+)\?[^\s\"'<>]*")
_GPU_UUID = re.compile(r"\b(GPU|MIG)-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)
_SERIAL_LINE = re.compile(
    r"^(\s*(?:Serial Number|GPU UUID|Board Part Number|GPU Part Number|"
    r"Module ID|Board ID|FRU Part Number|Inforom Image Version)\s*:\s*).+$",
    re.M | re.I)


def _secret_values() -> list[str]:
    vals = {v for k, v in os.environ.items() if _SECRET_NAME.search(k) and len(v) >= 8}
    if QUEUE_TOKEN:
        vals.add(QUEUE_TOKEN)
    # Longest first, so a value containing another is replaced whole.
    return sorted(vals, key=len, reverse=True)


def scrub(text: str, secrets: Optional[list[str]] = None) -> str:
    for s in _secret_values() if secrets is None else secrets:
        text = text.replace(s, "<redacted>")
    text = _URL_QUERY.sub(r"\1?<signature removed>", text)
    text = _GPU_UUID.sub(r"\1-<uuid>", text)
    return _SERIAL_LINE.sub(r"\1<redacted>", text)


def _env_text() -> str:
    lines = []
    for k in sorted(os.environ):
        v = "<redacted>" if _SECRET_NAME.search(k) else os.environ[k]
        lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ items

@dataclass
class Item:
    prio: int                     # lower goes in first
    name: str                     # path inside the bundle
    data: Optional[bytes] = None
    path: Optional[Path] = None
    required: bool = False        # level basic: in whatever the cap says

    @property
    def size(self) -> int:
        if self.data is not None:
            return len(self.data)
        try:
            return self.path.stat().st_size if self.path else 0
        except OSError:
            return 0


def _text(prio: int, name: str, text: str, secrets: list[str]) -> Item:
    return Item(prio, name, data=scrub(text, secrets).encode(), required=True)


def _run(argv: list[str], timeout: float = 30) -> str:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr.strip() else "") \
            + ("" if r.returncode == 0 else f"\n[exit {r.returncode}]")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[{' '.join(argv)} failed: {type(exc).__name__}: {exc}]"


def _read(path: str, limit: int = 1 << 20) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return f.read(limit).decode("utf-8", "replace")
    except OSError:
        return None


def _capped_log(p: Path) -> bytes:
    size = p.stat().st_size
    with p.open("rb") as f:
        if size <= LOG_CAP:
            return f.read()
        head = f.read(LOG_CAP // 4)
        f.seek(size - (LOG_CAP - LOG_CAP // 4))
        tail = f.read()
    return head + f"\n[... {size - LOG_CAP} bytes cut ...]\n".encode() + tail


def _in_window(p: Path, since: float) -> bool:
    try:
        return p.is_file() and p.stat().st_mtime >= since
    except OSError:
        return False


def _job_info(job_id: int) -> tuple[dict, list[dict], Optional[dict]]:
    row = db.get_job(job_id)
    job = dict(row) if row is not None else {"id": job_id}
    stages = [dict(s) for s in db.job_stages(job_id)]
    keys = None
    try:
        from .jobs import JobConfig
        cfg = JobConfig.model_validate(json.loads(job["config"]))
        keys = cfg.keys()
    except Exception:                                         # noqa: BLE001
        pass
    return job, stages, keys


def _failed_stage(stages: list[dict]) -> Optional[dict]:
    for want in (("failed", "cancelled"), ("running", "waiting")):
        for st in stages:
            if st["state"] in want:
                return st
    return None


def _repro(job: dict, stages: list[dict], secrets: list[str]) -> str:
    try:
        cfg = json.loads(job.get("config") or "{}")
    except ValueError:
        cfg = {}
    failed = _failed_stage(stages)
    argv = None
    if failed and failed.get("log_path"):
        first = (_read(failed["log_path"], 1 << 16) or "").splitlines()[:1]
        if first and first[0].startswith("$ "):
            argv = first[0][2:]
    clip = (cfg.get("input") or {}).get("file")
    lines = [
        "#!/bin/sh",
        f"# Reproduce job {job.get('id')} ({job.get('name')!r}) offline.",
        f"# Written {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} by OSVplat "
        f"{os.environ.get('OSVPLAT_VERSION') or '?'} ({os.environ.get('OSVPLAT_REVISION') or '?'}).",
        "#",
        "# 1. Start the same image (digest: see MANIFEST.json and your launch record)",
        f"#    with the same clip in samples/: {clip} (quick_hash "
        f"{(cfg.get('input') or {}).get('quick_hash')}).",
        "# 2. Queue the same config. Cached stages are rebuilt from the clip, so",
        "#    frames, selection, masks and SfM come out as they did here.",
        "curl -sS -X POST -H \"x-queue-token: $QUEUE_TOKEN\" -H 'content-type: application/json' \\",
        "  --data-binary @- localhost:8090/api/jobs <<'JSON'",
        json.dumps(cfg, indent=1),
        "JSON",
    ]
    if failed:
        lines += ["", f"# The stage that failed ({failed['stage']}) ran this; run it by hand",
                  "# inside the container to iterate faster once the cache holds its inputs:"]
        lines += [f"#   {argv}" if argv else "#   (its command line is the first line of its log)"]
    return scrub("\n".join(lines) + "\n", secrets)


def _collect(job_id: int, level: str) -> list[Item]:
    secrets = _secret_values()
    job, stages, keys = _job_info(job_id)
    started = float(job.get("started") or job.get("created") or 0) - 60
    items: list[Item] = []

    # ------------------------------------------------------------ basic
    rec = {k: v for k, v in job.items() if k != "config"}
    try:
        rec["config"] = json.loads(job.get("config") or "null")
    except ValueError:
        rec["config"] = job.get("config")
    rec["stages"] = stages
    rec["cache_keys"] = keys
    rec["software"] = {"version": os.environ.get("OSVPLAT_VERSION"),
                       "revision": os.environ.get("OSVPLAT_REVISION"),
                       "python": platform.python_version()}
    items.append(_text(0, "job.json", json.dumps(rec, indent=1, default=str), secrets))
    items.append(_text(0, "repro.sh", _repro(job, stages, secrets), secrets))

    for st in stages:
        lp = st.get("log_path")
        if not lp:
            continue
        p = Path(lp)
        try:
            if not p.is_file() or not p.resolve().is_relative_to(LOG_ROOT.resolve()):
                continue
            data = _capped_log(p).decode("utf-8", "replace")
        except OSError:
            continue
        items.append(_text(1, f"logs/{st['stage']}.log", data, secrets))

    telemetry.write(job_id)                 # the record as of now, not the last stage
    for name, p in (("telemetry/telemetry.json", telemetry.telemetry_path(job_id)),
                    ("telemetry/samples.jsonl.gz", telemetry.samples_path(job_id)),
                    ("telemetry/upload.json", outputs.status_path(job_id))):
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if name.endswith(".json"):
            items.append(_text(2, name, data.decode("utf-8", "replace"), secrets))
        else:
            items.append(Item(2, name, data=data, required=True))

    for pattern in CRASH_GLOBS:
        for p in sorted(Path(pattern).parent.glob(Path(pattern).name)):
            if _in_window(p, started):
                items.append(_text(3, f"crash/{p.name}", _read(str(p), 16 << 20) or "", secrets))

    items.append(_text(4, "gpu/nvidia-smi-q.txt", _run(["nvidia-smi", "-q"]), secrets))
    items.append(_text(4, "gpu/nvidia-smi.txt", _run(["nvidia-smi"]), secrets))
    items.append(_text(4, "gpu/compute-apps.json",
                       json.dumps(gpu.compute_procs(), indent=1), secrets))

    host = {
        "host/uname.txt": _run(["uname", "-a"]),
        "host/cpuinfo.txt": (_read("/proc/cpuinfo") or "").split("\n\n")[0],
        "host/meminfo.txt": _read("/proc/meminfo") or "",
        "host/loadavg.txt": _read("/proc/loadavg") or "",
        "host/core_pattern.txt": (_read("/proc/sys/kernel/core_pattern") or "")
        + _run(["sh", "-c", "ulimit -c"]),
        "host/df.txt": _run(["df", "-h"]),
        "host/ps.txt": _run(["ps", "-eo", "pid,ppid,stat,etime,rss,args", "--sort=-rss"]),
        "host/dmesg.txt": _run(["dmesg", "--ctime"]),
        "host/env.txt": _env_text(),
    }
    for f in ("pressure/cpu", "pressure/memory", "pressure/io"):
        host[f"host/{f.replace('/', '_')}.txt"] = _read(f"/proc/{f}") or ""
    for f in ("cpu.max", "cpu.stat", "memory.max", "memory.current", "memory.peak",
              "memory.events", "memory.stat", "pids.max"):
        host[f"host/cgroup_{f}"] = _read(f"/sys/fs/cgroup/{f}") or ""
    for name, text in host.items():
        if text:
            items.append(_text(5, name, text, secrets))

    if rank(level) < rank("artifacts"):
        return items

    # -------------------------------------------------------- artifacts
    dirs = {st["stage"]: Path(st["path"]) for st in stages if st.get("path")}
    train = dirs.get("train")
    if train and train.is_dir():
        for ext, prio in (("spz", 10), ("sog", 11), ("ply", 40)):
            for p in sorted(train.glob(f"*.{ext}")):
                items.append(Item(prio, f"train/{p.name}", path=p))
        for name in ("metrics.csv", "metrics_report.txt", ".done"):
            if (train / name).is_file():
                items.append(Item(12, f"train/{name}", path=train / name))
        for p in sorted(train.glob("*.licht")):
            items.append(Item(30, f"train/{p.name}", path=p))
    sfm = dirs.get("sfm")
    if sfm and sfm.is_dir():
        for p in sorted(sfm.glob("*.json")):
            items.append(Item(13, f"sfm/{p.name}", path=p))
        for p in sorted(sfm.rglob("sparse/**/*.bin")):
            items.append(Item(20, f"sfm/{p.relative_to(sfm)}", path=p))
    for stage in ("mask", "select", "frames"):
        d = dirs.get(stage)
        if d and d.is_dir():
            for p in sorted(d.glob("*.json")):
                items.append(Item(14, f"{stage}/{p.name}", path=p))

    if rank(level) < rank("heavy"):
        return items

    # ------------------------------------------------------------ heavy
    seen: set[Path] = set()
    for d in [SPLAT_ROOT, QUEUE_ROOT, Path("/tmp"), *dirs.values()]:
        for pat in CORE_NAMES:
            for p in sorted(Path(d).glob(pat)) if Path(d).is_dir() else []:
                if p.resolve() not in seen and _in_window(p, started):
                    seen.add(p.resolve())
                    items.append(Item(50, f"cores/{p.name}", path=p))
    if train and (train / "dataset").is_dir():
        view = train / "dataset"
        for p in sorted(view.rglob("*")):
            if p.is_file() and "images" not in p.relative_to(view).parts:
                items.append(Item(60, f"train/dataset/{p.relative_to(view)}", path=p))
    return items


# ------------------------------------------------------------------ build

def build(job_id: int, level: Optional[str] = None, reuse: bool = True) -> dict:
    """Write debug.tar for one job and return its status (debug.json).

    With reuse, a bundle already built at this level or above after the job
    ended is returned as it is: the failure hook and the watcher both ask.
    """
    level = (level or DEBUG_LEVEL).lower()
    if level not in DEBUG_LEVELS or level == "off":
        raise ValueError(f"level must be one of {DEBUG_LEVELS[1:]}")
    with _lock:
        job = db.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        old = status(job_id)
        ended = job["ended"] or 0
        if (reuse and old and old.get("state") in ("built", "uploaded", "upload_failed", "refused")
                and rank(old.get("level", "basic")) >= rank(level)
                and old.get("created", 0) >= ended and ended and bundle_path(job_id).is_file()):
            return old
        started = time.time()
        _set_status(job_id, state="building", level=level, started=round(started, 1))
        try:
            items = _collect(job_id, level)
            return _write(job_id, level, items, started)
        except Exception as exc:                              # noqa: BLE001
            return _set_status(job_id, state="failed", level=level,
                               error=f"{type(exc).__name__}: {exc}")


def _write(job_id: int, level: str, items: list[Item], started: float) -> dict:
    items.sort(key=lambda i: (not i.required, i.prio, i.name))
    budget = DEBUG_MAX_BYTES
    chosen, skipped, used = [], [], 0
    names: set[str] = set()
    for it in items:
        if it.name in names:
            continue
        names.add(it.name)
        size = it.size + TAR_OVERHEAD
        if not it.required and used + size > budget:
            skipped.append({"name": it.name, "bytes": it.size,
                            "reason": f"over QUEUE_DEBUG_MAX_GB ({budget / 2**30:g} GB)"})
            continue
        chosen.append(it)
        used += size
    manifest = {
        "schema": "osvplat.debug/1", "job": job_id, "level": level,
        "created": round(time.time(), 1),
        "software": {"version": os.environ.get("OSVPLAT_VERSION"),
                     "revision": os.environ.get("OSVPLAT_REVISION")},
        "files": [{"name": it.name, "bytes": it.size} for it in chosen],
        "skipped": skipped,
    }
    out = bundle_path(job_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    prefix = f"job{job_id:05d}-debug"
    with tarfile.open(tmp, "w", dereference=True) as tar:
        m = json.dumps(manifest, indent=1).encode()
        ti = tarfile.TarInfo(f"{prefix}/MANIFEST.json")
        ti.size, ti.mtime = len(m), int(time.time())
        tar.addfile(ti, io.BytesIO(m))
        for it in chosen:
            if it.data is not None:
                ti = tarfile.TarInfo(f"{prefix}/{it.name}")
                ti.size, ti.mtime, ti.mode = len(it.data), int(time.time()), 0o644
                tar.addfile(ti, io.BytesIO(it.data))
            else:
                try:
                    tar.add(it.path, arcname=f"{prefix}/{it.name}", recursive=False)
                except OSError as exc:
                    manifest["skipped"].append({"name": it.name, "bytes": it.size,
                                                "reason": f"unreadable: {exc}"})
    os.replace(tmp, out)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with out.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
    st = _set_status(job_id, state="built", level=level, created=manifest["created"],
                     build_s=round(time.time() - started, 1), path=str(out),
                     bytes=out.stat().st_size, sha256=sha.hexdigest(), md5=md5.hexdigest(),
                     files=len(chosen), skipped=manifest["skipped"])
    print(f"job {job_id}: debug bundle {out.name} ({level}, {st['bytes'] / 2**20:.1f} MiB, "
          f"{len(chosen)} files, {len(manifest['skipped'])} left out)")
    return st


# ----------------------------------------------------------------- upload

def _owner(dest: str, job_id: int) -> Optional[int]:
    """Another job that already uploaded its bundle to this same object."""
    for d in sorted(RUNS_ROOT.glob("job*")):
        if not d.name[3:].isdigit() or int(d.name[3:]) == job_id:
            continue
        st = status(int(d.name[3:]))
        if st and (st.get("upload") or {}).get("state") == "done" \
                and (st.get("upload") or {}).get("url") == dest:
            return int(d.name[3:])
    return None


def _request(target: dict, job_id: int, name: str, f, size: int) -> urllib.request.Request:
    """A streaming PUT (bundles run to GBs), or outputs' builder for a POST."""
    if (target.get("method") or "PUT").upper() == "POST":
        return outputs.request(target, job_id, name, f.read())
    url = target["url"].replace("{job}", f"job{job_id:05d}").replace("{file}", name)
    headers = {"Content-Type": "application/x-tar", **(target.get("headers") or {}),
               "Content-Length": str(size)}
    return urllib.request.Request(url, data=f, headers=headers, method="PUT")


def upload(job_id: int, target: Optional[dict] = None, attempts: int = 3) -> dict:
    target = target or DEBUG_UPLOAD
    st = status(job_id) or {}
    if not target or st.get("state") not in ("built", "upload_failed", "uploaded"):
        return st
    name = f"job{job_id:05d}-debug.tar"
    dest = outputs.safe_url(target["url"].replace("{job}", f"job{job_id:05d}")
                            .replace("{file}", name))
    with _lock:
        fixed = "{job}" not in target["url"] and "{job}" not in (target.get("key") or "")
        owner = _owner(dest, job_id) if fixed else None
        if owner is not None:
            st["upload"] = {"state": "refused", "url": dest,
                            "error": f"DEBUG_UPLOAD_URL names one object and job {owner}'s "
                                     "bundle is already there; this one stays local"}
            st["state"] = "refused"
            print(f"job {job_id}: debug upload refused: {st['upload']['error']}")
            return _set_status(job_id, **st)
        path, size = bundle_path(job_id), bundle_path(job_id).stat().st_size
        err, tries, transfer_s = None, 0, None
        for attempt in range(attempts):
            tries += 1
            try:
                with path.open("rb") as f:
                    req = _request(target, job_id, name, f, size)
                    t0 = time.monotonic()
                    with urllib.request.urlopen(req, timeout=1800, context=ssl_context()) as r:
                        r.read()
                        etag = r.headers.get("ETag")
                transfer_s = round(time.monotonic() - t0, 2)
                err = outputs.etag_problem(etag, st.get("md5", ""))
                if err is None:
                    break
            except Exception as exc:                          # noqa: BLE001
                err = (f"HTTP {exc.code} {exc.reason}" if hasattr(exc, "code")
                       else f"{type(exc).__name__}: {exc}")
            if attempt < attempts - 1:
                time.sleep(2 ** (attempt + 1))
        st["upload"] = {"state": "failed" if err else "done", "url": dest,
                        "attempts": tries, "transfer_s": None if err else transfer_s,
                        "error": err, "ended": round(time.time(), 1)}
        st["state"] = "upload_failed" if err else "uploaded"
        print(f"job {job_id}: debug bundle upload to {dest} "
              + (f"FAILED: {err} (kept at {path})" if err else f"done ({size} bytes)"))
        return _set_status(job_id, **st)


def on_failure(job_id: int) -> None:
    """Build (and upload) the bundle for a job that just failed. Never raises;
    runs in its own thread, so the GPU is free for the next job meanwhile."""
    if DEBUG_LEVEL == "off":
        return

    def run():
        try:
            st = build(job_id)
            if st.get("state") == "built" and DEBUG_UPLOAD:
                upload(job_id)
            elif DEBUG_UPLOAD_ERROR:
                print(f"job {job_id}: debug bundle kept local: {DEBUG_UPLOAD_ERROR}")
        except Exception as exc:                              # noqa: BLE001
            print(f"job {job_id}: debug bundle not written: {type(exc).__name__}: {exc}")
    threading.Thread(target=run, daemon=True, name=f"debug{job_id}").start()
