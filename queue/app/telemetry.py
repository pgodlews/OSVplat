"""Per-job telemetry: what ran, on what, how long each stage took.

Written by default, uploaded only when asked. Every job gets
QUEUE_ROOT/runs/job<id>/telemetry.json, rewritten after each stage and when the
job ends (done, failed, cancelled), plus logs.tar.gz with the redacted stage
logs once it ends. Nothing leaves the machine unless QUEUE_TELEMETRY_UPLOAD
names a destination. docs/job-telemetry.md lists every field.

Why it exists: estimate.py is calibrated on one 3090 and refits only from this
queue's own history. Comparing hardware -- another GPU, a 4-core host, a rented
instance -- needs the machine described next to the timings, which the metrics
table does not do.

Not in the export folder: export is a cache entry shared by every job with the
same key, and telemetry is per job. It also records no hostnames, paths, clip
names, job names, GPU UUIDs or serial numbers, and log bundles are redacted
(see _redactor), so a bundle can be shared as it is.

Linux-only probes (/proc, cgroups) degrade to null elsewhere; a telemetry
failure is printed and never fails a job.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import platform
import queue as _queue
import re
import socket
import statistics
import tarfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from . import db, gpu, resources
from .config import (LOG_ROOT, QUEUE_ROOT, RUNS_ROOT, SPLAT_ROOT,
                     TELEMETRY_ENABLED, TELEMETRY_PLACEMENT, TELEMETRY_UPLOAD,
                     WEBHOOK_SECRET, WEBHOOK_URL, ssl_context)

SCHEMA = "osvplat.telemetry/1"
SERVICE_STARTED = time.time()

# Per-log cap in the bundle: the head holds the command line and setup, the
# tail holds how it ended. A 30k-iteration LichtFeld log is tens of MB of
# progress lines in between.
LOG_HEAD = 256 * 1024
LOG_TAIL = 1024 * 1024

_resources: dict[int, dict[str, dict]] = {}    # job_id -> stage -> summary
_res_lock = threading.Lock()
_write_lock = threading.Lock()
# One upload at a time, each reading the file when its turn comes, so an older
# telemetry.json can never land after a newer one.
_upload_lock = threading.Lock()
_host: Optional[dict] = None


def run_dir(job_id: int) -> Path:
    return RUNS_ROOT / f"job{job_id:05d}"


def telemetry_path(job_id: int) -> Path:
    return run_dir(job_id) / "telemetry.json"


def logs_path(job_id: int) -> Path:
    return run_dir(job_id) / "logs.tar.gz"


# ------------------------------------------------------------------ host

def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _cpu() -> dict:
    info = _read("/proc/cpuinfo") or ""
    model = next((l.split(":", 1)[1].strip() for l in info.splitlines()
                  if l.startswith("model name")), None) or platform.processor() or None
    flags_line = next((l for l in info.splitlines() if l.startswith("flags")), "")
    flags = set(flags_line.split(":", 1)[1].split()) if ":" in flags_line else set()
    logical = os.cpu_count()
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = logical
    # A container limited with --cpus sees every host core in /proc and in its
    # affinity mask; only the cgroup quota says how much it may actually use.
    # resources.py reads cgroup v1 as well as v2: a RunPod host on v1 was
    # recorded as 64 effective CPUs against a quota of 13.6 (2026-09-22).
    quota = resources.cgroup_cpu_quota()
    quota = round(quota, 2) if quota is not None else None
    effective = min(x for x in (affinity, quota) if x) if (affinity or quota) else None
    # Physical cores, from unique (physical id, core id) pairs.
    cores, pid = set(), None
    for l in info.splitlines():
        if l.startswith("physical id"):
            pid = l.split(":", 1)[1].strip()
        elif l.startswith("core id"):
            cores.add((pid, l.split(":", 1)[1].strip()))
    return {
        "model": model,
        "arch": platform.machine(),
        "logical_cpus": logical,
        "physical_cores": len(cores) or None,
        "effective_cpus": effective,
        "cgroup_cpu_quota": quota,
        "flags": {f: (f in flags) if flags else None
                  for f in ("avx", "avx2", "fma", "avx512f")},
    }


def _memory() -> dict:
    total = None
    for l in (_read("/proc/meminfo") or "").splitlines():
        if l.startswith("MemTotal:"):
            total = int(l.split()[1]) * 1024
    limit = None
    raw = (_read("/sys/fs/cgroup/memory.max") or "").strip()
    if raw and raw != "max":
        try:
            limit = int(raw)
        except ValueError:
            pass
    shm = None
    try:
        st = os.statvfs("/dev/shm")
        shm = st.f_blocks * st.f_frsize
    except OSError:
        pass
    return {"total_bytes": total, "cgroup_limit_bytes": limit, "shm_bytes": shm}


def _disk() -> dict:
    try:
        st = os.statvfs(QUEUE_ROOT)
        return {"queue_root_total_bytes": st.f_blocks * st.f_frsize,
                "queue_root_free_bytes": st.f_bavail * st.f_frsize}
    except OSError:
        return {}


def _gpus() -> Optional[list[dict]]:
    # No uuid, no serial: they identify the card, and nothing here needs that.
    rows = gpu._nvidia_smi(
        "gpu=index,name,memory.total,compute_cap,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit")
    if rows is None:
        return None
    out = []
    for r in rows:
        r = r + [""] * (8 - len(r))

        def num(s, cast=float):
            try:
                return cast(s)
            except ValueError:
                return None
        out.append({"index": num(r[0], int), "name": r[1] or None,
                    "memory_mib": num(r[2], int), "compute_cap": r[3] or None,
                    "driver": r[4] or None, "pcie_gen": num(r[5], int),
                    "pcie_width": num(r[6], int), "power_limit_w": num(r[7])})
    return out


def _boot_time() -> Optional[float]:
    raw = _read("/proc/uptime")
    try:
        return round(time.time() - float(raw.split()[0]), 1) if raw else None
    except ValueError:
        return None


def host() -> dict:
    """Description of this machine. Probed once; it does not change under us."""
    global _host
    if _host is None:
        _host = {
            "os": platform.platform(terse=True),
            "container": Path("/.dockerenv").exists(),
            "cpu": _cpu(),
            "memory": _memory(),
            "gpus": _gpus(),
        }
    return _host


def _software() -> dict:
    return {
        "version": os.environ.get("OSVPLAT_VERSION") or None,
        "revision": os.environ.get("OSVPLAT_REVISION") or None,
        "python": platform.python_version(),
    }


# ------------------------------------------------------------- resources

def _proc_tree(root: int) -> list[int]:
    """root and every descendant, from /proc/<pid>/task/*/children."""
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        for t in Path(f"/proc/{p}/task").glob("*/children"):
            try:
                todo.extend(int(c) for c in t.read_text().split())
            except (OSError, ValueError):
                pass
    return out


_TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _proc_stat(pid: int) -> Optional[tuple[float, int]]:
    """(cpu seconds, rss bytes) for one process, or None if it is gone."""
    raw = _read(f"/proc/{pid}/stat")
    if not raw:
        return None
    # comm can contain spaces and parentheses; fields resume after the last ')'.
    f = raw[raw.rfind(")") + 2:].split()
    try:
        return (int(f[11]) + int(f[12])) / _TICK, int(f[21]) * _PAGE
    except (IndexError, ValueError):
        return None


class ResourceSampler(threading.Thread):
    """CPU, RSS and GPU use of one stage's process tree, sampled.

    CPU seconds are the sum, over every process seen, of its own utime+stime
    at the last sample. A child that starts and exits between two samples is
    missed, so this undercounts short-lived helpers; the stage scripts and
    LichtFeld are long-lived, so what it measures -- how many cores a stage
    actually kept busy -- holds. GPU figures are for the whole device, not
    just this job, which is exact with one job per GPU.
    """

    def __init__(self, gpu_index: Optional[int], pid: int, interval: float = 5.0):
        super().__init__(daemon=True, name="resources")
        self.gpu = gpu_index
        self.pid = pid
        self.interval = interval
        self.cpu: dict[int, float] = {}
        self.rss_peak = 0
        self.gpu_util: list[float] = []
        self.gpu_mem_peak = 0
        self.samples = 0
        # Not _stop: that name is threading.Thread's own method, which join()
        # calls; an Event there made join() raise TypeError.
        self._halt = threading.Event()

    def sample(self) -> None:
        rss = 0
        if Path("/proc").is_dir():
            for p in _proc_tree(self.pid):
                st = _proc_stat(p)
                if st:
                    self.cpu[p] = st[0]
                    rss += st[1]
        self.rss_peak = max(self.rss_peak, rss)
        if self.gpu is not None and self.gpu >= 0:
            rows = gpu._nvidia_smi("gpu=index,utilization.gpu,memory.used")
            for r in rows or []:
                try:
                    if int(r[0]) == self.gpu:
                        self.gpu_util.append(float(r[1]))
                        self.gpu_mem_peak = max(self.gpu_mem_peak, int(r[2]))
                except (IndexError, ValueError):
                    pass
        self.samples += 1

    def run(self) -> None:
        # First look early, so a stage shorter than one interval still gets
        # a sample; frames and select on a short clip take seconds.
        wait = min(0.5, self.interval)
        while not self._halt.wait(wait):
            wait = self.interval
            try:
                self.sample()
            except Exception:                                 # noqa: BLE001
                pass

    def stop(self) -> None:
        self._halt.set()

    def summary(self, wall_s: float) -> dict:
        cpu_s = round(sum(self.cpu.values()), 1)
        util = sorted(self.gpu_util)

        def pct(q):
            return round(util[min(len(util) - 1, int(q * len(util)))], 1) if util else None
        return {
            "samples": self.samples,
            "cpu_seconds": cpu_s if self.cpu else None,
            "avg_cores_busy": round(cpu_s / wall_s, 2) if self.cpu and wall_s > 0 else None,
            "rss_peak_bytes": self.rss_peak or None,
            "gpu_util_p50": pct(0.5), "gpu_util_p95": pct(0.95),
            "gpu_util_mean": round(statistics.fmean(util), 1) if util else None,
            "gpu_mem_peak_mib": self.gpu_mem_peak or None,
        }


def record_resources(job_id: int, stage: str, summary: dict) -> None:
    with _res_lock:
        _resources.setdefault(job_id, {})[stage] = summary


# ------------------------------------------------------------ the record

def _scalars(d: Optional[dict]) -> dict:
    """Numbers and booleans only: stage info also carries file paths."""
    return {k: v for k, v in (d or {}).items()
            if not k.startswith("_") and isinstance(v, (int, float, bool))}


def _config(cfg: dict) -> dict:
    """The job config minus what names things: the clip file and job name."""
    cfg = json.loads(json.dumps(cfg))
    cfg.pop("name", None)
    inp = cfg.get("input") or {}
    f = inp.pop("file", None)
    if f:
        inp["file_ext"] = Path(f).suffix.lower()
    return cfg


def _metrics(job_id: int) -> dict:
    rows = db.conn().execute(
        "SELECT name, value FROM metrics WHERE job_id=? AND value IS NOT NULL",
        (job_id,)).fetchall()
    return {r["name"]: r["value"] for r in rows}


def build(job_id: int) -> Optional[dict]:
    row = db.get_job(job_id)
    if row is None:
        return None
    try:
        cfg = json.loads(row["config"])
    except (TypeError, ValueError):
        cfg = {}
    try:
        plan = json.loads(row["plan"]) if row["plan"] else None
    except (TypeError, ValueError):
        plan = None
    input_bytes = None
    f = (cfg.get("input") or {}).get("file")
    if f:
        try:
            input_bytes = (SPLAT_ROOT / f).stat().st_size
        except OSError:
            pass
    with _res_lock:
        res = dict(_resources.get(job_id, {}))
    stages = []
    for st in db.job_stages(job_id):
        try:
            prog = json.loads(st["progress"]) if st["progress"] else None
        except (TypeError, ValueError):
            prog = None
        wall = (round(st["ended"] - st["started"], 1)
                if st["started"] and st["ended"] else None)
        stages.append({
            "stage": st["stage"], "state": st["state"],
            "cache_key": st["cache_key"],
            "started": st["started"], "ended": st["ended"], "wall_s": wall,
            "planned_s": (plan or {}).get(st["stage"]),
            "info": _scalars(prog),
            "resources": res.get(st["stage"]),
        })
    return {
        "schema": SCHEMA,
        "written": round(time.time(), 1),
        "job": {
            "id": job_id, "state": row["state"],
            "sweep_id": row["sweep_id"],
            "created": row["created"], "started": row["started"],
            "ended": row["ended"],
            # Errors quote paths and clip names ("input samples/x.OSV is gone").
            "error": _redactor(job_id)((row["error"] or "")[:500]) or None,
            "input_bytes": input_bytes,
            "config": _config(cfg),
        },
        "plan": plan,
        "stages": stages,
        "metrics": _metrics(job_id),
        "host": {**host(), "disk": _disk()},
        "software": _software(),
        "placement": TELEMETRY_PLACEMENT or None,
        "timeline": {"host_boot": _boot_time(),
                     "service_started": round(SERVICE_STARTED, 1)},
    }


def write(job_id: int, final: bool = False) -> None:
    """Rewrite this job's telemetry (and, at the end, its logs); then upload.

    Called after every stage and once when the job ends. Never raises.
    """
    if not TELEMETRY_ENABLED:
        return
    try:
        rec = build(job_id)
        if rec is None:
            return
        d = run_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        p = telemetry_path(job_id)
        with _write_lock:
            tmp = p.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
            tmp.write_text(json.dumps(rec, indent=1))
            os.replace(tmp, p)
        files = ["telemetry.json"]
        if final:
            _bundle_logs(job_id)
            files.append("logs.tar.gz")
            with _res_lock:
                _resources.pop(job_id, None)
        if TELEMETRY_UPLOAD:
            threading.Thread(target=_upload_files, args=(job_id, files),
                             daemon=True, name=f"telemetry{job_id}").start()
    except Exception as exc:                                  # noqa: BLE001
        print(f"job {job_id}: telemetry not written: {exc}")


# ------------------------------------------------------------------ logs

_SERIAL = re.compile(r"""(['"]?(?:serial|sn|serial_number)['"]?\s*[:=]\s*)(['"]?)[A-Za-z0-9._-]{4,}\2""", re.I)
_LATLON = re.compile(r"""(['"]?(?:lat|lon|lng|latitude|longitude|gps_?lat|gps_?lon)['"]?\s*[:=]\s*)-?\d+\.\d+""", re.I)


def _redactor(job_id: int):
    subs: list[tuple[str, str]] = []
    for path, tag in ((SPLAT_ROOT, "$SPLAT_ROOT"), (QUEUE_ROOT, "$QUEUE_ROOT"),
                      (Path.home(), "$HOME")):
        s = str(path)
        if len(s) > 1:
            subs.append((s, tag))
    # Longest first, so /home/x/splat/queue becomes $QUEUE_ROOT, not $HOME/...
    subs.sort(key=lambda t: -len(t[0]))
    host_names = {n for n in (socket.gethostname(), platform.node()) if n and len(n) > 2}
    row = db.get_job(job_id)
    clip = None
    if row is not None:
        try:
            f = (json.loads(row["config"]).get("input") or {}).get("file")
            clip = Path(f).stem if f else None
        except (TypeError, ValueError):
            pass

    def redact(text: str) -> str:
        for s, tag in subs:
            text = text.replace(s, tag)
        for h in host_names:
            text = re.sub(rf"\b{re.escape(h)}\b", "<host>", text)
        # A short or all-digit stem ("0198") would also hit unrelated numbers.
        if clip and len(clip) >= 6 and not clip.isdigit():
            text = text.replace(clip, "<clip>")
        text = _SERIAL.sub(r"\1<serial>", text)
        text = _LATLON.sub(r"\1<redacted>", text)
        return text
    return redact


def _capped(path: Path) -> bytes:
    size = path.stat().st_size
    with path.open("rb") as f:
        if size <= LOG_HEAD + LOG_TAIL:
            return f.read()
        head = f.read(LOG_HEAD)
        f.seek(size - LOG_TAIL)
        tail = f.read()
    cut = f"\n[... {size - LOG_HEAD - LOG_TAIL} bytes cut ...]\n".encode()
    return head + cut + tail


def _bundle_logs(job_id: int) -> None:
    redact = _redactor(job_id)
    out = logs_path(job_id)
    tmp = out.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for st in db.job_stages(job_id):
            # Only logs this job wrote. A cached stage's log belongs to the
            # job that built it, which bundles it itself.
            lp = st["log_path"]
            if not lp or st["state"] == "cached":
                continue
            p = Path(lp)
            if not p.is_file() or not p.resolve().is_relative_to(LOG_ROOT.resolve()):
                continue
            data = redact(_capped(p).decode("utf-8", "replace")).encode()
            ti = tarfile.TarInfo(f"{st['stage']}.log")
            ti.size, ti.mtime = len(data), int(p.stat().st_mtime)
            tar.addfile(ti, io.BytesIO(data))
    os.replace(tmp, out)


# ---------------------------------------------------------------- upload

def _multipart(fields: dict, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"'
                     f'\r\n\r\n{v}\r\n'.encode())
    # S3 requires the file to be the last field.
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="{filename}"\r\nContent-Type: application/octet-stream'
                 f'\r\n\r\n'.encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def upload_request(target: dict, job_id: int, name: str,
                   data: bytes) -> urllib.request.Request:
    """Build the request for one file. Pure, so tests can check it.

    target, from QUEUE_TELEMETRY_UPLOAD:
      {"method": "PUT", "url": ".../{job}/{file}", "headers": {...}}
      {"method": "POST", "url": "https://bucket...", "fields": {...},
       "key": "prefix/{job}/{file}"}   -- an S3 presigned POST whose policy
                                          allows that key prefix
    {job} is the zero-padded job id, {file} the file name.
    """
    job = f"job{job_id:05d}"
    method = (target.get("method") or "PUT").upper()
    url = target["url"].replace("{job}", job).replace("{file}", name)
    headers = dict(target.get("headers") or {})
    if method == "POST":
        fields = dict(target.get("fields") or {})
        if target.get("key"):
            fields["key"] = target["key"].replace("{job}", job).replace("{file}", name)
        body, ctype = _multipart(fields, name, data)
        headers["Content-Type"] = ctype
    else:
        body = data
        headers.setdefault("Content-Type", "application/json" if name.endswith(".json")
                           else "application/gzip")
    return urllib.request.Request(url, data=body, headers=headers, method=method)


def _upload_files(job_id: int, names: list[str]) -> None:
    with _upload_lock:
        _upload_locked(job_id, names)


def _upload_locked(job_id: int, names: list[str]) -> None:
    for name in names:
        p = run_dir(job_id) / name
        if not p.is_file():
            continue
        data = p.read_bytes()
        err = None
        for attempt in range(3):
            try:
                req = upload_request(TELEMETRY_UPLOAD, job_id, name, data)
                with urllib.request.urlopen(req, timeout=60, context=ssl_context()) as r:
                    r.read()
                err = None
                break
            except Exception as exc:                          # noqa: BLE001
                err = exc
                time.sleep(2 ** attempt)
        if err is not None:
            print(f"job {job_id}: telemetry upload of {name} failed: {err}")


# --------------------------------------------------------------- webhook

# Events go out in order, from one thread, so a receiver never sees a stage
# finish before it started.
_hook_q: Optional[_queue.Queue] = None
_hook_lock = threading.Lock()


def webhook_request(event: dict) -> urllib.request.Request:
    body = json.dumps(event, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json",
               "User-Agent": "OSVplat-queue"}
    if WEBHOOK_SECRET:
        headers["X-OSVplat-Signature"] = "sha256=" + hmac.new(
            WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return urllib.request.Request(WEBHOOK_URL, data=body, headers=headers,
                                  method="POST")


def _hook_worker() -> None:
    while True:
        event = _hook_q.get()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(webhook_request(event), timeout=15,
                                            context=ssl_context()) as r:
                    r.read()
                break
            except Exception as exc:                          # noqa: BLE001
                if attempt == 2:
                    print(f"webhook {event.get('event')} for job "
                          f"{event.get('job_id')} failed: {exc}")
                time.sleep(2 ** attempt)


def notify(event: str, job_id: int, stage: Optional[str] = None,
           state: Optional[str] = None, **extra) -> None:
    """Queue one webhook event; returns at once, never raises.

    Carries ids and states only -- no names, and no error text, which can
    quote a path. The full record is one GET of /api/jobs/<id>/telemetry away
    for a receiver that holds the token.
    """
    global _hook_q
    if not WEBHOOK_URL:
        return
    try:
        with _hook_lock:
            if _hook_q is None:
                _hook_q = _queue.Queue(maxsize=1000)
                threading.Thread(target=_hook_worker, daemon=True,
                                 name="webhook").start()
        row = db.get_job(job_id)
        ev = {"event": event, "job_id": job_id,
              "sweep_id": row["sweep_id"] if row else None,
              "stage": stage, "state": state, "ts": round(time.time(), 1),
              **{k: v for k, v in extra.items() if v is not None}}
        if TELEMETRY_PLACEMENT:
            ev["placement"] = TELEMETRY_PLACEMENT
        _hook_q.put_nowait(ev)
    except Exception as exc:                                  # noqa: BLE001
        print(f"webhook {event} for job {job_id} not queued: {exc}")
