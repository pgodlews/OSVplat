"""Per-job telemetry: what ran, on what, how long each stage took.

Written by default, uploaded only when asked. Every job gets
QUEUE_ROOT/runs/job<id>/telemetry.json, rewritten after each stage and when the
job ends (done, failed, cancelled), plus logs.tar.gz with the redacted stage
logs once it ends. Nothing leaves the machine unless QUEUE_TELEMETRY_UPLOAD
names a destination, and then both are sent once, when the job has ended and
its result upload (if any) is over. docs/job-telemetry.md lists every field.

Why it exists: estimate.py is calibrated on one 3090 and refits only from this
queue's own history. Comparing hardware -- another GPU, a 4-core host, a rented
instance -- needs the machine described next to the timings, which the metrics
table does not do.

Not in the export folder: export is a cache entry shared by every job with the
same key, and telemetry is per job. It also records no hostnames, paths, clip
names, job names, GPU UUIDs or serial numbers, and log bundles are redacted
(see _redactor), so a bundle can be shared as it is.

Linux-only probes (/proc, cgroups) degrade to null elsewhere. Nothing here may
stop or fail a job, or the service: every entry point a job or startup calls
(host, start_sampler, finish_sampler, write, notify) catches its own errors
and prints one line instead.
"""
from __future__ import annotations

import gzip
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

from . import db, gpu, hoststats, resources
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


def samples_path(job_id: int) -> Path:
    return run_dir(job_id) / "samples.jsonl.gz"


# A long job on a many-core host writes about 1 KB per 5 s sample; this cap
# is weeks of that, and only stops a runaway from filling the disk.
SAMPLES_MAX_BYTES = 32 * 1024 * 1024


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


def _disk_space() -> Optional[tuple[int, int, int]]:
    """(used, free, total) bytes of the filesystem under QUEUE_ROOT, or None.

    used counts every file on it (the clip, cache, runs, the image's writable
    layer in a container), so its peak is the disk a job needs.
    """
    try:
        st = os.statvfs(QUEUE_ROOT)
    except OSError:
        return None
    return ((st.f_blocks - st.f_bfree) * st.f_frsize, st.f_bavail * st.f_frsize,
            st.f_blocks * st.f_frsize)


def _disk() -> dict:
    sp = _disk_space()
    if sp is None:
        return {}
    return {"queue_root_total_bytes": sp[2], "queue_root_free_bytes": sp[1],
            "queue_root_used_bytes": sp[0]}


def _disk_peak(stages: list[dict]) -> dict:
    """Disk use over the whole job, from the stage samplers' figures."""
    res = [st for st in stages if st.get("resources")]
    peaks = [st["resources"].get("disk_used_bytes_peak") for st in res]
    peaks = [p for p in peaks if p is not None]
    frees = [st["resources"].get("disk_free_bytes_min") for st in res]
    frees = [f for f in frees if f is not None]
    if not peaks:
        return {}
    # The baseline is the stage that started first: before the job wrote anything.
    first = min((st for st in res if st["resources"].get("disk_used_bytes_start") is not None),
                key=lambda st: st.get("started") or float("inf"), default=None)
    return {"used_peak_bytes": max(peaks), "free_min_bytes": min(frees) if frees else None,
            "job_growth_peak_bytes": (max(peaks) - first["resources"]["disk_used_bytes_start"]
                                      if first else None)}


def _num(s: str, cast=float):
    """nvidia-smi prints "[N/A]" or "[Not Supported]" for what a card lacks."""
    try:
        return cast(s)
    except (TypeError, ValueError):
        return None


_GPU_HOST = ("gpu=index,name,memory.total,compute_cap,driver_version,"
             "pcie.link.gen.max,pcie.link.width.max,power.limit")
# Limits a host can lower below the card's own: a power cap under the default
# and max clocks are what a rented card is judged against.
_GPU_LIMITS = ",power.default_limit,power.max_limit,clocks.max.sm,clocks.max.mem"


def _gpus() -> Optional[list[dict]]:
    # No uuid, no serial: they identify the card, and nothing here needs that.
    # One field a driver does not know fails the whole query, so the limits
    # are asked for with a fallback to the fields every driver has.
    rows = gpu._nvidia_smi(_GPU_HOST + _GPU_LIMITS)
    if rows is None:
        rows = gpu._nvidia_smi(_GPU_HOST)
    if rows is None:
        return None
    out = []
    for r in rows:
        r = r + [""] * (12 - len(r))
        out.append({"index": _num(r[0], int), "name": r[1] or None,
                    "memory_mib": _num(r[2], int), "compute_cap": r[3] or None,
                    "driver": r[4] or None, "pcie_gen": _num(r[5], int),
                    "pcie_width": _num(r[6], int), "power_limit_w": _num(r[7]),
                    "power_default_w": _num(r[8]), "power_max_w": _num(r[9]),
                    "sm_clock_max_mhz": _num(r[10], int),
                    "mem_clock_max_mhz": _num(r[11], int)})
    return out


def _boot_time() -> Optional[float]:
    raw = _read("/proc/uptime")
    try:
        return round(time.time() - float(raw.split()[0]), 1) if raw else None
    except ValueError:
        return None


def _probe(name: str, fn):
    """fn(), or None and one line in the log: a probe never stops anything."""
    try:
        return fn()
    except Exception as exc:                                  # noqa: BLE001
        print(f"telemetry: {name} probe failed: {type(exc).__name__}: {exc}")
        return None


def host() -> dict:
    """Description of this machine. Probed once; it does not change under us.

    Never raises: the service calls it at startup, and a probe that trips on
    an unusual /proc or nvidia-smi must not keep the queue from starting.
    """
    global _host
    if _host is None:
        _host = {
            "os": _probe("os", lambda: platform.platform(terse=True)),
            "container": _probe("container", lambda: Path("/.dockerenv").exists()),
            "cpu": _probe("cpu", _cpu) or {},
            "memory": _probe("memory", _memory) or {},
            "gpus": _probe("gpu", _gpus),
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


# Why a GPU ran below its clocks (clocks_event_reasons, NVML's bit order).
# Idle is included: a stage that is mostly gpu_idle was not GPU-bound.
CLOCK_REASONS = {
    0x1: "gpu_idle", 0x2: "applications_clocks", 0x4: "sw_power_cap",
    0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal",
    0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks",
}
_GPU_SAMPLE = "gpu=index,utilization.gpu,memory.used"
_GPU_HEALTH = (",power.draw,clocks.sm,clocks.mem,temperature.gpu,"
               "pcie.link.gen.current,pcie.link.width.current,"
               "ecc.errors.uncorrected.volatile.total,")
# The limit in force now, after the clock reasons so their column stays put.
# Read every sample, not once: a host can lower it while a job runs.
_GPU_LIMIT_NOW = ",enforced.power.limit"
# Most complete first. clocks_event_reasons is the newer name (drivers from
# about 535); older drivers only know clocks_throttle_reasons. The last is
# what this sampler asked before it recorded health.
_SAMPLE_QUERIES = (_GPU_SAMPLE + _GPU_HEALTH + "clocks_event_reasons.active" + _GPU_LIMIT_NOW,
                   _GPU_SAMPLE + _GPU_HEALTH + "clocks_event_reasons.active",
                   _GPU_SAMPLE + _GPU_HEALTH + "clocks_throttle_reasons.active" + _GPU_LIMIT_NOW,
                   _GPU_SAMPLE + _GPU_HEALTH + "clocks_throttle_reasons.active",
                   _GPU_SAMPLE)
_sample_query: Optional[str] = None      # the first that worked on this driver
# A busy sample: clocks and power are read from these only, since an idle GPU
# drops its clocks and PCIe link on purpose.
BUSY_UTIL = 50.0
# Power limit last seen per GPU, across stages and samplers, so a change
# between two stages is caught too.
_limit_seen: dict[int, float] = {}
_limit_lock = threading.Lock()
LIMIT_CHANGES_MAX = 20


def _gpu_sample_rows() -> Optional[list[list[str]]]:
    global _sample_query
    if _sample_query:
        return gpu._nvidia_smi(_sample_query)
    for q in _SAMPLE_QUERIES:
        rows = gpu._nvidia_smi(q)
        if rows is not None:
            _sample_query = q
            return rows
    return None                           # nothing learned; try again next time


class ResourceSampler(threading.Thread):
    """CPU, RSS and GPU use of one stage's process tree, sampled.

    CPU seconds are the sum, over every process seen, of its own utime+stime
    at the last sample. A child that starts and exits between two samples is
    missed, so this undercounts short-lived helpers; the stage scripts and
    LichtFeld are long-lived, so what it measures -- how many cores a stage
    actually kept busy -- holds. GPU figures are for the whole device, not
    just this job, which is exact with one job per GPU.

    GPU health (power, clocks, temperature, PCIe link, clock reasons, ECC) is
    for telling a slow host from a slow stage: a power cap, a riser at x1 or a
    card that throttles hot shows here, not in utilisation.
    """

    def __init__(self, gpu_index: Optional[int], pid: int, interval: float = 5.0,
                 job_id: Optional[int] = None, stage: Optional[str] = None):
        super().__init__(daemon=True, name="resources")
        self.gpu = gpu_index
        self.pid = pid
        self.interval = interval
        # With a job: one line per sample in its samples.jsonl.gz.
        self.job_id = job_id
        self.stage = stage
        self._host_first: Optional[dict] = None
        self._host_prev: Optional[dict] = None
        self._cpu_prev: Optional[float] = None
        self._gpu_now: Optional[dict] = None
        self._series_off = job_id is None or not TELEMETRY_ENABLED
        self.cpu: dict[int, float] = {}
        self.rss_peak = 0
        self.gpu_util: list[float] = []
        self.gpu_mem_peak = 0
        # Busy samples only (utilisation >= BUSY_UTIL).
        self.power: list[float] = []
        self.sm_clock: list[float] = []
        self.mem_clock: list[float] = []
        self.temp_max: Optional[float] = None
        self.pcie_gen_max: Optional[int] = None
        self.pcie_width_max: Optional[int] = None
        self.ecc_max: Optional[int] = None
        self.reasons: dict[str, int] = {}
        self.reason_samples = 0
        # Busy samples whose clocks the power cap held down. The share over
        # all samples misleads: an idle card can report sw_power_cap too.
        self.busy_reason_samples = 0
        self.busy_power_capped = 0
        self.limit_min: Optional[float] = None
        self.limit_max: Optional[float] = None
        self.limit_changes: list[dict] = []
        self.disk_used_start: Optional[int] = None
        self.disk_used_peak: Optional[int] = None
        self.disk_free_min: Optional[int] = None
        self._disk_now: Optional[tuple[int, int, int]] = None
        self.samples = 0
        # Not _stop: that name is threading.Thread's own method, which join()
        # calls; an Event there made join() raise TypeError.
        self._halt = threading.Event()
        self._logged = False

    def _gpu_row(self, r: list[str]) -> None:
        util = float(r[1])
        self.gpu_util.append(util)
        self.gpu_mem_peak = max(self.gpu_mem_peak, int(r[2]))
        self._gpu_now = {"util": util, "mem_mib": int(r[2])}
        if len(r) < 11:
            return
        power, sm, mem, temp = (_num(x) for x in r[3:7])
        gen, width, ecc = (_num(x, int) for x in r[7:10])
        self._gpu_now.update({"power_w": power, "sm_mhz": sm, "mem_mhz": mem,
                              "temp_c": temp, "pcie_gen": gen, "pcie_width": width})
        if util >= BUSY_UTIL:
            for vals, v in ((self.power, power), (self.sm_clock, sm),
                            (self.mem_clock, mem)):
                if v is not None:
                    vals.append(v)
        # The link trains down when idle, so the highest seen is what the
        # slot can do; below the card's max under load means a narrow slot.
        for attr, v in (("temp_max", temp), ("pcie_gen_max", gen),
                        ("pcie_width_max", width), ("ecc_max", ecc)):
            if v is not None:
                cur = getattr(self, attr)
                setattr(self, attr, v if cur is None else max(cur, v))
        if len(r) > 11:
            self._limit(int(r[0]), _num(r[11]))
        try:
            bits = int(r[10], 16)
        except ValueError:
            return
        self.reason_samples += 1
        if util >= BUSY_UTIL:
            self.busy_reason_samples += 1
            self.busy_power_capped += bool(bits & 0x4)
        for bit, name in CLOCK_REASONS.items():
            if bits & bit:
                self.reasons[name] = self.reasons.get(name, 0) + 1
        self._gpu_now["clock_reasons"] = [n for b, n in CLOCK_REASONS.items() if bits & b]

    def _limit(self, index: int, limit: Optional[float]) -> None:
        """The enforced power limit: range over the stage, and every change."""
        if limit is None:
            return
        self._gpu_now["power_limit_w"] = limit
        self.limit_min = limit if self.limit_min is None else min(self.limit_min, limit)
        self.limit_max = limit if self.limit_max is None else max(self.limit_max, limit)
        with _limit_lock:
            prev = _limit_seen.get(index)
            _limit_seen[index] = limit
        if prev is not None and prev != limit:
            print(f"gpu {index}: power limit changed {prev:g} -> {limit:g} W")
            if len(self.limit_changes) < LIMIT_CHANGES_MAX:
                self.limit_changes.append({"t": round(time.time(), 1),
                                           "from_w": prev, "to_w": limit})

    def sample(self) -> None:
        rss = 0
        if Path("/proc").is_dir():
            for p in _proc_tree(self.pid):
                st = _proc_stat(p)
                if st:
                    self.cpu[p] = st[0]
                    rss += st[1]
        self.rss_peak = max(self.rss_peak, rss)
        self._gpu_now = None
        if self.gpu is not None and self.gpu >= 0:
            for r in _gpu_sample_rows() or []:
                try:
                    if int(r[0]) == self.gpu:
                        self._gpu_row(r)
                except (IndexError, ValueError):
                    pass
        self._disk_now = _disk_space()
        if self._disk_now:
            used, free, _ = self._disk_now
            if self.disk_used_start is None:
                self.disk_used_start = used
            self.disk_used_peak = used if self.disk_used_peak is None else max(self.disk_used_peak, used)
            self.disk_free_min = free if self.disk_free_min is None else min(self.disk_free_min, free)
        self.samples += 1
        host = hoststats.read()
        if self._host_first is None:
            self._host_first = host
        self._series(rss, host)
        self._host_prev = host

    def _series(self, rss: int, host: dict) -> None:
        """Append this sample to the job's samples.jsonl.gz.

        One gzip member per line: a service killed mid-write loses at most
        that line, and every earlier one still reads (gzip.open reads the
        members as one stream). The first sample only sets the baseline the
        next one's rates are measured from.
        """
        cpu_now = sum(self.cpu.values()) if self.cpu else None
        prev, cpu_prev = self._host_prev, self._cpu_prev
        self._cpu_prev = cpu_now
        if self._series_off or prev is None:
            return
        rates = hoststats.rates(prev, host, per_core=True)
        dt = (rates or {}).get("seconds")
        line = {"t": round(time.time(), 1), "stage": self.stage,
                "cores_busy": (round((cpu_now - cpu_prev) / dt, 2)
                               if dt and cpu_now is not None and cpu_prev is not None else None),
                "rss_mb": round(rss / 1e6) if rss else None,
                "disk_used_mb": round(self._disk_now[0] / 1e6) if self._disk_now else None,
                "disk_free_mb": round(self._disk_now[1] / 1e6) if self._disk_now else None,
                "host": rates, "gpu": self._gpu_now}
        try:
            p = samples_path(self.job_id)
            if p.exists() and p.stat().st_size > SAMPLES_MAX_BYTES:
                self._series_off = True
                print(f"job {self.job_id}: samples.jsonl.gz is over "
                      f"{SAMPLES_MAX_BYTES >> 20} MB; no more samples written")
                return
            p.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, "ab") as f:
                f.write(json.dumps(line, separators=(",", ":")).encode() + b"\n")
        except Exception as exc:                              # noqa: BLE001
            self._series_off = True
            print(f"job {self.job_id}: samples not written: {type(exc).__name__}: {exc}")

    def run(self) -> None:
        # First look early, so a stage shorter than one interval still gets
        # a sample; frames and select on a short clip take seconds.
        wait = min(0.5, self.interval)
        while not self._halt.wait(wait):
            wait = self.interval
            try:
                self.sample()
            except Exception as exc:                          # noqa: BLE001
                # Logged once, not every 5 s; the stage runs on either way.
                if not self._logged:
                    self._logged = True
                    print(f"telemetry: resource sample failed (pid {self.pid}): "
                          f"{type(exc).__name__}: {exc}")

    def stop(self) -> None:
        self._halt.set()

    def summary(self, wall_s: float) -> dict:
        cpu_s = round(sum(self.cpu.values()), 1)
        util = sorted(self.gpu_util)

        def pct(vals, q):
            vals = sorted(vals)
            return round(vals[min(len(vals) - 1, int(q * len(vals)))], 1) if vals else None
        return {
            "samples": self.samples,
            "cpu_seconds": cpu_s if self.cpu else None,
            "avg_cores_busy": round(cpu_s / wall_s, 2) if self.cpu and wall_s > 0 else None,
            "rss_peak_bytes": self.rss_peak or None,
            "gpu_util_p50": pct(util, 0.5), "gpu_util_p95": pct(util, 0.95),
            "gpu_util_mean": round(statistics.fmean(util), 1) if util else None,
            "gpu_mem_peak_mib": self.gpu_mem_peak or None,
            "gpu_busy_samples": len(self.power) if self.reason_samples else None,
            "gpu_power_w_busy_p50": pct(self.power, 0.5),
            "gpu_power_w_max": round(max(self.power), 1) if self.power else None,
            "gpu_sm_mhz_busy_p50": pct(self.sm_clock, 0.5),
            "gpu_mem_mhz_busy_p50": pct(self.mem_clock, 0.5),
            "gpu_temp_c_max": self.temp_max,
            "gpu_pcie_gen_max": self.pcie_gen_max,
            "gpu_pcie_width_max": self.pcie_width_max,
            "gpu_ecc_uncorrected": self.ecc_max,
            # Share of samples in which each reason held a clock down.
            "gpu_clock_reasons": ({k: round(v / self.reason_samples, 2)
                                   for k, v in sorted(self.reasons.items())}
                                  if self.reason_samples else None),
            "gpu_sw_power_cap_busy": (round(self.busy_power_capped / self.busy_reason_samples, 2)
                                      if self.busy_reason_samples else None),
            "gpu_power_limit_w_min": self.limit_min,
            "gpu_power_limit_w_max": self.limit_max,
            "gpu_power_limit_changes": self.limit_changes if self.limit_min is not None else None,
            # The filesystem under QUEUE_ROOT: what the stage needed of the disk.
            "disk_used_bytes_start": self.disk_used_start,
            "disk_used_bytes_peak": self.disk_used_peak,
            "disk_free_bytes_min": self.disk_free_min,
            # The machine over the whole stage: steal, pressure, throttling.
            "host": hoststats.rates(self._host_first, self._host_prev),
        }


def record_resources(job_id: int, stage: str, summary: dict) -> None:
    with _res_lock:
        _resources.setdefault(job_id, {})[stage] = summary


def start_sampler(gpu_index: Optional[int], pid: int, interval: float = 5.0,
                  job_id: Optional[int] = None,
                  stage: Optional[str] = None) -> Optional[ResourceSampler]:
    """A running sampler for one stage, or None. Never raises.

    With a job id, it also writes the job's time series (samples.jsonl.gz).
    """
    try:
        s = ResourceSampler(gpu_index, pid, interval, job_id=job_id, stage=stage)
        s.start()
        return s
    except Exception as exc:                                  # noqa: BLE001
        print(f"telemetry: no resource sampling for pid {pid}: "
              f"{type(exc).__name__}: {exc}")
        return None


def finish_sampler(job_id: Optional[int], stage: str,
                   sampler: Optional[ResourceSampler], wall_s: float) -> Optional[dict]:
    """Stop the sampler and return its summary, recorded for the job if one
    is given (the benchmark passes None). Never raises.

    Called from the stage's `finally`: an exception here would replace the
    stage's own outcome, so a telemetry bug would fail a healthy job.
    """
    if sampler is None:
        return None
    try:
        sampler.stop()
        summary = sampler.summary(wall_s)
        if job_id is not None:
            record_resources(job_id, stage, summary)
        return summary
    except Exception as exc:                                  # noqa: BLE001
        print(f"telemetry: {stage} resources not recorded"
              f"{'' if job_id is None else f' for job {job_id}'}: "
              f"{type(exc).__name__}: {exc}")
        return None


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


INPUT_FETCH = RUNS_ROOT / "input_fetch.json"     # docker/entrypoint.sh writes it


def _rate(nbytes, seconds) -> Optional[float]:
    return round(nbytes / 1e6 / seconds, 2) if nbytes and seconds else None


def _input_transfer(input_file: Optional[str]) -> Optional[dict]:
    """How INPUT_URL's download of this job's clip went, or None.

    None when the clip came some other way (copied in, mounted) or was
    already there: the entrypoint only writes a record when it downloads.
    Matched on the clip's name, which does not go into the record.
    """
    try:
        rec = json.loads(INPUT_FETCH.read_text())
    except (OSError, ValueError):
        return None
    if not input_file or not isinstance(rec, dict) or rec.get("file") != input_file:
        return None
    nbytes, secs = _num(rec.get("bytes"), int), _num(rec.get("seconds"))
    return {"bytes": nbytes, "seconds": secs, "mb_s": _rate(nbytes, secs),
            "first_byte_s": _num(rec.get("first_byte_s")),
            "ended": _num(rec.get("ended"))}


def _output_transfer(job_id: int) -> Optional[dict]:
    """OUTPUT_UPLOAD_URL's upload of this job's result (upload.json), or None.

    Only the sizes and times: upload.json also names the destination.
    """
    try:
        st = json.loads((run_dir(job_id) / "upload.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(st, dict):
        return None
    nbytes, secs = st.get("bytes"), st.get("transfer_s")
    return {"state": st.get("state"), "bytes": nbytes, "seconds": secs,
            "mb_s": _rate(nbytes, secs), "attempts": st.get("attempts"),
            "pack_s": st.get("pack_s"), "ended": st.get("ended")}


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
    # The previous record's resources, under what is in memory: a service
    # restart mid-job, or the rewrite after an output upload (which comes
    # after the final write dropped them from memory), would lose them.
    res = {}
    try:
        prev = json.loads(telemetry_path(job_id).read_text())
        res = {s["stage"]: s["resources"] for s in prev.get("stages", [])
               if s.get("resources")}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    with _res_lock:
        res.update(_resources.get(job_id, {}))
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
        "host": {**host(), "disk": {**_disk(), **_disk_peak(stages)}},
        "software": _software(),
        "transfers": {"input": _input_transfer(f), "output": _output_transfer(job_id)},
        "placement": TELEMETRY_PLACEMENT or None,
        "timeline": {"host_boot": _boot_time(),
                     "service_started": round(SERVICE_STARTED, 1)},
    }


def write(job_id: int, final: bool = False, upload: Optional[bool] = None) -> None:
    """Rewrite this job's telemetry (and, at the end, its logs).

    Called after every stage and once when the job ends. Never raises.

    Uploaded once, at the very end (`upload`, which defaults to `final`): one
    complete record per job rather than a partial one per stage. When the job
    also uploads its result, the worker defers this to outputs.py, which
    uploads the record after that transfer so it carries transfers.output.
    """
    if upload is None:
        upload = final
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
        if final:
            _bundle_logs(job_id)
            with _res_lock:
                _resources.pop(job_id, None)
        files = ["telemetry.json"]
        if logs_path(job_id).is_file():
            files.append("logs.tar.gz")
        if upload and TELEMETRY_UPLOAD:
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
        # Numbers only, nothing to redact. Read whole, so a sampler still
        # appending cannot cut the member short.
        try:
            data = samples_path(job_id).read_bytes()
        except OSError:
            data = b""
        if data:
            ti = tarfile.TarInfo("samples.jsonl.gz")
            ti.size, ti.mtime = len(data), int(time.time())
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
