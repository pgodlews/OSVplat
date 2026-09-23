"""Prometheus exposition for the queue.

Hand-rendered rather than pulling in prometheus_client: this service has one
runtime dependency and the text format is a few dozen lines to emit correctly.

What matters more than the renderer is what is deliberately NOT here. Per-job
series are emitted for RUNNING jobs only, so the label set is bounded by the
number of GPUs. One series per job id across every job the queue has ever run
would grow without limit, and an endpoint that does that eventually costs more
than the thing it measures -- this queue is at 166 jobs and climbing.
"""
from __future__ import annotations

import math
import os
import time
from typing import Iterable, Optional

from . import db, estimate, hoststats, retention, telemetry, worker
from .config import GPUS

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
PREFIX = "splatqueue"

MIB = 1024 * 1024


def _esc(v: str) -> str:
    """Escape a label VALUE: backslash, double quote and newline, in that order."""
    return (str(v).replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n"))


def _num(v) -> str:
    """Prometheus spells the specials differently to Python, and repr(float)
    would emit `nan`/`inf`, which a scraper rejects as a parse error."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NaN"
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "+Inf" if f > 0 else "-Inf"
    return repr(int(f)) if f.is_integer() and abs(f) < 1e15 else repr(f)


class Exposition:
    """Accumulates metric families in emission order."""

    def __init__(self) -> None:
        self._out: list[str] = []

    def add(self, name: str, help_text: str, mtype: str,
            samples: Iterable[tuple[Optional[dict], object]]) -> None:
        rows = [(lbl, val) for lbl, val in samples if val is not None]
        if not rows:
            # A family with no samples is legal but tells a dashboard nothing;
            # leaving it out keeps `up`-style checks honest about what exists.
            return
        full = f"{PREFIX}_{name}"
        self._out.append(f"# HELP {full} {help_text}")
        self._out.append(f"# TYPE {full} {mtype}")
        for labels, value in rows:
            tags = ""
            if labels:
                tags = "{" + ",".join(
                    f'{k}="{_esc(v)}"' for k, v in labels.items()
                    if v is not None) + "}"
            self._out.append(f"{full}{tags} {_num(value)}")

    def scalar(self, name: str, help_text: str, mtype: str, value) -> None:
        self.add(name, help_text, mtype, [(None, value)])

    def text(self) -> str:
        return "\n".join(self._out) + "\n"


def _gpu_health(e: Exposition) -> None:
    """Power, clocks, temperature, PCIe link and clock reasons, per GPU.

    The same nvidia-smi fields job telemetry samples; one query per scrape.
    """
    rows = []
    for r in telemetry._gpu_sample_rows() or []:
        if len(r) < 11:
            continue                      # a driver without the health fields
        try:
            rows.append((int(r[0]), r))
        except ValueError:
            pass
    num = telemetry._num

    def fam(name, help_text, i, scale=1.0, cast=float):
        e.add(name, help_text, "gauge",
              [({"gpu": g}, num(r[i], cast) * scale) for g, r in rows
               if num(r[i], cast) is not None])
    fam("gpu_power_watts", "GPU power draw.", 3)
    fam("gpu_sm_clock_hertz", "GPU SM clock.", 4, 1e6)
    fam("gpu_memory_clock_hertz", "GPU memory clock.", 5, 1e6)
    fam("gpu_temperature_celsius", "GPU temperature.", 6)
    fam("gpu_pcie_link_generation", "Current PCIe generation. The link trains "
        "down when idle; compare under load.", 7, cast=int)
    fam("gpu_pcie_link_width", "Current PCIe lanes. The link trains down when "
        "idle; compare under load.", 8, cast=int)
    fam("gpu_ecc_uncorrected_errors", "Uncorrected ECC errors since the driver "
        "loaded. Absent on cards without ECC.", 9, cast=int)
    reasons = []
    for g, r in rows:
        try:
            bits = int(r[10], 16)
        except ValueError:
            continue
        reasons += [({"gpu": g, "reason": n}, 1 if bits & b else 0)
                    for b, n in telemetry.CLOCK_REASONS.items()]
    e.add("gpu_clock_event_reason", "1 while this reason holds the GPU "
          "clocks down (sw_power_cap, hw_slowdown, sw_thermal...).", "gauge", reasons)


def _host(e: Exposition) -> None:
    """Machine-wide counters (hoststats.py); rate() them for shares and speeds.

    Counters rather than per-scrape rates: they need no state here, and any
    scrape interval gets the right answer.
    """
    h = hoststats.read()
    try:
        hz = os.sysconf("SC_CLK_TCK")
    except (AttributeError, ValueError, OSError):
        hz = 100
    st = h.get("stat")
    if st:
        e.add("host_cpu_seconds_total", "CPU time of the whole machine, summed "
              "over cores, by mode. steal is time the hypervisor gave to other "
              "guests; it is not in busy.", "counter",
              [({"mode": m}, st["total"][m] / hz)
               for m in ("busy", "idle", "iowait", "steal")])
    stalls = []
    for scope, key in (("system", "psi"), ("cgroup", "cgroup_psi")):
        for res, kinds in (h.get(key) or {}).items():
            for kind, us in (kinds or {}).items():
                stalls.append(({"scope": scope, "resource": res, "kind": kind}, us / 1e6))
    e.add("host_pressure_stalled_seconds_total", "Pressure stall time (PSI): "
          "some = at least one task waited for the resource, full = all did. "
          "scope system is the whole machine, cgroup this container.",
          "counter", stalls)
    cg = h.get("cgroup_cpu")
    if cg:
        e.scalar("cgroup_cpu_periods_total", "CPU quota periods elapsed.",
                 "counter", cg["periods"])
        e.scalar("cgroup_cpu_throttled_periods_total",
                 "Periods in which the CPU quota throttled this container.",
                 "counter", cg["throttled"])
        e.scalar("cgroup_cpu_throttled_seconds_total",
                 "Time this container spent throttled by its CPU quota.",
                 "counter", cg["throttled_us"] / 1e6)
    d = h.get("disk")
    if d:
        e.scalar("host_disk_read_bytes_total", "Bytes read, all disks.", "counter", d["read"])
        e.scalar("host_disk_written_bytes_total", "Bytes written, all disks.",
                 "counter", d["write"])
    n = h.get("net")
    if n:
        e.scalar("host_network_receive_bytes_total",
                 "Bytes received, all interfaces but lo.", "counter", n["rx"])
        e.scalar("host_network_transmit_bytes_total",
                 "Bytes sent, all interfaces but lo.", "counter", n["tx"])


def render(jobs: list[dict], now: Optional[float] = None) -> str:
    """`jobs` is the active (running and queued) job dicts, already assembled.

    Passed in rather than rebuilt here so there is one definition of what a job
    looks like, and so this module never has to import the API layer.
    """
    started = time.monotonic()
    now = now or time.time()
    e = Exposition()

    e.scalar("up", "Always 1; a scrape that fails is absence, not 0.",
             "gauge", 1)

    # ------------------------------------------------------------ queue
    counts = {r["state"]: r["n"] for r in db.conn().execute(
        "SELECT state, COUNT(*) n FROM jobs GROUP BY state").fetchall()}
    e.add("jobs", "Jobs by state, including ones hidden from the queue view.",
          "gauge", [({"state": s}, n) for s, n in sorted(counts.items())])
    e.scalar("jobs_hidden",
             "Jobs cleared from the queue view but kept for history.", "gauge",
             db.conn().execute(
                 "SELECT COUNT(*) n FROM jobs WHERE hidden=1").fetchone()["n"])

    e.scalar("paused", "1 when the dispatcher is holding jobs back.", "gauge",
             1 if worker.paused() else 0)
    e.scalar("max_concurrent", "Configured ceiling on jobs in flight.",
             "gauge", worker.max_concurrent())
    e.scalar("schedulable_capacity",
             "Jobs that can actually run now: the ceiling capped by GPUs not "
             "carrying foreign work.", "gauge", worker.schedulable_capacity())

    running = [j for j in jobs if j["state"] == "running"]
    queued = [j for j in jobs if j["state"] == "queued"]
    e.scalar("queue_pending_seconds",
             "Estimated work still to do, ignoring how it parallelises.",
             "gauge",
             sum((j["eta"].get("remaining") or 0.0) for j in running)
             + sum((j["eta"].get("plan_total") or 0.0) for j in queued))

    # ------------------------------------------------- running jobs only
    e.add("job_info",
          "Labels for a running job; the value is always 1.", "gauge",
          [({"job_id": j["id"], "name": j["name"],
             "stage": (j["eta"].get("stage") or ""),
             "gpu": ("" if j["gpu"] is None else j["gpu"])}, 1)
           for j in running])
    e.add("job_progress_ratio",
          "Fraction of a running job complete, weighted by estimated stage "
          "cost.", "gauge",
          [({"job_id": j["id"]}, j["eta"].get("fraction")) for j in running])
    e.add("job_elapsed_seconds", "Wall time since a running job started.",
          "gauge",
          [({"job_id": j["id"]}, j["eta"].get("elapsed")) for j in running])
    e.add("job_eta_seconds",
          "Estimated seconds remaining. Absent while the trainer is still "
          "ramping up and any figure would be badly optimistic.", "gauge",
          [({"job_id": j["id"]}, j["eta"].get("remaining")) for j in running])

    stage_rows, step_rows, target_rows, splat_rows = [], [], [], []
    for j in running:
        for s in j["stages"]:
            if s["state"] == "running" and s.get("fraction") is not None:
                stage_rows.append(
                    ({"job_id": j["id"], "stage": s["stage"]}, s["fraction"]))
            if s["stage"] == "train" and s.get("progress"):
                p = s["progress"]
                step_rows.append(({"job_id": j["id"]}, p.get("step")))
                target_rows.append(({"job_id": j["id"]}, p.get("total")))
                splat_rows.append(({"job_id": j["id"]}, p.get("splats")))
    e.add("job_stage_progress_ratio", "Fraction of the stage now in flight.",
          "gauge", stage_rows)
    e.add("job_train_step", "Training iteration reached.", "gauge", step_rows)
    e.add("job_train_target_steps", "Training iterations this job will run.",
          "gauge", target_rows)
    e.add("job_splats", "Gaussians in the model being trained.", "gauge",
          splat_rows)

    # -------------------------------------------------------------- GPUs
    st = worker.running_state()
    rows = st.get("gpus") or []
    e.add("gpu_memory_used_bytes", "GPU memory in use, by anything.", "gauge",
          [({"gpu": g["index"]}, g["mem_used"] * MIB)
           for g in rows if g.get("mem_used") is not None])
    e.add("gpu_memory_total_bytes", "GPU memory installed.", "gauge",
          [({"gpu": g["index"]}, g["mem_total"] * MIB)
           for g in rows if g.get("mem_total") is not None])
    e.add("gpu_utilization_ratio", "GPU utilisation, 0 to 1.", "gauge",
          [({"gpu": g["index"]}, g["util"] / 100.0)
           for g in rows if g.get("util") is not None])
    e.add("gpu_available",
          "1 when the queue would schedule onto this GPU right now.", "gauge",
          [({"gpu": g["index"]}, 1 if g["available"] else 0) for g in rows])
    e.add("gpu_foreign_processes",
          "Processes on the GPU that this queue did not start. Any at all "
          "makes the card unschedulable.", "gauge",
          [({"gpu": g["index"]}, len(g.get("foreign") or [])) for g in rows])
    e.scalar("gpus_configured", "GPUs this queue is allowed to schedule on.",
             "gauge", len(GPUS))
    e.scalar("gpu_probe_ok",
             "1 when nvidia-smi answered. A failed probe schedules nothing.",
             "gauge", 1 if (rows and rows[0].get("probe_ok")) else 0)

    _gpu_health(e)
    _host(e)

    # --------------------------------------------------------- estimator
    c = estimate.constants()
    e.scalar("train_rate_iterations_per_second",
             "Baseline training rate the ETA is built from, at width 3840 and "
             "a 3M cap.", "gauge", c["train_it_per_s"])
    e.scalar("train_rate_fitted",
             "1 when that rate was fitted from completed jobs, 0 when it is "
             "still the seed constant.", "gauge",
             1 if c.get("_train_rate_source") == "fitted" else 0)

    # ----------------------------------------------------------- storage
    by_stage: dict[str, list[int]] = {}
    for r in db.list_cache():
        agg = by_stage.setdefault(r["stage"], [0, 0])
        agg[0] += 1
        agg[1] += r["bytes"] or 0
    e.add("cache_entries", "Cached stage outputs on disk.", "gauge",
          [({"stage": s}, v[0]) for s, v in sorted(by_stage.items())])
    e.add("cache_bytes", "Bytes held by cached stage outputs.", "gauge",
          [({"stage": s}, v[1]) for s, v in sorted(by_stage.items())])

    disk = retention.status()
    e.scalar("disk_free_bytes", "Free space on the queue's filesystem.",
             "gauge", disk.get("free"))
    e.scalar("disk_min_free_bytes",
             "Floor the queue refuses to start a stage below.", "gauge",
             disk.get("min_free"))
    e.scalar("cache_total_bytes", "Total size of the stage cache.", "gauge",
             disk.get("cache_total"))

    e.scalar("scrape_duration_seconds", "Time spent building this response.",
             "gauge", round(time.monotonic() - started, 4))
    return e.text()
