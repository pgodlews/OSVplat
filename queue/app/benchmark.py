"""Host benchmark: run scripts/benchmark.py with the machine to itself.

Why: per-job timings cannot be compared across clips, and a rented host can be
slower than its labels (an oversubscribed CPU, a power-capped card, an x1
riser, a slow disk). A fixed workload gives numbers that compare across
machines; judging them is left to whatever launched this one.

POST /api/benchmark starts a run in the background and GET /api/benchmark
returns the state and the last result. QUEUE_BENCHMARK=1 runs it once at
startup. A run holds the whole machine (worker.exclusive): it is refused while
a job or render runs, and no job starts until it ends.

The script runs in venv_gs on the first free GPU, inside the same
ResourceSampler as a stage, so the record carries GPU health under load
(power, clocks, clock reasons, PCIe link) next to the scores. It is written to
QUEUE_ROOT/benchmark.json, with the anonymous host description from
telemetry.host(). Failures are loud (state "failed", the errors, a line in the
service log) and never touch the queue's jobs.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from typing import Optional

from . import telemetry, worker
from .config import (BENCHMARK, BENCHMARK_DISK_GB, GS_PY, LOG_ROOT, QUEUE_ROOT,
                     SPLAT_ROOT, TELEMETRY_PLACEMENT)

RESULT = QUEUE_ROOT / "benchmark.json"
LOG = LOG_ROOT / "benchmark.log"
TIMEOUT = 900             # a healthy run takes 1-2 minutes

_lock = threading.Lock()
_state: dict = {"state": "idle"}


def status() -> dict:
    """The current run (state running) or the last result, if any."""
    with _lock:
        if _state["state"] == "running":
            return dict(_state)
    try:
        return json.loads(RESULT.read_text())
    except (OSError, ValueError):
        return {"state": "idle"}


def _write(rec: dict) -> None:
    tmp = RESULT.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, RESULT)


def _run(gpus: list[int], hold) -> None:
    started = time.time()
    g = gpus[0] if gpus else None
    rec: dict = {"schema": "osvplat.benchmark.run/1", "state": "failed",
                 "started": round(started, 1), "gpu_index": g}
    out = QUEUE_ROOT / f".benchmark_{uuid.uuid4().hex[:6]}.json"
    try:
        cpus = telemetry.host()["cpu"].get("effective_cpus")
        argv = [str(GS_PY), str(BENCHMARK), "--out", str(out),
                "--scratch", str(QUEUE_ROOT), "--disk-gb", str(BENCHMARK_DISK_GB)]
        if cpus:
            argv += ["--cpus", str(max(1, int(cpus)))]
        env = dict(os.environ)
        if g is None:
            argv.append("--no-gpu")
            rec.setdefault("errors", {})["gpu"] = "no free GPU: GPU part not run"
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(g)
        with open(LOG, "wb") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, cwd=str(SPLAT_ROOT),
                                    start_new_session=True)
            sampler = telemetry.start_sampler(g, proc.pid, interval=2.0)
            try:
                code = proc.wait(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                worker._terminate(proc.pid, waiter=proc.wait)
                code = None
        rec["resources"] = telemetry.finish_sampler(None, "benchmark", sampler,
                                                    time.time() - started)
        try:
            scores = json.loads(out.read_text())
        except (OSError, ValueError):
            scores = {}
        for k in ("cpu", "memory", "disk", "gpu", "durations_s", "seconds_per_test"):
            if k in scores:
                rec[k] = scores[k]
        errors = {**rec.get("errors", {}), **scores.get("errors", {})}
        if code is None:
            errors["run"] = f"timed out after {TIMEOUT} s"
        elif code != 0 and not scores.get("errors"):
            errors["run"] = f"exited {code} without a result; see the benchmark log"
        rec["errors"] = errors
        rec["state"] = "failed" if errors else "done"
    except Exception as exc:                                  # noqa: BLE001
        rec.setdefault("errors", {})["run"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        # Whatever happens in here, the hold is released: a benchmark that
        # kept it would leave the queue dispatching nothing, for good.
        try:
            out.unlink(missing_ok=True)
            rec["ended"] = round(time.time(), 1)
            rec["wall_s"] = round(rec["ended"] - started, 1)
            rec["host"] = {**telemetry.host(), "disk": telemetry._disk()}
            rec["software"] = telemetry._software()
            rec["placement"] = TELEMETRY_PLACEMENT or None
            _write(rec)
            if rec["state"] == "failed":
                print(f"benchmark FAILED: {json.dumps(rec.get('errors'))} (log: {LOG})")
            else:
                print(f"benchmark done in {rec['wall_s']} s")
        except Exception as exc:                              # noqa: BLE001
            print(f"benchmark: result not written: {type(exc).__name__}: {exc}")
        finally:
            with _lock:
                _state.clear()
                _state["state"] = "idle"
            hold.__exit__(None, None, None)


def start() -> tuple[bool, str]:
    """Start a run in the background. (False, reason) when it cannot now."""
    with _lock:
        if _state["state"] == "running":
            return False, "a benchmark is already running"
        hold = worker.exclusive()
        gpus = hold.__enter__()
        if gpus is None:
            hold.__exit__(None, None, None)
            return False, ("a job or render is running; the benchmark needs the "
                           "machine to itself, so run it when the queue is idle")
        _state.clear()
        _state.update({"state": "running", "started": round(time.time(), 1),
                       "gpu_index": gpus[0] if gpus else None})
    threading.Thread(target=_run, args=(gpus, hold), daemon=True,
                     name="benchmark").start()
    return True, "started"


def running() -> bool:
    with _lock:
        return _state["state"] == "running"
