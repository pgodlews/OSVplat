"""GPU availability, including detection of processes the queue did not start.

The queue shares the box with hand-launched runs. A GPU carrying any compute
process that is not one of our own PIDs is treated as unavailable, so a manual
`LichtFeld-Studio` invocation can never be stomped on.

Every query is fail-CLOSED: if `nvidia-smi` cannot be run, times out, or returns
non-zero we do not know what is on the GPUs, and an unknown GPU is treated as
busy. Failing open here silently disables the foreign-process guard exactly when
the box is unhealthy, which is when it matters most.
"""
from __future__ import annotations

import subprocess
from typing import Iterable, Optional

from . import resources
from .config import GPUS

_caps: Optional[dict[int, str]] = None


def _nvidia_smi(query: str, extra: list[str] | None = None) -> Optional[list[list[str]]]:
    """Rows from one nvidia-smi query, or None if the query itself failed.

    None and [] are different answers: [] means "nvidia-smi ran and reported
    nothing", None means "we have no idea". Callers must not conflate them.
    """
    cmd = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"]
    cmd += extra or []
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    rows = []
    for line in out.stdout.strip().splitlines():
        line = line.strip()
        if line:
            rows.append([c.strip() for c in line.split(",")])
    return rows


def _uuid_to_index() -> Optional[dict[str, int]]:
    rows = _nvidia_smi("gpu=index,uuid")
    if rows is None:
        return None
    out = {}
    for r in rows:
        if len(r) >= 2:
            try:
                out[r[1]] = int(r[0])
            except ValueError:
                continue
    return out


def compute_procs() -> Optional[dict[int, list[dict]]]:
    """{gpu_index: [{pid, name, mib}]} for every running compute process.

    None when the probe failed -- see _nvidia_smi.
    """
    idx = _uuid_to_index()
    if idx is None:
        return None
    apps = _nvidia_smi("compute-apps=gpu_uuid,pid,process_name,used_memory")
    if apps is None:
        return None
    out: dict[int, list[dict]] = {g: [] for g in idx.values()}
    for r in apps:
        if len(r) < 4:
            continue
        gpu = idx.get(r[0])
        if gpu is None:
            continue
        try:
            pid = int(r[1])
        except ValueError:
            continue
        mib = int(r[3]) if r[3].isdigit() else 0
        out.setdefault(gpu, []).append({"pid": pid, "name": r[2], "mib": mib})
    return out


def compute_caps() -> dict[int, str]:
    """{index: "8.6"}, probed once: a card does not change under us. {} (and
    retried next time) when nvidia-smi failed, so no card is refused on a guess."""
    global _caps
    if _caps is None:
        rows = _nvidia_smi("gpu=index,compute_cap")
        if rows is None:
            return {}
        _caps = {int(r[0]): r[1] for r in rows if len(r) >= 2 and r[0].isdigit()}
    return _caps


def status(own_pids: Iterable[int] = (), held: Iterable[int] = ()) -> list[dict]:
    """Per-GPU status with a foreign-process flag and an availability verdict.

    `held` is the set of GPUs this service has already handed out (running jobs
    plus render reservations); they carry no foreign process but are not free.
    """
    own = set(own_pids)
    held = set(held)
    procs = compute_procs()
    probe_ok = procs is not None
    procs = procs or {}
    util_rows = _nvidia_smi(
        "gpu=index,memory.used,memory.total,utilization.gpu")
    util = {}
    if util_rows is not None:
        for r in util_rows:
            if len(r) >= 4:
                try:
                    util[int(r[0])] = {"mem_used": int(r[1]),
                                       "mem_total": int(r[2]),
                                       "util": int(r[3])}
                except ValueError:
                    continue
    rows = []
    for g in sorted(set(list(util) + list(procs) + GPUS)):
        ps = procs.get(g, [])
        foreign = [p for p in ps if p["pid"] not in own]
        # A card the image cannot run on (older than CUDA_ARCH's floor) is
        # listed but never scheduled: failing it up front beats a job dying in
        # gsplat or LichtFeld minutes in.
        unsupported = (resources.CUDA_ERROR
                       or resources.unsupported_reason(compute_caps().get(g)))
        schedulable = g in GPUS and not unsupported
        rows.append({
            "index": g,
            "schedulable": schedulable,
            "unsupported": unsupported,
            "procs": ps,
            "foreign": foreign,
            "busy_foreign": bool(foreign),
            "probe_ok": probe_ok,
            "held": g in held,
            # The single field the UI and the scheduler should both read.
            "available": (schedulable and probe_ok and not foreign
                          and g not in held),
            **util.get(g, {}),
        })
    return rows


def free_gpus(own_pids: Iterable[int] = (), held: Iterable[int] = ()) -> list[int]:
    """Schedulable GPUs with no foreign process and not held by our own jobs.

    Returns [] when the GPU probe failed: an unknown GPU is not a free GPU.
    """
    return [r["index"] for r in status(own_pids, held) if r["available"]]
