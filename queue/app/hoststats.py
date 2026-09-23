"""Machine-wide load counters: CPU time by mode, pressure stalls, cgroup
throttling, disk and network bytes. Standard library only, Linux only.

Why: a rented host can be slow for reasons no stage timing explains -- a
neighbour on the same box (steal, system-wide pressure), a CPU quota the job
keeps hitting (cgroup throttling), a slow disk or link. These are the numbers
that tell a slow host from a slow job.

read() takes one snapshot of raw counters; rates(prev, cur) turns two into
what happened in between. Both degrade to None per source where a file is
missing (macOS, a kernel without PSI, cgroup v1 without some files), and
never raise. Nothing here names a device or interface: disk and network are
summed across them.

Sources, and what they measure:
  /proc/stat                 whole machine (not namespaced): busy, iowait and
                             steal -- CPU time the hypervisor gave to others
  /proc/pressure/*           whole machine: share of time tasks were stalled
                             waiting for CPU, memory or IO (PSI)
  /sys/fs/cgroup/*.pressure  the same, for this container's cgroup only
  cgroup cpu.stat            periods in which the CPU quota throttled us
  /proc/diskstats            bytes read/written by whole disks
  /proc/net/dev              bytes in/out of this network namespace (in a
                             container: the container's own traffic)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

CGROUP = Path("/sys/fs/cgroup")
PROC = Path("/proc")
SECTOR = 512                     # /proc/diskstats counts 512-byte sectors
# Disks that are views of other disks, or not disks: summed with the real
# ones they would count the same bytes twice.
_VIRTUAL_DISKS = ("loop", "ram", "zram", "dm-", "md", "sr", "fd", "nbd")


def _read(p: Path) -> Optional[str]:
    try:
        return p.read_text()
    except OSError:
        return None


def _stat(proc: Path) -> Optional[dict]:
    """Aggregate and per-core CPU jiffies from /proc/stat."""
    raw = _read(proc / "stat")
    if not raw:
        return None
    total, cores = None, []
    for line in raw.splitlines():
        if not line.startswith("cpu"):
            continue
        name, *vals = line.split()
        v = [int(x) for x in vals[:8]] + [0] * (8 - len(vals[:8]))
        user, nice, system, idle, iowait, irq, softirq, steal = v
        # Four parts that add up to "all": busy is time spent working for
        # someone on this machine, steal is time the hypervisor gave away.
        row = {"all": sum(v), "idle": idle, "iowait": iowait, "steal": steal,
               "busy": sum(v) - idle - iowait - steal}
        if name == "cpu":
            total = row
        else:
            cores.append(row)
    return {"total": total, "cores": cores} if total else None


def _psi(path: Path) -> Optional[dict]:
    """PSI totals in microseconds: {"some": us, "full": us}."""
    raw = _read(path)
    if not raw:
        return None
    out = {}
    for line in raw.splitlines():
        kind, *fields = line.split()
        for f in fields:
            if f.startswith("total="):
                out[kind] = int(f[6:])
    return out or None


def _cgroup_cpu(root: Path) -> Optional[dict]:
    """Throttling counters: v2 cpu.stat, else v1 (throttled_time is in ns)."""
    for p, scale in ((root / "cpu.stat", 1), (root / "cpu" / "cpu.stat", 1000),
                     (root / "cpu,cpuacct" / "cpu.stat", 1000)):
        raw = _read(p)
        if not raw:
            continue
        kv = {}
        for line in raw.splitlines():
            k, _, v = line.partition(" ")
            try:
                kv[k] = int(v)
            except ValueError:
                pass
        if "nr_periods" not in kv:
            continue
        us = kv.get("throttled_usec")
        if us is None and "throttled_time" in kv:
            us = kv["throttled_time"] // scale
        return {"periods": kv["nr_periods"], "throttled": kv.get("nr_throttled", 0),
                "throttled_us": us or 0}
    return None


def _disk(proc: Path, sys_block: Path) -> Optional[dict]:
    raw = _read(proc / "diskstats")
    if not raw:
        return None
    try:
        whole = {d for d in os.listdir(sys_block) if not d.startswith(_VIRTUAL_DISKS)}
    except OSError:
        whole = None
    rd = wr = 0
    for line in raw.splitlines():
        f = line.split()
        if len(f) < 10:
            continue
        name = f[2]
        # Partitions are not in /sys/block; without it, skip the usual ones.
        if whole is not None and name not in whole:
            continue
        if whole is None and (name.startswith(_VIRTUAL_DISKS) or name[-1].isdigit()):
            continue
        rd += int(f[5]) * SECTOR
        wr += int(f[9]) * SECTOR
    return {"read": rd, "write": wr}


def _net(proc: Path) -> Optional[dict]:
    raw = _read(proc / "net" / "dev")
    if not raw:
        return None
    rx = tx = 0
    for line in raw.splitlines()[2:]:
        name, _, rest = line.partition(":")
        f = rest.split()
        if name.strip() == "lo" or len(f) < 9:
            continue
        rx += int(f[0])
        tx += int(f[8])
    return {"rx": rx, "tx": tx}


def read(proc: Path = PROC, cgroup: Path = CGROUP,
         sys_block: Path = Path("/sys/block")) -> dict:
    """One snapshot of raw counters. Missing sources are None. Never raises."""
    snap: dict = {"t": time.monotonic()}
    parts = {
        "stat": lambda: _stat(proc),
        "psi": lambda: {r: _psi(proc / "pressure" / r) for r in ("cpu", "memory", "io")},
        "cgroup_psi": lambda: {r: _psi(cgroup / f"{r}.pressure") for r in ("cpu", "memory", "io")},
        "cgroup_cpu": lambda: _cgroup_cpu(cgroup),
        "disk": lambda: _disk(proc, sys_block),
        "net": lambda: _net(proc),
    }
    for k, fn in parts.items():
        try:
            snap[k] = fn()
        except Exception:                                     # noqa: BLE001
            snap[k] = None
    return snap


def _frac(num, den) -> Optional[float]:
    return round(num / den, 4) if den and den > 0 and num >= 0 else None


def rates(prev: Optional[dict], cur: Optional[dict], per_core: bool = False) -> Optional[dict]:
    """What happened between two snapshots, as shares of the interval and
    bytes per second. None when there is no interval to speak of."""
    if not prev or not cur:
        return None
    dt = cur["t"] - prev["t"]
    if dt <= 0:
        return None
    out: dict = {"seconds": round(dt, 2)}
    try:
        a, b = prev.get("stat"), cur.get("stat")
        if a and b:
            d = {k: b["total"][k] - a["total"][k] for k in b["total"]}
            out["cpu_busy"] = _frac(d["busy"], d["all"])
            out["cpu_iowait"] = _frac(d["iowait"], d["all"])
            out["cpu_steal"] = _frac(d["steal"], d["all"])
            if per_core and len(a["cores"]) == len(b["cores"]):
                cores = []
                for x, y in zip(a["cores"], b["cores"]):
                    tot = y["all"] - x["all"]
                    cores.append(round(100 * (y["busy"] - x["busy"]) / tot) if tot > 0 else None)
                out["core_busy_pct"] = cores
        for key, name in (("psi", "psi"), ("cgroup_psi", "cgroup_psi")):
            a, b = prev.get(key) or {}, cur.get(key) or {}
            got = {}
            for r in ("cpu", "memory", "io"):
                if a.get(r) and b.get(r):
                    for kind in ("some", "full"):
                        if kind in a[r] and kind in b[r]:
                            got[f"{r}_{kind}"] = _frac((b[r][kind] - a[r][kind]) / 1e6, dt)
            # cpu "full" is always 0 outside a cgroup; keep the file's
            # answer anyway, the reader knows which it is.
            out[name] = got or None
        a, b = prev.get("cgroup_cpu"), cur.get("cgroup_cpu")
        if a and b:
            periods = b["periods"] - a["periods"]
            out["cgroup_throttled"] = _frac(b["throttled"] - a["throttled"], periods) \
                if periods > 0 else 0.0
            out["cgroup_throttled_s"] = round((b["throttled_us"] - a["throttled_us"]) / 1e6, 2)
        a, b = prev.get("disk"), cur.get("disk")
        if a and b:
            out["disk_read_mb_s"] = round((b["read"] - a["read"]) / 1e6 / dt, 2)
            out["disk_write_mb_s"] = round((b["write"] - a["write"]) / 1e6 / dt, 2)
        a, b = prev.get("net"), cur.get("net")
        if a and b:
            out["net_rx_mb_s"] = round((b["rx"] - a["rx"]) / 1e6 / dt, 3)
            out["net_tx_mb_s"] = round((b["tx"] - a["tx"]) / 1e6 / dt, 3)
    except Exception:                                         # noqa: BLE001
        pass                         # a counter that went backwards: keep what we have
    return out
