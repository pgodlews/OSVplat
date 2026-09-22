"""What this machine lets the queue use: CPUs, memory, GPUs it can run on.

Sizes the stages to the container, not to the host. On a rented GPU box the
two differ a lot: a RunPod pod showed 112 CPUs to nproc and os.cpu_count()
while its cgroup quota was 23.8 CPUs, and COLMAP's num_threads=-1 then ran 178
threads on that quota. SfM took 3267 s there against 1270 s on an 8-core Vast
host whose visible CPUs matched its allowance (notes, 2026-09-21).

Stages get the result as SPLAT_THREADS (and the usual OpenMP/BLAS variables);
scripts fall back to all cores when it is unset, so a script run by hand
behaves as before. QUEUE_CPUS overrides the discovered number.

Thread count is not part of any cache key: before this, -1 already meant a
different count on every machine.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Nothing older than compute capability 7.5 (RTX 20xx) can run gsplat or
# LichtFeld at the pinned versions, whatever the build.
MIN_COMPUTE_CAP = float(os.environ.get("QUEUE_MIN_COMPUTE_CAP", "7.5"))
# The architectures this build was compiled for, "7.5;8.6;12.0" (the Dockerfile
# records its CUDA_ARCH here). Unset on a native install: then only the floor
# above is checked.
BUILT_ARCH = os.environ.get("OSVPLAT_CUDA_ARCH", "").strip()


def _arch(s: str) -> Optional[tuple[int, int]]:
    s = s.strip().lower().replace("sm_", "").replace("+ptx", "")
    if "." not in s and s.isdigit() and len(s) >= 2:      # "86" -> 8.6, "120" -> 12.0
        s = f"{s[:-1]}.{s[-1]}"
    try:
        major, minor = s.split(".")
        return int(major), int(minor)
    except ValueError:
        return None


def built_archs(spec: str = None) -> list[tuple[int, int]]:
    spec = BUILT_ARCH if spec is None else spec
    return [a for a in (_arch(x) for x in spec.replace(",", ";").split(";") if x.strip()) if a]


_CGROUP = Path("/sys/fs/cgroup")


def _read(p: Path) -> Optional[str]:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def cgroup_cpu_quota(root: Path = _CGROUP) -> Optional[float]:
    """CPUs the cgroup may use, or None when unlimited or unknown.

    cgroup v2: cpu.max is "<quota> <period>" or "max <period>".
    cgroup v1: cpu/cpu.cfs_quota_us is -1 when unlimited.
    """
    raw = _read(root / "cpu.max")
    if raw:
        q, _, period = raw.partition(" ")
        if q == "max":
            return None
        try:
            return int(q) / int(period)
        except (ValueError, ZeroDivisionError):
            return None
    q = _read(root / "cpu" / "cpu.cfs_quota_us") or _read(root / "cpu.cfs_quota_us")
    p = _read(root / "cpu" / "cpu.cfs_period_us") or _read(root / "cpu.cfs_period_us")
    try:
        if q is not None and p is not None and int(q) > 0:
            return int(q) / int(p)
    except (ValueError, ZeroDivisionError):
        pass
    return None


def affinity_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):          # macOS has no sched_getaffinity
        return os.cpu_count() or 1


def effective_cpus(root: Path = _CGROUP) -> int:
    """Whole CPUs this process may keep busy: min(affinity, cgroup quota)."""
    override = os.environ.get("QUEUE_CPUS", "").strip()
    if override:
        return max(1, int(override))
    n = affinity_cpus()
    quota = cgroup_cpu_quota(root)
    if quota is not None:
        # Rounded down: 23.8 CPUs of quota keep 23 threads busy without
        # throttling; 24 get throttled every period.
        n = min(n, int(quota))
    return max(1, n)


def stage_threads(concurrent: int, root: Path = _CGROUP) -> int:
    """Threads for one stage when `concurrent` jobs may run side by side."""
    return max(1, effective_cpus(root) // max(1, concurrent))


def thread_env(threads: int) -> dict[str, str]:
    n = str(threads)
    return {"SPLAT_THREADS": n, "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n,
            "OPENBLAS_NUM_THREADS": n}


def memory_limit_bytes(root: Path = _CGROUP) -> Optional[int]:
    """The cgroup memory limit, else MemTotal; None if neither is readable."""
    raw = _read(root / "memory.max") or _read(root / "memory" / "memory.limit_in_bytes")
    if raw and raw != "max":
        try:
            v = int(raw)
            # v1 reports "unlimited" as a huge number near 2**63.
            if v < 1 << 60:
                return v
        except ValueError:
            pass
    for line in (_read(Path("/proc/meminfo")) or "").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def unsupported_reason(compute_cap: Optional[str], built: str = None) -> Optional[str]:
    """Why a GPU cannot run this build, or None if it can (or is unknown).

    CUDA code compiled for X.Y runs on X.Z for Z >= Y, never across a major
    version: an image built for 8.6 runs on an L4 (8.9) but not on an
    RTX 5090 (12.0) or an RTX 2080 (7.5).
    """
    cc = _arch(compute_cap) if compute_cap else None
    if cc is None:
        return None
    if cc[0] + cc[1] / 10 < MIN_COMPUTE_CAP:
        return f"compute capability {compute_cap} < {MIN_COMPUTE_CAP:g}"
    archs = built_archs(built)
    if archs and not any(a[0] == cc[0] and a[1] <= cc[1] for a in archs):
        have = ";".join(f"{a[0]}.{a[1]}" for a in archs)
        return (f"compute capability {compute_cap} is not in this build "
                f"(CUDA_ARCH {have}); rebuild with it")
    return None


# cuInit in a child process: a broken driver can hang or crash the caller.
_CUINIT = r"""
import ctypes, sys
try:
    cu = ctypes.CDLL("libcuda.so.1")
except OSError:
    sys.exit(0)                      # no driver in here: "no GPU visible" says so
r = cu.cuInit(0)
if r:
    name = ctypes.c_char_p()
    cu.cuGetErrorName(r, ctypes.byref(name))
    print((name.value or b"?").decode(), r)
"""
CUDA_ERROR: Optional[str] = None     # set once by probe_cuda() at startup


def probe_cuda(timeout: float = 60) -> Optional[str]:
    """Why CUDA cannot start on this machine, or None if it can.

    nvidia-smi can list a healthy-looking GPU on a host where CUDA itself does
    not initialise: a RunPod RTX 3090 on 2026-09-22 answered nvidia-smi but
    cuInit returned CUDA_ERROR_UNKNOWN, and every job then died in the frames
    stage after 0 s with an ffmpeg error. Checked once at startup so the
    service can say plainly that the host is broken, and refuse jobs.
    """
    global CUDA_ERROR
    try:
        r = subprocess.run([sys.executable, "-c", _CUINIT], capture_output=True,
                           text=True, timeout=timeout)
        out = r.stdout.strip()
        CUDA_ERROR = f"cuInit failed: {out}" if out else (
            f"cuInit probe exited {r.returncode}" if r.returncode else None)
    except subprocess.TimeoutExpired:
        CUDA_ERROR = f"cuInit did not return within {timeout:.0f} s"
    except OSError as exc:
        CUDA_ERROR = None
        print(f"cuda probe not run: {exc}")
    return CUDA_ERROR


def summary(gpus: Optional[list[dict]], concurrent: int) -> str:
    """One startup log line, e.g.
    'resources: 23 CPUs (cgroup quota 23.8 of 112 visible), 54 GB RAM,
     11 threads per stage; gpu0 RTX 3090 24576 MiB cc 8.6'."""
    quota = cgroup_cpu_quota()
    vis = affinity_cpus()
    eff = effective_cpus()
    why = ("QUEUE_CPUS" if os.environ.get("QUEUE_CPUS", "").strip()
           else f"cgroup quota {quota:.1f} of {vis} visible" if quota is not None and int(quota) < vis
           else "all visible")
    mem = memory_limit_bytes()
    parts = [f"{eff} CPUs ({why})"]
    if mem:
        parts.append(f"{mem / 1e9:.0f} GB RAM")
    parts.append(f"{stage_threads(concurrent)} threads per stage")
    line = "resources: " + ", ".join(parts)
    if gpus is None:
        return line + "; GPUs: nvidia-smi failed"
    if not gpus:
        return line + "; no GPU visible"
    g = "; ".join(f"gpu{x['index']} {x.get('name')} {x.get('memory_mib')} MiB "
                  f"cc {x.get('compute_cap')}"
                  + (f" UNSUPPORTED ({r})" if (r := unsupported_reason(x.get("compute_cap"))) else "")
                  for x in gpus)
    return f"{line}; {g}"
