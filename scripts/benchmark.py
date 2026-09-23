#!/usr/bin/env python3
"""Fixed-workload host benchmark: what this machine can do, in raw scores.

Run by the queue (POST /api/benchmark, or QUEUE_BENCHMARK=1 at startup) in
venv_gs, which has numpy, OpenCV, torch and gsplat. About 60-120 s. Every
input is generated here from a fixed seed, never a clip, so scores compare
across machines and nothing personal is read.

What each part is for (the per-clip timings in job telemetry cannot be
compared across clips, which is why this workload is fixed):
  cpu     JPEG decode/encode and SIFT extract + match, one thread and every
          allowed core. Frame extraction and SfM live on the CPU; a rented
          "16 vCPU" host that scales like 4 cores shows here.
  memory  single-thread copy bandwidth.
  disk    write + fsync, then read with the page cache dropped, under the
          --scratch directory (QUEUE_ROOT: where the cache lives).
  gpu     CUDA start-up time, fp32/fp16 matmul, device-to-device copy, pinned
          host<->device copies (a card on an x1 riser shows here), and a
          gsplat forward+backward loop, the nearest thing to training.

Raw scores only: judging them against a reference is the job of whatever
launched the machine. A part that fails records its error under "errors" and
the others still run; the queue marks the whole run failed if any part did.

    python benchmark.py --out result.json --scratch /data [--cpus 8] [--no-gpu]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

SCHEMA = "osvplat.benchmark/1"


def synth_image(w: int, h: int, seed: int = 0) -> np.ndarray:
    """A textured BGR image with corners and blobs: SIFT finds features in it
    and JPEG has real work, unlike flat noise or a gradient."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for cell, weight in ((4, 0.2), (16, 0.35), (64, 0.45)):
        n = rng.random((h // cell + 2, w // cell + 2, 3), dtype=np.float32)
        img += weight * cv2.resize(n, (w + 2 * cell, h + 2 * cell),
                                   interpolation=cv2.INTER_CUBIC)[:h, :w]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    for _ in range(max(50, w * h // 20000)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(3, max(4, w // 80)))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, color, -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + r), color, -1)
    return img


def rate(fn, seconds: float, min_iters: int = 3) -> float:
    """Calls per second of fn, run for about `seconds` (at least min_iters)."""
    fn()                                        # warm-up, not timed
    n, t0 = 0, time.perf_counter()
    while True:
        fn()
        n += 1
        dt = time.perf_counter() - t0
        if n >= min_iters and dt >= seconds:
            return n / dt


# ------------------------------------------------------------------- cpu

# Globals for pool workers, set by _worker_init: passing a 4K JPEG and SIFT
# descriptors with every task would measure pickling.
_W: dict = {}


def _worker_init(jpeg: bytes, gray: np.ndarray) -> None:
    cv2.setNumThreads(1)
    _W["jpeg"] = np.frombuffer(jpeg, np.uint8)
    _W["gray"] = gray
    _W["sift"] = cv2.SIFT_create(nfeatures=8000)


def _worker_run(args) -> int:
    kind, start_at, seconds = args
    # Start together, so the all-core figure is cores running at once.
    while time.time() < start_at:
        time.sleep(0.001)
    if kind == "decode":
        fn = lambda: cv2.imdecode(_W["jpeg"], cv2.IMREAD_COLOR)   # noqa: E731
    else:
        fn = lambda: _W["sift"].detectAndCompute(_W["gray"], None)  # noqa: E731
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        n += 1
    return n


def _all_cores(pool, procs: int, kind: str, seconds: float) -> float:
    start_at = time.time() + 0.5
    counts = pool.map(_worker_run, [(kind, start_at, seconds)] * procs)
    return sum(counts) / seconds


def bench_cpu(procs: int, seconds: float, frame: tuple[int, int],
              sift_size: tuple[int, int]) -> dict:
    cv2.setNumThreads(1)
    img = synth_image(*frame)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    jpeg = enc.tobytes()
    gray = cv2.cvtColor(cv2.resize(img, sift_size, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    other = cv2.cvtColor(cv2.resize(synth_image(*frame, seed=1), sift_size,
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=8000)
    _, d1 = sift.detectAndCompute(gray, None)
    _, d2 = sift.detectAndCompute(other, None)
    # About 7500 at full size (the 8000 cap is near), 370 at --quick. Far
    # fewer means the image or OpenCV is broken, and the score would be wrong.
    if d1 is None or d2 is None or len(d1) < 100:
        raise RuntimeError(f"only {0 if d1 is None else len(d1)} SIFT features "
                           f"in the test image")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    buf = np.frombuffer(jpeg, np.uint8)
    out = {
        "frame": f"{frame[0]}x{frame[1]}", "jpeg_bytes": len(jpeg),
        "sift_image": f"{sift_size[0]}x{sift_size[1]}", "sift_features": len(d1),
        "jpeg_decode_1t_per_s": round(rate(lambda: cv2.imdecode(buf, cv2.IMREAD_COLOR), seconds), 2),
        "jpeg_encode_1t_per_s": round(rate(lambda: cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95]), seconds), 2),
        "sift_1t_per_s": round(rate(lambda: sift.detectAndCompute(gray, None), seconds), 3),
        "sift_match_1t_per_s": round(rate(lambda: matcher.knnMatch(d1, d2, k=2), seconds), 3),
        "processes": procs,
    }
    # fork: the workers need nothing from a fresh interpreter, and spawn would
    # re-import cv2 and numpy in each one inside the timed window.
    with mp.get_context("fork").Pool(procs, _worker_init, (jpeg, gray)) as pool:
        out["jpeg_decode_all_per_s"] = round(_all_cores(pool, procs, "decode", seconds), 2)
        out["sift_all_per_s"] = round(_all_cores(pool, procs, "sift", seconds), 3)
    # How many cores' worth the machine delivered with every process busy.
    out["decode_scaling"] = round(out["jpeg_decode_all_per_s"] / out["jpeg_decode_1t_per_s"], 2)
    out["sift_scaling"] = round(out["sift_all_per_s"] / out["sift_1t_per_s"], 2)
    return out


def bench_memory(seconds: float, mib: int) -> dict:
    a = np.ones(mib * 2**20 // 8, np.float64)
    b = np.empty_like(a)
    per_s = rate(lambda: np.copyto(b, a), seconds)
    return {"buffer_mib": mib, "copy_gb_s": round(per_s * a.nbytes / 1e9, 2)}


def bench_disk(scratch: Path, total_bytes: int) -> dict:
    st = os.statvfs(scratch)
    free = st.f_bavail * st.f_frsize
    if free < total_bytes * 2 + 2**30:
        raise RuntimeError(f"only {free / 1e9:.1f} GB free under the scratch "
                           f"directory; the disk test needs {total_bytes * 2 / 1e9 + 1:.1f} GB")
    chunk = np.random.default_rng(0).bytes(min(64 * 2**20, total_bytes))
    path = scratch / f".benchmark_{os.getpid()}.tmp"
    try:
        t0 = time.perf_counter()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            written = 0
            while written < total_bytes:
                written += os.write(fd, chunk)
            os.fsync(fd)
        finally:
            os.close(fd)
        write_s = time.perf_counter() - t0
        fd = os.open(path, os.O_RDONLY)
        try:
            # Drop the file from the page cache, or this reads RAM.
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            t0 = time.perf_counter()
            read = 0
            while True:
                got = os.read(fd, len(chunk))
                if not got:
                    break
                read += len(got)
            read_s = time.perf_counter() - t0
        finally:
            os.close(fd)
    finally:
        path.unlink(missing_ok=True)
    return {"bytes": written, "write_fsync_mb_s": round(written / 1e6 / write_s, 1),
            "read_mb_s": round(read / 1e6 / read_s, 1),
            "cache_dropped": hasattr(os, "posix_fadvise")}


# ------------------------------------------------------------------- gpu

def bench_gpu(seconds: float, matmul_n: int, copy_mib: int, gaussians: int,
              size: tuple[int, int]) -> dict:
    t0 = time.perf_counter()
    import torch                                             # noqa: PLC0415
    import_s = time.perf_counter() - t0
    if not torch.cuda.is_available():
        raise RuntimeError("torch sees no CUDA device")
    t0 = time.perf_counter()
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    out = {"device": torch.cuda.get_device_name(0),
           "torch_import_s": round(import_s, 2),
           "cuda_init_s": round(time.perf_counter() - t0, 2)}

    def timed(fn) -> float:
        def step():
            fn()
            torch.cuda.synchronize()
        return rate(step, seconds)

    # Plain fp32, not TF32: the figure is meant to compare cards, and TF32
    # would make it a tensor-core number on some and not others.
    torch.backends.cuda.matmul.allow_tf32 = False
    for dtype, key in ((torch.float32, "matmul_fp32_tflops"),
                       (torch.float16, "matmul_fp16_tflops")):
        a = torch.randn(matmul_n, matmul_n, device="cuda", dtype=dtype)
        b = torch.randn(matmul_n, matmul_n, device="cuda", dtype=dtype)
        out[key] = round(timed(lambda: a @ b) * 2 * matmul_n**3 / 1e12, 2)
        del a, b

    n = copy_mib * 2**20
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    out["d2d_copy_gb_s"] = round(timed(lambda: dst.copy_(src)) * n / 1e9, 1)
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    out["h2d_pinned_gb_s"] = round(timed(lambda: src.copy_(host, non_blocking=True)) * n / 1e9, 2)
    out["d2h_pinned_gb_s"] = round(timed(lambda: host.copy_(src, non_blocking=True)) * n / 1e9, 2)
    del src, dst, host
    out["copy_mib"] = copy_mib

    from gsplat import rasterization                          # noqa: PLC0415
    g = torch.Generator(device="cuda").manual_seed(0)
    w, h = size
    means = (torch.rand(gaussians, 3, device="cuda", generator=g) - 0.5) * torch.tensor(
        [8.0, 4.0, 4.0], device="cuda") + torch.tensor([0.0, 0.0, 6.0], device="cuda")
    quats = torch.randn(gaussians, 4, device="cuda", generator=g)
    scales = torch.log(torch.rand(gaussians, 3, device="cuda", generator=g) * 0.03 + 0.005)
    opac = torch.rand(gaussians, device="cuda", generator=g) * 0.8 + 0.1
    colors = torch.rand(gaussians, 3, device="cuda", generator=g)
    params = [p.requires_grad_() for p in (means, quats, scales, opac, colors)]
    viewmat = torch.eye(4, device="cuda")[None]
    f = 0.8 * w
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], device="cuda")[None]
    target = torch.rand(1, h, w, 3, device="cuda", generator=g)

    def step():
        m, q, s, o, c = params
        img, _, _ = rasterization(m, torch.nn.functional.normalize(q, dim=-1),
                                  torch.exp(s), torch.sigmoid(o), c, viewmat, K, w, h)
        loss = (img - target).abs().mean()
        loss.backward()
        for p in params:
            p.grad = None
    out["gsplat_it_s"] = round(timed(step), 2)
    out["gsplat_gaussians"] = gaussians
    out["gsplat_image"] = f"{w}x{h}"
    out["peak_mem_mib"] = torch.cuda.max_memory_allocated() // 2**20
    return out


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scratch", required=True, type=Path,
                    help="directory for the disk test (its own filesystem is measured)")
    ap.add_argument("--cpus", type=int, default=0,
                    help="processes for the all-core test (default: CPUs this process may use)")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--seconds", type=float, default=4.0, help="per measurement")
    ap.add_argument("--disk-gb", type=float, default=2.0)
    ap.add_argument("--quick", action="store_true",
                    help="tiny sizes, for tests: the scores mean nothing")
    a = ap.parse_args()

    procs = a.cpus or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                       else os.cpu_count() or 1)
    q = a.quick
    parts = {
        "cpu": lambda: bench_cpu(procs, a.seconds, (640, 320) if q else (3840, 1920),
                                 (320, 160) if q else (1920, 960)),
        "memory": lambda: bench_memory(a.seconds, 16 if q else 512),
        "disk": lambda: bench_disk(a.scratch, int(a.disk_gb * 1e9)),
    }
    if not a.no_gpu:
        parts["gpu"] = lambda: bench_gpu(a.seconds, 1024 if q else 8192, 16 if q else 1024,
                                         10_000 if q else 1_000_000,
                                         (320, 180) if q else (1920, 1080))
    rec = {"schema": SCHEMA, "started": round(time.time(), 1), "seconds_per_test": a.seconds,
           "errors": {}, "durations_s": {}}
    for name, fn in parts.items():
        t0 = time.perf_counter()
        print(f"benchmark: {name} ...", flush=True)
        try:
            rec[name] = fn()
            print(f"benchmark: {name}: {json.dumps(rec[name])}", flush=True)
        except Exception as exc:                              # noqa: BLE001
            rec[name] = None
            rec["errors"][name] = f"{type(exc).__name__}: {exc}"[:500]
            print(f"benchmark: {name} FAILED\n{traceback.format_exc()}", flush=True)
        rec["durations_s"][name] = round(time.perf_counter() - t0, 1)
    rec["ended"] = round(time.time(), 1)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, a.out)
    return 1 if rec["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
