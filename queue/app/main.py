"""FastAPI app: queue, job config, sweeps, logs, compare."""
from __future__ import annotations

import asyncio
import hmac
import itertools
import json
import re
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from . import (db, estimate, gpu, metrics, outputs, progress, resources,
               retention, telemetry, worker)
from .config import (CACHE_ROOT, GPUS, GS_PY, METRICS_ENABLED, MODELS_ROOT, QUEUE_TOKEN,
                     RENDER_COMPARE, RENDER_ROOT, SPLAT_ROOT, TOKEN_COOKIE)
from .jobs import FISHEYE_EXTS, IMU_SELECT_DEFAULT, JobConfig, quick_hash
from . import mask_backends
from .stages import ARTIFACT_EXTS, ORDER, images_dir, is_cached

import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from scripts.osv_meta import detect_camera
except ImportError:
    try:
        from osv_meta import detect_camera
    except ImportError:
        detect_camera = None

app = FastAPI(title="splat queue")
STATIC = Path(__file__).parent / "static"

# Keyed by path and mtime, so it never serves a stale probe -- but every new
# clip (and every re-encode of one) adds an entry that is never read again.
_probe_cache: dict[str, dict] = {}
PROBE_CACHE_MAX = 64


# ---------------------------------------------------------------- auth
# Everything this service exposes is a control: submit GPU work, cancel it,
# delete experiment history, change scheduling. It listens on a LAN port, so it
# gets a shared secret. With no token configured it serves loopback only, which
# makes `ssh -L` the fallback rather than an open port the default.

LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


def _presented_token(request: Request) -> Optional[str]:
    # Authorization: Bearer is how metrics scrapers and most HTTP clients
    # expect to send a credential, and it is the same token compared the same
    # way -- accepting it costs nothing and saves every caller a special case.
    auth = request.headers.get("authorization") or ""
    scheme, _, value = auth.partition(" ")
    bearer = value.strip() if scheme.lower() == "bearer" else None
    return (request.headers.get("x-queue-token")
            or bearer
            or request.query_params.get("token")
            or request.cookies.get(TOKEN_COOKIE))


def _token_ok(got: str) -> bool:
    """compare_digest raises on a non-ASCII str, and a stray header or a mangled
    cookie can carry one. An unhandled raise in middleware is a 500 where the
    honest answer is 401."""
    try:
        return hmac.compare_digest(got, QUEUE_TOKEN)
    except (TypeError, ValueError):
        return False


@app.middleware("http")
async def _require_token(request: Request, call_next):
    if not QUEUE_TOKEN:
        host = request.client.host if request.client else ""
        if host not in LOOPBACK:
            return JSONResponse(
                {"detail": "this queue has no QUEUE_TOKEN set, so it answers "
                           "loopback only. Redeploy with queue/deploy.sh (it "
                           "generates one), or tunnel: "
                           "ssh -L 8090:127.0.0.1:8090 <box>"},
                status_code=401)
        return await call_next(request)

    got = _presented_token(request)
    if not got or not _token_ok(got):
        return JSONResponse(
            {"detail": "missing or wrong queue token; open the URL deploy.sh "
                       "printed, or send it as the X-Queue-Token header"},
            status_code=401)
    resp = await call_next(request)
    # Arriving with ?token=... is how a browser gets in the first time; hand it
    # a cookie so every later navigation and fetch works without the parameter.
    # Refresh it whenever it disagrees: a cookie left over from an older token
    # (or another queue on the same host and a different port) otherwise sticks,
    # and ?token=... with the CORRECT token could not replace it -- the page
    # loaded, then every fetch it made came back 401.
    if (request.query_params.get("token")
            and request.cookies.get(TOKEN_COOKIE) != QUEUE_TOKEN):
        resp.set_cookie(TOKEN_COOKIE, QUEUE_TOKEN, httponly=True,
                        samesite="lax", max_age=90 * 24 * 3600)
    return resp


@app.on_event("startup")
def _startup() -> None:
    db.init()
    print("auth: token required" if QUEUE_TOKEN else
          "auth: NO TOKEN SET -- serving loopback clients only")
    # What the stages will be sized to (resources.py); telemetry has already
    # probed the GPUs, so this costs no extra nvidia-smi call later.
    print(resources.summary(telemetry.host()["gpus"], worker.max_concurrent()))
    if GPUS and resources.probe_cuda():
        print(f"ERROR: CUDA does not start on this machine ({resources.CUDA_ERROR}) "
              f"although nvidia-smi lists the GPU. The host is faulty: no job can "
              f"run here, new jobs are refused. On a rented GPU, destroy it and "
              f"take another.")
    outputs.startup_check()
    if worker.enforce_start_paused():
        print("queue forced back to PAUSED on startup "
              "(QUEUE_START_PAUSED=0 to keep the stored state)")
    n = worker.reconcile()
    if n:
        print(f"reconciled {n} job(s) left running by a previous process")
    logs, rends = retention.prune_logs(), retention.prune_renders()
    if logs["removed"] or rends["removed"]:
        print(f"pruned {logs['removed']} log(s) and {rends['removed']} render(s)")
    print(f"disk: {retention.free_bytes()/retention.GB:.1f} GB free, "
          f"cache holds {retention.cache_total()/retention.GB:.1f} GB")
    worker.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    worker.stop()


# ------------------------------------------------------------------ inputs

def ffprobe(path: Path) -> dict:
    key = f"{path}:{path.stat().st_mtime_ns}"
    if key in _probe_cache:
        return _probe_cache[key]
    info: dict[str, Any] = {"duration": None, "width": None, "height": None,
                            "fps": None, "codec": None}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=width,height,codec_name,avg_frame_rate:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            j = json.loads(out.stdout)
            st = (j.get("streams") or [{}])[0]
            info["width"], info["height"] = st.get("width"), st.get("height")
            info["codec"] = st.get("codec_name")
            fr = st.get("avg_frame_rate", "0/1")
            if "/" in fr:
                n, d = fr.split("/")
                info["fps"] = round(float(n) / float(d), 3) if float(d) else None
            dur = (j.get("format") or {}).get("duration")
            info["duration"] = round(float(dur), 2) if dur else None
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError):
        pass
    if len(_probe_cache) >= PROBE_CACHE_MAX:
        _probe_cache.clear()
    _probe_cache[key] = info
    return info


STITCHED_EXTS = (".mp4", ".mov", ".mkv")
# DJI .OSV (Osmo 360, Avata 360) is raw dual fisheye too, but it carries the
# camera's lens calibration, so it runs the fisheye rig pipeline instead of
# being refused (jobs.FISHEYE_EXTS, docs/how-it-works.md, "Fisheye rig").
FISHEYE_NOTE = ("raw DJI dual fisheye — reconstructed as a calibrated two-lens "
                "rig, no stitch")
# The rest still cannot be used: Insta360's containers have no decoded
# calibration, and a low-res proxy is not something to reconstruct from.
RAW_EXTS = (".insv", ".insp", ".lrv")
RAW_NOTE = ("raw camera footage — stitch it to an equirectangular MP4 first "
            "(Insta360 Studio); only DJI .OSV is reconstructed directly")


@app.get("/api/inputs")
def api_inputs() -> list[dict]:
    samples = SPLAT_ROOT / "samples"
    rows = []
    if samples.is_dir():
        for p in sorted(samples.iterdir()):
            ext = p.suffix.lower()
            if ext not in STITCHED_EXTS + RAW_EXTS + FISHEYE_EXTS:
                continue
            st = p.stat()
            raw = ext in RAW_EXTS
            fisheye = ext in FISHEYE_EXTS
            cam_info = detect_camera(p) if (fisheye and detect_camera) else {}
            camera = cam_info.get("camera")
            cam_name = cam_info.get("camera_name")
            rec_mask = cam_info.get("recommended_mask", False)
            rows.append({
                "file": str(p.relative_to(SPLAT_ROOT)),
                "name": p.name,
                "bytes": st.st_size,
                "mtime": st.st_mtime,
                "usable": not raw,
                "pipeline": None if raw else ("fisheye_rig" if fisheye else "equirect"),
                "camera": camera,
                "camera_name": cam_name,
                "recommended_mask": rec_mask,
                # What _prepare fills in for select.imu when a request leaves it out.
                "recommended_imu": IMU_SELECT_DEFAULT if fisheye else False,
                "supports_distance_selection": fisheye and camera == "avata360",
                "note": RAW_NOTE if raw else (
                    f"{cam_name or 'raw DJI dual fisheye'} — reconstructed as a calibrated two-lens rig, no stitch"
                    if fisheye else None
                ),
                **({} if raw else ffprobe(p)),
            })
    return rows


# -------------------------------------------------------------------- jobs

def _validate(cfg_in: dict) -> JobConfig:
    """Turn pydantic validation failures into a readable 400, not a 500."""
    try:
        return JobConfig.model_validate(cfg_in)
    except ValidationError as exc:
        msgs = []
        for e in exc.errors():
            loc = ".".join(str(x) for x in e["loc"]) or "config"
            msgs.append(f"{loc}: {e['msg']}")
        raise HTTPException(400, "; ".join(msgs)) from exc


def input_path(rel: str) -> Path:
    """Resolve a config's input to a real file inside SPLAT_ROOT.

    `SPLAT_ROOT / rel` happily accepts an absolute path (Path discards the left
    operand) and ../.. traversal, so the input field was really "any file on the
    box". Everything here is checked after resolution, so a symlink pointing out
    of the root is caught too.
    """
    if not rel or os.path.isabs(rel) or rel.startswith("~"):
        raise HTTPException(400, "input must be a path relative to SPLAT_ROOT")
    try:
        src = (SPLAT_ROOT / rel).resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(400, f"input not found: {rel}") from None
    try:
        src.relative_to(SPLAT_ROOT.resolve())
    except ValueError:
        raise HTTPException(
            400, f"input resolves outside SPLAT_ROOT: {rel}") from None
    if not src.is_file():
        raise HTTPException(400, f"input is not a file: {rel}")
    if src.suffix.lower() in RAW_EXTS:
        raise HTTPException(400, f"{src.name}: {RAW_NOTE}")
    return src


def _prepare(cfg_in: dict) -> JobConfig:
    cfg = _validate(cfg_in)
    src = input_path(cfg.input.file)
    # Always recompute. A client-supplied hash is a claim about a file the
    # client may not even have; trusting it let a config copied from another job
    # inherit that job's entire cached pipeline for a different clip.
    cfg.input.quick_hash = quick_hash(src)

    # Auto-detect camera type if mask.enabled was not explicitly specified:
    # - Osmo 360 (handheld): operator always in shot -> mask.enabled defaults True
    # - Avata 360 (drone): no operator -> mask.enabled defaults False (can still be enabled if user specified it)
    mask_in = cfg_in.get("mask")
    explicit_mask = isinstance(mask_in, dict) and "enabled" in mask_in
    if not explicit_mask and detect_camera and cfg.is_fisheye:
        cam_info = detect_camera(src)
        cfg.mask.enabled = bool(cam_info.get("recommended_mask", False))

    # Refuse a mask backend this install cannot run now, with how to fix it,
    # instead of accepting the job and failing at the mask stage an hour in.
    # After the camera default: an Osmo clip turns masking on by itself, and a
    # request naming SAM 3 without "enabled" used to slip past this check.
    if cfg.mask.enabled:
        st = next(b for b in mask_backends.availability(MODELS_ROOT)
                  if b["name"] == cfg.mask.backend)
        if not st["available"]:
            raise HTTPException(400, f"mask backend {cfg.mask.backend!r} is not "
                                     f"available: {st['reason']}")

    select_in = cfg_in.get("select")
    explicit_window = isinstance(select_in, dict) and "window" in select_in
    if not explicit_window and cfg.is_fisheye:
        cfg.select.window = 3
    explicit_imu = isinstance(select_in, dict) and "imu" in select_in
    if not explicit_imu and cfg.is_fisheye:
        cfg.select.imu = IMU_SELECT_DEFAULT

    if cfg.select.mode == "distance":
        from avata_motion import load
        try:
            motion = load(src)
            if cfg.frames.fps > motion["fps"]:
                raise ValueError("Distance selection candidate fps must not exceed the source fps")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return cfg


def _cache_state(cfg: JobConfig) -> dict[str, bool]:
    return {s: is_cached(CACHE_ROOT / s / k) for s, k in cfg.keys().items()}


def _store_plan(job_id: int, cfg: JobConfig) -> None:
    """Freeze the estimate onto the job at submit time.

    Frozen rather than recomputed per request for three reasons: it needs an
    ffprobe, the queue view asks for it every four seconds, and an estimate
    that drifts under a running job cannot be compared against what that job
    actually did. A clip that will not probe is not a reason to refuse the job
    -- the UI just has no ETA for it.
    """
    try:
        dur = ffprobe(input_path(cfg.input.file))["duration"]
        if dur:
            db.set_plan(job_id, estimate.estimate(
                cfg, dur, cached=_cache_state(cfg)))
    except (HTTPException, OSError, ValueError, KeyError):
        pass


class CreateReq(BaseModel):
    config: dict
    priority: int = 0


def _refuse_if_cannot_deliver() -> None:
    """Fail at submission, not after the GPU time: no GPU here can run this
    build (resources.unsupported_reason), or the result could not be
    delivered (outputs.problems)."""
    if GPUS and resources.CUDA_ERROR:
        raise HTTPException(400, f"CUDA does not start on this machine "
                            f"({resources.CUDA_ERROR}): the host is faulty; "
                            f"use another one")
    caps = gpu.compute_caps()
    mine = [g for g in GPUS if g in caps]
    if mine and all(resources.unsupported_reason(caps[g]) for g in mine):
        raise HTTPException(400, "no GPU here can run this build: " + "; ".join(
            f"gpu{g}: {resources.unsupported_reason(caps[g])}" for g in mine))
    bad = outputs.problems()
    if bad:
        raise HTTPException(400, "; ".join(bad) + " -- no job can deliver its "
                            "result; fix OUTPUT_UPLOAD_URL and restart")
    left = outputs.seconds_left()
    if left is not None and left < 3600:
        print(f"WARNING: OUTPUT_UPLOAD_URL expires in {left / 60:.0f} min; "
              f"a job accepted now may finish after it")


@app.post("/api/jobs")
def api_create(req: CreateReq) -> dict:
    _refuse_if_cannot_deliver()
    cfg = _prepare(req.config)
    jid = db.create_job(cfg.name, cfg.model_dump(), priority=req.priority)
    for stage, key in cfg.keys().items():
        d = CACHE_ROOT / stage / key
        db.upsert_stage(jid, stage, key,
                        "cached" if is_cached(d) else "pending", path=str(d))
    _store_plan(jid, cfg)
    return {"id": jid, "keys": cfg.keys(), "cached": _cache_state(cfg),
            "upload_expires_in_s": outputs.seconds_left()}


def _set_dotted(d: dict, path: str, value) -> None:
    cur = d
    parts = path.split(".")
    for i, p in enumerate(parts[:-1]):
        cur = cur.setdefault(p, {})
        # A base like {"train": null} would otherwise surface as a 500.
        if not isinstance(cur, dict):
            raise HTTPException(
                400, f"sweep axis {path!r}: {'.'.join(parts[:i + 1])!r} in the "
                     f"base config is {type(cur).__name__}, not an object")
    cur[parts[-1]] = value


class Variant(BaseModel):
    label: str = ""
    set: dict[str, Any] = {}        # {"train.sh_degree": 3}
    priority: int = 0


# One paste should not be able to queue a fortnight of GPU time.
MAX_SWEEP_JOBS = int(os.environ.get("QUEUE_MAX_SWEEP", 64))


class SweepReq(BaseModel):
    base: dict
    axes: dict[str, list] = {}      # cross product: {"train.sh_degree": [1,3]}
    variants: list[Variant] = []    # one-factor-at-a-time from the baseline
    priority: int = 0


@app.post("/api/jobs/sweep")
def api_sweep(req: SweepReq) -> dict:
    """Queue a family of related jobs sharing one upstream cache.

    Two shapes, because experiments come in two shapes:

    * `axes` -- full cross product. Right when factors may interact.
    * `variants` -- one entry per job, each overriding the baseline. Right for a
      first sweep, where you want N one-factor changes against a control and a
      cross product would queue combinations nobody asked for.

    A variant with an empty `set` is the control.
    """
    _refuse_if_cannot_deliver()
    if not req.axes and not req.variants:
        raise HTTPException(400, "give either axes or variants")
    if req.axes and req.variants:
        raise HTTPException(400, "give axes or variants, not both")

    specs: list[tuple[str, dict, int]] = []

    if req.variants:
        if len(req.variants) > MAX_SWEEP_JOBS:
            raise HTTPException(
                400, f"{len(req.variants)} variants exceeds the "
                     f"{MAX_SWEEP_JOBS}-job sweep limit")
        for v in req.variants:
            label = v.label or (", ".join(
                f"{k.split('.')[-1]}={val}" for k, val in v.set.items())
                or "baseline")
            specs.append((label, v.set, v.priority or req.priority))
    else:
        names = list(req.axes)
        empty = [n for n in names if not req.axes[n]]
        if empty:
            raise HTTPException(
                400, f"axis with no values: {', '.join(sorted(empty))}")
        # Count the product BEFORE building it. Four axes of four values is 256
        # jobs and, at 95 minutes each, a fortnight of GPU time queued by one
        # careless paste.
        total = math.prod(len(req.axes[n]) for n in names)
        if total > MAX_SWEEP_JOBS:
            raise HTTPException(
                400, f"{' x '.join(str(len(req.axes[n])) for n in names)} = "
                     f"{total} jobs exceeds the {MAX_SWEEP_JOBS}-job sweep "
                     f"limit; narrow the axes or raise QUEUE_MAX_SWEEP")
        for combo in itertools.product(*(req.axes[n] for n in names)):
            overrides = dict(zip(names, combo))
            label = ", ".join(f"{n.split('.')[-1]}={v}"
                              for n, v in overrides.items())
            specs.append((label, overrides, req.priority))

    # Validate EVERY variant before creating any job. Validating and inserting
    # in one pass meant a typo in the last variant returned a 400 with the
    # earlier variants already queued -- an error the caller had no reason to
    # think left anything behind.
    prepared = []
    for label, overrides, priority in specs:
        cfg_d = json.loads(json.dumps(req.base))
        for dotted, value in overrides.items():
            _set_dotted(cfg_d, dotted, value)
        cfg_d["name"] = f"{req.base.get('name', 'sweep')} [{label}]"
        try:
            cfg = _prepare(cfg_d)
        except HTTPException as exc:
            raise HTTPException(
                exc.status_code,
                f"variant {label!r}: {exc.detail} (no jobs were queued)"
            ) from exc
        prepared.append((label, cfg, priority))

    sweep_id = uuid.uuid4().hex[:8]
    created = []
    for label, cfg, priority in prepared:
        jid = db.create_job(cfg.name, cfg.model_dump(), sweep_id=sweep_id,
                            priority=priority)
        for stage, key in cfg.keys().items():
            d = CACHE_ROOT / stage / key
            db.upsert_stage(jid, stage, key,
                            "cached" if is_cached(d) else "pending",
                            path=str(d))
        _store_plan(jid, cfg)
        created.append({"id": jid, "name": cfg.name, "label": label,
                        "keys": cfg.keys(), "cached": _cache_state(cfg)})
    return {"sweep_id": sweep_id, "jobs": created}


def _job_dict(row, now: Optional[float] = None) -> dict:
    cfg = json.loads(row["config"])
    now = now or time.time()
    try:
        plan = json.loads(row["plan"]) if row["plan"] else {}
    except (ValueError, TypeError):
        plan = {}
    stages = []
    for st in db.job_stages(row["id"]):
        prog = json.loads(st["progress"]) if st["progress"] else None
        # `seconds` stays what it always was -- the duration of a FINISHED
        # stage. A stage in flight reports elapsed separately, so a caller that
        # only understands the old field cannot mistake a partial run for a
        # completed one.
        stages.append({
            "stage": st["stage"], "state": st["state"],
            "cache_key": st["cache_key"], "progress": prog,
            "started": st["started"], "ended": st["ended"],
            "seconds": (round(st["ended"] - st["started"], 1)
                        if st["started"] and st["ended"] else None),
            "elapsed": (round(now - st["started"], 1)
                        if st["state"] == "running" and st["started"] else None),
            "est": plan.get(st["stage"]),
            "fraction": (progress.stage_fraction(st["stage"], prog, plan)
                         if st["state"] == "running" else None),
            "detail": (progress.stage_detail(st["stage"], prog)
                       if st["state"] == "running" else ""),
        })
    return {
        "id": row["id"], "name": row["name"], "state": row["state"],
        "gpu": row["gpu"], "error": row["error"], "sweep_id": row["sweep_id"],
        "created": row["created"], "started": row["started"],
        "ended": row["ended"], "config": cfg, "stages": stages,
        "metrics": db.job_metrics(row["id"]),
        "review_state": row["review_state"],
        "review_note": row["review_note"],
        "plan": plan or None,
        "eta": progress.job_progress(row, stages, now=now),
    }


@app.get("/api/jobs")
def api_jobs(limit: int = Query(200, ge=1, le=1000),
             offset: int = Query(0, ge=0)) -> list[dict]:
    # One clock reading for the whole page, so the rows agree with each other
    # and with the queue-wide drain time computed from them.
    now = time.time()
    active = [_job_dict(r, now=now) for r in db.active_jobs()]
    progress.queue_eta(active, worker.schedulable_capacity(), now=now,
                       paused=worker.paused())
    by_id = {j["id"]: j for j in active}
    return [by_id[r["id"]] if r["id"] in by_id else _job_dict(r, now=now)
            for r in db.list_jobs(limit=limit, offset=offset)]


@app.get("/api/jobs/{job_id}")
def api_job(job_id: int) -> dict:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    # upload: OUTPUT_UPLOAD_URL's result for this job (outputs.py), or None.
    return {**_job_dict(row), "upload": outputs.status(job_id)}


@app.delete("/api/jobs/{job_id}")
def api_cancel(job_id: int) -> dict:
    if not worker.cancel(job_id):
        raise HTTPException(400, "job is not queued or running")
    return {"ok": True}


def _export_files(job_id: int) -> dict[str, Path]:
    """The finished splat files of a job, by name. Only names that are really in
    the job's export stage directory are servable, so the download route cannot
    be pointed anywhere else on disk."""
    st = db.conn().execute(
        "SELECT state, path FROM stages WHERE job_id=? AND stage='export'",
        (job_id,)).fetchone()
    if not st or not st["path"] or st["state"] not in ("done", "cached"):
        return {}
    d = Path(st["path"])
    if not d.is_dir():
        return {}
    return {p.name: p for p in sorted(d.iterdir())
            if not p.name.startswith(".")
            and p.suffix.lstrip(".") in ARTIFACT_EXTS and p.exists()}


@app.get("/api/jobs/{job_id}/files")
def api_files(job_id: int) -> list[dict]:
    if db.get_job(job_id) is None:
        raise HTTPException(404, "no such job")
    return [{"name": n, "bytes": p.stat().st_size,
             "url": f"/api/jobs/{job_id}/files/{n}"}
            for n, p in _export_files(job_id).items()]


@app.get("/api/jobs/{job_id}/files/{name}")
def api_file(job_id: int, name: str):
    p = _export_files(job_id).get(name)
    if p is None:
        raise HTTPException(404, "no such file for this job")
    job = db.get_job(job_id)
    # Prefix the job name: every run exports splat_30000.*, so a folder of
    # downloads would otherwise be a pile of identically named files.
    # Starlette already percent-encodes whatever it is given, so this is for
    # tidy names on disk: sweep jobs are called "x [strategy=mcmc, sh=3]".
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", job["name"]).strip("._") or f"job{job_id}"
    return FileResponse(p.resolve(), media_type="application/octet-stream",
                        filename=f"{stem}_{name}")


@app.get("/api/jobs/{job_id}/telemetry")
def api_telemetry(job_id: int, logs: bool = Query(False)):
    """This job's telemetry.json, or with ?logs=1 its redacted log bundle.

    Written after every stage (docs/job-telemetry.md), so a running job returns
    what it has so far. 404 when telemetry is off or nothing is written yet.
    """
    p = telemetry.logs_path(job_id) if logs else telemetry.telemetry_path(job_id)
    if not p.is_file():
        raise HTTPException(404, "no telemetry for this job"
                            + ("" if telemetry.TELEMETRY_ENABLED else
                               " (QUEUE_TELEMETRY is off)"))
    if logs:
        return FileResponse(p, media_type="application/gzip",
                            filename=f"job{job_id:05d}_logs.tar.gz")
    return FileResponse(p, media_type="application/json")


@app.get("/api/jobs/{job_id}/log")
async def api_log(job_id: int, stage: str = Query("train"),
                  follow: bool = Query(False),
                  from_byte: int = Query(0, alias="from", ge=0)):
    st = db.conn().execute(
        "SELECT log_path FROM stages WHERE job_id=? AND stage=?",
        (job_id, stage)).fetchone()
    if not st or not st["log_path"]:
        raise HTTPException(404, "no log for that stage yet")
    path = Path(st["log_path"])
    if not path.exists():
        raise HTTPException(404, "log file missing")
    if not follow:
        return FileResponse(path, media_type="text/plain")

    async def gen():
        with path.open("r", errors="replace") as f:
            # Start exactly where the caller's plain GET stopped, which it
            # passes as ?from=<content-length>. This used to rewind 8 KB from
            # EOF unconditionally while the page fetched the whole log first,
            # so every follow replayed the last ~80 lines into a pane that
            # already had them. With no ?from it behaves like tail -f and
            # starts at the end rather than duplicating anything.
            size = path.stat().st_size
            f.seek(min(from_byte, size) if from_byte else size)
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    row = db.get_job(job_id)
                    if row and row["state"] not in ("running", "queued"):
                        yield "event: end\ndata: done\n\n"
                        return
                    await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------- estimate

class EstimateReq(BaseModel):
    config: dict


class ReviewReq(BaseModel):
    approved: bool
    note: str = ""


def _mask_dir_for(job_id: int) -> Path:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    try:
        cfg = JobConfig.model_validate(json.loads(row["config"]))
    except ValidationError as exc:
        # A job stored before a schema change is still readable in the job list;
        # only this endpoint needs a live config, so fail it clearly rather than
        # 500-ing.
        raise HTTPException(
            400, f"job config predates the current schema: {str(exc)[:200]}")
    if not cfg.mask.enabled:
        raise HTTPException(400, "this job has masking switched off")
    return CACHE_ROOT / "mask" / cfg.k_mask()


# Two sizes only. The URL is the cache key on the server AND in the browser, and
# an open-ended ?w= would let a reload storm fill the disk with near-duplicates.
THUMB_SIZES = {480, 1600}
THUMB_ROOT = RENDER_ROOT / "thumbs"


@app.get("/api/jobs/{job_id}/thumb.jpg")
def api_thumb(job_id: int, w: int = Query(480)):
    """The job's opening frame, downscaled.

    Taken from the SOURCE clip at trim_start rather than from the frames cache,
    so a job that has not run yet still has a picture -- which is the point,
    since the preview is most useful while deciding what to queue. Keyed on the
    clip's content hash, not the job, so every job sharing a clip and trim
    shares one file.
    """
    if w not in THUMB_SIZES:
        raise HTTPException(400, f"w must be one of {sorted(THUMB_SIZES)}")
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    try:
        cfg = JobConfig.model_validate(json.loads(row["config"]))
    except ValidationError as exc:
        raise HTTPException(
            400, f"job config predates the current schema: {str(exc)[:200]}")

    src = input_path(cfg.input.file)
    at = max(0.0, cfg.input.trim_start or 0.0)
    stem = f"{cfg.input.quick_hash or 'nohash'}_{at:g}_{w}"
    out = THUMB_ROOT / f"{stem}.jpg"
    if not out.is_file() or out.stat().st_size == 0:
        THUMB_ROOT.mkdir(parents=True, exist_ok=True)
        # Each writer owns its temporary file; publication stays atomic.
        with tempfile.TemporaryDirectory(prefix=f".{stem}.", dir=THUMB_ROOT) as tmp_dir:
            tmp = Path(tmp_dir) / "thumb.jpg"
            # -ss before -i is the fast keyframe seek; scale to an EVEN height
            # (-2, not -1) because the JPEG encoder rejects odd chroma dimensions
            # on a 2:1 equirect at some widths.
            # A raw .OSV's first stream is one lens; put both side by side so the
            # thumbnail is the same 2:1 shape as a stitched clip's.
            scale = (["-filter_complex", f"[0:v:0][0:v:1]hstack,scale={w}:-2"]
                     if cfg.is_fisheye else ["-vf", f"scale={w}:-2"])
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", f"{at:g}", "-i", str(src), "-frames:v", "1",
                 *scale, "-q:v", "4", str(tmp)],
                capture_output=True, text=True, timeout=120)
            if proc.returncode != 0 or not tmp.is_file():
                raise HTTPException(
                    502, f"could not read a frame from {src.name}: "
                         f"{proc.stderr.strip()[:200]}")
            os.replace(tmp, out)          # never serve a half-written JPEG
    # The URL never changes for a given clip, trim and size, so the browser can
    # keep it forever -- which is what stops the four-second table rebuild
    # re-fetching a thumbnail per row per poll.
    return FileResponse(out, media_type="image/jpeg", headers={
        "Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/jobs/{job_id}/review")
def api_review_get(job_id: int) -> dict:
    """What a human needs to decide: coverage, misses, and where to look."""
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    d = _mask_dir_for(job_id)
    info = {}
    marker = d / ".done"
    if marker.is_file():
        try:
            info = json.loads(marker.read_text())
        except (OSError, ValueError):
            info = {}
    overlays = sorted(p.name for p in (d / "overlay").glob("*.jpg")) \
        if (d / "overlay").is_dir() else []
    return {
        "id": job_id,
        "state": row["state"],
        "review_state": row["review_state"],
        "review_note": row["review_note"],
        "awaiting": row["state"] == "awaiting_review",
        "coverage_solid_angle_mean": info.get("coverage_solid_angle_mean"),
        "coverage_solid_angle_max": info.get("coverage_solid_angle_max"),
        "frames": info.get("frames"),
        "frames_without_detection": info.get("frames_without_detection", []),
        "warnings": info.get("warnings", []),
        "sheet_url": f"/api/jobs/{job_id}/review/sheet.jpg",
        "overlay_urls": [f"/api/jobs/{job_id}/review/overlay/{n}"
                         for n in overlays],
    }


@app.get("/api/jobs/{job_id}/review/sheet.jpg")
def api_review_sheet(job_id: int):
    path = _mask_dir_for(job_id) / "review_sheet.jpg"
    if not path.is_file():
        raise HTTPException(404, "no contact sheet for this mask set")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/review/overlay/{name}")
def api_review_overlay(job_id: int, name: str):
    # Resolve and containment-check: the name comes from the URL, and a mask
    # directory is not a place to serve arbitrary paths from.
    base = (_mask_dir_for(job_id) / "overlay").resolve()
    path = (base / name).resolve()
    if not str(path).startswith(str(base) + os.sep) or not path.is_file():
        raise HTTPException(404, "no such overlay")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/jobs/{job_id}/review")
def api_review_post(job_id: int, req: ReviewReq) -> dict:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    if row["state"] != "awaiting_review":
        raise HTTPException(
            400, f"job is {row['state']}, not awaiting review")
    if req.approved:
        db.set_review(job_id, "approved", req.note)
        # Straight back into the queue. Everything up to and including the
        # masks is a cache hit, so the job resumes rather than restarts.
        db.set_job_state(job_id, "queued", started=None, ended=None, gpu=None)
        return {"id": job_id, "state": "queued", "review_state": "approved"}
    db.set_review(job_id, "rejected", req.note)
    db.set_job_state(job_id, "cancelled", ended=time.time(),
                     error=f"masks rejected at review: {req.note[:500]}")
    return {"id": job_id, "state": "cancelled", "review_state": "rejected"}


@app.post("/api/estimate")
def api_estimate(req: EstimateReq) -> dict:
    cfg = _prepare(req.config)
    src = input_path(cfg.input.file)
    dur = ffprobe(src)["duration"]
    if not dur:
        raise HTTPException(400, "cannot probe input duration")
    try:
        return estimate.estimate(cfg, dur, cached=_cache_state(cfg))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ------------------------------------------------------------------ status

@app.get("/api/status")
def api_status() -> dict:
    now = time.time()
    st = worker.running_state()
    counts = {r["state"]: r["n"] for r in db.conn().execute(
        "SELECT state, COUNT(*) n FROM jobs GROUP BY state").fetchall()}
    # Only running and queued rows can affect when the queue drains, and there
    # are rarely more than a handful of those -- unlike the full job page.
    active = [_job_dict(r, now=now) for r in db.active_jobs()]
    return {**st, "counts": counts, "schedulable_gpus": GPUS,
            "mask_backends": mask_backends.availability(MODELS_ROOT),
            "visible_jobs": db.count_jobs(),
            "queue": progress.queue_eta(
                active, worker.schedulable_capacity(), now=now,
                paused=st["paused"]),
            "metrics": METRICS_ENABLED,
            "splat_root": str(SPLAT_ROOT)}


def api_metrics():
    """Prometheus exposition. Registered only when QUEUE_METRICS=1.

    Behind the same token as everything else -- it names input clips and job
    labels, and this service is reachable on the LAN. Prometheus sends the
    header itself:

        scrape_configs:
          - job_name: splat-queue
            static_configs: [{targets: ['gpu-workstation:8090']}]
            authorization: {type: Bearer, credentials: <token>}

    or, since the middleware also accepts the dedicated header:

            http_headers:
              X-Queue-Token: {values: [<token>]}
    """
    now = time.time()
    active = [_job_dict(r, now=now) for r in db.active_jobs()]
    progress.queue_eta(active, worker.schedulable_capacity(), now=now,
                       paused=worker.paused())
    return PlainTextResponse(metrics.render(active, now=now),
                             media_type=metrics.CONTENT_TYPE)


# Registered rather than decorated, so that with the option off the route does
# not exist at all -- no handler to reach, nothing to probe, and /metrics 404s
# the way any other unknown path does.
if METRICS_ENABLED:
    app.get("/metrics")(api_metrics)


class PauseReq(BaseModel):
    paused: bool


@app.post("/api/pause")
def api_pause(req: PauseReq) -> dict:
    worker.set_paused(req.paused)
    return {"paused": worker.paused()}


class RenderReq(BaseModel):
    ids: list[int]
    poses: list[str] = []
    n_poses: int = 4


@app.post("/api/compare/render")
def api_render(req: RenderReq) -> dict:
    """Render every selected model from the same held-out poses into one sheet.

    This is the payoff of the compare view: identical poses, identical crop,
    one image. Needs a GPU, so it refuses while both are busy rather than
    queueing behind a 95-minute training run.
    """
    if not req.ids:
        raise HTTPException(400, "no jobs selected")

    models, dataset, images = [], None, None
    held_out_split = True
    for jid in req.ids:
        row = db.get_job(jid)
        if row is None:
            raise HTTPException(404, f"no such job: {jid}")
        stages = {s["stage"]: s for s in db.job_stages(jid)}
        tr = stages.get("train")
        if not tr or tr["state"] not in ("done", "cached"):
            raise HTTPException(400, f"job {jid} has no finished training stage")
        info = json.loads(tr["progress"] or "{}")
        ply = (info.get("artifacts") or {}).get("ply", {}).get("path")
        if not ply:
            raise HTTPException(400, f"job {jid} exported no PLY")
        # Strip '=' from the label: the renderer takes LABEL=path, and sweep
        # names are built from the varied parameter ("... [sh_degree=3]"), so
        # an unsanitised name put a second '=' into the spec.
        label = f"{jid}:{row['name'][:20]}".replace("=", "-")
        models.append(f"{label}={ply}")
        sfm, sel = stages.get("sfm"), stages.get("select")
        d = str(Path(sfm["path"]) / "dataset") if sfm else None
        job_cfg = json.loads(row["config"])
        if str((job_cfg.get("input") or {}).get("file", "")).lower().endswith(FISHEYE_EXTS):
            raise HTTPException(
                400, f"job {jid} is a fisheye rig reconstruction; the comparison "
                     f"renderer draws pinhole views off equirect or perspective "
                     f"models and does not handle OPENCV_FISHEYE cameras yet")
        # A model trained without --eval saw every image, so nothing in this
        # reconstruction is held out for it. One such job makes the whole sheet
        # a training-view comparison, and it is labelled as one.
        if not (job_cfg.get("train") or {}).get("eval", True):
            held_out_split = False
        if dataset is None:
            dataset = d
            # The perspective SfM modes register generated pinhole views, not
            # the panoramas, so the GT column has to come from the same place
            # training read (see stages.images_dir).
            if sfm and sel:
                cfg = JobConfig.model_validate(job_cfg)
                images = str(images_dir(cfg, Path(sfm["path"]),
                                        Path(sel["path"])))
        elif d != dataset:
            raise HTTPException(
                400, "selected jobs do not share one SfM, so their poses are "
                     "not comparable; compare jobs from a single sweep")

    if not dataset or not Path(dataset).is_dir():
        raise HTTPException(400, "SfM dataset directory is missing")

    stamp = f"cmp_{'_'.join(str(i) for i in sorted(req.ids))}_{int(time.time())}.jpg"
    out = RENDER_ROOT / stamp
    argv = [str(GS_PY), str(RENDER_COMPARE), "--dataset", dataset,
            "--images", images or dataset, "--out", str(out),
            "--n-poses", str(req.n_poses),
            # LichtFeld's own default split. 0 disables it, for models that
            # trained on everything.
            "--test-every", "8" if held_out_split else "0"]
    if req.poses:
        argv += ["--poses", ",".join(req.poses)]
    for m in models:
        argv += ["--model", m]

    # Hold the GPU for the whole render. Picking a free one and then launching
    # is not enough: the render does not appear in nvidia-smi for a second or
    # two, which is long enough for the dispatcher -- or a second render
    # request -- to hand the same card to a training job.
    with worker.reserve_gpu() as g:
        if g is None:
            raise HTTPException(
                409, "no free GPU for rendering right now; try again when a "
                     "training slot frees up")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=900, env=env, cwd=str(SPLAT_ROOT))
        if proc.returncode != 0:
            raise HTTPException(
                500, f"render failed: {(proc.stderr or proc.stdout)[-600:]}")
        return {"url": f"/renders/{stamp}", "gpu": g,
                "held_out": held_out_split,
                "stdout": proc.stdout[-400:]}


@app.post("/api/jobs/clear")
def api_clear(purge: bool = Query(False)) -> dict:
    """Drop finished jobs from the queue view. Cache artifacts are untouched.

    Hides rather than deletes. Deleting cascaded to the stage and metric rows,
    so pressing a button labelled "clear finished" threw away every recorded
    PSNR, wall time and peak-VRAM figure -- including the completed-job history
    estimate.py fits its it/s constant from, which is the whole reason the
    estimate sharpens with use. `?purge=true` is the version that really
    deletes, for when that is what you mean.
    """
    if purge:
        ids = db.purge_hidden()
        for i in ids:
            shutil.rmtree(telemetry.run_dir(i), ignore_errors=True)
        return {"purged": len(ids)}
    return {"hidden": db.hide_finished()}


class ConcurrencyReq(BaseModel):
    max_concurrent: int


@app.post("/api/concurrency")
def api_concurrency(req: ConcurrencyReq) -> dict:
    worker.set_max_concurrent(req.max_concurrent)
    return {"max_concurrent": worker.max_concurrent()}


@app.get("/api/cache")
def api_cache() -> dict:
    return {"entries": [dict(r) for r in db.list_cache()],
            **retention.status()}


@app.post("/api/cache/gc")
def api_cache_gc(dry_run: bool = Query(False),
                 budget_gb: Optional[float] = Query(None)) -> dict:
    """Evict least-recently-used cache entries.

    Skips anything a queued or running job depends on, and anything whose
    directory is locked by a live process. `dry_run` lists what would go.
    """
    if budget_gb is None:
        # No explicit ceiling: reclaim only as far as the free-space floor, so
        # the obvious button is not also the one that wipes the cache.
        return retention.gc_cache(target_free=retention.MIN_FREE_BYTES,
                                  dry_run=dry_run)
    return retention.gc_cache(budget=budget_gb * retention.GB, dry_run=dry_run)


# ----------------------------------------------------------------- compare

@app.get("/api/compare")
def api_compare(ids: str = Query(...)) -> dict:
    try:
        job_ids = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "ids must be comma-separated integers")
    jobs = []
    for jid in job_ids:
        row = db.get_job(jid)
        if row:
            jobs.append(_job_dict(row))
    if not jobs:
        raise HTTPException(404, "no such jobs")

    # Config diff: only the leaf fields that actually differ.
    def flat(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            p = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flat(v, p + "."))
            else:
                out[p] = v
        return out

    flats = [flat(j["config"]) for j in jobs]
    all_keys = sorted({k for f in flats for k in f})
    diff = {k: [f.get(k) for f in flats] for k in all_keys
            if len({json.dumps(f.get(k), default=str) for f in flats}) > 1}
    return {"jobs": jobs, "diff": diff}


app.mount("/renders", StaticFiles(directory=str(RENDER_ROOT)), name="renders")
app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
