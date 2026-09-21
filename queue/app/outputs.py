"""Upload a finished job's splats to object storage (OUTPUT_UPLOAD_URL).

For rented GPUs (docs/cloud.md): the host has root over the container, so
nothing on it should be able to reach more than this run's own result. A
presigned PUT URL is exactly that: one object, until it expires. It is still a
credential the host can read, so the archive's sha256 goes into upload.json
and the log to check the download against.

After a job ends done, its export files (the .ply/.sog/.spz the download
buttons serve; symlinks into the train cache, followed) are packed as
job<id>.tar and sent with the same request builder as telemetry, so both
accept a plain PUT URL or a presigned S3 POST. A PUT also carries Content-MD5,
so S3 rejects a body that arrived damaged.

Fails loudly, never silently: upload.json in the job's run dir records
state/bytes/sha256/error, the job detail API returns it, the webhook sends
job.uploaded, and a failure is printed to the service log. It never changes the
job's own state: the splat exists and is cached either way.

One presigned URL names one object. A URL without a {job} placeholder is used
for the first successful upload only; later jobs are refused rather than
overwriting it.
"""
from __future__ import annotations

import base64
import calendar
import hashlib
import json
import os
import socket
import tarfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from . import db, telemetry
from .config import (OUTPUT_UPLOAD, OUTPUT_UPLOAD_ERROR, RUNS_ROOT, TLS_INSECURE,
                     ssl_context)
from .stages import ARTIFACT_EXTS

_lock = threading.Lock()          # one upload at a time; they share the uplink


def status_path(job_id: int) -> Path:
    return telemetry.run_dir(job_id) / "upload.json"


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


def safe_url(url: str) -> str:
    """scheme://host/path without the query: the signature is the credential."""
    u = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, "", ""))


# ------------------------------------------------------------- preflight
# A presigned PUT cannot be tried without writing: the method is part of the
# signature, so HEAD fails, and a test PUT would leave an object that looks
# like a result. What can be known without using it up: whether it is well
# formed, when it expires (it says so itself), and whether its host answers.

def expires_at(target: dict) -> Optional[float]:
    """When a presigned target stops working (epoch s), or None if it does not say.

    SigV4 (X-Amz-Date + X-Amz-Expires), SigV2 (Expires), Google
    (X-Goog-Date + X-Goog-Expires), and a presigned POST's policy expiration.
    """
    policy = (target.get("fields") or {}).get("policy") or (target.get("fields") or {}).get("Policy")
    if policy:
        try:
            exp = json.loads(base64.b64decode(policy))["expiration"]
            return calendar.timegm(time.strptime(exp[:19], "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, KeyError, TypeError):
            return None
    q = {k.lower(): v[0] for k, v in urllib.parse.parse_qs(
        urllib.parse.urlsplit(target.get("url", "")).query).items()}
    for date_k, exp_k in (("x-amz-date", "x-amz-expires"), ("x-goog-date", "x-goog-expires")):
        if date_k in q and exp_k in q:
            try:
                t0 = calendar.timegm(time.strptime(q[date_k], "%Y%m%dT%H%M%SZ"))
                return t0 + int(q[exp_k])
            except ValueError:
                return None
    if "expires" in q and q["expires"].isdigit():
        return float(q["expires"])
    return None


def problems(now: Optional[float] = None) -> list[str]:
    """Reasons the configured upload cannot work at all. The API refuses new
    jobs while there are any: better to fail before hours of GPU time than
    at the end of them."""
    if OUTPUT_UPLOAD_ERROR:
        return [OUTPUT_UPLOAD_ERROR]
    if not OUTPUT_UPLOAD:
        return []
    exp = expires_at(OUTPUT_UPLOAD)
    now = time.time() if now is None else now
    if exp is not None and exp <= now:
        return [f"OUTPUT_UPLOAD_URL expired {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(exp))}"]
    return []


def seconds_left(now: Optional[float] = None) -> Optional[float]:
    exp = expires_at(OUTPUT_UPLOAD) if OUTPUT_UPLOAD else None
    return None if exp is None else round(exp - (time.time() if now is None else now))


def reachable(url: str, timeout: float = 10) -> Optional[str]:
    """None if the upload host resolves and accepts a TCP connection on its
    port, else what went wrong. No HTTP request: nothing is sent to it."""
    u = urllib.parse.urlsplit(url)
    host = u.hostname
    port = u.port or (443 if u.scheme == "https" else 80)
    if not host:
        return "no host in the URL"
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return f"DNS lookup of {host} failed: {exc}"
    try:
        socket.create_connection((host, port), timeout=timeout).close()
    except OSError as exc:
        return f"TCP connect to {host}:{port} failed: {exc}"
    return None


def startup_check() -> None:
    """Log what is known about the upload target; never raises, never blocks."""
    if TLS_INSECURE:
        print("WARNING: QUEUE_TLS_INSECURE=1: TLS certificates are not checked on "
              "uploads, telemetry and the webhook")
    if not OUTPUT_UPLOAD and not OUTPUT_UPLOAD_ERROR:
        return
    for p in problems():
        print(f"ERROR: {p}. New jobs are refused until it is fixed; "
              f"the service stays up.")
    if OUTPUT_UPLOAD:
        left = seconds_left()
        print(f"output upload: {safe_url(OUTPUT_UPLOAD['url'])}, "
              + ("no expiry in the URL" if left is None else f"expires in {left / 3600:.1f} h"))

        def probe():
            err = reachable(OUTPUT_UPLOAD["url"])
            if err:
                print(f"WARNING: output upload host {urllib.parse.urlsplit(OUTPUT_UPLOAD['url']).netloc} "
                      f"not reachable: {err}")
        threading.Thread(target=probe, daemon=True, name="upload-probe").start()


# ---------------------------------------------------------------- upload

def export_files(job_id: int) -> list[Path]:
    st = db.conn().execute(
        "SELECT state, path FROM stages WHERE job_id=? AND stage='export'",
        (job_id,)).fetchone()
    if not st or not st["path"] or st["state"] not in ("done", "cached"):
        return []
    d = Path(st["path"])
    if not d.is_dir():
        return []
    return [p for p in sorted(d.iterdir())
            if not p.name.startswith(".") and p.suffix.lstrip(".") in ARTIFACT_EXTS
            and p.is_file()]


def build_archive(job_id: int, files: list[Path], out: Path) -> dict:
    """Pack files as job<id>/<name> (links followed); return size and digests."""
    prefix = f"job{job_id:05d}"
    tmp = out.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    with tarfile.open(tmp, "w", dereference=True) as tar:
        for p in files:
            tar.add(p, arcname=f"{prefix}/{p.name}", recursive=False)
        tel = telemetry.telemetry_path(job_id)
        if tel.is_file():
            tar.add(tel, arcname=f"{prefix}/telemetry.json", recursive=False)
    os.replace(tmp, out)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with out.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
    return {"bytes": out.stat().st_size, "sha256": sha.hexdigest(),
            "md5_b64": base64.b64encode(md5.digest()).decode()}


def request(target: dict, job_id: int, name: str, data: bytes,
            md5_b64: str) -> urllib.request.Request:
    """telemetry.upload_request, plus the tar content type and Content-MD5."""
    t = dict(target)
    t["headers"] = {"Content-Type": "application/x-tar", **(target.get("headers") or {})}
    req = telemetry.upload_request(t, job_id, name, data)
    if (target.get("method") or "PUT").upper() == "PUT":
        req.add_header("Content-MD5", md5_b64)
    return req


def _single_object_owner(target: dict, dest: str) -> Optional[int]:
    """The job that already uploaded to this same fixed object, if any.

    Compared by `dest` (the URL without its signature): a new URL for another
    object is fine, the same object again is not.
    """
    if "{job}" in target["url"] or "{job}" in (target.get("key") or ""):
        return None
    for d in sorted(RUNS_ROOT.glob("job*")):
        st = status(int(d.name[3:])) if d.name[3:].isdigit() else None
        if st and st.get("state") == "done" and st.get("url") == dest:
            return int(d.name[3:])
    return None


def upload(job_id: int, target: Optional[dict] = None, attempts: int = 3) -> dict:
    """Pack and upload one job's export. Returns the status it recorded."""
    target = target or OUTPUT_UPLOAD
    dest = safe_url(target["url"].replace("{job}", f"job{job_id:05d}")
                    .replace("{file}", f"job{job_id:05d}.tar"))
    with _lock:
        owner = _single_object_owner(target, dest)
        if owner is not None and owner != job_id:
            st = _set_status(job_id, state="refused", url=dest,
                             error=f"OUTPUT_UPLOAD_URL names one object and job "
                                   f"{owner} already uploaded to it; use a {{job}} "
                                   f"placeholder or a new URL per run")
            print(f"job {job_id}: output upload refused: {st['error']}")
            return st
        files = export_files(job_id)
        if not files:
            st = _set_status(job_id, state="failed", url=dest,
                             error="no export files to upload")
            print(f"job {job_id}: output upload FAILED: no export files")
            return st
        name = f"job{job_id:05d}.tar"
        arc = telemetry.run_dir(job_id) / name
        arc.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        _set_status(job_id, state="uploading", url=dest, started=round(started, 1))
        meta = build_archive(job_id, files, arc)
        data = arc.read_bytes()
        err = None
        for attempt in range(attempts):
            try:
                req = request(target, job_id, name, data, meta["md5_b64"])
                with urllib.request.urlopen(req, timeout=600, context=ssl_context()) as r:
                    r.read()
                err = None
                break
            except Exception as exc:                          # noqa: BLE001
                # HTTPError's text can echo the URL; keep only code and reason.
                err = (f"HTTP {exc.code} {exc.reason}" if hasattr(exc, "code")
                       else f"{type(exc).__name__}: {exc}")
                if attempt < attempts - 1:
                    time.sleep(2 ** (attempt + 1))
        st = _set_status(
            job_id, state="failed" if err else "done", url=dest, file=name,
            files=[p.name for p in files], bytes=meta["bytes"],
            sha256=meta["sha256"], started=round(started, 1),
            ended=round(time.time(), 1), error=err)
        if err:
            print(f"job {job_id}: output upload to {dest} FAILED: {err} "
                  f"(archive kept at {arc})")
        else:
            arc.unlink(missing_ok=True)
            print(f"job {job_id}: uploaded {name} ({meta['bytes']} bytes, "
                  f"sha256 {meta['sha256']}) to {dest}")
        return st


def upload_async(job_id: int) -> None:
    """Start the upload if OUTPUT_UPLOAD_URL is set. Never raises."""
    if not OUTPUT_UPLOAD:
        return

    def run():
        try:
            st = upload(job_id)
        except Exception as exc:                              # noqa: BLE001
            st = {"state": "failed", "error": str(exc)}
            print(f"job {job_id}: output upload FAILED: {exc}")
        telemetry.notify("job.uploaded", job_id, state=st.get("state"))
    threading.Thread(target=run, daemon=True, name=f"output{job_id}").start()
