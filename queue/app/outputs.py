"""Upload a finished job's splats to object storage (OUTPUT_UPLOAD_URL).

For rented GPUs (docs/cloud.md): the host has root over the container, so
nothing on it should be able to reach more than this run's own result. A
presigned PUT URL is exactly that: one object, until it expires. It is still a
credential the host can read, so the archive's sha256 goes into upload.json
and the log to check the download against.

After a job ends done, its export files (the .ply/.sog/.spz the download
buttons serve; symlinks into the train cache, followed) are packed as
job<id>.tar and sent with the same request builder as telemetry, so both
accept a plain PUT URL or a presigned S3 POST. After a PUT, the ETag S3 returns
(the object's MD5 for a single-part upload) is compared with the archive's MD5,
so a body that arrived damaged fails the upload. Not a Content-MD5 header: AWS
refuses a presigned PUT carrying headers the URL did not sign ("There were
headers present in the request which were not signed"), and the MD5 is not
known when the URL is made. versitygw accepted it, AWS did not (2026-09-22).

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
import io
import json
import os
import socket
import tarfile
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Optional

from . import db, telemetry, uploads
from .config import OUTPUT_UPLOAD, OUTPUT_UPLOAD_ERROR, TLS_INSECURE, ssl_context
from .stages import ARTIFACT_EXTS
from .uploads import safe_url

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


class _Hashing:
    """The archive file, hashed as it is written: no second pass over GBs."""

    def __init__(self, f):
        self.f, self.sha, self.md5 = f, hashlib.sha256(), hashlib.md5()

    def write(self, b) -> int:
        self.sha.update(b)
        self.md5.update(b)
        return self.f.write(b)


def build_archive(job_id: int, files: list[Path], out: Path) -> dict:
    """Pack files as job<id>/<name> (links followed); return size and digests."""
    prefix = f"job{job_id:05d}"
    tmp = out.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    with tmp.open("wb") as raw:
        h = _Hashing(raw)
        _pack(job_id, files, prefix, h)
    os.replace(tmp, out)
    return {"bytes": out.stat().st_size, "sha256": h.sha.hexdigest(),
            "md5": h.md5.hexdigest()}


def _pack(job_id: int, files: list[Path], prefix: str, fileobj) -> None:
    with tarfile.open(fileobj=fileobj, mode="w|", dereference=True) as tar:
        for p in files:
            tar.add(p, arcname=f"{prefix}/{p.name}", recursive=False)
        # Read whole first: a tar member cut short would break the result
        # archive, and a copy of the telemetry must never cost the upload.
        try:
            tel = telemetry.telemetry_path(job_id).read_bytes()
        except OSError as exc:
            tel = None
            print(f"job {job_id}: telemetry.json not in the result archive: {exc}")
        if tel:
            info = tarfile.TarInfo(f"{prefix}/telemetry.json")
            info.size, info.mtime, info.mode = len(tel), int(time.time()), 0o644
            tar.addfile(info, io.BytesIO(tel))


def _uploaded_to(job_id: int) -> Optional[str]:
    """Where this job's result went, if it went (compared without signature)."""
    st = status(job_id)
    return st.get("url") if st and st.get("state") == "done" else None


def upload(job_id: int, target: Optional[dict] = None, attempts: int = 3) -> dict:
    """Pack and upload one job's export. Returns the status it recorded."""
    target = target or OUTPUT_UPLOAD
    name = f"job{job_id:05d}.tar"
    dest = uploads.destination(target, job_id, name)
    with _lock:
        owner = uploads.owner_of(target, dest, job_id, _uploaded_to)
        if owner is not None:
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
        arc = telemetry.run_dir(job_id) / name
        arc.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        _set_status(job_id, state="uploading", url=dest, started=round(started, 1))
        meta = build_archive(job_id, files, arc)
        pack_s = round(time.time() - started, 1)
        # Streamed from disk: a 6M-splat result is GBs, and was read whole.
        res = uploads.send(target, job_id, name, arc, meta["md5"], timeout=600,
                           attempts=attempts, context=ssl_context())
        err, tries, transfer_s = res["error"], res["attempts"], res["transfer_s"]
        st = _set_status(
            job_id, state="failed" if err else "done", url=dest, file=name,
            files=[p.name for p in files], bytes=meta["bytes"],
            sha256=meta["sha256"], started=round(started, 1),
            ended=round(time.time(), 1), pack_s=pack_s,
            transfer_s=None if err else transfer_s, attempts=tries, error=err)
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
        # The job's one telemetry upload, deferred to here by the worker so
        # the record includes transfers.output.
        telemetry.write(job_id, upload=True)
        telemetry.notify("job.uploaded", job_id, state=st.get("state"))
    threading.Thread(target=run, daemon=True, name=f"output{job_id}").start()
