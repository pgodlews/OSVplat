"""One way to send a file to a presigned target.

Four things go to object storage: the result tar (outputs.py), the handoff
bundle (handoff.py), restore points (checkpoints.py) and debug bundles
(debugdump.py). Each used to carry its own copy of this: streaming PUT or
presigned POST, retries, the ETag check, the error text, and the rule that a
URL naming one object belongs to the first job that uploads to it. The copies
had already drifted (the result tar was read whole into memory, the others
streamed). Their status files and what they log stay their own.

A target is OUTPUT_UPLOAD_URL's shape: {"method": "PUT", "url": ...,
"headers": {...}} or a presigned S3 POST {"method": "POST", "url", "fields",
"key"}. {job} and {file} in the URL (and a POST's key) are filled in.
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from . import telemetry
from .config import RUNS_ROOT, ssl_context


def safe_url(url: str) -> str:
    """scheme://host/path without the query: the signature is the credential."""
    u = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, "", ""))


def object_url(target: dict, job_id: int, name: str) -> str:
    return target["url"].replace("{job}", f"job{job_id:05d}").replace("{file}", name)


def destination(target: dict, job_id: int, name: str) -> str:
    """Where it goes, safe to log and to store."""
    return safe_url(object_url(target, job_id, name))


def fixed_object(target: dict) -> bool:
    """True when every job's upload would land on the same object."""
    return "{job}" not in target["url"] and "{job}" not in (target.get("key") or "")


def owner_of(target: dict, dest: str, job_id: int,
             uploaded_to: Callable[[int], Optional[str]]) -> Optional[int]:
    """Another job that already uploaded to this same fixed object, or None.

    uploaded_to(job) is the destination that job's own status file records as
    successfully uploaded. A URL with a {job} placeholder names one object per
    job, and is never refused.
    """
    if not fixed_object(target):
        return None
    for d in sorted(RUNS_ROOT.glob("job*")):
        jid = d.name[3:]
        if not jid.isdigit() or int(jid) == job_id:
            continue
        if uploaded_to(int(jid)) == dest:
            return int(jid)
    return None


def etag_problem(etag: Optional[str], md5_hex: str) -> Optional[str]:
    """A damaged upload, judged by the ETag S3 returned, or None.

    Only a plain 32-hex ETag is an MD5: multipart uploads ("...-3") and
    SSE-KMS objects have other ETags, and some servers send none. Those are
    not checked rather than failed; the sha256 in the status file still is.
    """
    e = (etag or "").strip().strip('"').lower()
    if len(e) == 32 and all(c in "0123456789abcdef" for c in e) and e != md5_hex:
        return f"stored object's ETag {e} is not the archive's MD5 {md5_hex}: damaged in transit"
    return None


def request(target: dict, job_id: int, name: str, data: bytes,
            content_type: str = "application/x-tar") -> urllib.request.Request:
    """telemetry.upload_request with this content type. No other headers: a
    presigned PUT signs only `host`, and AWS refuses unsigned extras (a
    Content-Type is the one it tolerates)."""
    t = dict(target)
    t["headers"] = {"Content-Type": content_type, **(target.get("headers") or {})}
    return telemetry.upload_request(t, job_id, name, data)


def _open_request(target: dict, job_id: int, name: str, f, size: int,
                  content_type: str) -> urllib.request.Request:
    """A streaming PUT (bundles run to GBs), or a POST, which has to be built
    in memory as a multipart body."""
    if (target.get("method") or "PUT").upper() == "POST":
        return request(target, job_id, name, f.read(), content_type)
    headers = {"Content-Type": content_type, **(target.get("headers") or {}),
               "Content-Length": str(size)}
    return urllib.request.Request(object_url(target, job_id, name), data=f,
                                  headers=headers, method="PUT")


def send(target: dict, job_id: int, name: str, path: Path, md5: str, *,
         timeout: float, attempts: int = 3, retry_s: float = 2.0,
         content_type: str = "application/x-tar", context=None) -> dict:
    """Upload one file, retrying with backoff (retry_s, then twice that).

    Returns {"state": "done"|"failed", "bytes", "attempts", "transfer_s",
    "error"}. Never raises for a transfer problem. The error text never holds
    the URL: an HTTPError's message can echo it, so only code and reason.
    context: the TLS context (config.ssl_context() when None).
    """
    if context is None:
        context = ssl_context()
    size = path.stat().st_size
    err, tries, transfer_s = None, 0, None
    for attempt in range(attempts):
        tries += 1
        try:
            with path.open("rb") as f:
                req = _open_request(target, job_id, name, f, size, content_type)
                t0 = time.monotonic()
                with urllib.request.urlopen(req, timeout=timeout, context=context) as r:
                    r.read()
                    etag = r.headers.get("ETag")
            transfer_s = round(time.monotonic() - t0, 2)
            err = etag_problem(etag, md5)
            if err is None:
                break
        except Exception as exc:                              # noqa: BLE001
            err = (f"HTTP {exc.code} {exc.reason}" if hasattr(exc, "code")
                   else f"{type(exc).__name__}: {exc}")
        if attempt < attempts - 1:
            time.sleep(retry_s * 2 ** attempt)
    return {"state": "failed" if err else "done", "bytes": size, "attempts": tries,
            "transfer_s": None if err else transfer_s, "error": err}
