"""Reclaim-notice watcher for spot GPUs (issue #10, docs/cloud.md "Spot GPUs").

QUEUE_PREEMPT_WATCH=aws|gcp polls the cloud's metadata service every
QUEUE_PREEMPT_POLL_S seconds (5, as AWS recommends). On a notice every running
trainer is asked for a snapshot (worker.request_snapshots, SIGUSR1), which
becomes a restore point and goes to CHECKPOINT_UPLOAD_URL like a scheduled one:
measured 3.4 s from the request to a verified restore point at 3M splats, then
the upload. The queue is paused as well, so no job starts on a machine about to
go.

  aws  IMDSv2 spot/instance-action: the 2-minute interruption notice. Also
       events/recommendations/rebalance, which can come earlier but does not
       always end in a reclaim: a snapshot is taken, nothing is paused. A
       container needs the instance's metadata hop limit at 2 to reach IMDSv2.
  gcp  instance/preempted turning TRUE. By default a Spot VM gets no more than
       ~30 s (ACPI soft-off) before it is powered off; a 120 s notice is in
       Preview. Scheduled snapshots (train.checkpoint_every) matter most here.

Every notice is acted on once. Nothing here ever raises into the service.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

from . import checkpoints, telemetry
from .config import (AWS_METADATA_URL, GCP_METADATA_URL, PREEMPT_POLL_S, PREEMPT_WATCH,
                     RUNS_ROOT)

NOTICE_RECORD = RUNS_ROOT / "preempt.json"
TIMEOUT_S = 2.0

_thread: Optional[threading.Thread] = None
_stop = threading.Event()


def _get(url: str, headers: Optional[dict] = None, method: str = "GET") -> Optional[str]:
    """Body of a 200, None for a 404 (no notice); raises on anything else."""
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


class Aws:
    def __init__(self, base: str = AWS_METADATA_URL):
        self.base, self._token, self._token_exp = base, None, 0.0

    def _headers(self) -> dict:
        if not self._token or time.time() > self._token_exp:
            self._token = _get(f"{self.base}/latest/api/token", method="PUT",
                               headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"})
            self._token_exp = time.time() + 21000
        return {"X-aws-ec2-metadata-token": self._token or ""}

    def poll(self) -> Optional[dict]:
        h = self._headers()
        body = _get(f"{self.base}/latest/meta-data/spot/instance-action", headers=h)
        if body:
            try:
                info = json.loads(body)
            except ValueError:
                info = {}
            return {"kind": "reclaim", "provider": "aws",
                    "action": info.get("action"), "at": info.get("time")}
        body = _get(f"{self.base}/latest/meta-data/events/recommendations/rebalance", headers=h)
        if body:
            try:
                info = json.loads(body)
            except ValueError:
                info = {}
            return {"kind": "rebalance", "provider": "aws", "at": info.get("noticeTime")}
        return None


class Gcp:
    def __init__(self, base: str = GCP_METADATA_URL):
        self.base = base

    def poll(self) -> Optional[dict]:
        body = _get(f"{self.base}/computeMetadata/v1/instance/preempted",
                    headers={"Metadata-Flavor": "Google"})
        if body and body.strip().upper() == "TRUE":
            return {"kind": "reclaim", "provider": "gcp", "action": "preempted", "at": None}
        return None


def handle(notice: dict) -> list[int]:
    """Snapshot every running trainer; on a reclaim, pause the queue too."""
    from . import worker                    # worker imports checkpoints, not this
    t0 = time.time()
    asked = worker.request_snapshots()
    if notice["kind"] == "reclaim":
        worker.set_paused(True)
    rec = {**notice, "received": round(t0, 1), "jobs": asked,
           "paused": notice["kind"] == "reclaim"}
    try:
        NOTICE_RECORD.write_text(json.dumps(rec, indent=1))
    except OSError:
        pass
    for jid in asked:
        checkpoints.record_notice(jid, {k: rec[k] for k in ("kind", "provider", "action",
                                                              "received")
                                        if k in rec})
        telemetry.notify("job.preempt_notice", jid, "train", notice["kind"])
    print(f"preempt: {notice['provider']} {notice['kind']} notice"
          + (f" ({notice.get('action')} at {notice.get('at')})" if notice.get("action") else "")
          + f"; snapshot requested for job(s) {asked or 'none running'}"
          + ("; queue paused" if rec["paused"] else ""))
    return asked


def watch(source, poll_s: float = PREEMPT_POLL_S, stop: threading.Event = _stop) -> None:
    """Poll until stopped. Each kind of notice is acted on once."""
    seen: set[str] = set()
    errors = 0
    while not stop.is_set():
        try:
            notice = source.poll()
            errors = 0
            if notice and notice["kind"] not in seen:
                seen.add(notice["kind"])
                handle(notice)
        except Exception as exc:                              # noqa: BLE001
            errors += 1
            if errors in (1, 10, 100):
                print(f"preempt: metadata poll failed ({errors}x): {type(exc).__name__}: {exc}")
        stop.wait(poll_s)


def start() -> None:
    """Start the watcher when QUEUE_PREEMPT_WATCH asks for one."""
    global _thread
    if PREEMPT_WATCH == "off" or (_thread and _thread.is_alive()):
        return
    source = Aws() if PREEMPT_WATCH == "aws" else Gcp()
    _stop.clear()
    _thread = threading.Thread(target=watch, args=(source,), daemon=True, name="preempt")
    _thread.start()
    print(f"preempt: watching {PREEMPT_WATCH} metadata every {PREEMPT_POLL_S:g} s")


def stop() -> None:
    _stop.set()
