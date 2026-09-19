"""SQLite state. WAL so the worker thread and request handlers can share it."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Optional

from .config import DB_PATH

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    config      TEXT NOT NULL,           -- JobConfig as JSON
    state       TEXT NOT NULL,           -- queued|running|awaiting_review|done|failed|cancelled
    priority    INTEGER NOT NULL DEFAULT 0,
    gpu         INTEGER,
    error       TEXT,
    sweep_id    TEXT,
    created     REAL NOT NULL,
    started     REAL,
    ended       REAL,
    hidden      INTEGER NOT NULL DEFAULT 0,  -- cleared from the view, not gone
    plan        TEXT                         -- estimate.estimate() at submit time
);
CREATE TABLE IF NOT EXISTS stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    cache_key   TEXT NOT NULL,
    state       TEXT NOT NULL,           -- pending|cached|running|done|failed|skipped
    path        TEXT,
    log_path    TEXT,
    progress    TEXT,                    -- JSON blob, stage-specific
    started     REAL,
    ended       REAL,
    UNIQUE(job_id, stage)
);
CREATE TABLE IF NOT EXISTS cache (
    cache_key   TEXT PRIMARY KEY,
    stage       TEXT NOT NULL,
    path        TEXT NOT NULL,
    bytes       INTEGER NOT NULL DEFAULT 0,
    created     REAL NOT NULL,
    last_used   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    value       REAL,
    text        TEXT,
    PRIMARY KEY (job_id, name)
);
CREATE TABLE IF NOT EXISTS settings (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_stages_job ON stages(job_id);
"""


def conn() -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        c = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA busy_timeout=30000")
        _local.conn = c
    return c


def init() -> None:
    c = conn()
    c.executescript(SCHEMA)
    # "Clear finished" used to DELETE, taking the stage and metric rows with it
    # and shrinking the history estimate.py fits its it/s constant from. Jobs
    # are hidden instead, so add the column to databases created before that.
    cols = {r["name"] for r in c.execute("PRAGMA table_info(jobs)")}
    if "hidden" not in cols:
        c.execute("ALTER TABLE jobs ADD COLUMN hidden INTEGER NOT NULL "
                  "DEFAULT 0")
    # Human review of the mask stage. Kept on the job rather than the stage row
    # because the mask stage itself is shared through the cache: two jobs can
    # reuse one mask set and still need approving separately.
    if "review_state" not in cols:
        c.execute("ALTER TABLE jobs ADD COLUMN review_state TEXT")
    if "review_note" not in cols:
        c.execute("ALTER TABLE jobs ADD COLUMN review_note TEXT")
    # The per-stage estimate as it stood when the job was submitted. Frozen
    # rather than recomputed per request: it needs an ffprobe, the UI asks for
    # it every four seconds, and an estimate that moves under a running job
    # cannot be compared against what that job is actually doing.
    if "plan" not in cols:
        c.execute("ALTER TABLE jobs ADD COLUMN plan TEXT")


# ------------------------------------------------------------------ settings

def get_setting(k: str, default: str = "") -> str:
    r = conn().execute("SELECT v FROM settings WHERE k=?", (k,)).fetchone()
    return r["v"] if r else default


def set_setting(k: str, v: str) -> None:
    conn().execute(
        "INSERT INTO settings(k,v) VALUES(?,?) "
        "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))


# ---------------------------------------------------------------------- jobs

def create_job(name: str, config: dict, sweep_id: Optional[str] = None,
               priority: int = 0) -> int:
    cur = conn().execute(
        "INSERT INTO jobs(name,config,state,priority,sweep_id,created) "
        "VALUES(?,?,'queued',?,?,?)",
        (name, json.dumps(config), priority, sweep_id, time.time()))
    return int(cur.lastrowid)


def get_job(job_id: int) -> Optional[sqlite3.Row]:
    return conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


# Running first, then queued IN THE ORDER THE DISPATCHER WILL TAKE THEM, then
# everything finished newest-first. The queue table used to sort every group by
# id DESC, so the job shown at the top of the queued list was the LAST one that
# would run -- the display contradicted next_queued() for anything but a single
# job. The two CASE expressions are NULL for non-queued rows, which ties them and
# lets the final `id DESC` decide.
JOB_ORDER = """
    ORDER BY CASE state WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,
             CASE WHEN state='queued' THEN priority END DESC,
             CASE WHEN state='queued' THEN id END ASC,
             id DESC"""


def list_jobs(limit: int = 200, offset: int = 0) -> list[sqlite3.Row]:
    return conn().execute(
        "SELECT * FROM jobs WHERE hidden=0" + JOB_ORDER + " LIMIT ? OFFSET ?",
        (limit, offset)).fetchall()


def count_jobs() -> int:
    """Visible jobs, so the UI can say what its page is a page OF."""
    return int(conn().execute(
        "SELECT COUNT(*) n FROM jobs WHERE hidden=0").fetchone()["n"])


def set_plan(job_id: int, plan: dict) -> None:
    conn().execute("UPDATE jobs SET plan=? WHERE id=?",
                   (json.dumps(plan), job_id))


def set_review(job_id: int, state: str, note: str = "") -> None:
    conn().execute("UPDATE jobs SET review_state=?, review_note=? WHERE id=?",
                   (state, note[:2000], job_id))


def active_jobs() -> list[sqlite3.Row]:
    """Running and queued only, in the order the dispatcher will take them."""
    return conn().execute(
        "SELECT * FROM jobs WHERE hidden=0 AND state IN ('running','queued')"
        + JOB_ORDER).fetchall()


def next_queued() -> Optional[sqlite3.Row]:
    return conn().execute(
        "SELECT * FROM jobs WHERE state='queued' AND hidden=0 "
        "ORDER BY priority DESC, id ASC LIMIT 1").fetchone()


def hide_finished() -> int:
    """Drop finished jobs from the queue VIEW. Nothing is deleted."""
    cur = conn().execute(
        "UPDATE jobs SET hidden=1 "
        "WHERE state IN ('done','failed','cancelled') AND hidden=0")
    return cur.rowcount


def purge_hidden() -> int:
    """Actually delete hidden jobs, and with them their stages and metrics."""
    cur = conn().execute("DELETE FROM jobs WHERE hidden=1")
    return cur.rowcount


def set_job_state(job_id: int, state: str, **kw) -> None:
    cols, vals = ["state=?"], [state]
    for k, v in kw.items():
        cols.append(f"{k}=?")
        vals.append(v)
    vals.append(job_id)
    conn().execute(f"UPDATE jobs SET {','.join(cols)} WHERE id=?", vals)


# -------------------------------------------------------------------- stages

# States a stage is IN, not one it has finished in. Moving back into one of
# these means a fresh attempt, so the previous attempt's end time and progress
# blob have to go. COALESCE kept both: the detail pane showed a re-running stage
# with a duration of stale_ended minus new_started -- a negative number -- next
# to the PSNR of the run before it.
UNFINISHED = ("pending", "waiting", "running")


def upsert_stage(job_id: int, stage: str, cache_key: str, state: str,
                 **kw) -> None:
    fields = {"path": None, "log_path": None, "progress": None,
              "started": None, "ended": None}
    fields.update(kw)
    restart = state in UNFINISHED
    ended = "NULL" if restart and fields["ended"] is None else \
        "COALESCE(excluded.ended,stages.ended)"
    progress = "NULL" if restart and fields["progress"] is None else \
        "COALESCE(excluded.progress,stages.progress)"
    conn().execute(
        "INSERT INTO stages(job_id,stage,cache_key,state,path,log_path,"
        "progress,started,ended) VALUES(?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(job_id,stage) DO UPDATE SET "
        "cache_key=excluded.cache_key, state=excluded.state, "
        "path=COALESCE(excluded.path,stages.path), "
        "log_path=COALESCE(excluded.log_path,stages.log_path), "
        f"progress={progress}, "
        "started=COALESCE(excluded.started,stages.started), "
        f"ended={ended}",
        (job_id, stage, cache_key, state, fields["path"], fields["log_path"],
         fields["progress"], fields["started"], fields["ended"]))


def stage_progress(job_id: int, stage: str, progress: dict) -> None:
    conn().execute("UPDATE stages SET progress=? WHERE job_id=? AND stage=?",
                   (json.dumps(progress), job_id, stage))


def job_stages(job_id: int) -> list[sqlite3.Row]:
    return conn().execute(
        "SELECT * FROM stages WHERE job_id=? ORDER BY id", (job_id,)).fetchall()


# --------------------------------------------------------------------- cache

def cache_get(cache_key: str) -> Optional[sqlite3.Row]:
    return conn().execute("SELECT * FROM cache WHERE cache_key=?",
                          (cache_key,)).fetchone()


def cache_put(cache_key: str, stage: str, path: str, nbytes: int = 0) -> None:
    now = time.time()
    conn().execute(
        "INSERT INTO cache(cache_key,stage,path,bytes,created,last_used) "
        "VALUES(?,?,?,?,?,?) ON CONFLICT(cache_key) DO UPDATE SET "
        "bytes=excluded.bytes, last_used=excluded.last_used",
        (cache_key, stage, path, nbytes, now, now))


def cache_touch(cache_key: str) -> None:
    conn().execute("UPDATE cache SET last_used=? WHERE cache_key=?",
                   (time.time(), cache_key))


def list_cache() -> list[sqlite3.Row]:
    return conn().execute(
        "SELECT * FROM cache ORDER BY last_used DESC").fetchall()


# ------------------------------------------------------------------- metrics

def put_metric(job_id: int, name: str, value: Any) -> None:
    num = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    txt = None if num is not None else str(value)
    conn().execute(
        "INSERT INTO metrics(job_id,name,value,text) VALUES(?,?,?,?) "
        "ON CONFLICT(job_id,name) DO UPDATE SET value=excluded.value, "
        "text=excluded.text", (job_id, name, num, txt))


def job_metrics(job_id: int) -> dict[str, Any]:
    rows = conn().execute("SELECT name,value,text FROM metrics WHERE job_id=?",
                          (job_id,)).fetchall()
    return {r["name"]: (r["value"] if r["value"] is not None else r["text"])
            for r in rows}
