"""Disk retention: a free-space floor, LRU cache eviction, log/render pruning.

Nothing here ever ran before, and the queue is a machine for filling a disk: a
7 GB clip at 10 fps is tens of thousands of 8K JPEGs, every sweep variant adds a
training directory, and both are kept forever so the next job can reuse them.
The failure mode is a full disk 80 minutes into a 95-minute training run.

Everything is opt-out via env vars, and eviction is conservative: it never
touches a cache entry that a queued or running job depends on, and never one
whose directory is locked by a live process.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from . import db, telemetry
from .config import CACHE_ROOT, LOG_ROOT, QUEUE_ROOT, RENDER_ROOT
from .stages import lock_holder_alive, read_lock

GB = 1_000_000_000

# Refuse to start a stage with less than this free. 20 GB is roughly one 8K
# frames directory plus a training run's exports, so a job that passes the check
# can finish rather than dying half way.
MIN_FREE_BYTES = float(os.environ.get("QUEUE_MIN_FREE_GB", 20)) * GB
# The floor is for a stage whose size nothing predicts: every stage is sized
# from the job (stages.space_estimate) and wants its estimate plus this much
# headroom instead, unless the clip does not probe or QUEUE_MIN_FREE_GB was set
# by hand, which then holds for every stage as before. The flat 20 GB refused a
# train-only box twice on 2026-09-29 (train with 15.9 GB free, then export with
# 20.0 GB free after 114 min, for a job that added 1.3 GB), and a prep box had
# to be launched with the floor set by hand (2026-09-30, 21 GB disk, peak 13.7).
MIN_FREE_SET = "QUEUE_MIN_FREE_GB" in os.environ
ESTIMATED_SLACK_BYTES = float(os.environ.get("QUEUE_ESTIMATED_SLACK_GB", 3)) * GB
# 0 = no ceiling; eviction then only happens to satisfy MIN_FREE_BYTES.
CACHE_BUDGET_BYTES = float(os.environ.get("QUEUE_CACHE_BUDGET_GB", 0)) * GB
LOG_KEEP_DAYS = float(os.environ.get("QUEUE_LOG_KEEP_DAYS", 30))
RENDER_KEEP_DAYS = float(os.environ.get("QUEUE_RENDER_KEEP_DAYS", 30))


def free_bytes(path: Path = QUEUE_ROOT) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def cache_total() -> int:
    row = db.conn().execute("SELECT COALESCE(SUM(bytes),0) n FROM cache").fetchone()
    return int(row["n"])


# A job in any of these states is going to read its cache entries again.
# awaiting_review is the one that is easy to miss and the worst to get wrong: it
# is defined by waiting, so it is exactly the job whose entries sit untouched
# long enough to become the least-recently-used ones. Evicting them deleted the
# contact sheet out from under the person reviewing it (the sheet endpoint
# starts 404ing), and approving then re-ran the mask stage -- so the approval
# applied to a mask set nobody had ever seen.
LIVE_STATES = ("queued", "running", "awaiting_review")


def protected_keys() -> set[str]:
    """Cache keys a job that has not finished is counting on.

    Evicting one of these is not merely wasteful: a job mid-flight would find
    its upstream stage gone between stages, and a queued sweep would silently
    lose the shared SfM the whole sweep was built around.
    """
    rows = db.conn().execute(
        "SELECT DISTINCT s.cache_key FROM stages s JOIN jobs j ON j.id = s.job_id "
        f"WHERE j.state IN ({','.join('?' * len(LIVE_STATES))})",
        LIVE_STATES).fetchall()
    return {r["cache_key"] for r in rows}


# Evicted only after every other stage's entries. A finished job's train
# directory holds the only copy of its snapshots (project.licht) and its resume
# point, and is hours of GPU to rebuild; frames, select, mask, sfm and export
# come back from the clip or from it. LRU alone took a finished job's
# project.licht, nine snapshots, minutes after it ended (issue #28).
EVICT_LAST = ("train",)


def eviction_order(rows: list) -> list:
    """Cache rows oldest use first, with EVICT_LAST stages after the rest."""
    oldest_first = list(reversed(rows))             # list_cache is newest-first
    return sorted(oldest_first, key=lambda r: r["stage"] in EVICT_LAST)


def gc_cache(target_free: float = 0.0, budget: float | None = None,
             dry_run: bool = False) -> dict:
    """Evict cache entries, least recently used first, train entries last.

    Stops as soon as both conditions hold: total cache bytes within `budget`
    and free disk at or above `target_free`.
    """
    # None means "no ceiling" and 0 means "evict everything evictable", so the
    # two cannot be collapsed into a falsy check.
    if budget is None:
        budget = CACHE_BUDGET_BYTES or None
    protected = protected_keys()
    total = cache_total()
    evicted, freed, skipped = [], 0, 0
    # A dry run deletes nothing, so the disk never gets freer and every eligible
    # entry used to be listed. Count what the listed entries would free instead
    # (their recorded size: an upper bound, since select/ hardlinks frames/).
    would_free = 0

    for row in eviction_order(db.list_cache()):
        over_budget = budget is not None and total > budget
        short_of_free = free_bytes() + would_free < target_free
        if not over_budget and not short_of_free:
            break
        if row["cache_key"] in protected:
            skipped += 1
            continue
        d = Path(row["path"])
        if lock_holder_alive(read_lock(d)):
            skipped += 1
            continue
        if dry_run:
            evicted.append({"key": row["cache_key"], "stage": row["stage"],
                            "bytes": row["bytes"]})
            total -= int(row["bytes"] or 0)
            would_free += int(row["bytes"] or 0)
            continue
        before = free_bytes()
        try:
            if d.is_dir():
                shutil.rmtree(d)
        except OSError:
            skipped += 1
            continue
        db.conn().execute("DELETE FROM cache WHERE cache_key=?",
                          (row["cache_key"],))
        # Measure rather than trust the recorded size: select/ hardlinks its
        # panoramas out of frames/, so deleting one of the pair frees nothing
        # until the other goes too.
        freed += max(0, free_bytes() - before)
        total -= int(row["bytes"] or 0)
        evicted.append({"key": row["cache_key"], "stage": row["stage"],
                        "bytes": row["bytes"]})
    return {"evicted": evicted, "n_evicted": len(evicted), "freed": freed,
            "skipped_in_use": skipped, "cache_total": cache_total(),
            "free": free_bytes()}


def _prune_dir(d: Path, keep_days: float, patterns: tuple[str, ...]) -> dict:
    if keep_days <= 0 or not d.is_dir():
        return {"removed": 0, "bytes": 0}
    cutoff = time.time() - keep_days * 86400
    removed, freed = 0, 0
    for pat in patterns:
        for p in d.glob(pat):
            try:
                st = p.stat()
                if st.st_mtime >= cutoff or not p.is_file():
                    continue
                p.unlink()
            except OSError:
                continue
            removed += 1
            freed += st.st_size
    return {"removed": removed, "bytes": freed}


def prune_logs() -> dict:
    return _prune_dir(LOG_ROOT, LOG_KEEP_DAYS, ("*.log",))


def prune_renders() -> dict:
    # glob is not recursive, so the thumbs subdirectory needs naming: they are
    # small and regenerate on demand, but a sweep over trim values would
    # otherwise leave one per variant forever.
    return _prune_dir(RENDER_ROOT, RENDER_KEEP_DAYS,
                      ("cmp_*.jpg", "*.jpg", "thumbs/*.jpg"))


# Margin on the largest entry a stage has ever produced. The next frames dump
# can be bigger than the biggest so far (a longer clip, a higher fps), so the
# prior is scaled up rather than taken flat. 0 disables the history term and
# leaves the plain MIN_FREE_BYTES floor.
NEED_SAFETY = float(os.environ.get("QUEUE_NEED_SAFETY", 1.25))


def stage_need(stage: str) -> float:
    """What this stage is likely to want, measured rather than guessed.

    There is no a-priori size model here: an 8K frames dump is tens of GB and a
    downscaled one is hundreds of MB, and the same is true of a training
    directory. So the flat floor let through exactly the failure it was written
    to prevent -- a stage several times larger than MIN_FREE_BYTES passes a
    20 GB check, fills the disk, and dies 80% of the way in. The largest entry
    this stage has already produced is the honest prior; with no history yet
    there is nothing to say and the floor stands on its own.
    """
    if NEED_SAFETY <= 0:
        return 0.0
    row = db.conn().execute(
        "SELECT MAX(bytes) n FROM cache WHERE stage=?", (stage,)).fetchone()
    return float((row["n"] if row else 0) or 0) * NEED_SAFETY


def stage_floor(estimated: bool) -> float:
    """The flat floor, or, when the stage's own need is estimated and the floor
    was not set by hand, the headroom that goes on top of the estimate."""
    if estimated and not MIN_FREE_SET:
        return ESTIMATED_SLACK_BYTES
    return MIN_FREE_BYTES


def evictions_path(job_id: int) -> Path:
    return telemetry.run_dir(job_id) / "evictions.json"


def evictions(job_id: int) -> list[dict]:
    """Cache entries evicted to make room for this job's stages."""
    try:
        out = json.loads(evictions_path(job_id).read_text())
    except (OSError, ValueError):
        return []
    return [e for e in out if isinstance(e, dict)] if isinstance(out, list) else []


def record_evictions(job_id: int, stage: str, res: dict) -> None:
    """Keep what ensure_space evicted for `stage` with the job that needed it.

    It was a line in the service's stdout only, so the client of a job whose
    train directory went (issue #28) had no way to see it. Never raises: a
    record that cannot be written must not fail the stage.
    """
    rec = {"stage": stage, "at": round(time.time(), 1), "want": res.get("want"),
           "basis": res.get("basis_kind"), "free_before": res.get("free_before"),
           "freed": res["freed"], "evicted": res["evicted"]}
    try:
        p = evictions_path(job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(evictions(job_id) + [rec], indent=1))
        os.replace(tmp, p)
    except OSError as exc:
        print(f"job {job_id}: evictions before {stage} not recorded: {exc}")


def ensure_space(stage: str, estimate: float | None = None,
                 job_id: int | None = None) -> dict | None:
    """Preflight before a stage writes. Frees what it can, then refuses.

    Wants the largest of: the floor (stage_floor), the stage's history
    (stage_need) and the caller's estimate of what it will write. Raising here
    costs nothing; discovering it at 80% of a 95-minute training run costs the
    run, and leaves a truncated export behind for the finalizer to reject.

    Returns the eviction it made (gc_cache's result plus `want`, `basis_kind`
    and `free_before`) when it evicted anything, else None; with a job id, it
    is also recorded with the job (record_evictions).
    """
    floor = stage_floor(estimate is not None)
    hist = stage_need(stage)
    est = estimate or 0.0
    headroom = estimate is not None and not MIN_FREE_SET
    # An estimated stage wants its estimate plus the headroom; history still
    # wins when this stage has written more before.
    want = max(hist, est + floor) if headroom else max(floor, hist, est)
    if want == hist and hist > (est + floor if headroom else max(floor, est)):
        kind, basis = "history", f"{NEED_SAFETY:g}x the largest {stage} entry on record"
    elif headroom:
        kind, basis = "estimate", (f"the estimate for this job's {stage}, {est/GB:.1f} GB, "
                                   f"plus {floor/GB:g} GB headroom")
    elif want == est and est > floor:
        kind, basis = "estimate", f"the estimate for this job's {stage}"
    else:
        kind, basis = "floor", f"the {MIN_FREE_BYTES/GB:.0f} GB floor"
    before = free_bytes()
    if before >= want:
        return None
    res = gc_cache(target_free=want)
    made = None
    if res["n_evicted"]:
        made = {**res, "want": round(want), "basis_kind": kind, "free_before": before}
        print(f"cache GC before {stage}{'' if job_id is None else f' of job {job_id}'}: "
              f"evicted {res['n_evicted']} entries "
              f"({', '.join(e['stage'] + ' ' + e['key'] for e in res['evicted'])}), "
              f"freed {res['freed']/GB:.1f} GB toward {want/GB:.1f} GB ({basis})")
        if job_id is not None:
            record_evictions(job_id, stage, made)
    prune_logs()
    prune_renders()
    have = free_bytes()
    if have < want:
        raise RuntimeError(
            f"only {have/GB:.1f} GB free on {QUEUE_ROOT} and {stage} needs at "
            f"least {want/GB:.1f} GB ({basis}); {res['skipped_in_use']} cache "
            f"entries could not be evicted because unfinished jobs depend on "
            f"them. Free space, lower QUEUE_MIN_FREE_GB or QUEUE_NEED_SAFETY, "
            f"or clear the cache")
    return made


def status() -> dict:
    return {"free": free_bytes(), "cache_total": cache_total(),
            "min_free": MIN_FREE_BYTES,
            "min_free_set": MIN_FREE_SET,
            "estimated_slack": ESTIMATED_SLACK_BYTES,
            "cache_budget": CACHE_BUDGET_BYTES or None,
            "log_keep_days": LOG_KEEP_DAYS,
            "render_keep_days": RENDER_KEEP_DAYS}
