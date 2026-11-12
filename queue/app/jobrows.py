"""A new job's stage rows, written with the job.

Every way a job is created writes them: the API (main.py) and a resume from a
restore point (checkpoints.py). A handoff import writes its own (its upstream
stages are "imported"). Without them a queued job is invisible to
retention.protected_keys(), so the cache entries it is waiting to use could be
evicted under it, and the UI shows it with no stages.
"""
from __future__ import annotations

from . import db
from .config import CACHE_ROOT
from .jobs import JobConfig
from .stages import ORDER, is_cached


def stage_rows(job_id: int, cfg: JobConfig) -> None:
    """A row per stage: cached, pending, or skipped past run_until."""
    last = ORDER.index(cfg.run_until) if cfg.run_until else len(ORDER)
    for stage, key in cfg.keys().items():
        d = CACHE_ROOT / stage / key
        state = ("skipped" if ORDER.index(stage) > last
                 else "cached" if is_cached(d) else "pending")
        db.upsert_stage(job_id, stage, key, state, path=str(d))
