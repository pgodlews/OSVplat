"""How far a running job has got, and how long it has left.

Every number here is computed with the SERVER's clock and shipped as a
duration, never a timestamp. The browser runs on a different machine to the
service, and skew between the two would otherwise surface as an elapsed time
that is minutes wrong -- for a two-hour job, silently.

Three qualities of answer, and the UI is told which one it got:
  * `measured` -- the stage reports a real counter (frames, mask, sfm, train)
  * `plan`     -- no counter, so elapsed against estimate.estimate()
  * `none`     -- nothing to go on yet
"""
from __future__ import annotations

import json
import time
from typing import Optional

from . import estimate
from .jobs import JobConfig
from .stages import ORDER

# Share of the SfM stage each phase accounts for. Measured on the spherical
# path (job 52: 22 s extraction, 6 s matching, the rest mapping) and on the
# perspective path from the project notes (1.6 / 5.5 / 19 min). Those two disagree by
# a lot, so this is a compromise rather than a constant: mapping dominates
# either way, which is the part that matters for where the bar sits.
SFM_PHASE_SPAN = {"extract": (0.00, 0.10),
                  "match":   (0.10, 0.25),
                  "map":     (0.25, 1.00)}

PHASE_LABEL = {"extract": "features", "match": "matching", "map": "mapping",
               "loading": "loading dataset", "training": "training"}

# Under this fraction the trainer's iteration rate is still several times its
# steady-state value -- densification has not yet grown the splat count -- and
# any remaining-time number is optimistic by enough to be worth not showing.
TRAIN_ETA_FLOOR = 0.05


def _clamp(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _plan_of(row) -> dict:
    try:
        return json.loads(row["plan"]) if row["plan"] else {}
    except (ValueError, TypeError, IndexError, KeyError):
        return {}


# ------------------------------------------------------------------- stages

def stage_fraction(stage: str, prog: Optional[dict], plan: dict) -> Optional[float]:
    """0..1 through one stage from its own reported counters, or None."""
    prog = prog or {}
    done, total = prog.get("done"), prog.get("total")

    if stage == "frames":
        # Position in the SOURCE clip, not output frames: the denominator for
        # frames is fps-dependent, and the clip length is what the plan used.
        pos = prog.get("clip_pos_s")
        clip = (plan.get("_assumptions") or {}).get("clip_seconds")
        if prog.get("finished"):
            return 1.0
        return _clamp(pos / clip) if pos is not None and clip else None

    if stage == "mask":
        return _clamp(done / total) if done is not None and total else None

    if stage == "sfm":
        span = SFM_PHASE_SPAN.get(prog.get("phase"))
        if span is None:
            return None
        lo, hi = span
        within = _clamp(done / total) if done is not None and total else 0.0
        frac = lo + (hi - lo) * within
        # The fisheye rig reconstructs twice; each pass runs the three phases
        # again, so the pass decides which share of the bar this one is.
        passes, at = prog.get("sfm_passes"), prog.get("sfm_pass")
        if passes and at:
            return _clamp(((at - 1) + frac) / passes)
        return frac

    if stage == "train":
        step, total = prog.get("step"), prog.get("total")
        if step is None or not total:
            # The dataset load reports nothing and can run for minutes.
            return 0.0 if prog.get("phase") == "loading" else None
        return _clamp(step / total)

    return None                       # select and export are seconds long


def stage_detail(stage: str, prog: Optional[dict]) -> str:
    """The short human string under the bar: what it is counting, and to what."""
    prog = prog or {}
    phase = PHASE_LABEL.get(prog.get("phase"), "")
    if stage == "train":
        if prog.get("phase") == "loading":
            n = prog.get("images")
            return f"loading dataset{f' ({n} images)' if n else ''}"
        step, total = prog.get("step"), prog.get("total")
        if step is not None and total:
            bits = [f"{step}/{total}"]
            if prog.get("splats"):
                bits.append(f"{prog['splats'] / 1e6:.2f}M splats")
            return " · ".join(bits)
        return ""
    done, total = prog.get("done"), prog.get("total")
    bits = [b for b in (phase, f"{done}/{total}" if done is not None and total
                        else "") if b]
    if stage == "sfm" and prog.get("sfm_passes"):
        bits.insert(0, f"pass {prog.get('sfm_pass')}/{prog['sfm_passes']}")
    if stage == "sfm" and prog.get("restarts"):
        # Otherwise the bar just slides backwards and says nothing about why.
        bits.append(f"restarted {prog['restarts']}x")
    if stage == "frames" and prog.get("speed"):
        bits.append(f"{prog['speed']:g}x realtime")
    return " · ".join(bits)


def stage_remaining(stage: str, cfg: JobConfig, prog: Optional[dict],
                    plan: dict, elapsed: float) -> tuple[Optional[float], str]:
    """Seconds left in a stage that is running now, and where that came from.

    The quality of the bar and the quality of the countdown are not the same
    thing -- the trainer's dataset load has a position (nought) but no useful
    remaining time -- so the source is reported for the number, not the bar.
    """
    planned = plan.get(stage)
    frac = stage_fraction(stage, prog, plan)

    if stage == "train":
        # NOT elapsed x (1-f)/f, which is what the trainer's own ETA does and
        # why it reads 2-4x optimistic for the first half of a run: iterations
        # get steadily slower as densification grows the splat count from ~40k
        # to the cap. The fitted constant is a whole-run average over completed
        # jobs, so that slowdown is already inside it.
        prog = prog or {}
        step, total = prog.get("step"), prog.get("total")
        if step is None or not total:
            return planned, ("plan" if planned is not None else "none")
        if step / total < TRAIN_ETA_FLOOR:
            return None, "none"            # too early to mean anything
        return max(0.0, (total - step) / estimate.train_rate(cfg)), "measured"

    plan_rem = max(0.0, planned - elapsed) if planned is not None else None
    if frac is None or frac <= 0.0:
        return plan_rem, ("plan" if plan_rem is not None else "none")
    observed = elapsed * (1.0 - frac) / frac
    if plan_rem is None:
        return observed, "measured"
    # Trust the plan while the stage has barely started, and what it is
    # actually doing by the time it is half done.
    return observed * frac + plan_rem * (1.0 - frac), "measured"


# ---------------------------------------------------------------------- job

FINISHED = ("done", "cached", "skipped")


def job_progress(row, stages: list[dict], now: Optional[float] = None) -> dict:
    """Whole-job position, elapsed and remaining, from its stage rows.

    `stages` is the list already assembled for the API response, so this does
    not go back to the database.
    """
    now = now or time.time()
    plan = _plan_of(row)
    state = row["state"]

    elapsed = None
    if row["started"]:
        elapsed = (row["ended"] or now) - row["started"]

    out = {"elapsed": round(elapsed, 1) if elapsed is not None else None,
           "stage": None, "detail": "", "fraction": None,
           "remaining": None, "finish_at": None, "source": "none",
           "plan_total": plan.get("total")}

    if state in ("done", "failed", "cancelled"):
        out["fraction"] = 1.0 if state == "done" else None
        return out

    try:
        cfg = JobConfig.model_validate(json.loads(row["config"]))
    except (ValueError, TypeError):
        return out

    # Weight each stage by what it was predicted to cost, counting only the
    # stages this job will actually run: a job whose frames and select came from
    # the cache is not 40% done before it starts.
    by_stage = {s["stage"]: s for s in stages}
    weights, live = {}, None
    for name in ORDER:
        st = by_stage.get(name)
        if st is None or st["state"] in ("cached", "skipped"):
            continue
        weights[name] = max(0.0, plan.get(name) or 0.0)
        if st["state"] == "running":
            live = st

    total_w = sum(weights.values())
    if live is None:
        # Queued, or between stages. Everything still to run is still to run.
        out["stage"] = next((n for n in ORDER
                             if (by_stage.get(n) or {}).get("state")
                             in ("pending", "waiting")), None)
        done_w = sum(w for n, w in weights.items()
                     if (by_stage.get(n) or {}).get("state") in FINISHED)
        out["fraction"] = round(done_w / total_w, 4) if total_w else None
        rest = total_w - done_w
        out["remaining"] = round(rest, 1) if plan else None
        out["source"] = "plan" if plan else "none"
    else:
        name = live["stage"]
        prog = live["progress"]
        st_elapsed = (now - live["started"]) if live["started"] else 0.0
        frac = stage_fraction(name, prog, plan)
        rem, out["source"] = stage_remaining(name, cfg, prog, plan, st_elapsed)

        out["stage"] = name
        out["detail"] = stage_detail(name, prog)

        done_w = sum(w for n, w in weights.items()
                     if (by_stage.get(n) or {}).get("state") in FINISHED)
        if total_w:
            out["fraction"] = round(
                (done_w + weights.get(name, 0.0) * (frac or 0.0)) / total_w, 4)

        later = sum(w for n, w in weights.items()
                    if ORDER.index(n) > ORDER.index(name))
        if rem is not None:
            out["remaining"] = round(rem + later, 1)

    if out["remaining"] is not None:
        out["finish_at"] = round(now + out["remaining"], 1)
    return out


# -------------------------------------------------------------------- queue

def _occupancy(job: dict) -> float:
    """How long a running job will keep its slot.

    A job whose own ETA is not meaningful yet -- the trainer below the floor,
    say -- still occupies a GPU, and `remaining or 0.0` said it did not. The
    next job in line was told it started immediately and then sat queued for
    the better part of an hour, with the countdown insisting otherwise.
    """
    e = job["eta"]
    if e.get("remaining") is not None:
        return e["remaining"]
    return max(0.0, (e.get("plan_total") or 0.0) - (e.get("elapsed") or 0.0))


def queue_eta(jobs: list[dict], max_concurrent: int,
              now: Optional[float] = None, *, paused: bool = False) -> dict:
    """Fill in `starts_in` per queued job, and say when the queue drains.

    Mirrors what the dispatcher will actually do: jobs come off in JOB_ORDER
    (which is the order `jobs` already arrives in) and each takes the slot that
    frees up first.
    """
    now = now or time.time()
    running = [_occupancy(j) for j in jobs if j["state"] == "running"]
    pending = [j for j in jobs if j["state"] in ("running", "queued")]
    queued = [j for j in jobs if j["state"] == "queued"]
    if queued and (paused or max_concurrent <= 0):
        for j in queued:
            j["eta"].update(starts_in=None, remaining=None, finish_at=None)
        return {"jobs": len(pending), "drains_in": None, "drains_at": None}

    # Retire the earliest finishing excess slots when concurrency shrinks.
    # Their jobs still finish, but cannot be replaced until the running count
    # falls below the new ceiling.
    n = max(0, max_concurrent)
    slots = sorted(running, reverse=True)[:n]
    slots += [0.0] * (n - len(slots))

    for j in jobs:
        if j["state"] != "queued":
            continue
        i = min(range(len(slots)), key=lambda k: slots[k])
        starts_in = slots[i]
        cost = (j["eta"]["plan_total"] or 0.0)
        slots[i] = starts_in + cost
        j["eta"]["starts_in"] = round(starts_in, 1)
        if cost:
            j["eta"]["remaining"] = round(starts_in + cost, 1)
            j["eta"]["finish_at"] = round(now + starts_in + cost, 1)

    drain = max(running + slots, default=0.0)
    return {"jobs": len(pending),
            "drains_in": round(drain, 1),
            "drains_at": round(now + drain, 1) if pending else None}
