"""Stage definitions: cache dirs, argv construction, output parsing, finalizers.

Stage order: frames -> select -> mask -> sfm -> train -> export.
A raw dual-fisheye .OSV runs the same stages with their own implementations
(see "fisheye rig" below); the dispatch happens in the STAGES registry.
Each stage writes its artifacts into CACHE_ROOT/<stage>/<cache_key>/ and drops a
.done marker on success; a stage whose marker exists is reused verbatim.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shlex
import time
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import (CACHE_ROOT, FISHEYE_FRAMES, FISHEYE_MASKS, FISHEYE_SFM,
                     FISHEYE_TRAIN, GS_PY, LFS_BIN, OSV_META, PERSON_MASKS,
                     MASK_PY, MODELS_ROOT, RUN_SFM, SELECT_SHARP, SCRIPTS,
                     SFM_PY, SPLAT_ROOT)
from .jobs import JobConfig
from .mask_backends import MASK_BACKENDS, weights_dir

# mask reads the selected panoramas and feeds BOTH sfm (suppressing features on
# people) and train (excluding them from the photometric loss), so it has to run
# before sfm, not merely be displayed there.
ORDER = ["frames", "select", "mask", "sfm", "train", "export"]
PICK_MODEL = SCRIPTS / "32_pick_model.py"


@dataclass
class Ctx:
    """Everything a stage needs, plus values resolved by earlier stages."""
    job_id: int
    cfg: JobConfig
    gpu: int
    keys: dict[str, str]
    derived: dict = field(default_factory=dict)

    def dir(self, stage: str) -> Path:
        return CACHE_ROOT / stage / self.keys[stage]


def done_marker(d: Path) -> Path:
    return d / ".done"


def lock_path(d: Path) -> Path:
    return d / ".lock"


def proc_ident(pid: int) -> str:
    """A stable identity for a pid: the kernel's start time for that process.

    A pid on its own is not an identity. A lock file outlives the process that
    wrote it, pids are reused, and reconcile() signals whatever it finds --- so
    without this a restart hours later can SIGTERM an unrelated process that
    happens to have inherited the number. Empty when it cannot be determined,
    in which case callers fall back to a liveness check alone.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
        # comm can contain spaces and parens, so fields start after the LAST ')'.
        fields = data[data.rfind(b")") + 2:].split()
        return fields[19].decode()           # starttime, field 22 overall
    except (OSError, IndexError, UnicodeDecodeError):
        pass
    try:                                     # macOS and anything without /proc
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def lock_holder_alive(lk: dict) -> bool:
    """Is the process that wrote this lock still the process running now?"""
    pid = lk.get("pid")
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if not pid_alive(pid):
        return False
    ident = lk.get("ident") or ""
    if not ident:
        return True                          # pre-identity lock, or unknowable
    current = proc_ident(pid)
    return not current or current == ident


def take_lock(d: Path, pid: int) -> bool:
    """Claim the cache dir. False if somebody else already holds it.

    O_CREAT|O_EXCL, not write_text: two jobs of one sweep reach the same shared
    stage within milliseconds of each other, and a plain write lets both of them
    "take" the lock and then write the same directory concurrently.
    """
    d.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": pid, "ident": proc_ident(pid),
                          "at": time.time()}).encode()
    try:
        fd = os.open(lock_path(d), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError:
        return False
    with os.fdopen(fd, "wb") as fh:
        fh.write(payload)
    return True


def restamp_lock(d: Path, pid: int) -> None:
    """Rewrite a lock we already hold, e.g. with the real child pid."""
    lock_path(d).write_text(json.dumps({"pid": pid, "ident": proc_ident(pid),
                                        "at": time.time()}))


def release_lock(d: Path) -> None:
    try:
        lock_path(d).unlink()
    except OSError:
        pass


def read_lock(d: Path) -> dict:
    try:
        return json.loads(lock_path(d).read_text())
    except (OSError, ValueError):
        return {}


def pid_state(pid: int) -> str:
    """One-letter process state, or "" when it cannot be determined."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
        # comm can contain spaces and parens, so fields start after the LAST ')'.
        return data[data.rfind(b")") + 2:].split()[0].decode()[:1]
    except (OSError, IndexError, UnicodeDecodeError):
        pass
    try:                                     # macOS and anything without /proc
        out = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()[:1]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def pid_alive(pid: int) -> bool:
    """Is this pid a process that can still do anything?

    A zombie answers os.kill(pid, 0) exactly like a live process, so a naive
    check reports a killed-but-unreaped stage subprocess as still running --
    which made restart recovery hold back a lock it should have released, and
    would let a dead lock holder block a shared stage until the wait timed out.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                          # someone else's: alive, not ours
    except (OSError, TypeError):
        return False
    return pid_state(pid) != "Z"


def is_cached(d: Path) -> bool:
    """A cache hit needs a marker that PARSES, not merely one that exists.

    mark_done used to write in place, so a service killed mid-write left a
    truncated .done that every later job read as a completed stage.
    """
    m = done_marker(d)
    if not m.is_file():
        return False
    try:
        json.loads(m.read_text())
    except (OSError, ValueError):
        return False
    return True


def mark_done(d: Path, info: dict) -> None:
    """Publish the marker atomically: full file or no file, never a stub."""
    tmp = d / ".done.tmp"
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, done_marker(d))


def read_done(d: Path) -> dict:
    try:
        return json.loads(done_marker(d).read_text())
    except (OSError, ValueError):
        return {}


def reset_stage_dir(d: Path) -> None:
    """Empty a cache dir we are about to (re)build, keeping only our lock.

    A stage that failed or was cancelled leaves its partial output behind, and
    the next attempt with the same key writes into it. That mixes runs: old
    checkpoints stay next to new ones (and train_finalize then has to guess
    which is current), and 20_select_sharp.py skips any destination file that
    already exists, so a rebuild after a window change keeps the OLD picks.
    Only ever called while holding the lock and with no valid .done present.
    """
    if not d.is_dir():
        return
    for p in d.iterdir():
        if p.name == ".lock":
            continue
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
        except OSError:
            pass


def dir_bytes(d: Path) -> int:
    """Bytes held by a cache dir, charging hardlinked files proportionally.

    20_select_sharp.py hardlinks the chosen panoramas out of the frames dir, so
    counting full size in both places double-counts and makes the cache total
    meaningless for GC decisions. Dividing each file by its link count keeps the
    sum over all cache dirs equal to the real disk usage.
    """
    total = 0.0
    for p in d.rglob("*"):
        if p.is_symlink() or not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        total += st.st_size / max(1, st.st_nlink)
    return int(total)


# ------------------------------------------------------------------ frames

def frames_argv(ctx: Ctx) -> list[str]:
    cfg = ctx.cfg
    src = SPLAT_ROOT / cfg.input.file
    out = ctx.dir("frames")
    out.mkdir(parents=True, exist_ok=True)
    # -progress writes machine-readable key=value blocks to stdout, which is
    # otherwise unused here (the frames go to files). -nostats drops the
    # human progress line it would otherwise smear across stderr. Without this
    # the longest cheap stage in the pipeline reported nothing at all.
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-progress", "pipe:1", "-nostats",
            "-hwaccel", "cuda"]
    # -ss before -i is a fast keyframe seek; -t after -i bounds the decode.
    if cfg.input.trim_start:
        argv += ["-ss", f"{cfg.input.trim_start:g}"]
    argv += ["-i", str(src)]
    if cfg.input.trim_end is not None:
        dur = cfg.input.trim_end - (cfg.input.trim_start or 0.0)
        argv += ["-t", f"{dur:g}"]
    argv += ["-vf", f"fps={cfg.frames.fps:g}",
             "-q:v", str(cfg.frames.jpeg_q),
             # Six digits, not five: 99999 frames is under 3 hours at the
             # default 10 fps, and ffmpeg silently wraps past the width.
             str(out / "%06d.jpg")]
    return argv


# ffmpeg -progress emits one key=value per line, repeating a block every
# second. Only four of the keys are worth anything here; the rest is noise.
FFMPEG_PROGRESS = re.compile(r"^(frame|out_time_us|speed|progress)=(.+)$")


def frames_parse(line: str, prev: dict) -> dict:
    """Position in the SOURCE clip, which is what the fraction is taken against.

    Frame count alone cannot give a percentage: the output frame count is
    fps x duration, and a client that changed fps after the plan was written
    would be measured against the wrong denominator. Elapsed clip seconds is
    the same number the estimate was built from.
    """
    m = FFMPEG_PROGRESS.match(line.strip())
    if not m:
        return {}
    key, val = m.group(1), m.group(2).strip()
    if key == "progress":
        return {"finished": True} if val == "end" else {}
    if key == "frame":
        return {"frames_out": int(val)} if val.isdigit() else {}
    try:
        # Both are "N/A" until the first frame is muxed.
        if key == "out_time_us":
            return {"clip_pos_s": round(int(val) / 1e6, 2)}
        return {"speed": float(val.rstrip("x"))}
    except ValueError:
        return {}


def frames_finalize(ctx: Ctx) -> dict:
    out = ctx.dir("frames")
    n = len(list(out.glob("*.jpg")))
    if n == 0:
        raise RuntimeError("frame extraction produced no JPEGs")
    ctx.derived["n_candidates"] = n
    return {"candidates": n}


def frames_verify(ctx: Ctx, info: dict) -> None:
    """A .done marker is a claim about the directory; check the claim holds."""
    want = info.get("candidates")
    n = len(list(ctx.dir("frames").glob("*.jpg")))
    if want and n != want:
        raise RuntimeError(f"cache claims {want} candidate frames, found {n}")
    if not n:
        raise RuntimeError("cached frames directory is empty")


# ------------------------------------------------------------------ select

def select_prepare(ctx: Ctx) -> None:
    """Resolve target-pano mode into a concrete sharpness window."""
    sel = ctx.cfg.select
    n = ctx.derived.get("n_candidates") or len(
        list(ctx.dir("frames").glob("*.jpg")))
    ctx.derived["n_candidates"] = n
    if sel.mode == "target":
        w = max(1, round(n / max(1, sel.target_panos)))
    else:
        w = sel.window
    ctx.derived["window"] = w


def select_argv(ctx: Ctx) -> list[str]:
    out = ctx.dir("select")
    out.mkdir(parents=True, exist_ok=True)
    return [str(SFM_PY), str(SELECT_SHARP), str(ctx.derived["window"]),
            str(ctx.dir("frames")), str(out)]


def select_finalize(ctx: Ctx) -> dict:
    n = len(list(ctx.dir("select").glob("pano_*.jpg")))
    if n < 8:
        raise RuntimeError(f"only {n} panoramas selected; need at least 8")
    ctx.derived["n_panos"] = n
    return {"panos": n, "window": ctx.derived["window"]}


def select_verify(ctx: Ctx, info: dict) -> None:
    want = info.get("panos")
    n = len(list(ctx.dir("select").glob("pano_*.jpg")))
    if want and n != want:
        raise RuntimeError(f"cache claims {want} panoramas, found {n}")
    if n < 8:
        raise RuntimeError(f"cached selection holds only {n} panoramas")


# -------------------------------------------------------------------- mask

def mask_argv(ctx: Ctx) -> list[str]:
    out = ctx.dir("mask")
    out.mkdir(parents=True, exist_ok=True)
    m = ctx.cfg.mask
    # Each backend runs under its own interpreter; the script imports its model
    # library lazily so it starts in either venv.
    return [str(MASK_PY), str(PERSON_MASKS),
            "--panos", str(ctx.dir("select")),
            "--out", str(out / "masks"),
            "--overlay", str(out / "overlay"),
            "--colmap-out", str(out / "masks_colmap"),
            "--sheet", str(out / "review_sheet.jpg"),
            "--summary", str(out / "summary.json"),
            *_masker_options(m)]


def _masker_options(m) -> list[str]:
    """70_person_masks.py options that decide WHAT gets masked.

    Shared by the stitched and the fisheye mask stages, so the two cannot drift
    apart on a backend option and hash one config into two different mask sets.
    """
    opts = ["--backend", m.backend, "--dilate", str(m.dilate),
            "--work-width", str(m.work_width)]
    if m.score is not None:
        opts += ["--score", f"{m.score:g}"]
    # Argument order is part of the stage's command line and log, not its cache
    # key, but keep it as it was for sam3 so old logs still compare line by line.
    if MASK_BACKENDS[m.backend]["prompts"]:
        opts += ["--prompts", ",".join(m.prompts)]
    wdir = weights_dir(m.backend, MODELS_ROOT)
    if wdir is not None:
        opts += ["--model", str(wdir)]
    if MASK_BACKENDS[m.backend]["prompts"]:
        opts += ["--nadir-view"] if m.nadir_view else ["--no-nadir-view"]
    return opts


# 70_person_masks.py prints "  62/110  43s" as it goes.
MASK_PROGRESS = re.compile(r"^\s*(\d+)/(\d+)\s+\d+s\s*$")


def mask_parse(line: str, prev: dict) -> dict:
    m = MASK_PROGRESS.match(line)
    return {"done": int(m.group(1)), "total": int(m.group(2))} if m else {}


def mask_finalize(ctx: Ctx) -> dict:
    """Record coverage, and refuse a mask set that plainly did not work.

    The failure this guards against is silent: the detector finds nobody, every
    mask comes out fully white, training proceeds exactly as if masking were
    off, and the run looks like evidence that masking does not help. A mask
    stage that masked nothing is a failure, not a result.
    """
    d = ctx.dir("mask") / "masks"
    n = len(list(d.glob("*.png"))) if d.is_dir() else 0
    n_panos = ctx.derived.get("n_panos") or len(
        list(ctx.dir("select").glob("pano_*.jpg")))
    if n != n_panos:
        raise RuntimeError(f"wrote {n} masks for {n_panos} panoramas")

    summary = json.loads((ctx.dir("mask") / "summary.json").read_text()) \
        if (ctx.dir("mask") / "summary.json").is_file() else {}
    cov = summary.get("coverage_solid_angle_mean", 0.0)
    empty = summary.get("frames_without_detection") or []
    if cov <= 0.0:
        raise RuntimeError(
            "person masking found nobody in any frame; training would be "
            "identical to an unmasked run. Check the overlay images")
    n_colmap = len(list((ctx.dir("mask") / "masks_colmap").glob("*.png"))) \
        if (ctx.dir("mask") / "masks_colmap").is_dir() else 0
    if ctx.cfg.mask.use_for_sfm and n_colmap != n_panos:
        raise RuntimeError(
            f"wrote {n_colmap} COLMAP-named masks for {n_panos} panoramas; "
            f"COLMAP treats a missing mask as 'extract everywhere', so this "
            f"would be a partially masked reconstruction")
    summary["masks"] = n
    summary["colmap_masks"] = n_colmap
    summary["warnings"] = (
        [f"{len(empty)} frame(s) had no person detected: "
         f"{', '.join(empty[:6])}"] if empty else [])
    return summary


def mask_verify(ctx: Ctx, info: dict) -> None:
    d = ctx.dir("mask") / "masks"
    want = info.get("masks")
    n = len(list(d.glob("*.png"))) if d.is_dir() else 0
    if want and n != want:
        raise RuntimeError(f"cached mask set claims {want} masks, found {n}")


def mask_skip(ctx: Ctx) -> bool:
    return not ctx.cfg.mask.enabled


# --------------------------------------------------------------------- sfm

def sfm_argv(ctx: Ctx) -> list[str]:
    out = ctx.dir("sfm")
    out.mkdir(parents=True, exist_ok=True)
    argv = [str(SFM_PY), str(RUN_SFM), ctx.cfg.sfm.render, ctx.cfg.sfm.mapper,
            str(ctx.dir("select")), str(out)]
    # Only the spherical path can take them; the perspective renders generate
    # their own per-virtual-camera masks and overwriting that breaks the rig.
    # 30_run_sfm.py refuses the combination rather than silently ignoring it.
    if ctx.cfg.mask.enabled and ctx.cfg.mask.use_for_sfm:
        argv += ["--masks", str(ctx.dir("mask") / "masks_colmap")]
    return argv


def sfm_output_dir(ctx: Ctx) -> Path:
    return ctx.dir("sfm") / f"sfm_{ctx.cfg.sfm.render}_{ctx.cfg.sfm.mapper}"


def images_dir(cfg: JobConfig, sfm_dir: Path, select_dir: Path) -> Path:
    """The directory the reconstruction's image NAMES refer to.

    Only the spherical path reconstructs the panoramas themselves. The two
    perspective modes make 12 pinhole views per panorama into
    <sfm>/sfm_<render>_<mapper>/images/ and register those, so a model chosen
    from that run names pinhole files. Passing the panorama directory as
    --images (which is what happened) hands the trainer a set of names its
    dataset does not contain: 8K equirects matched against PINHOLE intrinsics.
    """
    if cfg.is_fisheye:
        # The rig dataset renames lens0/frame_0001.jpg to lens0_frame_0001.jpg
        # (LichtFeld matches masks by stem across the whole dataset) and links
        # the images under that name.
        return sfm_dir / "dataset" / "images"
    if cfg.sfm.render == "spherical":
        return select_dir
    return sfm_dir / f"sfm_{cfg.sfm.render}_{cfg.sfm.mapper}" / "images"


def _sfm_warnings(summary: dict, n_panos: int) -> list[str]:
    """Guardrails: warn loudly rather than silently burning 90 minutes.

    Shared by the stitched and fisheye SfM finalizers; both summaries carry the
    same diagnostic fields (the fisheye one counts rig frames, not images).
    """
    warnings = []
    if summary["registration_pct"] < 90:
        warnings.append(
            f"only {summary['num_reg_frames']}/{n_panos} frames registered "
            f"({summary['registration_pct']:.0f}%)")
    bod = summary.get("baseline_over_depth")
    if bod is not None and bod < 0.5:
        warnings.append(
            f"baseline/median-depth {bod:.2f} < 0.5 — scene is mostly at "
            f"infinity and will reconstruct as haze")
    if summary.get("mean_reproj") and summary["mean_reproj"] > 2.0:
        warnings.append(
            f"mean reprojection error {summary['mean_reproj']:.2f} px is high")
    if len(summary.get("models", [])) > 1:
        warnings.append(
            f"SfM returned {len(summary['models'])} models; using "
            f"sparse/{summary['chosen']} with {summary['num_reg_frames']} frames")
    # A stray registration is not a lost frame, it is a corrupted scene scale for
    # every frame -- worth saying out loud even though it has already been fixed.
    dropped = summary.get("dropped_cameras") or []
    if dropped:
        names = ", ".join(f"{d['name']} ({d['num_points3D']} pts)" for d in dropped[:4])
        warnings.append(
            f"dropped {len(dropped)} camera(s) registered far outside the "
            f"trajectory: {names}")
    if summary.get("outlier_note"):
        warnings.append(summary["outlier_note"])
    return warnings


MIN_STITCHED_REGISTRATION_PCT = 50


def sfm_finalize(ctx: Ctx) -> dict:
    """Pick the largest model, link it as sparse/0, capture the diagnostic."""
    sfm_dir = sfm_output_dir(ctx)
    dataset = ctx.dir("sfm") / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([str(SFM_PY), str(PICK_MODEL), str(sfm_dir),
                           str(dataset)],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        # 32_pick_model.py reports its own errors as JSON on stdout; stderr is
        # usually empty, which used to leave this message ending in a colon.
        err = proc.stderr.strip()
        try:
            err = json.loads(proc.stdout.strip().splitlines()[-1]).get("error") or err
        except (ValueError, IndexError, AttributeError):
            err = err or proc.stdout.strip()[-500:]
        if err == "no valid reconstruction models":
            raise RuntimeError(
                "SfM could not start a reconstruction: no two frames overlapped "
                "enough. The selected frames are probably too far apart -- raise "
                "Candidate fps, lower the Sharpness window, or use a longer clip.")
        raise RuntimeError(f"model selection failed: {err[:500]}")
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    if "error" in summary:
        raise RuntimeError(summary["error"])

    n_panos = ctx.derived.get("n_panos") or len(
        list(ctx.dir("select").glob("pano_*.jpg")))
    summary["n_panos"] = n_panos
    summary["registration_pct"] = (
        100.0 * summary["num_reg_frames"] / n_panos if n_panos else 0.0)

    summary["warnings"] = _sfm_warnings(summary, n_panos)
    # A fragment is a failure, not a warning: 2 of 100 frames used to pass,
    # get cached, and train for an hour on a sliver of the scene. The fisheye
    # path refuses under 90% (85_fisheye_dataset.py); a stitched clip whose
    # later half fails to register still yields a usable partial scene, hence
    # the lower floor here.
    if summary["registration_pct"] < MIN_STITCHED_REGISTRATION_PCT:
        raise RuntimeError(
            f"SfM registered only {summary['num_reg_frames']} of {n_panos} "
            f"frames ({summary['registration_pct']:.0f}%, need "
            f"{MIN_STITCHED_REGISTRATION_PCT}%); the clip did not reconstruct "
            f"as one scene. Raise Candidate fps, lower the Sharpness window, "
            f"or trim to the part with steady, overlapping motion")

    imgs = images_dir(ctx.cfg, ctx.dir("sfm"), ctx.dir("select"))
    if not imgs.is_dir() or not any(imgs.iterdir()):
        raise RuntimeError(
            f"reconstruction images directory {imgs} is missing or empty; "
            f"training would have no pixels for these poses")
    summary["images_dir"] = str(imgs)
    ctx.derived["dataset"] = str(dataset)
    ctx.derived["images"] = str(imgs)
    ctx.derived["needs_gut"] = summary.get("needs_gut", False)
    return summary


def sfm_verify(ctx: Ctx, info: dict) -> None:
    """The dataset symlink is what LichtFeld reads; a dangling one trains on
    nothing and the failure surfaces 95 minutes later as an empty model."""
    link = ctx.dir("sfm") / "dataset" / "sparse" / "0"
    if not link.exists():
        raise RuntimeError(
            "cached SfM has no dataset/sparse/0 (deleted or dangling symlink)")
    if not any(link.glob("images.*")):
        raise RuntimeError("cached SfM model holds no images.bin/images.txt")


SFM_PATTERNS = [
    (re.compile(r"num_reg_frames\s*=\s*(\d+)"), "num_reg_frames", int),
    (re.compile(r"num_points3D\s*=\s*(\d+)"), "num_points3D", int),
    (re.compile(r"mean_reproj\w*\s*=?\s*([\d.]+)"), "mean_reproj", float),
]

# COLMAP runs three phases back to back and counts through each one, but the
# counters look nothing alike and only the first two carry their own total.
SFM_PANOS = re.compile(r"START .*\b(?:panos|frames)=(\d+)")
# 88_fisheye_sfm.py announces each of its two rig passes.
SFM_PASS = re.compile(r"FISHEYE PASS (\d+)/(\d+)")
SFM_EXTRACT = re.compile(r"Processed file \[(\d+)/(\d+)\]")
SFM_MATCH = re.compile(r"Processing image \[(\d+)/(\d+)\]")
SFM_MAP_START = re.compile(r"incremental_pipeline\.cc:\d+\] Loading database")
SFM_REGISTERING = re.compile(r"Registering image #\d+ \(num_reg_frames=(\d+)\)")

# Only ever moves forward. The mapper's own logging revisits earlier-sounding
# lines (it reloads the database when it restarts), and a phase label that
# flickers backwards reads as a stall.
PHASE_RANK = {"extract": 0, "match": 1, "map": 2}


def _forward(prev: dict, phase: str) -> str:
    at = prev.get("phase")
    return phase if PHASE_RANK[phase] >= PHASE_RANK.get(at, -1) else at


def sfm_parse(line: str, prev: dict) -> dict:
    out = {}
    for pat, key, cast in SFM_PATTERNS:
        m = pat.search(line)
        if m:
            try:
                out[key] = cast(m.group(1))
            except ValueError:
                pass

    m = SFM_PASS.search(line)
    if m:
        # Each pass counts through all three phases again. Let the phase start
        # over and have the pass number carry the forward motion instead, or the
        # second pass reads as a bar stuck at "mapping".
        out.update(sfm_pass=int(m.group(1)), sfm_passes=int(m.group(2)),
                   phase=None, done=0, total=0)
        return out

    m = SFM_PANOS.search(line)
    if m:
        out["panos"] = int(m.group(1))

    # Feature extraction and matching both count to a total they state.
    for pat, phase in ((SFM_EXTRACT, "extract"), (SFM_MATCH, "match")):
        m = pat.search(line)
        if m:
            out.update(phase=_forward(prev, phase),
                       done=int(m.group(1)), total=int(m.group(2)))
            return out

    m = SFM_REGISTERING.search(line)
    if m or SFM_MAP_START.search(line):
        # Mapping states no total of its own; the only denominator is the
        # panorama count 30_run_sfm.py logged when it started.
        entering = prev.get("phase") != "map"
        out.update(phase=_forward(prev, "map"),
                   total=prev.get("panos") or out.get("panos") or 0)
        if entering:
            # The matching phase counted to 110 too, and its tally is not this
            # phase's. Left in place it made the first registered frame look
            # like a mapper that had just thrown 109 frames away.
            out["done"] = 0
        if m:
            n = int(m.group(1))
            # COLMAP can discard a reconstruction and start over, taking its
            # frame counter back to 1. Count that, so the UI can say the mapper
            # restarted rather than show a bar sliding backwards unexplained.
            seen = prev.get("num_reg_frames")
            if not entering and seen is not None and n < seen:
                out["restarts"] = (prev.get("restarts") or 0) + 1
            out["done"] = n
    return out


# ------------------------------------------------------------------- train

def _resolve_gut(ctx: Ctx, camera_model: str) -> bool:
    """--gut when the camera model can only train through 3DGUT; refuse gut=false then.

    EQUIRECTANGULAR (and OPENCV_FISHEYE, used through it here) trains only
    through the 3DGUT path; without --gut it dies at iteration 0 with an opaque
    InvalidArgument/Training. The guardrail is documented as "--gut forced when
    the camera model needs it", but an explicit gut=false silently won, so a
    sweep with that axis burned a GPU slot to reach the same opaque failure.
    """
    t = ctx.cfg.train
    needs_gut = ctx.derived.get("needs_gut", False)
    if needs_gut and t.gut is False:
        raise RuntimeError(
            f"this reconstruction uses the {camera_model} camera model, which "
            f"LichtFeld can only train through --gut; train.gut=false would die "
            f"at iteration 0. Leave gut unset to let the camera model decide")
    gut = t.gut if t.gut is not None else needs_gut
    ctx.derived["gut"] = gut
    return gut


def _lichtfeld_argv(ctx: Ctx, dataset: str, images: str, gut: bool,
                    masked: bool) -> list[str]:
    cfg, t = ctx.cfg, ctx.cfg.train
    argv = [str(LFS_BIN), "--headless",
            "-d", dataset,
            "--images", images,
            "-o", str(ctx.dir("train")),
            "--strategy", t.strategy,
            "--max-cap", str(t.max_cap),
            "--sh-degree", str(t.sh_degree)]

    # Budget: steps_scaler scales iterations AND every schedule. Passing both
    # --iter and --steps-scaler double-scales, so emit exactly one of them.
    if t.steps_scaler != 1.0:
        argv += ["--steps-scaler", f"{t.steps_scaler:g}"]
    else:
        argv += ["--iter", str(t.iter)]

    if t.max_width:
        argv += ["--max-width", str(t.max_width)]
    if gut:
        argv.append("--gut")
    if t.eval:
        argv.append("--eval")
    if t.enable_mip:
        argv.append("--enable-mip")
    if t.background_improvements:
        argv.append("--background-improvements")
    if t.exposure_correction:
        argv.append("--exposure-correction")
    if t.bilateral_grid:
        argv.append("--bilateral-grid")
    if t.min_opacity is not None:
        argv += ["--min-opacity", f"{t.min_opacity:g}"]
    if t.max_screen_share is not None:
        argv += ["--max-screen-share", f"{t.max_screen_share:g}"]

    # Ignore mode zeroes the photometric weight where the mask is black, so
    # masked pixels contribute no gradient at all -- which is what we want for
    # a person who is never seen from a second viewpoint.
    if masked:
        argv += ["--mask-mode", "ignore"]

    fmts = ",".join(cfg.export.formats)
    if fmts:
        argv.append(f"--export={fmts}")

    if t.extra_args.strip():
        # shlex, not split(): --far-seed-dose "4000 5000" is one argument, and
        # str.split() would have handed the trainer two malformed ones.
        argv += shlex.split(t.extra_args)
    return argv


def train_argv(ctx: Ctx) -> list[str]:
    cfg = ctx.cfg
    out = ctx.dir("train")
    out.mkdir(parents=True, exist_ok=True)
    dataset = ctx.derived.get("dataset") or str(ctx.dir("sfm") / "dataset")
    images = ctx.derived.get("images") or str(
        images_dir(cfg, ctx.dir("sfm"), ctx.dir("select")))

    # LichtFeld looks for masks in <-d>/masks/, so a masked run needs a dataset
    # root that has both the reconstruction and the masks under it. That root
    # cannot be the SfM cache dir: the SfM is deliberately shared between masked
    # and unmasked variants, and dropping a masks/ folder into it would silently
    # mask every job that reuses that reconstruction. So build a private view,
    # under this train key, out of two symlinks.
    if cfg.mask.enabled:
        view = out / "dataset"
        (view / "sparse").mkdir(parents=True, exist_ok=True)
        for link, target in ((view / "sparse" / "0",
                              Path(dataset) / "sparse" / "0"),
                             (view / "masks", ctx.dir("mask") / "masks")):
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target.resolve(), target_is_directory=True)
        dataset = str(view)

    gut = _resolve_gut(ctx, "EQUIRECTANGULAR")
    return _lichtfeld_argv(ctx, dataset, images, gut, masked=cfg.mask.enabled)


TRAIN_PROGRESS = re.compile(
    r"(\d+)/(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Splats:\s*(\d+)")
# LichtFeld reports PSNR and SSIM only -- there is no LPIPS in its eval output
# (the LPIPS numbers in the project notes came from gsplat). Matching SSIM is optional
# so a format change cannot silently drop PSNR too.
TRAIN_EVAL = re.compile(
    r"PSNR:\s*([\d.]+)(?:.*?SSIM:\s*([\d.]+))?", re.IGNORECASE)
TRAIN_DONE = re.compile(r"completed in\s+([\d.]+)\s*(\w+)", re.IGNORECASE)


# Nothing is reported between launch and iteration 0, and at full resolution
# that gap is minutes -- long enough that a step-based bar sitting at zero looks
# like a hung job. It is why FIRST_PROGRESS_GRACE is half an hour.
TRAIN_LOADING = re.compile(r"Loading dataset from:")
TRAIN_IMAGES = re.compile(r"Training with (\d+) images")


def train_parse(line: str, prev: dict) -> dict:
    out = {}
    m = TRAIN_PROGRESS.search(line)
    if m:
        step, total = int(m.group(1)), int(m.group(2))
        out.update({"step": step, "total": total, "loss": float(m.group(3)),
                    "splats": int(m.group(4)), "phase": "training",
                    "pct": round(100.0 * step / total, 1) if total else 0.0})
    elif prev.get("step") is None:
        # Only before the first iteration: the trainer reloads nothing later,
        # but a stray match must never drag the phase back out of training.
        if TRAIN_LOADING.search(line):
            out["phase"] = "loading"
        m2 = TRAIN_IMAGES.search(line)
        if m2:
            out["images"] = int(m2.group(1))
    m = TRAIN_EVAL.search(line)
    if m:
        out["psnr"] = float(m.group(1))
        if m.group(2):
            out["ssim"] = float(m.group(2))
    return out


def read_metrics_csv(train_dir: Path) -> dict:
    """Parse LichtFeld's metrics.csv (written when --eval is on).

    Far more reliable than scraping the progress log: it is a real CSV with a
    header, one row per eval step. We take the last row.
    """
    path = train_dir / "metrics.csv"
    if not path.is_file():
        return {}
    try:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return {}
    if not rows:
        return {}
    last = rows[-1]
    out = {}
    for src, dst in (("psnr", "psnr"), ("ssim", "ssim"),
                     ("num_gaussians", "splats"), ("iteration", "final_step"),
                     ("time_per_image", "eval_s_per_image")):
        v = (last.get(src) or "").strip()
        if v:
            try:
                out[dst] = float(v)
            except ValueError:
                pass
    out["eval_steps"] = len(rows)
    return out


ARTIFACT_EXTS = ("ply", "sog", "spz", "html")
# LichtFeld names every export after the iteration it was taken at:
# splat_30000.ply. That number is the only completion evidence that does not
# come from the trainer's own summary of itself.
STEP_IN_NAME = re.compile(r"_(\d+)\.[A-Za-z0-9]+$")


def _artifact_step(p: Path) -> Optional[int]:
    m = STEP_IN_NAME.search(p.name)
    return int(m.group(1)) if m else None


def _final_artifact(hits: list[Path]) -> tuple[Path, Optional[int]]:
    """The LAST checkpoint among same-extension exports, with its step.

    Not the largest: a run that exports an intermediate checkpoint alongside the
    final one can easily have the bigger file at the earlier step, because
    densification prunes. Picking by size therefore serves a mid-training model
    as if it were the finished one.
    """
    numbered = [(p, s) for p in hits if (s := _artifact_step(p)) is not None]
    if numbered:
        return max(numbered, key=lambda ps: ps[1])
    # Unnumbered output: newest wins, and the step has to come from metrics.csv.
    return max(hits, key=lambda p: p.stat().st_mtime), None


def train_finalize(ctx: Ctx) -> dict:
    out = ctx.dir("train")
    arts = {}
    file_steps = []
    for ext in ARTIFACT_EXTS:
        hits = [p for p in sorted(out.glob(f"*.{ext}")) if p.is_file()]
        if not hits:
            continue
        chosen, step = _final_artifact(hits)
        arts[ext] = {"path": str(chosen), "bytes": chosen.stat().st_size,
                     "step": step}
        if step is not None:
            file_steps.append(step)
    if not arts:
        raise RuntimeError("training produced no exportable artifacts")

    empty = sorted(e for e, m in arts.items() if not m["bytes"])
    if empty:
        raise RuntimeError(
            f"training wrote a zero-byte {'/'.join(empty)} artifact")

    # Every format the job asked for has to be on disk. This is also what makes
    # export formats safe to leave out of the cache key: a cached PLY-only run
    # reused by a PLY+SOG job fails here, is invalidated, and retrains --- while
    # the reverse (a PLY+SOG cache serving a PLY-only job) still hits.
    missing = sorted({f for f in ctx.cfg.export.formats
                      if f in ARTIFACT_EXTS and f not in arts})
    if missing:
        raise RuntimeError(
            f"training exported {'/'.join(sorted(arts))} but the job asked for "
            f"{'/'.join(missing)}; not caching an incomplete export set")

    gut = ctx.derived.get("gut")
    if gut is None:                       # cached stage: train_argv never ran
        gut = read_done(out).get("gut")
    info = {"artifacts": arts, "gut": gut}
    info.update(read_metrics_csv(out))

    # Refuse to cache a short run. A trainer that exits 0 having stopped early
    # (SIGTERM, an internal abort) would otherwise be indistinguishable from a
    # completed one, and its partial PLY would be served to every later job
    # sharing this config. Two independent witnesses -- the exported filename
    # and metrics.csv -- and the pessimistic one decides, so a truncated run
    # cannot pass by having written one convincing artifact.
    # Every format has to come from the same, final checkpoint. Taking the
    # highest step across formats let splat_30000.ply vouch for a
    # splat_1000.sog beside it, which was then served as the finished model.
    steps_by_fmt = {e: m["step"] for e, m in arts.items() if m["step"] is not None}
    if len(set(steps_by_fmt.values())) > 1:
        raise RuntimeError(
            "exports come from different checkpoints ("
            + ", ".join(f"{e}@{s}" for e, s in sorted(steps_by_fmt.items()))
            + "); not caching a mixed export set")
    expected = ctx.cfg.train.effective_iters
    csv_step = info.get("final_step")
    evidence = [int(s) for s in (min(file_steps) if file_steps else None,
                                 csv_step) if s is not None]
    if not evidence:
        raise RuntimeError(
            "cannot tell how far training got: no splat_<step>.<ext> export "
            "and no metrics.csv; refusing to cache an unverifiable result")
    reached = min(evidence)
    if reached < expected:
        raise RuntimeError(
            f"training stopped at step {reached} of {expected}; refusing "
            f"to cache a partial result")
    info["final_step"] = reached
    # Fall back to the marker, exactly as `gut` does above. This finalizer also
    # runs against an EXISTING entry (see _cache_ok), where ctx.derived is empty
    # and whatever it returns is written back over the marker -- so reading the
    # peak only from derived erased it the first time a later job reused the
    # run, permanently, and the reusing job recorded no VRAM metric either.
    # This is the one number there is no a-priori model for, so losing it costs
    # the very history VramSampler exists to accumulate.
    peak = ctx.derived.get("peak_vram_mib") or read_done(out).get("peak_vram_mib")
    if peak:
        info["peak_vram_mib"] = peak
    return info


# ------------------------------------------------------------------ export

def export_argv(ctx: Ctx) -> Optional[list[str]]:
    """No-op for now: training already emits --export formats in place.

    Kept as a stage so extra conversions (spz, rad, decimated html) can be added
    without disturbing the cache chain above it.
    """
    return None


def export_finalize(ctx: Ctx) -> dict:
    src, dst = ctx.dir("train"), ctx.dir("export")
    dst.mkdir(parents=True, exist_ok=True)
    want = {p.name: p for pat in ("*.ply", "*.sog", "*.spz", "*.html")
            for p in sorted(src.glob(pat))}
    if not want:
        # This finalizer re-runs on every cache hit (see _cache_ok), so raising
        # here is what invalidates an export entry whose training run has been
        # rebuilt away or evicted, instead of serving a directory of links to
        # nothing.
        raise RuntimeError(
            f"no training artifacts to export from {src}; the train cache "
            f"directory is empty or gone")
    # Names left by a PREVIOUS train run under this key are either dangling or a
    # different model, and nothing else ever removed them. Dot files are ours
    # (.lock is still held here, .done is the marker) and are left alone.
    for p in dst.iterdir():
        if p.name.startswith(".") or p.name in want:
            continue
        try:
            p.unlink()
        except OSError:
            pass
    linked = []
    for name, p in want.items():
        target = dst / name
        # is_symlink() as well as exists(): a DANGLING link answers exists()
        # with False while still occupying the name, and copy2 onto one follows
        # it and writes through to a path whose parent may be gone.
        if target.is_symlink() or target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        try:
            target.symlink_to(p.resolve())
        except OSError:
            shutil.copy2(p, target)
        linked.append(name)
    return {"artifacts": sorted(linked)}


# ------------------------------------------------------------ fisheye rig
# A raw DJI dual-fisheye .OSV is reconstructed WITHOUT stitching: both lenses go
# to SfM as a calibrated two-camera rig seeded from the camera's own
# calibration, and train as two OPENCV_FISHEYE cameras (docs/how-it-works.md, "Fisheye rig"). Stage
# names, the cache chain, the review gate and the UI are the stitched path's;
# only what each stage runs differs. The heavy lifting lives in scripts 80-89,
# because the service's own venv has no numpy or OpenCV.

def _lens_counts(d: Path) -> list[int]:
    return [len(list((d / f"lens{i}").glob("*.jpg"))) for i in (0, 1)]


def fisheye_frames_argv(ctx: Ctx) -> list[str]:
    cfg = ctx.cfg
    out = ctx.dir("frames")
    for i in (0, 1):
        (out / f"lens{i}").mkdir(parents=True, exist_ok=True)
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-progress", "pipe:1", "-nostats", "-hwaccel", "cuda"]
    if cfg.input.trim_start:
        argv += ["-ss", f"{cfg.input.trim_start:g}"]
    if cfg.input.trim_end is not None:
        argv += ["-t", f"{cfg.input.trim_end - (cfg.input.trim_start or 0.0):g}"]
    # Input-scoped duration bounds both lens outputs.
    argv += ["-i", str(SPLAT_ROOT / cfg.input.file)]
    fps, q = f"{cfg.frames.fps:g}", str(cfg.frames.jpeg_q)
    # One decode, both lenses: streams 0 and 1 are the two fisheyes, sampled by
    # the same fps filter so candidate N of each lens is the same instant.
    argv += ["-filter_complex", f"[0:v:0]fps={fps}[l0];[0:v:1]fps={fps}[l1]",
             "-map", "[l0]", "-q:v", q, str(out / "lens0" / "%06d.jpg"),
             "-map", "[l1]", "-q:v", q, str(out / "lens1" / "%06d.jpg")]
    return argv


def fisheye_frames_finalize(ctx: Ctx) -> dict:
    out = ctx.dir("frames")
    n0, n1 = _lens_counts(out)
    if n0 == 0 or n0 != n1:
        raise RuntimeError(
            f"fisheye frame extraction wrote {n0} and {n1} candidates for the "
            f"two lenses; a rig frame needs both")
    src = SPLAT_ROOT / ctx.cfg.input.file
    proc = subprocess.run([str(SFM_PY), str(OSV_META), str(src), str(out)],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(
            f"could not read a lens calibration from {src.name}: "
            f"{(proc.stderr or proc.stdout).strip()[-400:]}. The fisheye rig "
            f"needs the camera's calibration; stitch this clip instead")
    cal = json.loads((out / "calibration.json").read_text())
    streams = sorted(l["stream"] for l in cal.get("lenses", []) if "stream" in l)
    if streams != [0, 1]:
        raise RuntimeError(
            f"{src.name}: found {len(cal.get('lenses', []))} lens calibration "
            f"slot(s) but no active pair for streams 0 and 1")
    dev = cal.get("device", {})
    ctx.derived["n_candidates"] = n0
    return {"candidates": n0, "pipeline": "fisheye_rig",
            "camera": dev.get("model"), "proto": dev.get("proto"),
            "firmware": dev.get("firmware"),
            "stream_size": [dev.get("video_width"), dev.get("video_height")]}


def fisheye_frames_verify(ctx: Ctx, info: dict) -> None:
    out = ctx.dir("frames")
    n0, n1 = _lens_counts(out)
    want = info.get("candidates")
    if want and (n0 != want or n1 != want):
        raise RuntimeError(f"cache claims {want} candidates per lens, found {n0} and {n1}")
    if not n0:
        raise RuntimeError("cached fisheye frames directory is empty")
    if not (out / "calibration.json").is_file():
        raise RuntimeError("cached fisheye frames have no calibration.json")


def fisheye_select_prepare(ctx: Ctx) -> None:
    sel = ctx.cfg.select
    n = ctx.derived.get("n_candidates") or _lens_counts(ctx.dir("frames"))[0]
    ctx.derived["n_candidates"] = n
    ctx.derived["window"] = (max(1, round(n / max(1, sel.target_panos)))
                             if sel.mode == "target" else sel.window)


def fisheye_select_argv(ctx: Ctx) -> list[str]:
    cfg = ctx.cfg
    out = ctx.dir("select")
    out.mkdir(parents=True, exist_ok=True)
    argv = [str(SFM_PY), str(FISHEYE_FRAMES), "--select-only",
            str(ctx.dir("frames")), str(out),
            "--window", str(ctx.derived["window"]),
            "--fps", f"{cfg.frames.fps:g}"]
    if cfg.select.mode == "distance":
        argv += ["--distance-m", str(cfg.select.distance_m),
                 "--max-gap-s", str(cfg.select.max_gap_s),
                 "--motion-src", str(SPLAT_ROOT / cfg.input.file)]
    if cfg.input.trim_start:
        # The frame dump restarts its clock at the trim, so candidate times --
        # and the video frame behind each candidate -- count from there.
        argv += ["--start", f"{cfg.input.trim_start:g}"]
    if cfg.select.imu:
        # The stream is read from the clip itself; the frames stage keeps only
        # the calibration, whose focal length turns a body rate into pixels.
        argv += ["--imu", str(SPLAT_ROOT / cfg.input.file),
                 "--calib", str(ctx.dir("frames") / "calibration.json")]
    return argv


def fisheye_select_finalize(ctx: Ctx) -> dict:
    n0, n1 = _lens_counts(ctx.dir("select") / "images")
    if n0 != n1:
        raise RuntimeError(f"selected {n0} and {n1} images for the two lenses")
    if n0 < 8:
        raise RuntimeError(f"only {n0} rig frames selected; need at least 8")
    ctx.derived["n_panos"] = n0
    info = {"panos": n0, "rig_frames": n0, "window": ctx.derived["window"]}
    if ctx.cfg.select.mode == "distance":
        report = json.loads((ctx.dir("select") / "selection.json").read_text())
        if report.get("mode") != "distance" or not report.get("distance"):
            raise RuntimeError("distance selection produced no distance summary")
        info.update(mode="distance", window=None, distance=report["distance"],
                    distance_m=ctx.cfg.select.distance_m)
        if report["distance"].get("warning"):
            info["warnings"] = [report["distance"]["warning"]]
    if ctx.cfg.select.imu:
        sel = ctx.dir("select") / "selection.json"
        imu = json.loads(sel.read_text()).get("imu") if sel.is_file() else None
        if not imu:
            # The key says the orientation stream chose these frames; a run
            # that never read it must not be cached under that claim.
            raise RuntimeError("select ran with select.imu but selection.json "
                               "records no IMU summary")
        info["imu"] = imu
    return info


def fisheye_select_verify(ctx: Ctx, info: dict) -> None:
    n0, n1 = _lens_counts(ctx.dir("select") / "images")
    want = info.get("panos")
    if want and (n0 != want or n1 != want):
        raise RuntimeError(f"cache claims {want} rig frames, found {n0} and {n1}")
    if n0 < 8:
        raise RuntimeError(f"cached selection holds only {n0} rig frames")
    if ctx.cfg.select.mode == "distance":
        if not info.get("distance") or not (ctx.dir("select") / "motion_path.json").is_file():
            raise RuntimeError("cached distance selection is missing its motion report")


def fisheye_mask_argv(ctx: Ctx) -> list[str]:
    out = ctx.dir("mask")
    out.mkdir(parents=True, exist_ok=True)
    m = ctx.cfg.mask
    return [str(GS_PY), str(FISHEYE_MASKS),
            "--images", str(ctx.dir("select") / "images"),
            "--calib", str(ctx.dir("frames") / "calibration.json"),
            "--out", str(out), "--masker-python", str(MASK_PY),
            "--", *_masker_options(m)]


def _fisheye_mask_counts(ctx: Ctx) -> list[int]:
    d = ctx.dir("mask") / "fisheye_masks"
    return [len(list((d / f"lens{i}").glob("*.jpg.png"))) for i in (0, 1)]


def fisheye_mask_finalize(ctx: Ctx) -> dict:
    ctx.derived["n_panos"] = (ctx.derived.get("n_panos")
                              or _lens_counts(ctx.dir("select") / "images")[0])
    # The stitched masks, overlays, sheet and coverage are exactly what the
    # stitched path produces, so the same checks -- and the review gate -- apply.
    info = mask_finalize(ctx)
    n0, n1 = _fisheye_mask_counts(ctx)
    if n0 != info["masks"] or n1 != info["masks"]:
        raise RuntimeError(
            f"carried {n0} and {n1} masks back to the lenses for "
            f"{info['masks']} rig frames")
    info["fisheye_masks"] = n0
    return info


def fisheye_mask_verify(ctx: Ctx, info: dict) -> None:
    mask_verify(ctx, info)
    n0, n1 = _fisheye_mask_counts(ctx)
    want = info.get("fisheye_masks")
    if want and (n0 != want or n1 != want):
        raise RuntimeError(f"cache claims {want} fisheye masks per lens, found {n0} and {n1}")


def fisheye_sfm_argv(ctx: Ctx) -> list[str]:
    out = ctx.dir("sfm")
    out.mkdir(parents=True, exist_ok=True)
    argv = [str(SFM_PY), str(FISHEYE_SFM),
            "--images", str(ctx.dir("select") / "images"),
            "--calib", str(ctx.dir("frames") / "calibration.json"),
            "--out", str(out)]
    if ctx.cfg.mask.enabled and ctx.cfg.mask.use_for_sfm:
        argv += ["--person-masks", str(ctx.dir("mask") / "fisheye_masks")]
    return argv


def fisheye_sfm_finalize(ctx: Ctx) -> dict:
    path = ctx.dir("sfm") / "summary.json"
    if not path.is_file():
        raise RuntimeError("fisheye SfM finished without writing summary.json")
    summary = json.loads(path.read_text())
    n_frames = ctx.derived.get("n_panos") or summary.get("n_panos") or 0
    summary["warnings"] = _sfm_warnings(summary, n_frames)
    dataset = ctx.dir("sfm") / "dataset"
    images = dataset / "images"
    if not (dataset / "sparse" / "0").exists() or not images.is_dir() \
            or not any(images.iterdir()):
        raise RuntimeError(f"fisheye dataset {dataset} is missing its model or images")
    if not any((dataset / "masks").glob("*.png")):
        raise RuntimeError(
            f"fisheye dataset {dataset} has no valid-circle masks; training "
            f"would learn the black corners around each image circle")
    summary["images_dir"] = str(images)
    summary["needs_gut"] = True
    ctx.derived["dataset"] = str(dataset)
    ctx.derived["images"] = str(images)
    ctx.derived["needs_gut"] = True
    return summary


def fisheye_sfm_verify(ctx: Ctx, info: dict) -> None:
    sfm_verify(ctx, info)
    ds = ctx.dir("sfm") / "dataset"
    if not any((ds / "masks").glob("*.png")):
        raise RuntimeError("cached fisheye dataset has no masks")
    if not (ds / "images").is_dir() or not any((ds / "images").iterdir()):
        raise RuntimeError("cached fisheye dataset has no images")


def fisheye_train_argv(ctx: Ctx) -> list[str]:
    out = ctx.dir("train")
    out.mkdir(parents=True, exist_ok=True)
    dataset = ctx.derived.get("dataset") or str(ctx.dir("sfm") / "dataset")
    images = ctx.derived.get("images") or str(Path(dataset) / "images")
    view = out / "dataset"
    # Always masked: even with people left in, the valid circle has to be --
    # the corners around each image circle are black, and beyond the circle's
    # edge the lens model is not trustworthy.
    lfs = _lichtfeld_argv(ctx, str(view), images,
                          _resolve_gut(ctx, "OPENCV_FISHEYE"), masked=True)
    argv = [str(SFM_PY), str(FISHEYE_TRAIN), "--dataset", dataset, "--view", str(view)]
    if ctx.cfg.mask.enabled:
        argv += ["--person-masks", str(ctx.dir("mask") / "fisheye_masks")]
    return argv + ["--"] + lfs


# ------------------------------------------------------------------ registry

# "verify" is the cheap check run against an EXISTING cache entry. train/export
# have none because the worker re-runs their full finalizer instead --- that is
# what catches a partial or short export --- while re-running sfm_finalize would
# shell out to the model picker on every cache hit, which is not free.
def _by_input(stitched, fisheye):
    """One registry entry per stage, choosing the implementation per job.

    None when neither kind has one, so `if spec["prepare"]` keeps working.
    """
    if stitched is None and fisheye is None:
        return None

    def pick(ctx: Ctx, *args):
        fn = fisheye if ctx.cfg.is_fisheye else stitched
        return fn(ctx, *args) if fn is not None else None
    pick.__name__ = (stitched or fisheye).__name__
    return pick


STAGES = {
    "frames": {"argv": _by_input(frames_argv, fisheye_frames_argv),
               "finalize": _by_input(frames_finalize, fisheye_frames_finalize),
               "prepare": None, "parse": frames_parse, "gpu": True,
               "verify": _by_input(frames_verify, fisheye_frames_verify)},
    "select": {"argv": _by_input(select_argv, fisheye_select_argv),
               "finalize": _by_input(select_finalize, fisheye_select_finalize),
               "prepare": _by_input(select_prepare, fisheye_select_prepare),
               "parse": None, "gpu": False,
               "verify": _by_input(select_verify, fisheye_select_verify)},
    "mask": {"argv": _by_input(mask_argv, fisheye_mask_argv),
             "finalize": _by_input(mask_finalize, fisheye_mask_finalize),
             "prepare": None, "parse": mask_parse, "gpu": True,
             "verify": _by_input(mask_verify, fisheye_mask_verify),
             "skip": mask_skip},
    "sfm": {"argv": _by_input(sfm_argv, fisheye_sfm_argv),
            "finalize": _by_input(sfm_finalize, fisheye_sfm_finalize),
            "prepare": None, "parse": sfm_parse, "gpu": True,
            "verify": _by_input(sfm_verify, fisheye_sfm_verify)},
    "train": {"argv": _by_input(train_argv, fisheye_train_argv),
              "finalize": train_finalize,
              "prepare": None, "parse": train_parse, "gpu": True,
              "verify": None},
    "export": {"argv": export_argv, "finalize": export_finalize,
               "prepare": None, "parse": None, "gpu": False,
               "verify": None},
}
