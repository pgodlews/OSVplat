#!/usr/bin/env python3
"""Unit tests for stage finalizers, cache-key logic and retention. No server needed.

Run on the box with the service venv:
    ~/splat/queue_app/venv/bin/python ~/splat/queue_app/test_stages.py

Covers the cache-poisoning regression: LichtFeld handles SIGTERM by exporting
whatever it has and exiting 0, so a cancelled run is indistinguishable from a
completed one unless the finalizer checks how far it actually got. Serving that
partial PLY to every later job with the same config is silent data corruption,
which is exactly what happened to jobs 48 and 49.
"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

TEST_ROOT = tempfile.TemporaryDirectory(prefix="queue_stages_test_")
os.environ["QUEUE_ROOT"] = TEST_ROOT.name
os.environ["SPLAT_ROOT"] = TEST_ROOT.name
os.environ["QUEUE_GPUS"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.jobs import JobConfig, key_of                      # noqa: E402
from app.stages import (Ctx, frames_parse, is_cached, mark_done,   # noqa: E402
                        mask_parse, read_done, read_lock, read_metrics_csv,
                        release_lock, reset_stage_dir, sfm_parse, take_lock,
                        train_finalize, train_parse)
from app import metrics, progress                               # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def make_ctx(tmp: Path, iters: int, scaler: float = 1.0,
             formats=("ply", "sog")) -> Ctx:
    cfg = JobConfig.model_validate({
        "name": "t", "input": {"file": "samples/x.mp4", "quick_hash": "deadbeef"},
        "train": {"iter": iters, "steps_scaler": scaler},
        "export": {"formats": list(formats)},
    })
    ctx = Ctx(job_id=1, cfg=cfg, gpu=0, keys=cfg.keys())
    # Point the train cache dir at our temp dir.
    ctx.dir = lambda stage, _t=tmp: _t                       # type: ignore[assignment]
    return ctx


def write_run(tmp: Path, step: int, with_csv=True):
    (tmp / f"splat_{step}.ply").write_bytes(b"x" * 2048)
    (tmp / f"splat_{step}.sog").write_bytes(b"x" * 512)
    if with_csv:
        (tmp / "metrics.csv").write_text(
            "iteration,psnr,ssim,time_per_image,num_gaussians\n"
            f"{step},29.4,0.85,0.18,300000\n")


# 1. A complete run is accepted.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    write_run(tmp, 30000)
    try:
        info = train_finalize(make_ctx(tmp, 30000))
        check("complete run accepted", info.get("final_step") == 30000,
              f"final_step={info.get('final_step')}")
        check("psnr parsed from metrics.csv", info.get("psnr") == 29.4)
    except Exception as e:                                    # noqa: BLE001
        check("complete run accepted", False, str(e))

# 2. A partial run is REJECTED (the regression).
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    write_run(tmp, 3380)                       # cancelled at 3380 of 15000
    try:
        train_finalize(make_ctx(tmp, 15000))
        check("partial run rejected", False, "finalize returned instead of raising")
    except RuntimeError as e:
        check("partial run rejected", "3380" in str(e) and "15000" in str(e), str(e)[:90])

# 3. Partial detected from the FILENAME even with no metrics.csv.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    write_run(tmp, 2734, with_csv=False)
    try:
        train_finalize(make_ctx(tmp, 15000))
        check("partial detected without metrics.csv", False, "did not raise")
    except RuntimeError as e:
        check("partial detected without metrics.csv", "2734" in str(e), str(e)[:90])

# 4. steps_scaler defines the expected step count, not the raw iter field.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    write_run(tmp, 15000)                      # 30000 * 0.5 == 15000, complete
    try:
        info = train_finalize(make_ctx(tmp, 30000, scaler=0.5))
        check("steps_scaler run accepted at its scaled length",
              info.get("final_step") == 15000)
    except Exception as e:                                    # noqa: BLE001
        check("steps_scaler run accepted at its scaled length", False, str(e))

# 5. No artifacts at all is an error, not a silent pass.
with tempfile.TemporaryDirectory() as d:
    try:
        train_finalize(make_ctx(Path(d), 100))
        check("empty output rejected", False, "did not raise")
    except RuntimeError as e:
        check("empty output rejected", "no exportable artifacts" in str(e))

# 6. metrics.csv parsing takes the LAST row.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    (tmp / "metrics.csv").write_text(
        "iteration,psnr,ssim,time_per_image,num_gaussians\n"
        "700,25.29,0.838,0.19,300000\n"
        "3000,29.41,0.855,0.18,300000\n")
    m = read_metrics_csv(tmp)
    check("metrics.csv uses the final row",
          m.get("psnr") == 29.41 and m.get("final_step") == 3000 and
          m.get("eval_steps") == 2, str(m))

# 6b. The FINAL checkpoint wins, not the biggest file. Densification prunes, so
# an intermediate export is routinely larger than the finished one; picking by
# size served a mid-training model as if it were the requested result.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    (tmp / "splat_7000.ply").write_bytes(b"x" * 40960)     # bigger, earlier
    (tmp / "splat_7000.sog").write_bytes(b"x" * 4096)
    write_run(tmp, 30000)                                  # smaller, final
    try:
        info = train_finalize(make_ctx(tmp, 30000))
        check("final checkpoint chosen over the larger intermediate one",
              info["artifacts"]["ply"]["path"].endswith("splat_30000.ply"),
              info["artifacts"]["ply"]["path"])
    except Exception as e:                                    # noqa: BLE001
        check("final checkpoint chosen over the larger intermediate one",
              False, str(e))

# 6c. A cache entry missing a requested export format is not a hit. This is what
# keeps export formats safely out of the train cache key: a PLY-only run reused
# by a PLY+SOG job is rejected here and retrained, while the reverse still hits.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    (tmp / "splat_30000.ply").write_bytes(b"x" * 2048)
    (tmp / "metrics.csv").write_text(
        "iteration,psnr,ssim,time_per_image,num_gaussians\n"
        "30000,29.4,0.85,0.18,300000\n")
    try:
        train_finalize(make_ctx(tmp, 30000, formats=("ply", "sog")))
        check("missing requested format rejected", False, "did not raise")
    except RuntimeError as e:
        check("missing requested format rejected", "sog" in str(e), str(e)[:90])
    try:
        info = train_finalize(make_ctx(tmp, 30000, formats=("ply",)))
        check("a superset cache still satisfies a narrower request",
              info.get("final_step") == 30000)
    except Exception as e:                                    # noqa: BLE001
        check("a superset cache still satisfies a narrower request", False, str(e))

# 6d. A zero-byte export is not a result.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    write_run(tmp, 30000)
    (tmp / "splat_30000.sog").write_bytes(b"")
    try:
        train_finalize(make_ctx(tmp, 30000))
        check("zero-byte artifact rejected", False, "did not raise")
    except RuntimeError as e:
        check("zero-byte artifact rejected", "zero-byte" in str(e), str(e)[:90])

# 6e. Unnumbered export with no metrics.csv cannot be verified, so it is not
# cached -- previously it passed completion validation with no evidence at all.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    (tmp / "splat.ply").write_bytes(b"x" * 2048)
    (tmp / "splat.sog").write_bytes(b"x" * 512)
    try:
        train_finalize(make_ctx(tmp, 30000))
        check("unverifiable run rejected", False, "did not raise")
    except RuntimeError as e:
        check("unverifiable run rejected", "unverifiable" in str(e), str(e)[:90])

# 6f. metrics.csv claiming completion does not override a truncated export.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    (tmp / "splat_9000.ply").write_bytes(b"x" * 2048)
    (tmp / "splat_9000.sog").write_bytes(b"x" * 512)
    (tmp / "metrics.csv").write_text(
        "iteration,psnr,ssim,time_per_image,num_gaussians\n"
        "30000,29.4,0.85,0.18,300000\n")
    try:
        train_finalize(make_ctx(tmp, 30000))
        check("filename contradicting metrics.csv rejected", False, "did not raise")
    except RuntimeError as e:
        check("filename contradicting metrics.csv rejected",
              "9000" in str(e), str(e)[:90])

# 6g. Export formats are validated and canonically ordered, so ["sog","ply"]
# and ["ply","sog"] are the same cache entry rather than two.
try:
    JobConfig.model_validate({**{"name": "e", "input": {"file": "x", "quick_hash": "d"}},
                              "export": {"formats": ["ply", "nope"]}})
    check("unknown export format rejected", False, "did not raise")
except Exception as e:                                        # noqa: BLE001
    check("unknown export format rejected", "nope" in str(e), str(e)[:60])
_ekb = {"name": "e", "input": {"file": "x", "quick_hash": "d"}}
check("export format order does not change the cache key",
      JobConfig.model_validate({**_ekb, "export": {"formats": ["sog", "ply"]}}).keys()
      == JobConfig.model_validate({**_ekb, "export": {"formats": ["ply", "sog"]}}).keys())

# 7. Cache keys: a train-only change must not disturb upstream keys.
base = {"name": "k", "input": {"file": "samples/x.mp4", "quick_hash": "deadbeef"}}
a = JobConfig.model_validate({**base, "train": {"sh_degree": 1}}).keys()
b = JobConfig.model_validate({**base, "train": {"sh_degree": 3}}).keys()
check("sh_degree change keeps frames/select/sfm keys",
      (a["frames"], a["select"], a["sfm"]) == (b["frames"], b["select"], b["sfm"]))
check("sh_degree change alters the train key", a["train"] != b["train"])
c = JobConfig.model_validate({**base, "frames": {"fps": 5},
                              "train": {"sh_degree": 1}}).keys()
check("fps change alters every downstream key",
      c["select"] != a["select"] and c["sfm"] != a["sfm"] and c["train"] != a["train"])

# 7b. Masking now feeds BOTH the reconstruction and the training, so enabling
# it has to invalidate the sfm key -- otherwise a masked job would be served the
# unmasked reconstruction it was supposed to replace. use_for_sfm=false is the
# escape hatch that keeps the old cheap comparison available: training-side
# masking only, on byte-identical geometry.
m_off = JobConfig.model_validate({**base}).keys()
m_on = JobConfig.model_validate({**base, "mask": {"enabled": True}}).keys()
check("enabling masks keeps frames/select keys",
      (m_off["frames"], m_off["select"]) == (m_on["frames"], m_on["select"]))
check("enabling masks alters the mask, sfm and train keys",
      m_off["mask"] != m_on["mask"] and m_off["sfm"] != m_on["sfm"]
      and m_off["train"] != m_on["train"])

m_train_only = JobConfig.model_validate(
    {**base, "mask": {"enabled": True, "use_for_sfm": False}}).keys()
check("use_for_sfm=false leaves the reconstruction shared with an unmasked run",
      m_train_only["sfm"] == m_off["sfm"])
check("use_for_sfm=false still masks the training",
      m_train_only["train"] != m_off["train"]
      and m_train_only["mask"] == m_on["mask"])

m_score = JobConfig.model_validate(
    {**base, "mask": {"enabled": True, "score": 0.8}}).keys()
check("a mask parameter change alters mask, sfm and train",
      m_score["mask"] != m_on["mask"] and m_score["sfm"] != m_on["sfm"]
      and m_score["train"] != m_on["train"])

# review is a workflow gate, not an input. If it reached any key, turning it on
# would recompute every mask and retrain -- the opposite of what a preview gate
# is for.
m_review = JobConfig.model_validate(
    {**base, "mask": {"enabled": True, "review": True}}).keys()
check("asking for review changes no cache key at all", m_review == m_on)

# 7c. Two segmentation backends. They must not share a mask cache entry, and a
# prompt list only one of them can honour must be refused rather than ignored.
m_sam = JobConfig.model_validate(
    {**base, "mask": {"enabled": True, "backend": "sam3"}}).keys()
check("switching backend alters the mask key", m_sam["mask"] != m_on["mask"])
m_sam_dog = JobConfig.model_validate(
    {**base, "mask": {"enabled": True, "backend": "sam3",
                      "prompts": ["person", "dog"]}}).keys()
check("a different prompt set alters the mask key",
      m_sam_dog["mask"] != m_sam["mask"])
try:
    JobConfig.model_validate(
        {**base, "mask": {"enabled": True, "prompts": ["person", "dog"]}})
    check("prompts the maskrcnn backend cannot honour are rejected", False,
          "did not raise")
except Exception as e:                                        # noqa: BLE001
    check("prompts the maskrcnn backend cannot honour are rejected",
          "sam3" in str(e), str(e)[:70])

# 8. Locking: only one of N racing workers may own a cache dir, and a lock left
# behind by a dead process must be reclaimable rather than wedging the stage
# forever. This is the mechanism a sweep leans on when four variants reach the
# same uncached SfM key at once.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d) / "stagedir"
    winners = []
    barrier = threading.Barrier(8)

    def race():
        barrier.wait()
        winners.append(take_lock(tmp, os.getpid()))

    threads = [threading.Thread(target=race) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("exactly one racer takes the lock",
          winners.count(True) == 1, f"{winners.count(True)} of 8 won")
    check("the lock records its holder", read_lock(tmp).get("pid") == os.getpid())
    release_lock(tmp)
    check("released lock can be retaken", take_lock(tmp, os.getpid()))
    release_lock(tmp)

# 9. A truncated .done is not a cache hit. mark_done writes through a temp file
# so this can only happen to markers written by an older build -- but those
# exist on the box right now.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    mark_done(tmp, {"panos": 120})
    check("a complete marker is a cache hit", is_cached(tmp))
    check("no .done.tmp left behind", not (tmp / ".done.tmp").exists())
    (tmp / ".done").write_text('{"panos": 12')          # killed mid-write
    check("a truncated marker is not a cache hit", not is_cached(tmp))

# 10. Rebuilding a stage starts from an empty directory, keeping only the lock.
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    take_lock(tmp, os.getpid())
    (tmp / "splat_3380.ply").write_bytes(b"stale")
    (tmp / "sub").mkdir()
    (tmp / "sub" / "old.txt").write_text("stale")
    reset_stage_dir(tmp)
    check("rebuild clears stale output",
          [p.name for p in tmp.iterdir()] == [".lock"],
          str(sorted(p.name for p in tmp.iterdir())))
    release_lock(tmp)

# 10b. A cached training run keeps its measured peak VRAM. _cache_ok re-runs
# train_finalize against the existing entry and writes the result back over the
# marker, so a value read only from ctx.derived was erased the first time a
# later job reused the run -- the one number there is no a-priori model for.
with tempfile.TemporaryDirectory() as d:
    ctx = make_ctx(Path(d), 1000, formats=("ply",))
    out = ctx.dir("train")
    out.mkdir(parents=True, exist_ok=True)
    (out / "splat_1000.ply").write_bytes(b"x" * 100)
    ctx.derived["peak_vram_mib"] = 4700
    mark_done(out, train_finalize(ctx))
    check("the run records its peak VRAM",
          read_done(out).get("peak_vram_mib") == 4700)
    later = make_ctx(Path(d), 1000, formats=("ply",))   # a later job, no derived
    fresh = train_finalize(later)
    check("a cache hit recomputes the same marker",
          fresh == read_done(out), json.dumps(fresh))
    check("so the peak survives being reused",
          fresh.get("peak_vram_mib") == 4700)

# 10c. A disabled mask stage produces nothing, so every disabled config has to
# hash the same way -- editing a dormant backend or dilate used to fork k_train
# and buy a 95-minute retrain for a bit-identical model.
off_a = JobConfig.model_validate({**base})
off_b = JobConfig.model_validate({**base, "mask": {
    "enabled": False, "backend": "sam3", "prompts": ["dog"], "dilate": 31}})
check("dormant mask fields do not fork the train key",
      off_a.k_train() == off_b.k_train())
check("nor the mask key itself", off_a.k_mask() == off_b.k_mask())
on_a = JobConfig.model_validate({**base, "mask": {"enabled": True}})
on_b = JobConfig.model_validate({**base, "mask": {"enabled": True, "dilate": 31}})
check("a live mask parameter still forks it", on_a.k_train() != on_b.k_train())
check("and masked never collides with unmasked",
      on_a.k_train() != off_a.k_train())

# 11. Retention never evicts what an unfinished job still needs. A job parked
# in awaiting_review is the case that used to slip through: it is defined by
# waiting, so its entries are the first to become least-recently-used, and
# losing the mask entry deletes the contact sheet out from under the reviewer.
from app import db, retention                                 # noqa: E402

db.init()
mask_dir = Path(retention.CACHE_ROOT) / "mask" / "reviewkey"
mask_dir.mkdir(parents=True, exist_ok=True)
(mask_dir / "review_sheet.jpg").write_bytes(b"contact sheet")
db.cache_put("reviewkey", "mask", str(mask_dir), 13)
jid = db.create_job("waiting on a human", {})
db.upsert_stage(jid, "mask", "reviewkey", "done", path=str(mask_dir))
db.set_job_state(jid, "awaiting_review")

res = retention.gc_cache(target_free=1 << 60)       # maximum disk pressure
check("a job awaiting review keeps its cache",
      "reviewkey" not in [e["key"] for e in res["evicted"]],
      f"skipped_in_use={res['skipped_in_use']}")
check("so the reviewer's contact sheet survives",
      (mask_dir / "review_sheet.jpg").is_file())

db.set_job_state(jid, "cancelled")                  # rejected at review
res = retention.gc_cache(target_free=1 << 60)
check("and it is evictable once the job is finished",
      "reviewkey" in [e["key"] for e in res["evicted"]])

# 12. The preflight is sized from what the stage has actually produced before,
# not from a flat floor a large stage sails straight through.
db.cache_put("bigtrain", "train", str(retention.CACHE_ROOT), 40 * retention.GB)
check("stage_need reads the largest entry on record",
      retention.stage_need("train") == 40 * retention.GB * retention.NEED_SAFETY,
      f"{retention.stage_need('train')/retention.GB:.0f} GB")
check("a stage with no history falls back to the floor",
      retention.stage_need("select") == 0.0)

# 13. Two smaller cache-correctness rules, both of which showed the previous
# run's output as if it were this one's.
with tempfile.TemporaryDirectory() as d:
    from app.stages import export_finalize                     # noqa: E402
    ectx = make_ctx(Path(d), 1000, formats=("ply",))
    tr, ex = Path(d) / "train", Path(d) / "export"
    tr.mkdir(); ex.mkdir()
    ectx.dir = lambda stage, _t=Path(d): _t / stage            # type: ignore
    (ex / ".lock").write_text("held")
    (tr / "splat_30000.ply").write_bytes(b"x" * 10)
    export_finalize(ectx)
    (tr / "splat_30000.ply").unlink()                          # retrained shorter
    (tr / "splat_20000.ply").write_bytes(b"y" * 20)
    got = export_finalize(ectx)
    check("export drops links to a superseded training run",
          got["artifacts"] == ["splat_20000.ply"], str(got))
    check("but leaves the stage lock alone", (ex / ".lock").is_file())
    (tr / "splat_20000.ply").unlink()
    try:
        export_finalize(ectx)
        check("an export with nothing behind it is rejected", False)
    except RuntimeError as exc:
        check("an export with nothing behind it is rejected",
              "no training artifacts" in str(exc))

db.init()
_j = db.create_job("restarted stage", {})
db.upsert_stage(_j, "train", "k", "done", started=100.0, ended=200.0,
                progress=json.dumps({"psnr": 29.4}))
db.upsert_stage(_j, "train", "k", "running", started=1000.0)
_r = db.conn().execute(
    "SELECT started,ended,progress FROM stages WHERE job_id=?", (_j,)).fetchone()
check("a re-running stage does not inherit the old end time",
      _r["ended"] is None, f"ended={_r['ended']} started={_r['started']}")
check("nor the previous attempt's progress", _r["progress"] is None)


# --------------------------------------------------------------- progress
# Every line below is copied out of a real stage log on the box. The parsers
# exist to be fed exactly this and nothing tests them but this.

def feed(parse, lines, prog=None):
    prog = dict(prog or {})
    for ln in lines:
        prog.update(parse(ln, prog))
    return prog


_f = feed(frames_parse, ["frame=142", "out_time_us=N/A", "speed=N/A",
                         "progress=continue", "frame=1418",
                         "out_time_us=141800000", "speed=  2.4x"])
check("ffmpeg progress gives clip position", _f.get("clip_pos_s") == 141.8, str(_f))
check("and survives the N/A it emits before the first frame",
      _f.get("speed") == 2.4 and _f.get("frames_out") == 1418, str(_f))
check("end of stream is recorded",
      feed(frames_parse, ["progress=end"]).get("finished") is True)

check("mask progress line parses",
      feed(mask_parse, ["  62/110  43s"]) == {"done": 62, "total": 110})
check("but an unrelated line does not",
      feed(mask_parse, ["  skipping unreadable pano_0007.jpg"]) == {})

_SFM = [
    "2026-09-08 12:48:21,476 START render=spherical mapper=incremental "
    "panos=110 cuda=True",
    "I1 1 feature_extraction.cc:498] === Feature extraction ===",
    "I1 1 feature_extraction.cc:270] Processed file [2/111]",
    "I1 1 pairing.cc:436] Generating sequential image pairs...",
    "I1 1 pairing.cc:100] Processing image [110/110]",
    "I1 1 incremental_pipeline.cc:353] Loading database",
    "I1 1 incremental_pipeline.cc:620] Registering image #75 (num_reg_frames=2)",
    "I1 1 incremental_pipeline.cc:620] Registering image #90 (num_reg_frames=44)",
]
_s = feed(sfm_parse, _SFM)
check("sfm reaches the mapping phase with the pano count as its total",
      (_s.get("phase"), _s.get("done"), _s.get("total")) == ("map", 44, 110), str(_s))
# Matching counts to 110 as well. Carried across the phase boundary it made the
# first registered frame look like a mapper that had just discarded 109 of them.
check("and matching's tally is not mistaken for a mapper restart",
      not _s.get("restarts"), str(_s.get("restarts")))
_s2 = feed(sfm_parse,
           ["I1 1 incremental_pipeline.cc:620] Registering image #3 "
            "(num_reg_frames=1)"], _s)
check("a real restart is counted", _s2.get("restarts") == 1, str(_s2))
check("sfm phases only ever move forward",
      feed(sfm_parse, ["I1 1 feature_extraction.cc:270] Processed file [4/111]"],
           _s2).get("phase") == "map")

_LOAD = "[12:51:25.395] [info] training_setup.cpp:750  Loading dataset from: /x"
_STEP = ("Training [x] 13% [04m:33s<27m:59s] 4100/30000 | Loss: 0.0654 | "
         "Splats: 130439")
check("the pre-iteration dead zone is named",
      feed(train_parse, [_LOAD, "colmap.cpp:3262  Training with 110 images"])
      == {"phase": "loading", "images": 110})
_t = feed(train_parse, [_LOAD, _STEP])
check("and gives way to real steps",
      (_t["phase"], _t["step"], _t["splats"]) == ("training", 4100, 130439), str(_t))
check("which nothing can drag back",
      feed(train_parse, [_LOAD], _t)["phase"] == "training")

_PLAN = {"frames": 100.0, "select": 10.0, "sfm": 600.0, "train": 6000.0,
         "export": 60.0, "total": 6770.0, "_assumptions": {"clip_seconds": 141.8}}
check("frames fraction is taken against the clip, not the frame count",
      progress.stage_fraction("frames", {"clip_pos_s": 70.9}, _PLAN) == 0.5)
check("sfm mapping sits in the back three quarters of its stage",
      progress.stage_fraction("sfm", {"phase": "map", "done": 55,
                                      "total": 110}, _PLAN) == 0.625)
check("the trainer's dataset load has a position but no progress",
      progress.stage_fraction("train", {"phase": "loading"}, _PLAN) == 0.0)

_cfg = JobConfig.model_validate({
    "name": "t", "input": {"file": "samples/x.mp4", "quick_hash": "d"},
    "train": {"iter": 30000}})
_early = progress.stage_remaining("train", _cfg, {"step": 300, "total": 30000},
                                  _PLAN, 60.0)
check("no ETA while the trainer is still ramping up", _early == (None, "none"),
      str(_early))
_mid, _src = progress.stage_remaining(
    "train", _cfg, {"step": 15000, "total": 30000}, _PLAN, 3000.0)
# elapsed x (1-f)/f -- the trainer's own arithmetic -- would say 3000 s here.
# It is wrong because iterations keep slowing as densification runs, which is
# exactly what the fitted whole-run rate already accounts for.
check("and the ETA past that is not a straight extrapolation of elapsed",
      _src == "measured" and abs(_mid - 3000.0) > 1.0, f"{_mid:.0f}s")

# A job still in its first few percent has no ETA of its own, but it is holding
# the GPU all the same.
_floor = [{"state": "running",
           "eta": {"remaining": None, "elapsed": 60.0, "plan_total": 4200.0}},
          {"state": "queued", "eta": {"remaining": None, "plan_total": 4200.0}}]
progress.queue_eta(_floor, 1)
check("a job with no ETA yet still occupies its slot",
      _floor[1]["eta"]["starts_in"] == 4140.0, str(_floor[1]["eta"]))

_q = [{"state": "running", "eta": {"remaining": 3600.0, "plan_total": 7000.0}},
      {"state": "queued", "eta": {"remaining": None, "plan_total": 6000.0}},
      {"state": "queued", "eta": {"remaining": None, "plan_total": 6000.0}}]
_sum = progress.queue_eta(_q, 2)
check("a queued job is told when its slot frees up",
      [j["eta"].get("starts_in") for j in _q[1:]] == [0.0, 3600.0], str(_q))
check("and the queue knows when it drains", _sum["drains_in"] == 9600.0, str(_sum))
# Lowering max_concurrent under two running jobs must not drop one of them.
_busy = [{"state": "running", "eta": {"remaining": 5000.0, "plan_total": 0}},
         {"state": "running", "eta": {"remaining": 3000.0, "plan_total": 0}}]
check("shrinking concurrency does not lose a job already running",
      progress.queue_eta(_busy, 1)["drains_in"] == 5000.0)

# ------------------------------------------------------------- /metrics
# The exposition format is unforgiving: one bad label value and the scraper
# drops the whole scrape, silently, and the dashboard just goes flat.

check("a label value is escaped, not merely quoted",
      metrics._esc('a"b\\c\nd') == 'a\\"b\\\\c\\nd', metrics._esc('a"b\\c\nd'))
check("NaN and infinities use Prometheus's spelling, not Python's",
      (metrics._num(float("nan")), metrics._num(float("inf")),
       metrics._num(float("-inf"))) == ("NaN", "+Inf", "-Inf"))
check("whole numbers are not rendered in exponent form",
      metrics._num(3000000.0) == "3000000", metrics._num(3000000.0))

_ex = metrics.Exposition()
_ex.add("nothing", "h", "gauge", [])
_ex.add("some", "h", "gauge", [({"a": "1"}, 2), (None, None)])
_txt = _ex.text()
check("a family with no samples is left out entirely", "nothing" not in _txt)
check("and a None-valued sample is dropped, not rendered",
      _txt.count("\n") == 3 and _txt.endswith("\n"), repr(_txt))

# The cardinality guard. Per-job series must cover RUNNING jobs only: one
# series per job id ever run grows without limit, and this queue is already
# past 160 jobs.
def _job(jid, state, frac=0.5):
    return {"id": jid, "name": f"j{jid}", "state": state, "gpu": 0,
            "stages": [{"stage": "train", "state":
                        "running" if state == "running" else "done",
                        "progress": {"step": 10, "total": 100, "splats": 5},
                        "fraction": frac if state == "running" else None}],
            "eta": {"stage": "train", "fraction": frac, "elapsed": 10.0,
                    "remaining": 20.0, "plan_total": 30.0}}

_out = metrics.render([_job(1, "running"), _job(2, "queued"),
                       _job(3, "done"), _job(4, "failed")])
check("only running jobs get per-job series",
      _out.count("splatqueue_job_progress_ratio{") == 1
      and 'job_id="1"' in _out and 'job_id="3"' not in _out
      and 'job_id="4"' not in _out)
check("a queued job counts toward pending work but gets no series",
      'splatqueue_job_info{job_id="2"' not in _out
      and "splatqueue_queue_pending_seconds 50" in _out, "50 = 20 running + 30 queued")
check("every emitted line is a comment or a sample",
      all(l.startswith("#") or " " in l for l in _out.splitlines() if l))

# ------------------------------------------------------- fisheye rig (.OSV)
# A raw DJI .OSV runs the same six stages with its own implementations. What
# must hold: stitched jobs keep every cache key they already have, the two kinds
# never share an entry, and each fisheye stage builds the command the scripts
# expect.
from app import estimate as _estimate                         # noqa: E402
from app import stages as _stages                             # noqa: E402
from app.jobs import FISHEYE_PIPELINE                         # noqa: E402


def _cfg(file, **kw):
    d = {"name": "t", "input": {"file": file, "quick_hash": "deadbeef"}}
    d.update(kw)
    return JobConfig.model_validate(d)


_mp4, _osv = _cfg("samples/x.mp4"), _cfg("samples/x.OSV")
check("a .OSV input selects the fisheye pipeline, whatever the case",
      _osv.is_fisheye and _cfg("samples/x.osv").is_fisheye and not _mp4.is_fisheye)
check("stitched cache keys are exactly what they were before fisheye support",
      _mp4.k_frames() == key_of("frames", _mp4.config_version, "deadbeef", None, None,
                                _mp4.frames.model_dump()))
check("the fisheye pipeline term is part of the fisheye frames key",
      _osv.k_frames() == key_of("frames", _osv.config_version, "deadbeef", None, None,
                                _osv.frames.model_dump(), FISHEYE_PIPELINE))
check("no stage of a fisheye job can share a cache entry with a stitched one",
      all(_osv.keys()[k] != _mp4.keys()[k] for k in _mp4.keys()))
try:
    _cfg("samples/x.osv", sfm={"render": "perspective_overlapping"})
    _refused = False
except Exception as _e:                                          # noqa: BLE001
    _refused = "fisheye" in str(_e)
check("a stitched-only SfM mode is refused for raw fisheye, not silently ignored", _refused)

with tempfile.TemporaryDirectory() as _d:
    _root = Path(_d)

    def _ctx_for(cfg):
        c = Ctx(job_id=1, cfg=cfg, gpu=0, keys=cfg.keys())
        c.dir = lambda stage, _r=_root: _r / stage             # type: ignore[assignment]
        return c

    _f = _stages.STAGES["frames"]["argv"](_ctx_for(_osv))
    check("fisheye frames decode both lenses in one ffmpeg call, into lens0/ and lens1/",
          "-filter_complex" in _f and _f[-1].endswith("lens1/%06d.jpg")
          and any(a.endswith("lens0/%06d.jpg") for a in _f))
    _s = _stages.STAGES["frames"]["argv"](_ctx_for(_mp4))
    check("stitched frames still use the plain fps filter",
          "-vf" in _s and "-filter_complex" not in _s)

    _mc = _ctx_for(_cfg("samples/x.osv", mask={"enabled": True, "dilate": 7}))
    _m = _stages.STAGES["mask"]["argv"](_mc)
    check("fisheye masks run the stitch-mask-warp driver with the masker's own options after --",
          "87_fisheye_masks.py" in _m[1] and _m[_m.index("--") + 1:][:2] == ["--backend", "maskrcnn"]
          and "7" in _m[_m.index("--"):])

    _sc = _ctx_for(_cfg("samples/x.osv", mask={"enabled": True}))
    _sa = _stages.STAGES["sfm"]["argv"](_sc)
    check("fisheye SfM gets the person masks when the job uses them for SfM",
          "88_fisheye_sfm.py" in _sa[1] and "--person-masks" in _sa)
    _sa2 = _stages.STAGES["sfm"]["argv"](_ctx_for(_cfg("samples/x.osv", mask={"enabled": True, "use_for_sfm": False})))
    check("and not when use_for_sfm is off", "--person-masks" not in _sa2)

    _tc = _ctx_for(_cfg("samples/x.osv", mask={"enabled": True}))
    _tc.derived.update(dataset=str(_root / "sfm" / "dataset"),
                       images=str(_root / "sfm" / "dataset" / "images"), needs_gut=True)
    _t = _stages.STAGES["train"]["argv"](_tc)
    _lfs = _t[_t.index("--") + 1:]
    check("fisheye training goes through the mask-combining launcher",
          "89_fisheye_train_view.py" in _t[1] and "--person-masks" in _t)
    check("fisheye training always masks and always uses --gut",
          "--gut" in _lfs and _lfs[_lfs.index("--mask-mode") + 1] == "ignore")
    check("fisheye training reads its dataset view and the rig's renamed images",
          _lfs[_lfs.index("-d") + 1].endswith("train/dataset")
          and _lfs[_lfs.index("--images") + 1].endswith("sfm/dataset/images"))
    _uc = _ctx_for(_osv)
    _uc.derived.update(needs_gut=True)
    _u = _stages.STAGES["train"]["argv"](_uc)
    _ul = _u[_u.index("--") + 1:]
    check("an unmasked fisheye job still masks the valid circle",
          "--person-masks" not in _u and _ul[_ul.index("--mask-mode") + 1] == "ignore")
    _gc = _ctx_for(_cfg("samples/x.osv", train={"gut": False}))
    _gc.derived.update(needs_gut=True)
    try:
        _stages.STAGES["train"]["argv"](_gc)
        _gut_refused = False
    except RuntimeError as _e:
        _gut_refused = "OPENCV_FISHEYE" in str(_e)
    check("train.gut=false is refused for a fisheye rig", _gut_refused)

    _st = _ctx_for(_cfg("samples/x.mp4", mask={"enabled": True}))
    _st.derived.update(dataset=str(_root / "sfm" / "dataset"), images=str(_root / "select"), needs_gut=True)
    _sl = _stages.STAGES["train"]["argv"](_st)
    check("stitched training is unchanged: LichtFeld directly, masked view, --gut",
          _sl[0].endswith("LichtFeld-Studio") and "--gut" in _sl
          and _sl[_sl.index("--mask-mode") + 1] == "ignore")
    check("eval keeps its metrics but writes no per-view PNGs, on both pipelines",
          "--eval" in _sl and "--no-save-eval-images" in _sl
          and "--eval" in _lfs and "--no-save-eval-images" in _lfs)
    _ne = _ctx_for(_cfg("samples/x.mp4", train={"eval": False}))
    _ne.derived.update(dataset=str(_root / "sfm" / "dataset"), images=str(_root / "select"), needs_gut=True)
    _nl = _stages.STAGES["train"]["argv"](_ne)
    check("and neither flag without train.eval",
          "--eval" not in _nl and "--no-save-eval-images" not in _nl)
    check("the eval-image switch is not a cache-key term",
          _mp4.k_train() == key_of("train", _mp4.k_sfm(), _mp4.k_mask(),
                                   _mp4.train.model_dump()))
    try:
        _cfg("samples/x.mp4", train={"extra_args": "--no-save-eval-images"})
        _dup_refused = False
    except Exception as _e:                                      # noqa: BLE001
        _dup_refused = "--no-save-eval-images" in str(_e)
    check("extra_args cannot repeat the eval-image switch", _dup_refused)

check("images_dir points a fisheye job at the rig dataset's renamed images",
      str(_stages.images_dir(_osv, Path("/c/sfm"), Path("/c/select"))) == "/c/sfm/dataset/images")


def _feed_sfm(lines):
    prog = {}
    for ln in lines:
        prog.update(sfm_parse(ln, prog))
    return prog


_p = _feed_sfm(["FISHEYE PASS 1/2 frames=104 radius_px=[1744, 1745]",
                "2026-09-11 START refine_intrinsics=True fscale=1.0 frames=104 overlap=10 cuda=True",
                "I0911 incremental_pipeline.cc:620] Registering image #3 (num_reg_frames=50)",
                "FISHEYE PASS 2/2 frames=104",
                "I0911 feature_extraction.cc:12] Processed file [52/104]"])
check("a second fisheye pass restarts the phases instead of sticking at mapping",
      _p.get("sfm_pass") == 2 and _p.get("phase") == "extract" and _p.get("panos") == 104)
_fr = progress.stage_fraction("sfm", _p, {})
check("and the bar puts it in the second half", _fr is not None and 0.5 < _fr < 0.6, _fr)

_e = _estimate.estimate(_cfg("samples/x.osv", mask={"enabled": True}, select={"window": 3}), 31.0)
check("the estimator prices a fisheye job with fisheye constants",
      _e["_assumptions"]["pipeline"] == "fisheye_rig"
      and _e["sfm"] == round(_estimate.fisheye_sfm_seconds(_e["_assumptions"]["panos"], _estimate.SEED), 1)
      and _e["frames"] == round(31.0 * _estimate.SEED["fisheye_frames_realtime"], 1))
# Job 211: a 7-minute clip at window 3 is 1404 rig frames, and its SfM measured
# 5 h 14 min. A linear price called that 2 h.
_big = _estimate.fisheye_sfm_seconds(1404, _estimate.SEED)
check("fisheye SfM is priced superlinearly, near the measured 1404-frame run",
      4.5 * 3600 < _big < 6 * 3600
      and _estimate.fisheye_sfm_seconds(2808, _estimate.SEED) > 2.5 * _big, _big)

# Camera auto-detection and smart mask defaults
from app.main import _prepare
from scripts.osv_meta import detect_camera

_repo_dir = Path(__file__).resolve().parent.parent
_osmo_sample = _repo_dir / "CAM_20260813121712_0040_D.OSV"
_avata_sample = _repo_dir / "DJI_20260621181643_0010_D.OSV"

if _osmo_sample.is_file() and _avata_sample.is_file():
    _osmo_det = detect_camera(_osmo_sample)
    check("camera detector identifies Osmo 360", _osmo_det["camera"] == "osmo360" and _osmo_det["recommended_mask"] is True)
    _avata_det = detect_camera(_avata_sample)
    check("camera detector identifies Avata 360", _avata_det["camera"] == "avata360" and _avata_det["recommended_mask"] is False)

    _test_samples = Path(os.environ["SPLAT_ROOT"]).resolve() / "samples"
    _test_samples.mkdir(parents=True, exist_ok=True)
    _t_osmo = _test_samples / "cam_test.osv"
    _t_avata = _test_samples / "dji_test.osv"
    try:
        os.link(_osmo_sample, _t_osmo)
        os.link(_avata_sample, _t_avata)
    except OSError:
        pass

    if _t_osmo.is_file() and _t_avata.is_file():
        _p_osmo = _prepare({"name": "osmo-job", "input": {"file": "samples/cam_test.osv"}})
        check("Osmo 360 defaults mask.enabled to True", _p_osmo.mask.enabled is True)

        _p_avata = _prepare({"name": "avata-job", "input": {"file": "samples/dji_test.osv"}})
        check("Avata 360 defaults mask.enabled to False", _p_avata.mask.enabled is False)

        _p_avata_on = _prepare({"name": "avata-on", "input": {"file": "samples/dji_test.osv"}, "mask": {"enabled": True}})
        check("Avata 360 retains mask.enabled=True when explicitly set", _p_avata_on.mask.enabled is True)

        _p_osmo_off = _prepare({"name": "osmo-off", "input": {"file": "samples/cam_test.osv"}, "mask": {"enabled": False}})
        check("Osmo 360 retains mask.enabled=False when explicitly set", _p_osmo_off.mask.enabled is False)

        _t_osmo.unlink(missing_ok=True)
        _t_avata.unlink(missing_ok=True)

# ------------------------------------------------ IMU-aware selection (.OSV)
# select.imu off must leave every key a cached run already has; on forks select
# and everything after it; only a raw .OSV can ask for it; and the select stage
# hands the script the clip, its calibration and the trim it needs to find each
# candidate's video frame.
from app.jobs import IMU_SELECT, IMU_SELECT_DEFAULT            # noqa: E402

_sel_dump = {"mode": "window", "window": 5, "target_panos": 300}
_imu = _cfg("samples/x.OSV", select={"imu": True})
check("select.imu off hashes the select key exactly as before the option existed",
      _osv.k_select() == key_of("select", _osv.k_frames(), _sel_dump))
check("select.imu on forks select and every key after it, but not frames",
      _imu.k_frames() == _osv.k_frames()
      and _imu.k_select() == key_of("select", _osv.k_frames(), _sel_dump, IMU_SELECT)
      and all(_imu.keys()[k] != _osv.keys()[k] for k in ("select", "mask", "sfm", "train", "export")))
try:
    _cfg("samples/x.mp4", select={"imu": True})
    _imu_refused = False
except Exception as _e:                                          # noqa: BLE001
    _imu_refused = "select.imu" in str(_e)
check("select.imu is refused for stitched input, which has no orientation stream", _imu_refused)

# FISHEYE_SFM re-runs a fisheye reconstruction without redoing frames,
# selection or masks, and never touches a stitched key.
from app.jobs import FISHEYE_SFM                               # noqa: E402

# The sfm dump is the two fields that existed before sfm.upright, spelled out,
# so a new SfmCfg field cannot slip into the key unnoticed.
_sfm_dump = {"render": "spherical", "mapper": "incremental"}
check("the fisheye sfm term forks sfm, train and export of fisheye jobs only",
      _osv.k_sfm() == key_of("sfm", _osv.k_select(), _sfm_dump,
                             _osv._sfm_mask_term(), FISHEYE_SFM)
      and _mp4.k_sfm() == key_of("sfm", _mp4.k_select(), _sfm_dump,
                                 _mp4._sfm_mask_term()))
check("the fisheye sfm term leaves frames, select and mask keys alone",
      _osv.k_frames() == key_of("frames", _osv.config_version, "deadbeef", None, None,
                                _osv.frames.model_dump(), FISHEYE_PIPELINE)
      and _osv.k_select() == key_of("select", _osv.k_frames(), _sel_dump))

with tempfile.TemporaryDirectory() as _d:
    _root = Path(_d)

    def _select_ctx(cfg):
        c = Ctx(job_id=1, cfg=cfg, gpu=0, keys=cfg.keys())
        c.dir = lambda stage, _r=_root: _r / stage             # type: ignore[assignment]
        c.derived["window"] = 3
        return c

    _a = _stages.STAGES["select"]["argv"](_select_ctx(_imu))
    check("fisheye select with imu reads the clip's own stream and the frames stage's calibration",
          "80_fisheye_frames.py" in _a[1]
          and _a[_a.index("--imu") + 1] == str(_stages.SPLAT_ROOT / "samples" / "x.OSV")
          and _a[_a.index("--calib") + 1] == str(_root / "frames" / "calibration.json"))
    _a = _stages.STAGES["select"]["argv"](_select_ctx(_osv))
    check("and without imu passes neither, nor a start for an untrimmed clip",
          not {"--imu", "--calib", "--start"} & set(_a))
    _a = _stages.STAGES["select"]["argv"](_select_ctx(
        _cfg("samples/x.OSV", input={"file": "samples/x.OSV", "quick_hash": "deadbeef", "trim_start": 2.5})))
    check("a trimmed clip tells select where its candidates start", _a[_a.index("--start") + 1] == "2.5")

    _c = _select_ctx(_imu)
    for _l in (0, 1):
        (_root / "select" / "images" / f"lens{_l}").mkdir(parents=True, exist_ok=True)
        for _i in range(8):
            (_root / "select" / "images" / f"lens{_l}" / f"frame_{_i:04d}.jpg").touch()
    try:
        _stages.STAGES["select"]["finalize"](_c)
        _unsummarised = False
    except RuntimeError as _e:
        _unsummarised = "IMU summary" in str(_e)
    check("a select run under select.imu with no IMU summary is not cached", _unsummarised)
    (_root / "select" / "selection.json").write_text(json.dumps({"imu": {"overruled": 4}}))
    check("with one, the summary lands in the stage record",
          _stages.STAGES["select"]["finalize"](_c).get("imu") == {"overruled": 4})

# ------------------------------------------------ upright and in metres (.OSV)
# sfm.upright off keeps every key; on forks sfm and after, never frames, select
# or mask; stitched input is refused; the sfm stage hands 88 the clip, the
# selection and the trim; and a run that never recorded an alignment is not
# cached, while one that fell back is, with a warning.
from app.jobs import UPRIGHT, UPRIGHT_DEFAULT                  # noqa: E402

_up = _cfg("samples/x.OSV", sfm={"upright": True})
check("sfm.upright on forks sfm, train and export, and nothing before them",
      _up.k_sfm() == key_of("sfm", _osv.k_select(), _sfm_dump, _osv._sfm_mask_term(),
                            FISHEYE_SFM, UPRIGHT)
      and all(_up.keys()[k] == _osv.keys()[k] for k in ("frames", "select", "mask"))
      and all(_up.keys()[k] != _osv.keys()[k] for k in ("sfm", "train", "export")))
try:
    _cfg("samples/x.mp4", sfm={"upright": True})
    _up_refused = False
except Exception as _e:                                          # noqa: BLE001
    _up_refused = "sfm.upright" in str(_e)
check("sfm.upright is refused for stitched input", _up_refused)
check("the API turns sfm.upright on for .OSV by default", UPRIGHT_DEFAULT is True)

with tempfile.TemporaryDirectory() as _d:
    _root = Path(_d)

    def _sfm_ctx(cfg):
        c = Ctx(job_id=1, cfg=cfg, gpu=0, keys=cfg.keys())
        c.dir = lambda stage, _r=_root: _r / stage             # type: ignore[assignment]
        c.derived["n_panos"] = 8
        return c

    _a = _stages.STAGES["sfm"]["argv"](_sfm_ctx(_cfg(
        "samples/x.OSV", sfm={"upright": True},
        input={"file": "samples/x.OSV", "quick_hash": "deadbeef", "trim_start": 4})))
    check("fisheye sfm with upright reads the clip, the selection and the trim",
          _a[_a.index("--upright") + 1] == str(_stages.SPLAT_ROOT / "samples" / "x.OSV")
          and _a[_a.index("--selection") + 1] == str(_root / "select" / "selection.json")
          and _a[_a.index("--start") + 1] == "4")
    check("and without it passes none of them",
          not {"--upright", "--selection", "--start"} & set(_stages.STAGES["sfm"]["argv"](_sfm_ctx(_osv))))

    _ds = _root / "sfm" / "dataset"
    (_ds / "sparse" / "0").mkdir(parents=True)
    (_ds / "images").mkdir()
    (_ds / "images" / "lens0_frame_0000.jpg").touch()
    (_ds / "masks").mkdir()
    (_ds / "masks" / "lens0_frame_0000.png").touch()
    _base = {"num_reg_frames": 8, "registration_pct": 100.0, "n_panos": 8}

    def _finalize(extra, cfg=_up):
        (_root / "sfm" / "summary.json").write_text(json.dumps(dict(_base, **extra)))
        return _stages.STAGES["sfm"]["finalize"](_sfm_ctx(cfg))

    try:
        _finalize({})
        _unaligned = False
    except RuntimeError as _e:
        _unaligned = "no alignment" in str(_e)
    check("an upright sfm run that recorded no alignment is not cached", _unaligned)
    _w = _finalize({"alignment": {"upright": False, "metric": False, "reason": "barely rotated"}})["warnings"]
    check("one that fell back is cached, with the reason as a warning",
          any("not levelled" in w and "barely rotated" in w for w in _w))
    _w = _finalize({"alignment": {"upright": True, "metric": False,
                                  "gps_skipped": "the clip carries no GPS fix"}})["warnings"]
    check("levelled without GPS is the normal Osmo case, not a warning",
          not any("level" in w for w in _w))
    _w = _finalize({"alignment": {"upright": True, "metric": False,
                                  "gps_skipped": "the GPS path spans 4.0 m, under 20 m"}})["warnings"]
    check("GPS that was there but unusable is a warning", any("not scaled" in w for w in _w))
    check("the stage without the option ignores alignment entirely",
          not any("level" in w for w in _finalize({}, cfg=_osv)["warnings"]))

if _osmo_sample.is_file():
    _samples = Path(os.environ["SPLAT_ROOT"]).resolve() / "samples"
    _samples.mkdir(parents=True, exist_ok=True)
    _t_imu, _t_mp4 = _samples / "imu_test.osv", _samples / "imu_test.mp4"
    try:
        os.link(_osmo_sample, _t_imu)
    except OSError:
        pass
    _t_mp4.write_bytes(b"not really a video")
    if _t_imu.is_file():
        check(f"an .OSV request without select.imu gets the default ({IMU_SELECT_DEFAULT})",
              _prepare({"name": "imu-default", "input": {"file": "samples/imu_test.osv"}}).select.imu
              is IMU_SELECT_DEFAULT)
        check("an explicit select.imu is kept either way",
              _prepare({"name": "imu-off", "input": {"file": "samples/imu_test.osv"},
                        "select": {"imu": False}}).select.imu is False
              and _prepare({"name": "imu-on", "input": {"file": "samples/imu_test.osv"},
                            "select": {"imu": True}}).select.imu is True)
    check("a stitched request never gets select.imu",
          _prepare({"name": "imu-mp4", "input": {"file": "samples/imu_test.mp4"}}).select.imu is False)
    if _t_imu.is_file():
        check("an .OSV request without sfm.upright gets the default, an explicit one is kept",
              _prepare({"name": "up", "input": {"file": "samples/imu_test.osv"}}).sfm.upright
              is UPRIGHT_DEFAULT
              and _prepare({"name": "up-off", "input": {"file": "samples/imu_test.osv"},
                            "sfm": {"upright": False}}).sfm.upright is False)
    check("a stitched request never gets sfm.upright",
          _prepare({"name": "up-mp4", "input": {"file": "samples/imu_test.mp4"}}).sfm.upright is False)
    _t_imu.unlink(missing_ok=True)
    _t_mp4.unlink(missing_ok=True)

print()
print("FAILURES:", fails if fails else "none")
raise SystemExit(1 if fails else 0)
