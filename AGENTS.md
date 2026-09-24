# AGENTS.md

Notes for coding agents and contributors. User-facing docs are in README.md and
docs/; this file is the rules that are not visible from the code.

## Layout

- `queue/app/` — FastAPI service, job model (`jobs.py`), stage runners
  (`stages.py`), scheduler (`worker.py`), web UI (`static/index.html`, no build step).
- `scripts/` — the pipeline stages the queue shells out to, one venv each:
  `venv` (pycolmap, OpenCV) for selection/SfM/.OSV readers, `venv_gs` (torch,
  gsplat) for masks and renders. The service's own venv has no numpy/OpenCV,
  so heavy work belongs in a script, not in `queue/app/`.
- `scripts/setup_*.sh` — tool builds, shared by the native install and the
  Dockerfile. Keep them runnable without a GPU (`CUDA_ARCH` set explicitly).

## Tests

Run from the repo root, no GPU needed:

```bash
python3 queue/test_stages.py && python3 queue/test_worker.py
python3 queue/test_regressions.py && python3 queue/test_distance.py
python3 queue/test_telemetry.py && python3 queue/test_remote.py
python3 queue/test_benchmark.py && python3 queue/test_hoststats.py
python3 scripts/test_avata_motion.py && python3 scripts/test_imu_select.py
python3 scripts/test_benchmark.py                      # numpy + OpenCV; GPU part with torch
~/splat/venv/bin/python scripts/test_fisheye.py        # needs pycolmap
```

`queue/test_api.py` needs a running service and its token.

Every test must set `SPLAT_ROOT`/`QUEUE_ROOT` to a temporary directory and
`QUEUE_GPUS=""` **before** importing `app.*` — `config.py` reads them at import
time, and a test that forgot once wrote into a live queue database.

## Rules

- **Cache keys.** Each stage is cached by a hash of the options that affect
  it (`JobConfig.keys()`). If you change what a script outputs for the same
  options, bump its version term in `queue/app/jobs.py` so stale caches are not
  served: `FISHEYE_SFM` for the fisheye reconstruction (`82`/`88_fisheye_sfm.py`,
  `colmap_incremental.py`), `FISHEYE_PIPELINE` for any other fisheye script,
  `IMU_SELECT` for gyro selection scoring, `config_version` for everything. Adding an option that
  does not change pixels (like `mask.review`) must *not* enter the key.
  `test_stages.py` pins existing keys; update the pins only on purpose.
- **Never `pkill -f` / `pgrep -f`.** They match the shell running them.
  Processes are tracked and killed by PID and process group.
- **Fail loudly.** A stage that produced nothing useful (no masks, a 2-frame
  SfM model, a cancelled training's partial export) must fail its finalizer,
  never cache. Silent empty results are the most expensive bug class here.
- **Pins are deliberate** (`setup_*.sh`, docs/how-it-works.md "Pinned
  toolchain"). Do not bump torch, pycolmap, gsplat or LichtFeld casually;
  quality numbers in the docs were measured against these.
- **docs/troubleshooting.md numbering is stable.** Code cites `#N`; append new
  entries, never renumber.
- **Claims in docs need a measurement** (clip, numbers, what was compared).
  Nulls are recorded too — see the gyro veto section.
- Never commit clips (`.OSV`, `.mp4`), splats (`.ply`, `.sog`, `.spz`) or
  anything with GPS data, serial numbers or personal paths.
- **Telemetry stays anonymous.** `queue/app/telemetry.py` records no
  hostnames, paths, clip or job names, GPU UUIDs or serials, and redacts logs.
  A new field goes in `docs/job-telemetry.md`, and anything that could name a
  person, place or machine stays out. Writing is on by default; sending is
  only ever opt-in (`QUEUE_TELEMETRY_UPLOAD`).
- The UI is one static HTML file with inline JS/CSS; keep it dependency-free.
- New mask models (a future SAM, etc.) start in `queue/app/mask_backends.py`;
  its docstring lists the other three places to touch. Never ship gated weights.
