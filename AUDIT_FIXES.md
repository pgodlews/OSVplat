# Audit fixes: instructions for coding agents

Audit of the tree at `1a6d71a` (main), 2026-11-12. Every tracked file was read
except the three images under `docs/images/`. Each item below says what is
wrong, where, how to fix it, and how to know it is fixed.

## Status

| Item | State |
|---|---|
| A1 | Fixed on `claude/audit-fixes-a`: `32_pick_model.py` counts frames; `STITCHED_SFM` term. Not run against pycolmap: run a `perspective_overlapping` job on a CUDA host before merging. |
| A2 | **Not a bug.** LichtFeld 09d33818 scales with `lround` in float32 (`OptimizationParameters::scale_steps`), which recovers every `iter` from 100 to 300000 from the 6-digit flag. A test in `test_stages.py` now locks that in. |
| A3 | Fixed on `claude/audit-fixes-a` (orphans, `QUEUE_RUN_KEEP_DAYS`). |
| A4 | Fixed on `claude/audit-fixes-a` (Stop after, import panel, bundle link); checked in a browser against prep and train services. |
| A5 | Fixed on `claude/audit-fixes-a`. Not checked with `docker compose config` (no Docker on the machine that made the change). |
| A6 | Fixed on `claude/audit-fixes-a`. |
| A7–A10 | Fixed (`app/jobrows.py`; one 90 % threshold passed from 88 to 85; `extra_args` reduced to flag names; `exclusive()` clears its hold on a failed probe). |
| B1–B4, B8–B12 | Fixed. B1: comments only; the SfM estimate seed is *not* refit (no fisheye-sfm-4 history to fit from here). |
| B5 | Presets fixed. "currently 0.1.4-rc6" is accurate (`:latest` = 0.1.4-rc6, checked 2026-11-12). Decided: update README and docs/docker.md to 0.2.0 once rc5 is tested, signed and published. |
| B6 | The tables already name their build and say they predate v0.5.4. Decided: one Standard run on clip 0005 during the rc5 test adds a 0.5.4 row. |
| B7 | Fixed: `app/osv.py`; the import failure is printed at startup and answered with a 400. |
| B13 | Camera serial blanked in debug bundles. Decided: the GPS reference in `alignment.json` stays, documented in docs/cloud.md. |
| B14 | `publish_image.sh` fixed. `numpy` in `venv_gs`: 0.2.0-rc4 carries 2.5.3 (checked 2026-11-12; `venv` is pinned at 2.5.2). Decided: pin `numpy==2.5.3` in `setup_gsplat_venv.sh`, with a check after the transformers install like the torch one, the next time that script changes for another reason. Editing it now would make rc5 recompile gsplat for six architectures (~4.5 h) for no change in the image. `30_run_sfm.py` vs pycolmap 4.2.1's `panorama.py`: needs a CUDA host. |
| C1 | Fixed: `app/uploads.py`; the result tar is hashed while written and streamed. A damaged-ETag result upload now backs off between retries, as the other three already did. |
| C2 | uuid map cached, `VramSampler` removed (peak from `ResourceSampler`). The 2 s `/api/status` cache was **not** done: a stale `held`/`foreign` flag for little gain. |
| C3, C4, C6 | Fixed. C6 adds no key term: nothing read `frames/telemetry.csv`. |
| C5 | Decided and done: `87_fisheye_masks.py` deletes `erp/` after a successful run. No key change. |
| C7 | Decided and done: the select key hashes the unused field at its default. UI jobs keep their keys; target-mode API jobs (and window mode with a custom `target_panos`) move once. |
| C8 | Log stream ends with its stage; redundant prune pattern and per-frame constant fixed. Dockerfile dedup skipped (layer sharing would need measuring). |
| D | Done. Decided: `scripts/detect_camera.py` deleted; `QUEUE_MASK_PY` stays (documented). |

## How to work through this file

- Read `AGENTS.md` first. Its rules win over anything here.
- One item per commit (or PR). Items are independent unless one says otherwise.
- Run the suites from `AGENTS.md`, "Tests", before and after every change.
  Baseline on 2026-11-12: all 17 documented suites pass on a Mac with no GPU
  (`queue/test_api.py` and `scripts/test_fisheye.py` were not run: they need a
  running service and pycolmap).
- **Cache keys.** Each item states whether a version term must move. If you
  change what a stage writes and the item does not mention keys, stop and ask.
  `queue/test_stages.py` pins keys; change a pin only where an item says so.
- **Do not change pins** (`scripts/setup_*.sh`, `LFS_REF`, torch, pycolmap).
- **Do not write a number into the docs that you did not measure.** Where an
  item needs a measurement you cannot make (no GPU, no clip), leave the doc
  claim alone and say so in the PR.
- Add a regression test for every behaviour fix. New tests set `SPLAT_ROOT`,
  `QUEUE_ROOT`, `QUEUE_GPUS=""` and `QUEUE_PREP_BACKEND="cuda"` before importing
  `app.*`.
- Confidence labels: **confirmed** = follows from the code as read;
  **needs a run** = the reasoning is sound but was not executed here (no GPU,
  pycolmap or LichtFeld on the audit machine). Confirm before fixing.

Priority: A (broken) → B (inconsistent) → C (inefficient) → D (dead).
Suggested first five: A1, A3, A2, A4, A5.

---

## A. Broken

### A1. Stitched perspective SfM counts images as frames, so its registration guard never fires — needs a run

- **Where:** `scripts/32_pick_model.py:50,123` (`num_reg_frames: len(kept_ids)`,
  where `kept_ids` are image ids); `queue/app/stages.py:582-597`
  (`registration_pct = 100 * num_reg_frames / n_panos`,
  `MIN_STITCHED_REGISTRATION_PCT`).
- **Problem:** `sfm.render=perspective_*` registers 12 pinhole images per
  panorama. The percentage is then up to 1200 %, so neither the 50 % failure
  nor the "< 90 %" warning can trigger, and a fragment gets cached and trained.
  `88_fisheye_sfm.py:201-203` already works around the same thing for the rig.
- **Fix:** in `32_pick_model.py` report frames and images separately:
  `num_reg_frames` = distinct `frame_id` among kept images, plus a new
  `num_reg_images`. Sort models by registered frames. Keep the per-image
  `dropped_cameras` list.
- **Keys:** the model on disk does not change, only `summary`/`.done` numbers.
  No term covers stitched SfM alone and `config_version` would redo everything.
  Add a term (for example `STITCHED_SFM = "stitched-sfm-1"`) that joins `k_sfm`
  only when `not is_fisheye and sfm.render != "spherical"`, so spherical and
  fisheye pins stay. Add the term to `AGENTS.md` "Cache keys". Adding it to
  `handoff.version_terms()` as well would make every bundle from an older
  build fail the version check on import, so only do that if the owner
  agrees; the per-key comparison in `check_manifest` already catches a
  mismatch.
- **Done when:** a unit test feeds `sfm_finalize` a summary with 24 images
  from 2 of 100 panoramas and it raises; a real perspective run still passes.

### A2. `--steps-scaler` is printed with 6 significant digits — needs a run

- **Where:** `queue/app/stages.py:775` (`f"{scaler:g}"`), `:779` (save steps),
  checked against `effective_iters` in `train_finalize` (`:1015-1027`).
- **Problem:** `iter=10000` becomes `--steps-scaler 0.333333`, which is
  9999.99 steps; `iter=7000` → 6999.99, `31000` → 30999.9, `100000` → 99999.9.
  If LichtFeld truncates instead of rounding, a run that completed is failed
  with "training stopped at step 9999 of 10000" and its result discarded.
- **Fix:** first read how LichtFeld at `LFS_REF` turns the scaled value into an
  iteration count (the comment at `:776-778` says `--save-steps` uses `lround`;
  check iterations too). Regardless, format with `format(scaler, ".12g")`.
- **Keys:** none (argv only; `k_train` hashes `iter`, not the flag text).
- **Done when:** `test_stages.py` asserts that for iter in
  7000, 10000, 20000, 31000, 100000, `round(30000 * float(flag)) == iter` and
  `abs(30000 * float(flag) - iter) < 1e-6`.

### A3. Disk used by failed stages and by `runs/` is invisible to retention — confirmed

- **Where:** `queue/app/worker.py:797` (`db.cache_put` only after success),
  `:536-543` (`_cache_ok` deletes the cache row, leaves the directory);
  `queue/app/retention.py:114` (`gc_cache` walks `db.list_cache()` only);
  nothing prunes `QUEUE_ROOT/runs/job*/`.
- **Problem:** a failed or cancelled stage leaves its directory (a frames dump
  is tens of GB; a failed training holds `project.licht`). It is in no table,
  so `ensure_space` cannot evict it and `cache_total` does not count it. It
  goes away only if the same key is rebuilt. Likewise `runs/` keeps every
  `debug.tar` (up to 4.5 GB), handoff tar, restore tar and failed-upload result
  tar until `POST /api/jobs/clear?purge=true`. This is the "full disk at 80 %"
  failure the module exists to prevent.
- **Fix:**
  1. `retention.orphans()`: directories `CACHE_ROOT/<stage>/<key>` that have no
     parsable `.done`, no live lock (`lock_holder_alive`), and whose key is not
     in `protected_keys()`. Also stale `CACHE_ROOT/.handoff-*` staging dirs.
  2. `gc_cache` evicts orphans **before** any valid entry, and reports them.
     Do not delete at failure time: `debugdump` (level `artifacts`) reads the
     failed stage's directory.
  3. `retention.status()` and `/api/cache` report orphan bytes.
  4. Add `QUEUE_RUN_KEEP_DAYS` (default 30, 0 = keep): for jobs in a finished
     state older than that, delete the large files in the run dir (`*.tar`),
     keep the small JSON records. Never touch a job that is queued, running or
     awaiting review, or a restore tar of a failed job younger than the limit.
     Call it where `prune_logs` is called. Document it in `queue/README.md`.
- **Keys:** none.
- **Done when:** tests cover: an orphan is evicted before a valid entry; a
  locked or protected dir is not; a debug tar past the limit is removed and
  `debug.json` stays.

### A4. The web UI cannot submit a job on a prep, Mac or train install — confirmed

- **Where:** `queue/app/static/index.html` (no `run_until` control; grep finds
  none), `queue/app/main.py:379-405` (`_refuse_for_image`), `/api/status`
  already returns `image_variant` (`main.py:1092`) and the UI ignores it.
  `queue/run_mac.sh:36` prints the UI URL for an install where every job needs
  `run_until`.
- **Fix:** in `index.html`, dependency-free:
  - a "Stop after" select (`frames`, `select`, `mask`, `sfm`, empty = full run)
    sent as `config.run_until`;
  - when `image_variant == "prep"`: default it to `sfm`, remove the empty
    option, and grey out the training fields;
  - when `image_variant == "train"`: replace the New job form with the list
    from `GET /api/handoff/bundles` and an Import button
    (`POST /api/handoff/import`);
  - show the built handoff bundle's download link on a job that has one.
- **Done when:** on a service started with `OSVPLAT_VARIANT=prep`, a job can be
  queued from the browser and ends `done` with a bundle link. Update the UI
  section and the state diagram in `queue/README.md` if labels change.

### A5. `compose.yaml` does not pass settings that `docs/docker.md` says go in `.env` — confirmed by reading; check with `docker compose config`

- **Where:** `compose.yaml:17-25` forwards 8 variables. `docs/docker.md:94-101`
  lists `QUEUE_PREP_BACKEND`, `QUEUE_CPUS`, `SSH_PUBLIC_KEYS`, `QUEUE_BIND`,
  `INPUT_URL`, `OUTPUT_UPLOAD_URL` as `.env` settings. Compose reads `.env`
  for `${...}` substitution only, so these never reach the container.
- **Fix:** add to the service
  `env_file: [{path: .env, required: false}]` and keep the explicit
  `environment:` entries that have defaults. Check that `UID`/`GID`/`IMAGE`
  arriving in the container environment do no harm. Note in `docs/docker.md`
  that SSH needs the container to run as root (`docker/entrypoint.sh:85`),
  which the default `user:` line prevents.
- **Done when:** `docker compose config` shows `QUEUE_CPUS` in the service
  environment after it is set in `.env`.

### A6. Time estimates ignore `run_until` — confirmed

- **Where:** `queue/app/estimate.py:162-213` (no reference to `run_until`);
  `queue/app/progress.py:256,288` use `plan["total"]` for slot occupancy.
- **Problem:** a prep job's plan includes train and export, so its
  `plan_total`, the queue drain time and the next job's `starts_in` are too
  long by the whole training time.
- **Fix:** in `estimate()`, zero every stage after `cfg.run_until` before
  summing. Add `run_until` to `_assumptions`.
- **Done when:** `test_stages.py` (or `test_regressions.py`) checks that a
  `run_until="sfm"` estimate has `train == export == 0`.

### A7. A job resumed without a bundle has no stage rows — confirmed

- **Where:** `queue/app/checkpoints.py:552` (`db.create_job` only), compare
  `main.py:408-415` (`_stage_rows`) and `handoff.py:779-787`.
- **Problem:** until the worker reaches each stage, the job has no rows in
  `stages`. `retention.protected_keys()` reads that table, so the upstream
  entries the resume depends on can be evicted while it is queued; the UI
  shows no stages and there is no plan.
- **Fix:** move `_stage_rows` out of `main.py` into a module both can import
  (for example `db.py` or a small `jobrows.py`), call it from `import_restore`
  after `create_job`, and store a plan when the clip probes.
- **Done when:** `test_checkpoint.py` asserts the queued resume job's keys are
  in `protected_keys()`.

### A8. Three different registration thresholds for the fisheye rig — confirmed

- **Where:** `scripts/88_fisheye_sfm.py:49` (`MIN_REGISTERED = 0.9`, unused),
  `:127` (`--min-registered` default 0.85, the one actually checked at
  `:163`), `scripts/85_fisheye_dataset.py:104` (refuses under 90 % of
  instants), `queue/app/stages.py:587-590` (comment says 85 enforces 90 %).
- **Problem:** between 85 % and 90 % the run passes 88's check, spends time on
  84, 83 and `upright`, then dies in 85 with a message about "a fragment" and
  sibling models.
- **Fix:** one constant, 0.90 (today's effective limit), in 88; delete the
  dead constant; have 88 pass the value to 85 through a new
  `--min-registered` argument so the two cannot drift. Fix the comment in
  `stages.py`.
- **Keys:** none (the same runs fail, earlier and with the right message). If
  the owner wants 85 % instead, that changes what gets cached: bump
  `FISHEYE_SFM`.

### A9. Job telemetry carries free text — confirmed

- **Where:** `queue/app/telemetry.py:704-712` (`_config` drops only `name` and
  `input.file`).
- **Problem:** `train.extra_args` is copied verbatim and can hold paths
  (`--eval-mask "masks:/home/…"`); `mask.prompts` and `mask.attached` are user
  text. `AGENTS.md` says telemetry records no paths.
- **Fix:** replace `extra_args` with the sorted flag names only
  (`shlex.split`, keep tokens starting with `-`, cut at `=`). Keep prompts
  (they are a pipeline option) but pass them through `_redactor`. State both
  in `docs/job-telemetry.md`.
- **Done when:** `test_telemetry.py` asserts a path in `extra_args` does not
  appear in `telemetry.json`.

### A10. `worker.exclusive()` can leave the queue held forever — confirmed, low probability

- **Where:** `queue/app/worker.py:196-197`: `_exclusive.set()` then
  `gpu.free_gpus(...)` outside any `try`.
- **Fix:** wrap so that any exception between `set()` and the `yield` clears
  the event and re-raises.

---

## B. Inconsistent

### B1. Comments and one estimate still describe a two-pass rig SfM

- `88_fisheye_sfm.py:147` prints `FISHEYE PASS 1/1`; its docstring says one
  pass is enough. Stale text: `queue/app/stages.py:631,661-663`,
  `queue/app/progress.py:78-79`, `queue/app/jobs.py:451`,
  `scripts/87_fisheye_masks.py:16`. Rewrite them; keep the pass parsing (it
  costs nothing and `test_stages.py:972` covers it).
- `queue/app/estimate.py:28-36`: the mapping fit `0.0794 * n^1.7` and the
  comment "recurs at every ~10 % of growth" predate `FISHEYE_SFM` 2 to 4
  (`82_fisheye_sfm.py:156-158` now uses ratio 2.0 and 2 refinements, and
  reports mapping roughly halved). The seed is therefore stale. **Refit only
  from measured runs** (the queue's own history, fisheye jobs keyed
  `fisheye-sfm-4`); if you have no data, fix the comment to say the fit
  predates version 4 and leave the numbers.

### B2. `88_fisheye_sfm.py` docstring says `--overlap 10`, the default is 15

`scripts/88_fisheye_sfm.py:25` vs `:126`; `82_fisheye_sfm.py:77` defaults to
10 and is always given 88's value. Fix the docstring. Do not change the value
(it would need a `FISHEYE_SFM` bump).

### B3. Hard-coded worker counts ignore `SPLAT_THREADS`

`queue/app/resources.py` sizes stages to the container's CPU quota, but:
`scripts/80_fisheye_frames.py:223` and `scripts/20_select_sharp.py:20` use
`ProcessPoolExecutor(16)`; `scripts/88_fisheye_sfm.py:114` and
`scripts/89_fisheye_train_view.py:73` use `ThreadPoolExecutor(8)`. Use
`int(os.environ.get("SPLAT_THREADS") or 0) or os.cpu_count()` capped as now
(16 and 8), or `io_pool.io_threads(cap)`. No key change (thread count is in no
key, see the `resources.py` docstring).

### B4. `deploy.sh` ships 9 of the 12 queue suites and never removes old scripts

- `queue/deploy.sh:52-56` omits `test_debugdump.py`, `test_handoff.py`,
  `test_checkpoint.py`, `test_prep_backend.py`. Use `"$REPO"/queue/test_*.py`
  as the Dockerfile does.
- `queue/deploy.sh:60` and `queue/run_mac.sh:23` rsync `scripts/` without
  `--delete`, so a script removed from the repo stays on the target, against
  the promise in `queue/app/config.py:17-18`. Add `--delete --exclude
  '__pycache__'` after confirming nothing else lives in `$ROOT/scripts`.

### B5. README presets and version text

- `README.md` "Usage" lists four presets; `index.html:333-339` has five
  (`High`, 4.5M cap, is missing). "Max: … and SH degree 3" is stale: Standard
  is SH 3 too (`index.html:336`, `jobs.py:239`). Max differs by
  `max_width 7680` only.
- "currently 0.1.4-rc6" (`README.md` twice, `docs/docker.md:47`) while tags go
  to `v0.2.0-rc4` and `Dockerfile:55` cites 0.2.0-rc1. `:latest` is the last
  *signed* release, so this may be right. **Ask the owner; do not guess.**

### B6. The trainer moved to LichtFeld 0.5.4 "unmeasured"

Commit `de235a2` says so, and `TRAINER = "lichtfeld-09d33818"`. `README.md`
("Standard: … the settings every measurement in the docs used") and the tables
in `docs/how-it-works.md` ("Trainer options, measured") were measured on the
previous trainer. Per `AGENTS.md` a claim needs a measurement: either re-run
the reference clip and update, or add one line under each table naming the
trainer build it was measured with. Owner's call which; an agent without a GPU
does the second only after asking.

### B7. `main.py` finds `scripts/` from its own file path, and fails silently

- **Where:** `queue/app/main.py:35-49` (sys.path edits, `detect_camera = None`
  fallback), `:314` and `queue/app/estimate.py:178` (`from avata_motion import
  …` relying on that path edit).
- **Problem:** it works today only because `deploy.sh`, `run_mac.sh` and the
  Dockerfile all put `scripts/` two levels above `app/`. If the import fails,
  nothing is logged and behaviour changes: Osmo clips stop defaulting to
  masks, distance selection is hidden, and a distance-mode submit raises
  `ModuleNotFoundError` (a 500).
- **Fix:** one small module (for example `queue/app/osv.py`) that inserts
  `config.SCRIPTS` into `sys.path` once and exposes `detect_camera`,
  `avata_load`, `avata_plan`. On import failure print one ERROR line at
  startup and make the affected API paths return a 400 that names the cause.
  Both scripts are stdlib-only, so the CI guardrail still holds.

### B8. Host probes implemented twice, differently

`queue/app/telemetry.py:154-174` reads only cgroup v2 `memory.max`;
`queue/app/resources.py:116-130` reads v1 as well. `telemetry._cpu`
(`:116-151`) recomputes affinity and effective CPUs and ignores `QUEUE_CPUS`.
Make telemetry call `resources.memory_limit_bytes()`,
`resources.affinity_cpus()` and `resources.effective_cpus()`; keep the field
names in `docs/job-telemetry.md` unchanged.

### B9. Small duplications that have already drifted or will

- Flag parsing: `config.py:111-112` repeats `_flag` (defined at `:114`);
  `handoff.py:73` and `checkpoints.py:58` parse by hand; `START_PAUSED`
  (`config.py:99`) accepts only `"1"`. Move `_flag` to the top of `config.py`
  and use it everywhere.
- `config.py:64-65`: a comment about schedulable GPUs separated from the code
  it describes (`:87-90`). Move it.
- `.osv` literal: `estimate.py:63,81`, `main.py:1182`. Use
  `jobs.FISHEYE_EXTS` (in the SQL, build the condition from it).
- `30000`: `jobs.py:233,289` and `stages.py:767` (`LFS_DEFAULT_ITER`). One
  constant in `jobs.py`, imported by `stages.py`.
- Export globs: `stages.py:1056` and `:1513` repeat `ARTIFACT_EXTS` (`:935`).
- `db.py:19,35`: schema comments omit states `waiting`/`cancelled` and the
  columns added by `init()` (`review_state`, `review_note`, `handoff`,
  `resume`).
- `worker.py:397`: local `resources` shadows the imported module inside
  `run_stage`. Rename the local (`res_sampler`).

### B10. Undocumented settings and routes

Not in any doc: `QUEUE_STALL_TIMEOUT`, `QUEUE_FIRST_PROGRESS_GRACE`,
`QUEUE_MAX_CONCURRENT`, `QUEUE_LFS_PYTHON`, `QUEUE_MODELS_DIR`,
`QUEUE_PREEMPT_POLL_S`; routes `POST /api/concurrency`,
`GET /api/jobs/{id}/thumb.jpg`, `GET /api/jobs/{id}/restore/download`. Add them
to `queue/README.md` (settings table and API list). The reverse check passed:
every variable and path the docs name exists in the code.

### B11. Mask messages say "person" whatever was prompted

`queue/app/stages.py:445-447,458-459` and `scripts/70_person_masks.py:465`.
With `prompts: ["boat"]` the failure reads "person masking found nobody". Name
the prompts in the message.

### B12. Stale references to things not in this repository

- `scripts/85_fisheye_dataset.py:6,87`: `86_fisheye_eval.py`,
  `94_fisheye_ab_eval.py` do not exist here.
- `scripts/82_fisheye_sfm.py:16`: "fmasks_colmap/ from 81_fisheye_stitch.py";
  the masks come from `88_fisheye_sfm.py:build_masks`.
- `scripts/20_select_sharp.py:3,7-8`, `scripts/30_run_sfm.py:41-42`: defaults
  `run1/erp10`, `run1/panos` from the lab notebook. Make the arguments
  required.
- `queue/app/worker.py:4-5`: "two concurrent jobs on this box".
- `queue/app/stages.py:1045-1049`: `export_argv` docstring lists `spz` as a
  future conversion; training exports it already.

### B13. Debug bundles carry a GPS reference and the camera serial; the docs do not say so

At level `artifacts`, `queue/app/debugdump.py:358,365` add `sfm/*.json` and
`frames/*.json`. That includes `sfm/alignment.json`
(`geo.reference_lat_lon_alt`, `scripts/upright.py:364`) and
`calibration.json` / `calibration_refined.json` (`device.serial`,
`scripts/osv_meta.py:335`). `docs/cloud.md` ("Debug bundles", near line 500)
lists only what is stripped. Bundles are meant not to be anonymous, so:
document both in that section, and blank `device.serial` in the copies that go
into the bundle. Ask the owner before removing the GPS reference.

### B14. Shell and packaging nits

- `scripts/publish_image.sh:248`: `$VERSION[-target]` inside a string;
  shellcheck SC1087. Write `${VERSION}[-target]`.
- `scripts/setup_gsplat_venv.sh:26` and `scripts/setup_mac.sh:83` install
  `numpy` unpinned in `venv_gs`, while `venv` pins `numpy==2.5.2`. Report to
  the owner; do not pin on your own (pins are measured).
- `scripts/30_run_sfm.py:89-98` is a near-copy of
  `pycolmap.panorama.run_spherical` with the note "re-check after a pycolmap
  upgrade"; pycolmap moved 4.2.0 → 4.2.1 (`jobs.py:383`). Diff it against the
  installed `panorama.py` on a CUDA host and record the result in the comment.

---

## C. Inefficient

### C1. Four copies of the upload code; the result upload holds the whole tar in memory

- **Where:** `outputs.py:256-318`, `handoff.py:406-475`,
  `checkpoints.py:254-310`, `debugdump.py:481-547`. `handoff._request` and
  `debugdump._request` are identical. `outputs.upload` does
  `data = arc.read_bytes()` (`:282`) and hashes the archive in a second pass
  (`:210-214`); the other three stream the file and (handoff, checkpoints)
  hash while writing. A 6M-splat SH 3 result is over 1.5 GB of RAM, doubled
  for a multipart POST.
- **Fix:** one module (`queue/app/uploads.py`): `send(target, job_id, name,
  path, md5, timeout, attempts)` with streaming PUT, the POST fallback, retry
  with backoff, the ETag check and the error text rule ("HTTP code reason"
  only); and `single_object_owner(target, dest, job_id, status_fn)`. Make all
  four callers use it; give `build_archive` a hashing writer. Behaviour and
  the JSON status files must not change: the existing tests in
  `test_remote.py`, `test_handoff.py`, `test_checkpoint.py`,
  `test_debugdump.py` are the contract.

### C2. Too many `nvidia-smi` processes

- `gpu.compute_procs()` runs two queries each call; the uuid → index map
  (`gpu.py:45-56`) never changes. Cache it; drop the cache when a query fails.
- `telemetry.ResourceSampler.sample` (`telemetry.py:532-538`) runs the sample
  query plus `compute_procs()` every second for a stage's first 120 s: three
  processes a second per job.
- `worker.VramSampler` (`worker.py:313-341`) measures the same number the
  resource sampler already has (`own_mib`, `telemetry.py:512-517`). Expose a
  `own_mem_peak_mib` on `ResourceSampler`, set `peak_vram_mib` from it in
  `run_stage`, and delete `VramSampler`. `train_finalize` and the
  `peak_vram_mib` metric must keep working (`test_worker.py`).
- `gpu.status()` runs three queries and is called by `/api/status` every 4 s
  per open browser tab (`worker.running_state`, `worker.py:1112`). Give the
  API path a 2 s cache; the dispatcher and `reserve_gpu` must keep reading
  fresh state.

### C3. `/api/jobs` runs two queries and a config validation per row, every 4 s

`queue/app/main.py:546-598`: `_job_dict` calls `db.job_stages` and
`db.job_metrics` per job for up to 200 rows; `/api/status` (`:1084`) builds
the active jobs again. Fetch stages and metrics for the page in two
`WHERE job_id IN (…)` queries and pass them in. Output must be byte-identical
(`test_api.py`, plus a new unit test comparing old and new for a fixture DB).

### C4. `detect_camera` reads the whole telemetry track each time

`scripts/osv_meta.py:255-256` loads the full `camd` payload (every IMU sample;
for an Avata clip, all samples of the track joined) to read one header record.
It runs per `.OSV` on every `GET /api/inputs` and again on every submit and
estimate (`main.py:199,288`). Either cache by `(path, size, mtime_ns)` as
`probe.ffprobe` does, or better, stop after the first record: read only the
start of the payload. Keep `scripts/test_*` passing; add a test that two
calls read the file once.

### C5. The fisheye mask stage keeps its stitched panoramas

`scripts/87_fisheye_masks.py:53-61` writes `MASK_DIR/erp/` (JPEG q95,
2W × W per rig frame) for the detector, then never reads it again; nothing
else does (`grep -rn "erp" queue/app scripts`). It is most of the mask
stage's 0.248 B/px (`stages.py:1428`) and is not shipped in handoff bundles.
Delete `erp/` at the end of a successful run. Keys: the files later stages
read are unchanged, so no bump, but confirm with the owner under the
cache-key rule. Leave `PREP_BYTES_PER_PX` alone (it is a measured peak) and
note in its comment that the kept size is now smaller.

### C6. Every fisheye frames stage writes an unused `telemetry.csv` with GPS

`queue/app/stages.py:1137` runs `osv_meta.py <clip> <frames dir>`, whose
`main` (`osv_meta.py:371-409`) also writes `telemetry.csv`: one row per record
with latitude and longitude. The pipeline reads only `calibration.json`. Add
`--calibration-only` to `osv_meta.py`, pass it from the stage, keep the
default for hand use. No key change.

### C7. The select key hashes fields the mode does not use

`queue/app/jobs.py:524`: in `window` mode `target_panos` is hashed, in
`target` mode `window` is (the API also rewrites `window` to 3 for `.OSV`,
`main.py:302-305`). Two jobs that select the same frames get different keys
and redo select, masks, SfM and training. Same class as the `k_mask` fix
described at `jobs.py:555-562`. Normalise the unused field to its default
before hashing. **This moves existing keys** for target-mode jobs and for
window-mode jobs with a non-default `target_panos`: update the pins in
`test_stages.py` on purpose and say so in the commit. Low value if targets
are rarely changed; ask the owner before doing it.

### C8. Smaller ones

- `main.py:874-893`: the log follower polls until the *job* ends, not the
  stage it follows. End the stream when that stage row leaves `running`.
- `retention.py:180-181`: pattern `cmp_*.jpg` is covered by `*.jpg`.
- `scripts/70_person_masks.py:413-414,426-427`: the solid-angle denominator is
  recomputed per frame; compute it once.
- `Dockerfile:107-129,141-164,189-220`: the same COPY/chmod/label block three
  times. Optional: an intermediate stage or a shared ARG block. Check image
  layer sharing and size before and after; skip if either gets worse.

---

## D. No longer needed

Remove, one commit, tests green:

| What | Where |
|---|---|
| `cache_get` (no caller) | `queue/app/db.py:293` |
| `TRAIN_DONE` (no user) | `queue/app/stages.py:868` |
| `FISHEYE_NOTE` (superseded by the f-string at `:219`) | `queue/app/main.py:178` |
| `MIN_REGISTERED` | `scripts/88_fisheye_sfm.py:49` (with A8) |
| unused import `Optional` | `queue/app/benchmark.py:28` |
| unused import `shutil` | `queue/app/checkpoints.py:37` |
| unused import `CACHE_ROOT` | `queue/app/retention.py:21` |
| unused import `numpy` | `scripts/81_fisheye_stitch.py:37` |
| unused imports `signal`, `subprocess`, `ORDER` | `queue/test_checkpoint.py:18,19,67` |
| f-strings with no placeholder | `scripts/detect_camera.py:28,29,34,35,39` |
| `VramSampler` | `queue/app/worker.py:313` (with C2) |

Check before touching:

- `queue/test_remote.py:30` imports `app.worker` and does not use the name; it
  may be imported for its side effects. Remove only if the suite still passes.
- `scripts/test_benchmark.py:22` imports `gsplat` as an availability probe.
  Leave it.
- `scripts/detect_camera.py` is referenced by nothing (docs, tests, code) and
  prints the camera's serial number. Ask the owner: document it in
  `docs/osmo360-telemetry.md` or delete it.
- `QUEUE_MASK_PY` (`config.py:26-30`) exists for "an older install's separate
  venv_sam". Ask the owner whether any install still uses it.

Leave alone (looks dead, is not):

- `TrainCfg.background_improvements` (`jobs.py:245-247`): every stored config
  carries it.
- `export` as a stage with no command (`stages.py:1044`): it is part of the
  cache chain and of every pinned key.
- `avata_motion._load`'s unused `size`/`mtime_ns` parameters: they are the
  `lru_cache` key.
- `scripts/benchmark.py:234-243`: pyflakes reports undefined names there; it
  is a false positive from `del` after lambdas.

---

## Checked and found sound (do not re-audit)

- All 17 documented suites pass; `queue/app` imports no numpy, cv2 or torch.
- Every `troubleshooting #N` cited in code or docs exists (1 to 38).
- Every environment variable and API path named in the docs exists in code.
- Path handling on the routes that take names (`input_path`, review overlay,
  `bundle_file`, `restore_file`, handoff `_extract`) resolves and contains.
- Lock handling in `stages.py` / `worker.py` (exclusive create, pid identity,
  re-stamp before finalize, `.done` before release).

## Not verified by this audit

- Anything needing a GPU, pycolmap, LichtFeld or a clip: A1 and A2 above,
  the LichtFeld patches applying at `LFS_REF`, the Docker build.
- That the pinned package versions in `scripts/setup_*.sh` resolve today.
- `queue/test_api.py` and `scripts/test_fisheye.py`.
