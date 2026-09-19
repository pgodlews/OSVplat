# OSVplat queue service

Job model, stage runner, stage cache, FastAPI + web UI. See the
[top-level README](../README.md) for setting up the GPU workstation first.

Runs as a systemd service on the GPU workstation, so jobs survive closing the
browser and logging out.

## Deploy

With Docker the container runs this service; see [docs/docker.md](../docs/docker.md).
Natively:

```bash
./queue/deploy.sh
```

Syncs `queue/app/` and the test files → `~/splat/queue_app/`, `scripts/` →
`~/splat/scripts/`, builds `~/splat/queue_app/venv`, installs a `splat-queue`
systemd unit, restarts it. It prints a URL carrying the access token; open that
once and the cookie it sets covers every later visit.

`BIND=127.0.0.1 ./queue/deploy.sh` keeps the port off the LAN entirely, so the
UI is reachable only from the workstation itself.

## Safety

- **Every request needs the access token.** This service submits GPU work,
  cancels it, deletes history and changes scheduling, so the port is not open.
  `deploy.sh` generates a token on first deploy and keeps it in
  `~/splat/queue_app/.queue_env` (0600), loaded by systemd as an
  `EnvironmentFile`. Send it as `?token=…` (once — it becomes a cookie) or as
  the `X-Queue-Token` header. **With no token configured the service answers
  loopback only**, so a hand-started `uvicorn --host 0.0.0.0` cannot
  accidentally expose an unauthenticated queue.
- **The queue starts paused** — every start, not just the first. `paused` is
  persisted, so a queue someone had resumed used to come back scheduling and the
  guarantee held only until the first press of Resume. Startup now forces it
  back; `QUEUE_START_PAUSED=0` keeps the stored state instead.
- **Cancellation has a deadline.** SIGTERM to the process group, then SIGKILL
  after `QUEUE_CANCEL_GRACE` (default 60 s), so a trainer that will not exit
  cannot hold a GPU with its job already recorded as cancelled.
- **Locks carry process identity**, not just a pid. Pids get reused; restart
  recovery used to signal whatever process held the number, and released the
  lock without waiting for the orphan to actually exit — handing the cache dir
  to a second writer.
- **Foreign-process detection.** Before scheduling, `gpu.py` asks `nvidia-smi` which
  compute processes are on each GPU and skips any GPU carrying a PID the queue did not
  start. A manually launched `LichtFeld-Studio` blocks that GPU for as long as it runs.
- **No `pkill -f`.** Cancellation kills the process group by PID
  (docs/troubleshooting.md #1 — `pkill -f` can kill the shell that runs it).

## Layout on disk

```
~/splat/queue_app/         service code + venv
~/splat/queue/queue.db     SQLite state
~/splat/queue/cache/<stage>/<key>/    content-addressed stage artifacts
~/splat/queue/logs/        one log per job stage
~/splat/scripts/           version-controlled copies of the pipeline scripts
```

## Disk retention

The queue is a machine for filling a disk: a 7 GB clip at 10 fps is tens of
thousands of 8K JPEGs, every sweep variant adds a training directory, and both
are kept so the next job can reuse them. Nothing used to reclaim any of it, and
the failure mode is a full disk 80 minutes into a 95-minute run.

| Knob | Default | Effect |
|---|---|---|
| `QUEUE_MIN_FREE_GB` | 20 | A stage refuses to start below this. It first evicts and prunes to try to get there. |
| `QUEUE_CACHE_BUDGET_GB` | 0 (none) | Ceiling the cache is evicted down to. |
| `QUEUE_LOG_KEEP_DAYS` | 30 | Stage logs older than this are pruned at startup. |
| `QUEUE_RENDER_KEEP_DAYS` | 30 | Comparison sheets older than this are pruned at startup. |
| `QUEUE_NEED_SAFETY` | 1.25 | Margin on the largest entry a stage has produced before, used as its space requirement. 0 falls back to the flat floor. |

The floor alone was not enough: an 8K frames dump is tens of GB and a training
directory can be more, so a stage several times larger than `QUEUE_MIN_FREE_GB`
passed the check, filled the disk, and died 80% of the way in --- the exact
failure the check exists to prevent. Each stage now asks for `QUEUE_NEED_SAFETY`
times the largest entry it has ever produced, whichever is greater, measured
from the cache table rather than guessed from a formula. A stage with no history
yet has nothing to say and the floor stands on its own.

Eviction is least-recently-used and conservative: it skips any entry an
unfinished job depends on --- queued, running, **or parked in
`awaiting_review`** --- and any entry whose directory is locked by a live
process. The review state is the one that is easy to miss and the worst to get
wrong: a job waiting on a human is by definition the one whose entries go
longest untouched, so they are the first the LRU reaches, and evicting the mask
entry deletes the contact sheet out from under the reviewer (its endpoint starts
404ing) and makes approval re-run the masks --- signing off a set nobody saw. A
sweep would likewise lose the shared SfM it was built around. It measures
what it actually reclaimed rather than trusting the recorded size, because
`select/` hardlinks its panoramas out of `frames/` — deleting one of the pair
frees nothing until the other goes too.

## Stage cache

Keys chain: `frames → select → sfm → train → export`. Each key hashes its own
params plus the upstream key, so changing a training flag reuses frames/select/sfm
and every variant trains on byte-identical geometry. A stage is reused when its
cache dir contains a `.done` marker that parses.

**Sharing an *unfinished* stage is the normal case, not an error.** The four
variants of a sweep reach the same uncached `sfm` key within milliseconds of each
other. Exactly one wins the directory (`O_CREAT|O_EXCL` on `.lock`) and the rest
show as `waiting` until its result lands, then take the cache hit. A lock whose
holder pid is dead is reclaimed. `QUEUE_CACHE_WAIT` (default 6 h) bounds the wait.

A stage that is about to be rebuilt starts from an emptied directory: leftovers
from a failed attempt would otherwise mix with the new run's output.

Large inputs are identified by `quick_hash` — size + head, middle and tail chunks, or the
whole file under 24 MB — rather than a full sha256, so a 7 GB clip hashes in well under a
second. The hash is **always computed server-side**; a `quick_hash` in a submitted config is
ignored, or a config copied between clips would inherit the other clip's whole pipeline.
`input.file` must resolve, after symlinks, to a file inside `SPLAT_ROOT`. A queued job
re-checks its input at dispatch and refuses to run if the file was replaced while it waited.

`config_version` (default 2) participates in every key. Bump it to invalidate the cache
after changing a pipeline script or upgrading the trainer — nothing else notices those.

## Avata distance selection (experimental)

Choose **Frame selection → Avata distance · sharpest nearby frame** for a raw
Avata 360 `.OSV`. The existing time-window default is unchanged. The initial
settings are **5 m shot spacing** and a **2 s maximum time gap**; they are starting
values, not an overlap guarantee. Smaller distances retain more frames.

API configuration:

```json
"select": {"mode": "distance", "distance_m": 5, "max_gap_s": 2, "imu": false}
```

The selector integrates the recorded North/East/Down **velocity** at its original
rate. It handles duplicate lens metadata (averaging copies that differ by at most
0.25 m/s per axis) and aligns candidates to ffmpeg's
source frame indices, including trims. It partitions distance into intervals,
shortlists every readable candidate within 20% of the spacing of the one nearest
each interval's centre, and picks the sharpest synchronized pair there. Slow-flight
time splits away from the centre therefore still rank all their candidates.
Fast flight is the limit: at 18 m/s and 10 candidate fps the candidates are
1.8 m apart, so a 5 m spacing usually leaves one choice per target. Widening
the shortlist there makes the spacing as irregular as time windows (checked on
DJI_..._0010), so raise candidate fps (about 20 at 18 m/s) for sharpness choice. The optional
gyro blur veto still applies within this shortlist.

Long intervals are split into shorter time spans to keep consecutive selected
frames at most `max_gap_s` apart, even when hovering. Consequently, this is
approximate distance spacing and the count can exceed path length / spacing.
The 0.15 m/s speed floor suppresses hover noise but also ignores very slow travel;
it is an initial heuristic. No GPS correction or visual overlap test is applied.

The estimate reads the actual telemetry to count selection groups, and warns
when candidate FPS is too low for the requested spacing. Missing/invalid
velocity, unsupported cameras, discontinuous telemetry, and upsampling beyond
the source FPS are refused. This mode is not available for stitched MP4 or Osmo.

`selection.json` records per-frame source index, distance, speed and a relative
NED position estimate, plus spacing diagnostics. `motion_path.json` retains the
provisional path for all candidates. Those positions are dead reckoning for
diagnostics, not calibrated SfM poses. Automatic SfM feedback/reselection is a
future step; this option implements the initial selection only.

Distance mode shares decoded-frame caches with time mode, then uses a separate
versioned selection key. Changing distance or maximum gap invalidates selection
and downstream stages; inactive distance settings do not affect existing keys.

CLI (the candidate folder contains `lens0/` and `lens1/`):

```sh
venv/bin/python scripts/80_fisheye_frames.py --select-only "$CAND" "$OUT" \
  --motion-src samples/flight.OSV --distance-m 5 --max-gap-s 2 --fps 10
```

For trimmed candidates, also pass the extraction's `--start` value. Tests:
`python3 scripts/test_avata_motion.py`, `python3 queue/test_distance.py`.

## Guardrails encoded

| Guard | Background |
|---|---|
| Largest `sparse/` model auto-selected and linked as `sparse/0` | #18 |
| `--gut` forced when the camera model is `EQUIRECTANGULAR`, and `gut=false` refused rather than obeyed | #15 |
| `--images` follows the SfM mode: panoramas for `spherical`, the generated pinhole views for either `perspective_*` | — |
| `.insv` refused (no decoded calibration); a DJI `.OSV` runs the fisheye rig pipeline instead | docs/how-it-works.md, "Fisheye rig" |
| Fisheye training always masks (valid circle) and always uses `--gut`; `train.gut=false` refused | — |
| A clip whose camd holds no lens calibration fails at `frames`, with the reason | — |
| A mask stage that masked nothing is a failure, not a result | #21 |
| `--iter` and `--steps-scaler` rejected together (they multiply) | verified in LichtFeld source |
| Post-SfM gate: registration %, reprojection error, baseline/median-depth | #18 |
| Stray cameras dropped before training, and the gate measured around the *median* camera centre | #19 |
| Kill by PID, never `pkill -f` | #1 |

## Raw DJI `.OSV`: the fisheye rig pipeline

Pick a `.OSV` (Osmo 360 or Avata 360) as the input and the job reconstructs it **without stitching**: both fisheye lenses go to SfM as a two-camera rig seeded from the calibration inside the file, and train as `OPENCV_FISHEYE` cameras. Measured better than stitching first (docs/how-it-works.md, "Fisheye rig"). Nothing to configure: `JobConfig.is_fisheye` is the file extension, and `sfm.render`/`sfm.mapper` must stay at their defaults, because only stitched input has a choice.

The six stages, their names, the cache chain, the review gate and the UI are the stitched path's. Each registry entry dispatches on the input (`stages._by_input`):

| Stage | Stitched MP4 | Raw `.OSV` |
|---|---|---|
| `frames` | ffmpeg fps dump | one ffmpeg decode of both lens streams into `lens0/`, `lens1/`; then `osv_meta.py` writes `calibration.json`, and a clip with no active lens pair fails here |
| `select` | `20_select_sharp.py` | `80_fisheye_frames.py --select-only`: the sharpest instant per window, scored on the blurrier lens; with `select.imu`, candidates the clip's own orientation stream predicts to smear more than 0.5 px beyond the stillest in their window are vetoed first (docs/how-it-works.md, "Gyro blur veto") |
| `mask` | `70_person_masks.py` | `87_fisheye_masks.py`: stitch with the stored calibration, run the same masker (same backends, overlays, sheet, summary, so review works unchanged), carry the masks back into each fisheye |
| `sfm` | `30_run_sfm.py` + `32_pick_model.py` | `88_fisheye_sfm.py`: one rig pass inside an 88° circle, `32_pick_model.py`, refined calibration, angular error, and a dataset with flattened names and valid-circle masks |
| `train` | LichtFeld, masked view when masking | `89_fisheye_train_view.py`: training masks = valid circle AND person, then `exec` LichtFeld (same PID, so cancel, watchdog and progress see the trainer) |
| `export` | shared | shared |

**Cache keys.** `JobConfig.k_frames` hashes a `FISHEYE_PIPELINE` term for `.OSV` input only, so every fisheye key differs from every stitched key, and every stitched job keeps the key it was cached under (`test_stages.py` pins both). Bump `FISHEYE_PIPELINE` after changing a fisheye script to invalidate fisheye caches alone.

**Gyro blur veto, `select.imu`.** Raw `.OSV` only; a stitched job that sets it is refused, because only the `.OSV` carries an orientation stream. The select stage adds `--imu <clip> --calib frames/calibration.json` (and `--start` for a trimmed clip, so each candidate is matched to the right video frame), and the stage record carries the script's summary: windows overruled, and predicted smear with and without the veto. `JobConfig.k_select` hashes an `IMU_SELECT` term only while the option is on, so every select entry cached before it existed keeps its key; bump `IMU_SELECT` after changing how `80_fisheye_frames.py` scores. A request that leaves it out gets `IMU_SELECT_DEFAULT` for `.OSV` input (`main._prepare`), and the UI's checkbox follows the same default through `recommended_imu` in `/api/inputs`. `IMU_SELECT_DEFAULT` is False, so the veto is opt-in: it picks measurably sharper frames indoors, but the splat A/B on `home.OSV` came out a null (+0.04 dB over 64 held-out views). Why and what it measured: docs/how-it-works.md, "Gyro blur veto".

**Always masked, always `--gut`.** Even with person masking off, the valid circle has to be masked: the corners around each image circle are black, and past 90° COLMAP's camera model does not apply. LichtFeld trains fisheye cameras through 3DGUT.

**Estimates** use separate seed constants (`fisheye_*` in `estimate.SEED`), and fisheye jobs fit their own training rate from history, so the two pipelines do not skew each other's. Fisheye SfM is priced superlinearly, because every global bundle adjustment covers the whole model. The price is per-frame extraction, matching and post-processing plus `0.0794 · frames^1.7` s of mapping. That puts 1404 rig frames (a 7-minute clip at window 3) at 5.3 h, which is what a real clip of that size took. Clips of equal length still differ by more than 3×, so read the SfM figure as an order of magnitude.

**Not supported yet:** the compare renderer (`POST /api/compare/render`) refuses fisheye jobs; `93_render_compare.py` draws pinhole views off equirect or perspective models.

**Status:** unit-tested (`test_stages.py`, run locally with temporary `SPLAT_ROOT`/`QUEUE_ROOT`) and used end to end on Osmo 360 and Avata 360 clips.

## Person masking (optional)

Off by default. Enable per job when someone rides along in the camera frame --
handheld, selfie stick, an operator or a companion in shot:

```json
"mask": {"enabled": true, "backend": "maskrcnn"}
"mask": {"enabled": true, "backend": "sam3", "prompts": ["person", "dog"]}
```

Two backends, both supported. `maskrcnn` (default) is torchvision's COCO
person detector: BSD-licensed, nothing to stage, 13 tangent views per frame.
`sam3` is open-vocabulary, reads the equirect directly (2 inferences instead of
13), measured faster and tighter -- but needs its gated weights at
`~/splat/models/sam3` (`scripts/get_mask_weights.sh sam3`) and comes under Meta's
SAM licence. Both run in `venv_gs`. Backends are declared in
`app/mask_backends.py`; one whose weights are missing is listed as unavailable
in `/api/status` and refused at submit with setup instructions. A backend
without prompt support given prompts other than `["person"]` is refused rather
than silently ignored.

The `mask` stage runs `scripts/70_person_masks.py` between `select` and `sfm`
(77 s for 110 8K panoramas on one 3090). The masks then feed **both** consumers:
COLMAP's feature extraction (`--masks`, so no keypoints are taken on people) and
the trainer (`--mask-mode ignore`, which zeroes the photometric weight there).

Deliberate choices in the key chain:

- **`mask` feeds the `sfm` key through a sentinel**, never a conditionally
  omitted field: `_sfm_mask_term()` returns either `k_mask()` or the constant
  `"no-sfm-mask"`, so every unmasked config still agrees on one reconstruction
  while a masked one cannot be served an unmasked SfM. A key that sometimes
  hashes an input and sometimes does not is how two configs come to share an
  entry.
- **`use_for_sfm: false`** keeps the reconstruction shared with an unmasked run
  and masks only the training. That is the cheap A/B: identical geometry, one
  variable.
- **`mask` is in the `train` key even when disabled.** Same reasoning as above.
- **`review` and `use_for_sfm` are NOT in the mask key.** Neither changes a mask
  pixel; hashing them would recompute every mask when you toggle a workflow flag.
- **A disabled mask config is normalised before hashing.** A stage that produces
  nothing has to hash the same way whatever its dormant fields say, or editing
  `backend` or `dilate` on a job with `enabled: false` forks `k_train` and buys a
  95-minute retrain for a bit-identical model. It normalises to the default
  disabled form rather than a fresh sentinel, so already-cached unmasked runs
  keep the keys they are stored under.

## Review gate

`"mask": {"review": true}` holds the job after masking, before anything
expensive. It **releases its GPU** and waits in `awaiting_review`:

```
GET  /api/jobs/{id}/review            coverage, frames with no detection, URLs
GET  /api/jobs/{id}/review/sheet.jpg  contact sheet, 12 overlays across the clip
GET  /api/jobs/{id}/review/overlay/{name}
POST /api/jobs/{id}/review            {"approved": true|false, "note": "..."}
```

The same three actions are on the job's detail pane in the UI --- coverage,
the frames where nobody was detected, the contact sheet, and Approve / Reject
--- and a parked job can be stopped from the queue table. It used to be
API-only: the state had no colour, the stop button was not offered (and `DELETE`
refused it), so a job that reached `awaiting_review` sat there inert.

Approving re-queues the job; frames, select and mask are cache hits, so it
resumes rather than restarts. That is why the gate is an early exit rather than
a block inside the stage loop -- blocking would hold a GPU idle for as long as
the human took. The check sits before every stage downstream of `mask`, not
after the mask stage, so a job whose masks came from the cache is held too:
reused masks still have not been looked at.

The train stage builds a private `dataset/` view under its own cache key, two
symlinks deep, because LichtFeld reads masks from `<-d>/masks/` and the SfM
cache dir is shared with unmasked variants -- a `masks/` folder dropped in there
would silently mask every job that reuses that reconstruction.

`mask_finalize` refuses a mask set whose coverage is zero. That failure is
otherwise invisible: every mask comes out white, training proceeds exactly as
if masking were off, and the run reads as evidence that masking does not help.

## API

```
GET    /api/inputs              clips in ~/splat/samples with ffprobe metadata
POST   /api/jobs                {config} -> job id + per-stage cache hits
POST   /api/jobs/sweep          {base, axes|variants} -> a family of jobs.
                                All-or-nothing: one invalid variant queues none.
                                Capped at QUEUE_MAX_SWEEP jobs (default 64).
GET    /api/jobs[?limit&offset] queue. Running first, then queued in the order
                                the dispatcher will take them, then finished
                                newest-first.
GET    /api/jobs/{id}           config, stages, progress, metrics
GET    /api/jobs/{id}/files     finished .ply/.sog/.spz: name, bytes, url
GET    /api/jobs/{id}/files/{name} download one (only names in that job's export dir)
GET    /api/jobs/{id}/log?stage=train[&follow=true]   plain text, or SSE tail
DELETE /api/jobs/{id}           cancel
POST   /api/estimate            {config} -> per-stage seconds
GET    /api/compare?ids=1,2,3   metrics table + config diff
POST   /api/compare/render      {ids, poses|n_poses} -> contact sheet of the same
                                held-out poses across every selected model
POST   /api/jobs/clear          hide finished jobs from the view. Metrics, stage
                                records and cache are all kept — the estimator
                                fits its it/s constant from that history.
                                ?purge=true deletes the hidden ones for real.
GET    /api/status              pause state, per-GPU status
POST   /api/pause               {paused: bool}
GET    /api/cache               cache entries, sizes, free space, retention settings
POST   /api/cache/gc[?dry_run&budget_gb]
                                evict least-recently-used entries. Never touches
                                a cache a queued or running job needs, or one
                                whose dir is locked by a live process.
```

## Tests

Five suites, installed by `deploy.sh`. Only the API suite needs a running
service and reads `QUEUE_TOKEN` from the service's own env file:

```bash
set -a; . ~/splat/queue_app/.queue_env; set +a; python3 ~/splat/queue_app/test_api.py
```

```bash
~/splat/queue_app/venv/bin/python ~/splat/queue_app/test_stages.py
~/splat/queue_app/venv/bin/python ~/splat/queue_app/test_worker.py
~/splat/queue_app/venv/bin/python ~/splat/queue_app/test_regressions.py
~/splat/venv/bin/python ~/splat/scripts/test_fisheye.py
```

`test_worker.py` covers the lifecycle paths that used to be reasoned about and
never exercised, each one a way a job could end up holding a GPU or a cache
directory could end up with two writers: a cancel racing the dispatcher, a
cancel landing before the subprocess is registered, a process that ignores
SIGTERM, restart recovery releasing a lock before the orphan has exited, and
restart recovery signalling a pid that has since been reused.

`test_api.py` cancels **only the jobs it created** — it used to cancel every
queued job it could see, which destroys a sweep somebody else has waiting.

`test_stages.py` unit-tests the finalizers and cache keys with no server, and carries the
regression for the cache-poisoning bug below — using the real step numbers from the incident
(3380 and 2734).

The API suite is stdlib only, with no venv needed. It forces the queue paused, creates jobs, asserts the cache-key
properties, cancels everything it made, and restores the original pause state. The
central assertion is that a sweep over training flags shares one `frames`/`select`/`sfm`
key while each variant gets a distinct `train` key.

`test_regressions.py` uses ffmpeg and an isolated database to check dual-lens
trimming, concurrent thumbnails, scheduling estimates, pagination and fitted
history. `scripts/test_fisheye.py` uses numpy and pycolmap on CPU to check stale
model rejection and the distortion refit; neither suite starts GPU work.

## Validated end to end

Three smoke jobs on a 60 s trim of `drone.mp4` (120 panos), run on GPU 1 while a
hand-launched `run_lub` training held GPU 0:

| Job | What | Result |
|---|---|---|
| 19 | full pipeline from scratch | frames 39.7 s · select 4.4 s · SfM 149 s (120/120 registered, 0.856 px, baseline/depth 2.76) · train 76.7 s · **270 s total** |
| 20 | same, `max_cap` 250k | frames/select/sfm **cached**, train 69.8 s · **70 s total** — the cache saved 200 s |
| 21 | byte-identical to 19 | **all five stages cached**, completed instantly |

Job 20 vs 21 (250k vs 300k splats, 3k iterations): PSNR 29.21 vs 29.42, SSIM 0.854 vs
0.856, PLY 26.0 MB vs 31.2 MB, peak VRAM 1.43 GB. First like-for-like numbers on the
equirect path.

The queue also confirmed the `--steps-scaler` semantics empirically: passed alone at 0.1,
LichtFeld ran 3,000 iterations (30,000 x 0.1) with no `--iter` on the command line.

## Gotchas found the hard way

**A killed trainer exits 0.** LichtFeld handles SIGTERM by exporting whatever it has and
exiting with status 0, so a cancelled run is indistinguishable from a finished one by exit
code alone. The stage runner treated that as success, ran the finalizer and wrote a `.done`
marker over a *partial* export — after which every later job with the same config was served
a 3,380-iteration PLY as if it were the requested 15,000. Silent wrong results, not a crash.
Three guards now: a cancelled stage is never marked done; `train_finalize` refuses to cache a
run that did not reach its requested step (checked against `metrics.csv` **and** the
`splat_<step>.ply` filename, taking the *pessimistic* one, and refusing outright when neither
exists); and a cached entry that fails re-validation is invalidated and rebuilt rather than
served. Keep all three.

**The final checkpoint is not the largest file.** `train_finalize` used to pick the biggest
matching artifact, which is wrong whenever a run exports an intermediate checkpoint too:
densification prunes, so `splat_7000.ply` is routinely larger than `splat_30000.ply`. The
finalizer now picks the highest step number in the filename, and falls back to newest-mtime
only when nothing is numbered.

**A gate built on means can be blinded by one bad row.** The post-SfM gate existed precisely to
catch a scene that would reconstruct as haze, and on osmo360 it reported `baseline_over_depth: 110.0`
with no warnings — for a scene whose real value is 2.01. One of 110 cameras had been registered 4.8e6
units away on 3 observations, which is enough to move the *mean* camera centre the gate measured
everything from. The tell was the depth histogram: p50 44042, p90 44046, p99 44055, a spread of 13
units across the whole scene, which is not a thing real geometry does. `32_pick_model.py` now works
off the median centre and median radius, and drops the outlier before the trainer reads a scene scale
from it. Any statistic a guardrail depends on has to be robust, or the guardrail is decoration.

**Export formats are checked, not cached into the key.** `--export=ply,sog` changes the
training command but deliberately does *not* change the `train` cache key: instead
`train_finalize` requires every format the job asked for to be present. A cached PLY-only run
reused by a PLY+SOG job fails validation, is invalidated and retrains, while the useful
direction — a PLY+SOG cache serving a PLY-only job — still hits.

**`BufferedReader.read(n)` blocks until it has all n bytes.** The stage log streamer used
`stream.read(4096)`; LichtFeld progress lines are ~100 bytes, so output only surfaced in
~40-line batches, and since the log file is written from inside that loop, both the log's
mtime and the DB progress went stale while jobs ran normally. Healthy runs looked frozen for
minutes, which cost two cancelled-for-nothing training jobs and a spurious concurrency cap
before the baseline settled it by completing. `_iter_lines()` now uses `read1()`, and the log
is flushed per line. If you touch that function, keep `read1()`.

Corollary: **100% GPU utilisation with low memory utilisation is normal training**, not a
busy-wait. Do not read it as a stall on its own.

**COLMAP's global bundle adjustment turns iterative at 1000 images.**
`CeresBundleAdjustmentOptions.max_num_images_direct_sparse_cpu_solver` defaults to 1000, and
COLMAP's incremental pipeline never sets it. A fisheye rig registers two images per frame, so
from 500 frames on every global pass ran `ITERATIVE_SCHUR`. Job 211 (a 7-minute Osmo walk, 1404
frames) spent 109, 123, 138 and 124 minutes on consecutive passes and was on course for more
than a day. Both GPUs sat idle, and the log showed nothing but slow timestamps.

`82_fisheye_sfm.py` now maps through `scripts/colmap_incremental.py`, a copy of COLMAP's Python
mirror of the mapper with the limit lifted. The same database mapped all 1404 frames in
4 h 57 min. On the finished model, one global bundle adjustment took 6.6 min direct (converged)
against 56 min iterative (stopped at its iteration cap with a slightly higher cost). The copy is
tied to pycolmap 4.2; on any other version it falls back to the stock mapper and warns in the
sfm log.

GPU bundle adjustment does not rescue this. cuDSS reached the same cost in 8.3 min, and COLMAP's
Caspar backend supports neither fisheye cameras nor rig refinement.

## Comparison renders

`scripts/93_render_compare.py` renders several models from the same held-out poses into one
contact sheet, with the source panorama as the last column. "Held-out" is now literal: poses
come from the trainer's validation split (`--test-every`, image *i* is held out when
`i % N == 0`, default 8), not from an even sample over every registered image — which
returned a training view seven times in eight and hid the overfitting the sheet exists to
show. `--test-every 0` samples all images and labels the tiles as training views; the queue
passes 0 automatically when a selected job trained without `--eval`. The ground-truth column is
**reprojected into the same pinhole camera** as the renders — a plain centre-crop of the
equirect is a different projection and drifts increasingly toward the edges, which makes
fine-detail comparison against it meaningless. Use `--tile` large enough for the question:
at 90 degrees FOV, 512 px is ~6 px/degree and cannot show a resolution difference; 1920 px is
~21 px/degree, matching an 8K equirect source.

## Not built yet

- Scheduled cache GC. Eviction runs on demand and automatically when the disk
  preflight is short, but nothing calls it on a timer.
- VRAM pre-check. The queue now *records* `peak_vram_mib` per job so the check can be
  fitted from data instead of a guessed formula.
- A decimated, self-contained viewer page (`splat-transform --decimate-adaptive`).
