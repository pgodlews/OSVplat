# Mapping-only SfM experiments (COLMAP 4.2.0)

This prototype imports the worktree's unchanged `scripts/colmap_incremental.py`.
It does no extraction, matching, training, or cache integration. Inputs are an
**existing rig-configured database with verified matches**, plus the corresponding
image directory. Image names in the database must resolve relative to that directory.
Each arm takes a SQLite backup into its own output directory and writes `sparse/`,
`mapping.log`, and `report.json`. Existing output directories are refused.

Motivation supplied with this task: clip 0005, 957 two-lens OPENCV_FISHEYE rig
frames, 16 threads; mapping 40.7 min, comprising 20.1 min global BA (23 passes)
and 20.5 min registration/local BA, with 82's ratios 1.4 and two refinements.
Those are prior measurements, not results from this prototype.

## Arms and options

| Arm | Change from shipped 82's default mapping configuration |
| --- | --- |
| `baseline` | None: focal + k1-k4 refined as in production (88 passes `--refine-intrinsics`; `--fixed-intrinsics` for 82's standalone default), sensor-from-rig refinement, seed 0, ratios 1.4, two global refinements, lifted direct sparse CPU solver limit 1,000,000 images |
| `redundant` | `opts.mapper.ba_global_ignore_redundant_points3D = True`; default coverage gain 0.05 |
| `ratio2` | Both `ba_global_frames_ratio` and `ba_global_points_ratio` set to 2.0 |
| `redundant_ratio2` | Both preceding changes |
| `localmt` | Only the BA options passed to local refinement get `ceres.min_num_residuals_for_cpu_multi_threading = 5000` |

All other options, including absolute global BA frequency limits, local bundle
size, color extraction, and multiple-model policy, retain 82's defaults. This
baseline corresponds to 82 **with** `--refine-intrinsics`, as 88 runs it in production (`--fixed-intrinsics` gives 82's standalone default). Ratios are triggers,
not a promise of exactly half as many passes; frequency triggers and the final
refinement still apply. Options and the imported mapper's SHA-256 are recorded.
The database snapshot SHA-256 is recorded before mapping, for input consistency.

In the 4.2.0 Python binding the coverage option is named
`ba_global_prune_points_min_coverage_gain`, despite the C++ member being named
`ba_global_ignore_redundant_points3D_min_coverage_gain`. Redundancy selection is
inactive below 10 registered **frames**. Ignored points are subsequently optimized
with poses/intrinsics fixed; they are not deleted. A single global mapper call
can therefore contain two Ceres solves. See the
[4.2 mapper implementation](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/sfm/incremental_mapper.cc)
and [Python option binding](https://github.com/colmap/colmap/blob/4.2.0/src/pycolmap/sfm/incremental_mapper.cc).

Do not set the pipeline's `ba_min_num_residuals_for_cpu_multi_threading` to implement
`localmt`: it changes both local and global BA. `get_local_bundle_adjustment()`
returns a new options object, so editing one unused return value has no effect.
The prototype edits the object actually passed to native local BA.

## Run all arms on one CPU box

The following is a Bash session on a box with Docker and image `88a55e02cd2e`
already installed. Run it from this worktree's root. Set the three input/output
paths first; use absolute host paths. The database should be quiescent. Its whole
parent directory is mounted read-only so any committed WAL content is visible
to SQLite backup. The image's Python must report exactly 4.2.0; other versions
are rejected rather than falling back to a different mapper.

```bash
set -euo pipefail
REPO="$(pwd -P)"
DB=/absolute/path/to/existing/rig/database.db
IMAGES=/absolute/path/to/selected/images
RUNS=/absolute/path/to/new/sfm-arms-results
IMAGE=88a55e02cd2e
THREADS=16
INPUT_DIR="$(dirname "$DB")"
DB_NAME="$(basename "$DB")"
mkdir -p "$RUNS/logs"

cpu_python() {
  docker run --rm --network none --cpus "$THREADS" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -e QUEUE_GPUS= \
    -e SPLAT_THREADS="$THREADS" -e OMP_NUM_THREADS="$THREADS" \
    -e MKL_NUM_THREADS="$THREADS" -e OPENBLAS_NUM_THREADS="$THREADS" \
    --mount "type=bind,src=$REPO,dst=/prototype,readonly" \
    --mount "type=bind,src=$INPUT_DIR,dst=/input,readonly" \
    --mount "type=bind,src=$IMAGES,dst=/images,readonly" \
    --mount "type=bind,src=$RUNS,dst=/runs" \
    --entrypoint /opt/splat/venv/bin/python "$IMAGE" "$@"
}

cpu_python -c 'import pycolmap; print(pycolmap.__version__, pycolmap.COLMAP_build); assert pycolmap.__version__ == "4.2.0"'
for arm in baseline redundant ratio2 redundant_ratio2 localmt; do
  cpu_python /prototype/experiments/sfm_mapper_arms/ab_map.py \
    --db "/input/$DB_NAME" --images /images --out "/runs/$arm" --arm "$arm" \
    > "$RUNS/logs/$arm.log" 2>&1
done

cpu_python /prototype/experiments/sfm_mapper_arms/compare.py \
  /runs/baseline/sparse \
  redundant=/runs/redundant/sparse \
  ratio2=/runs/ratio2/sparse \
  redundant_ratio2=/runs/redundant_ratio2/sparse \
  localmt=/runs/localmt/sparse \
  --out /runs/compare.json | tee "$RUNS/logs/compare.log"
```

No GPU device is exposed to these containers; BA GPU options remain false. Run
arms sequentially on an otherwise idle box. For repeat measurements choose a new
`RUNS` directory and reverse or rotate the arm order to assess warm-cache effects.
Do not infer a speedup from the tiny synthetic test below. Native progress is in
`$RUNS/<arm>/mapping.log`; the outer log receives the final `STAGE map` line.

If there are multiple models, `compare.py` refuses to choose one silently. Inspect
`report.json`, compare registration coverage, and pass explicit directories such
as `sparse/0`. Missing images are reported; low ATE on a partial common subset is
not evidence of a successful full reconstruction.

## What the report measures

- `map_s`: wall time around the entire mapping call, including database-cache
  load, colors, model write, and instrumentation; excluding SQLite backup/hash.
- `passes.global`: one entry per native `adjust_global_bundle` call, including
  initialization. `wall_s` includes problem construction, negative-depth filtering,
  redundancy selection, and both solve phases when present. It excludes the
  surrounding retriangulation, track completion, and post-BA filtering.
- `points_before` / `points_after`: model point counts around that native call.
  `points_at_selection` and `ignored_redundant_points` come from the native
  `Ignoring N / M redundant 3D points` log, after negative-depth filtering.
  The selection count is null when redundancy is inactive. Missing expected
  selection logs fail the run rather than being interpreted as zero.
- `passes.local`: one entry per native `adjust_local_bundle`, not one per registered
  frame. `wall_s` includes that function's track merging/completion and filtering.
  `local_call_count` / `local_wall_s` sum these calls across all model attempts.
- `residuals_estimate`: pre-solve scalar residual count from a read-only temporary
  BA configuration: all images in the selected rig frames, plus out-of-bundle
  observations of modified new/short-track points. It follows 4.2's membership
  rules and uses native `BundleAdjustmentConfig.num_residuals`. Each 2D observation
  contributes **two** residuals. Estimation time is outside the local BA timer and
  summed as `local_residual_estimation_s`; it remains part of `map_s`.
- `residuals_reduced`: twice the native local report's adjusted-observation count;
  Ceres may remove constant residuals, so this can differ from the estimate.
  Histograms and `local_share_under_50000` use the **pre-reduction estimate**.
  The threshold checks the unreduced native problem size. These counts indicate
  eligibility for threading, not a measurement of thread utilization.
- `solves`: each native Ceres summary's reduced residual count and reported solver
  time (rounded by COLMAP). This separates solve time from mapper-call wall time.
- Model statistics: registered frames/images, points, mean reprojection error,
  mean track length, observations. Empty/under-three-frame outputs fail.

The prototype temporarily substitutes a Python forwarding wrapper inside the
imported mirror. Native BA implementations are unchanged; only the two small
refinement loops are mirrored to expose each call. Stop/change criteria, loss
switching, normalization, and modified-point clearing follow 4.2.0. Python 4.2
exposes neither cancellation callbacks nor the mapper's prior-position flag;
these arms use no cancellation and 82's `use_prior_position=False`. Logging and
read-only residual estimation add overhead. The wrapper and native logging
settings are restored after the run. It is a single-process CLI, not a library
for concurrent mapping in one process.

`compare.py` matches registered images by name, fits one least-squares Sim(3)
from every common camera centre (no outlier rejection or reflection), then reports
median, maximum and RMSE ATE. Percentages use the **full baseline path length**,
computed along one lens in natural filename order, avoiding jumps between lenses.
By default it chooses the camera with most registered images, breaking ties by ID;
`--path-camera-id N` overrides this. Filenames must encode temporal order, as 82's
frame names do. The JSON records the denominator, camera ID, overlap and transform.
Degenerate coincident/collinear trajectories are rejected.

## One global BA replay

This reads a finished model, optionally perturbs its points and rig-frame poses,
and runs **one** native global Ceres solve. Intrinsics stay fixed and sensor poses
are refined, as in 82. Gauge fixing uses `TWO_CAMS_FROM_WORLD`. There is no
retriangulation, observation filtering, scene normalization, or redundant-point
second phase. The input is never overwritten. `setup_s`, `solve_wall_s`, their sum
`ba_wall_s`, model statistics and the full native Ceres summary are saved.

A JSON file is merged into `BundleAdjustmentOptions.ceres`; nested
`solver_options` accepts the binding's Ceres settings and enum strings. Unknown
options fail. **Set `auto_select_solver_type=false` when forcing a solver** or
COLMAP will choose based on image count. For example, in the session above:

```bash
cat > "$RUNS/ceres-cpu.json" <<'JSON'
{
  "use_gpu": false,
  "auto_select_solver_type": false,
  "min_num_residuals_for_cpu_multi_threading": 50000,
  "solver_options": {
    "linear_solver_type": "SPARSE_SCHUR",
    "max_num_iterations": 100,
    "num_threads": 16
  }
}
JSON
cpu_python /prototype/experiments/sfm_mapper_arms/replay_ba.py \
  --model /runs/baseline/sparse --out /runs/replay-cpu \
  --ceres-options /runs/ceres-cpu.json --no-use-gpu \
  --seed 0 --perturb-points 0.001 --perturb-translation 0.001 --perturb-rotation 0.01
```

Point and translation sigmas are in model units; rotation-vector sigma is in
degrees. Default sigmas are zero. Use the same model, seed and sigmas in every
replay comparison. Rig cameras remain coupled through frame poses. The replay
is intentionally distinct from a full mapper refinement pass.

For **later** custom CUDA-Ceres wheel testing, use the same script with
`--use-gpu` (overrides JSON), and a separately prepared image containing that
4.2.0 wheel. No GPU replay was run during this task. Example future invocation:

```bash
CUDA_IMAGE=your-local-image-with-custom-cuda-ceres-wheel
cat > "$RUNS/ceres-gpu.json" <<'JSON'
{
  "use_gpu": true,
  "min_num_images_gpu_solver": 0,
  "auto_select_solver_type": true,
  "max_num_images_direct_sparse_gpu_solver": 1000000,
  "solver_options": {"num_threads": 16, "max_num_iterations": 100}
}
JSON
docker run --rm --network none --gpus all --cpus 16 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e OMP_NUM_THREADS=16 -e MKL_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 \
  --mount "type=bind,src=$REPO,dst=/prototype,readonly" \
  --mount "type=bind,src=$RUNS,dst=/runs" \
  --entrypoint /opt/splat/venv/bin/python "$CUDA_IMAGE" \
  /prototype/experiments/sfm_mapper_arms/replay_ba.py \
  --model /runs/baseline/sparse --out /runs/replay-gpu --threads 16 \
  --ceres-options /runs/ceres-gpu.json --use-gpu \
  --seed 0 --perturb-points 0.001 --perturb-translation 0.001 --perturb-rotation 0.01
```

`use_gpu=true` is a request, not proof of CUDA execution: COLMAP can fall back to
CPU based on build capabilities or solve failure. Inspect `ba.log` and
`report.json`'s `summary.ceres_summary` (solver/library used, threads, costs,
termination) before labeling a result GPU. See the
[4.2 Ceres backend](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/estimators/bundle_adjustment_ceres.cc).

## CPU synthetic validation

```bash
python3.11 -m venv /tmp/sfm-arms-py311
/tmp/sfm-arms-py311/bin/pip install pycolmap==4.2.0
/tmp/sfm-arms-py311/bin/python experiments/sfm_mapper_arms/smoke_test.py
python3 queue/test_stages.py
```

The smoke test creates a fresh `/tmp/sfm-arms-smoke-*` directory, using COLMAP's
synthetic dataset: one rig, two OPENCV_FISHEYE cameras, 14 frames, 400 points,
0.25 px Gaussian keypoint noise, seed 0 (noise seed 42). It generates constant
PNG images for native color extraction. No real clip or GPU is used. The source
database hash must remain unchanged. Temporary synthetic data are not committed.

Measured on 2026-09-29 with the macOS ARM64 CPU wheel, COLMAP commit `be5e291`:
all five arms registered 14/14 frames, 28/28 images, 400 points, track length 28.
The instrumented baseline had exactly equal frame poses, camera parameters,
point coordinates and tracks to the uninstrumented shipped mirror. Its reprojection
error was 0.305666229 px. The aligned camera-centre maximum difference was about
4.3e-15 percent of path length (alignment arithmetic).

| Arm | Global calls | Local calls | Reprojection px | Max ATE vs baseline, % path |
| --- | ---: | ---: | ---: | ---: |
| baseline | 7 | 12 | 0.305666229 | 0 |
| redundant | 7 | 12 | 0.306106416 | 0.00187609 |
| ratio2 | 4 | 12 | 0.305666229 | 0.00000016 |
| redundant_ratio2 | 4 | 12 | 0.306106441 | 0.00187623 |
| localmt (one thread) | 7 | 12 | 0.305666229 | 0 |

The redundant arm ignored 200, 147, 147 points in its three eligible passes;
redundant_ratio2 ignored 147 in its eligible pass. All local residual estimates
were below 50,000, and 11/12 were at least 5,000. An additional `localmt` run with
two threads also recovered the full model. Both unperturbed and perturbed
single-BA replays converged with an explicitly selected `SPARSE_SCHUR` solver.
A known scale/rotation/translation test validates the Sim(3) convention independently.
Existing-output refusal, failed empty mapping, and degenerate-alignment checks also passed.
The stage regression suite passed, including cache-key pins.

All arms are possible in 4.2.0. No pipeline defaults, pins, or cache versions were
changed. The synthetic scene only validates behavior and instrumentation; it is
too small to predict a speedup for clip 0005. Docker commands are provided for the
later one-box experiment and were not executed in this local prototype task.
