"""Paths and resource configuration. Everything is derived from SPLAT_ROOT."""
import os
import subprocess
from pathlib import Path

SPLAT_ROOT = Path(os.environ.get("SPLAT_ROOT", Path.home() / "splat")).expanduser()
QUEUE_ROOT = Path(os.environ.get("QUEUE_ROOT", SPLAT_ROOT / "queue")).expanduser()

CACHE_ROOT = QUEUE_ROOT / "cache"
RUNS_ROOT = QUEUE_ROOT / "runs"
LOG_ROOT = QUEUE_ROOT / "logs"
RENDER_ROOT = QUEUE_ROOT / "renders"
DB_PATH = QUEUE_ROOT / "queue.db"

# Tools. scripts/ is copied from the repo by deploy.sh so the queue
# always runs version-controlled copies, not the loose files in ~/splat.
SCRIPTS = SPLAT_ROOT / "scripts"
SFM_PY = SPLAT_ROOT / "venv/bin/python"
SELECT_SHARP = SCRIPTS / "20_select_sharp.py"
RUN_SFM = SCRIPTS / "30_run_sfm.py"
LFS_BIN = SPLAT_ROOT / "LichtFeld-Studio/build/LichtFeld-Studio"
GS_PY = SPLAT_ROOT / "venv_gs/bin/python"          # torch + gsplat, for renders
PERSON_MASKS = SCRIPTS / "70_person_masks.py"
# Every mask backend runs in venv_gs: setup_gsplat_venv.sh pins a transformers
# that leaves its torch alone, and fails if it would not. Gated weights live under MODELS_ROOT, one
# directory per backend (mask_backends.py). QUEUE_MASK_PY points elsewhere, e.g.
# an older install's separate venv_sam.
MASK_PY = Path(os.environ.get("QUEUE_MASK_PY", GS_PY)).expanduser()
MODELS_ROOT = Path(os.environ.get("QUEUE_MODELS_DIR", SPLAT_ROOT / "models")).expanduser()
RENDER_COMPARE = SCRIPTS / "93_render_compare.py"

# Raw DJI dual fisheye (.OSV) runs its own stage implementations -- no stitch.
# See stages.py "fisheye rig" and docs/how-it-works.md, "Fisheye rig".
OSV_META = SCRIPTS / "osv_meta.py"                 # calibration.json from the camd box/track
FISHEYE_FRAMES = SCRIPTS / "80_fisheye_frames.py"  # rig-aware sharpness pick
FISHEYE_MASKS = SCRIPTS / "87_fisheye_masks.py"    # stitch -> person masks -> back to fisheye
FISHEYE_SFM = SCRIPTS / "88_fisheye_sfm.py"        # rig SfM, refinement, dataset
FISHEYE_TRAIN = SCRIPTS / "89_fisheye_train_view.py"  # training masks, then exec LichtFeld

# GPUs the queue is allowed to schedule on. A GPU carrying a foreign compute
# process is skipped at runtime regardless of this list (see gpu.py).

# Stall watchdog for the training stage. LichtFeld prints progress every 100
# iterations, so even a slow 0.3 it/s run reports every ~5 min; going quiet for
# STALL_TIMEOUT means it is wedged, not working. Job 41 (--max-width 7680) sat
# at 100% GPU with no progress for 13+ min and would have held a GPU for hours.
# FIRST_PROGRESS_GRACE is separate and much longer because the dataset load
# before the first iteration legitimately took ~10 min at full 8K resolution.
STALL_TIMEOUT = float(os.environ.get("QUEUE_STALL_TIMEOUT", 900))
FIRST_PROGRESS_GRACE = float(os.environ.get("QUEUE_FIRST_PROGRESS_GRACE", 1800))

def _all_gpus() -> list[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        return [int(x) for x in out.split()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


# "all" (the default) = every GPU nvidia-smi lists at startup; "0" or "0,1" =
# those only; "" = none, which the tests use to run without a GPU.
_gpus = os.environ.get("QUEUE_GPUS", "all").strip()
GPUS = _all_gpus() if _gpus == "all" else [int(x) for x in _gpus.split(",") if x != ""]

# Maximum training jobs in flight; one per GPU by default. (A cap of 1 was
# briefly the default while a suspected concurrent-training hang was
# investigated. That hang was not real -- it was a blocking read in this
# service's own log reader making healthy jobs look frozen. See _iter_lines.)
DEFAULT_MAX_CONCURRENT = int(os.environ.get("QUEUE_MAX_CONCURRENT", len(GPUS)))

# Start paused so deploying the service can never disturb a hand-launched run.
START_PAUSED = os.environ.get("QUEUE_START_PAUSED", "1") == "1"

# Prometheus exposition at /metrics. Off unless asked for: it names input clips
# and job labels, and a route that exists is a route that can be probed. When
# off the path is not registered at all rather than answering 404 from a live
# handler. deploy.sh takes METRICS=1; setting QUEUE_METRICS=1 in .queue_env
# overrides the deployed default, because systemd reads that file after the
# unit's own Environment= lines. Either way it stays behind the queue token.
# `== "1"` is the convention elsewhere in this file, and it is fine for flags
# only ever set by deploy.sh. This one is documented as hand-editable in
# .queue_env, so it accepts the spellings someone would actually type there
# rather than reading QUEUE_METRICS=true as "off" and saying nothing.
METRICS_ENABLED = os.environ.get("QUEUE_METRICS", "0").strip().lower() in (
    "1", "true", "yes", "on")

# Shared secret for every request. This service can start GPU jobs, cancel them,
# delete history and change scheduling, so it is not something to leave open on
# a LAN port. When unset, only loopback clients are served -- reach it through
# `ssh -L 8090:127.0.0.1:8090 <box>`. deploy.sh generates one and installs it as
# a 0600 EnvironmentFile.
QUEUE_TOKEN = os.environ.get("QUEUE_TOKEN", "").strip()
TOKEN_COOKIE = "queue_token"

for d in (QUEUE_ROOT, CACHE_ROOT, RUNS_ROOT, LOG_ROOT, RENDER_ROOT):
    d.mkdir(parents=True, exist_ok=True)
