"""Paths and resource configuration. Everything is derived from SPLAT_ROOT."""
import json
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

def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _json_env(name: str) -> dict:
    """A JSON object from the environment; {} if unset. Malformed is fatal:
    a typo would otherwise silently turn an upload off."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    val = json.loads(raw)
    if not isinstance(val, dict):
        raise ValueError(f"{name} must be a JSON object")
    return val


# Per-job telemetry (queue/app/telemetry.py, docs/job-telemetry.md). On by default,
# and local: telemetry.json and logs.tar.gz go to QUEUE_ROOT/runs/job<id>/ and
# stay there. QUEUE_TELEMETRY=0 stops writing them. QUEUE_TELEMETRY_UPLOAD is an
# upload target (presigned S3 POST or a PUT URL); without it nothing is sent
# anywhere. QUEUE_TELEMETRY_PLACEMENT is copied into each record as-is, for
# whoever launched this machine to say what it is (provider, region, price).
TELEMETRY_ENABLED = _flag("QUEUE_TELEMETRY", "1")

# QUEUE_TLS_INSECURE=1 accepts self-signed or otherwise unverifiable TLS
# certificates on this service's own outbound requests: OUTPUT_UPLOAD_URL,
# telemetry uploads, the webhook (and INPUT_URL, in the entrypoint). Off by
# default. On, anyone on the network path can read and alter those transfers,
# presigned URLs included, so it is for a private endpoint on a network you
# trust, not the internet.
TLS_INSECURE = _flag("QUEUE_TLS_INSECURE", "0")


def ssl_context():
    """None (verify as usual) unless QUEUE_TLS_INSECURE is on."""
    if not TLS_INSECURE:
        return None
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _telemetry_env(name: str, need_url: bool = False) -> dict:
    """Like _json_env, but a mistake here must never keep jobs from running:
    it is reported loudly and that setting is off; telemetry.json is still
    written locally. (Raising at import stopped the whole service, and in a
    container that is a restart loop that keeps billing.)"""
    try:
        val = _json_env(name)
        if need_url and val and not val.get("url"):
            raise ValueError(f'{name} needs a "url"')
        return val
    except ValueError as exc:
        print(f"WARNING: {name} ignored: {exc}")
        return {}


TELEMETRY_UPLOAD = _telemetry_env("QUEUE_TELEMETRY_UPLOAD", need_url=True)
TELEMETRY_PLACEMENT = _telemetry_env("QUEUE_TELEMETRY_PLACEMENT")


def _upload_target(name: str) -> dict:
    """A plain URL means one presigned PUT; a JSON object is a target in the
    QUEUE_TELEMETRY_UPLOAD shape (PUT with headers, or a presigned S3 POST)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        val = _json_env(name)
        if not val.get("url"):
            raise ValueError(f"{name} needs a \"url\"")
        return val
    if not raw.startswith("https://") and not raw.startswith("http://"):
        raise ValueError(f"{name} must be an http(s) URL or a JSON object")
    return {"method": "PUT", "url": raw}


# Finished splats to object storage (queue/app/outputs.py, docs/cloud.md): after
# each job that ends done, its export files go up as one tar. For rented GPUs,
# where nothing should have to be pulled over SSH and no cloud credentials
# belong on the box: a presigned URL can write one object and nothing else.
# Unlike telemetry, this is the run's delivery: a malformed value does not stop
# the service (SSH and the UI stay up) but outputs.preflight() makes the API
# refuse new jobs until it is fixed.
OUTPUT_UPLOAD_ERROR = None
try:
    OUTPUT_UPLOAD = _upload_target("OUTPUT_UPLOAD_URL")
except ValueError as exc:
    OUTPUT_UPLOAD, OUTPUT_UPLOAD_ERROR = {}, f"OUTPUT_UPLOAD_URL: {exc}"

# Optional webhook: a small JSON event POSTed when a stage starts or finishes
# and when a job ends (docs/job-telemetry.md, "Webhook"). Off unless a URL is
# set. With QUEUE_WEBHOOK_SECRET, each request carries an HMAC-SHA256 of its
# body in X-OSVplat-Signature, so the receiver can tell it came from here.
WEBHOOK_URL = os.environ.get("QUEUE_WEBHOOK_URL", "").strip()
WEBHOOK_SECRET = os.environ.get("QUEUE_WEBHOOK_SECRET", "").strip()

# Shared secret for every request. This service can start GPU jobs, cancel them,
# delete history and change scheduling, so it is not something to leave open on
# a LAN port. When unset, only loopback clients are served -- reach it through
# `ssh -L 8090:127.0.0.1:8090 <box>`. deploy.sh generates one and installs it as
# a 0600 EnvironmentFile.
QUEUE_TOKEN = os.environ.get("QUEUE_TOKEN", "").strip()
TOKEN_COOKIE = "queue_token"

for d in (QUEUE_ROOT, CACHE_ROOT, RUNS_ROOT, LOG_ROOT, RENDER_ROOT):
    d.mkdir(parents=True, exist_ok=True)
