#!/bin/bash
# Run the queue on an Apple silicon Mac, in the foreground (Ctrl-C stops it).
#   ./queue/run_mac.sh
# The Mac counterpart of deploy.sh, without systemd: copies the code next to the
# tools scripts/setup_mac.sh built, makes the service venv and an access token
# on first use, and starts the service. It preps only (frames, select, mask,
# sfm): a job needs run_until, and "sfm" writes the handoff bundle a CUDA box
# trains from (docs/how-it-works.md, "Prep on Apple silicon").
# BIND=0.0.0.0 to reach the UI from another machine; PORT, SPLAT_ROOT as usual.
set -euo pipefail
[ "$(uname -sm)" = "Darwin arm64" ] || { echo "run_mac.sh is for Apple silicon Macs; use deploy.sh" >&2; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SPLAT_ROOT=${SPLAT_ROOT:-$HOME/splat}
PORT=${PORT:-8090}
BIND=${BIND:-127.0.0.1}
APP="$SPLAT_ROOT/queue_app"
[ -x "$SPLAT_ROOT/venv/bin/python" ] && [ -x "$SPLAT_ROOT/venv_gs/bin/python" ] \
  || { echo "no tools in $SPLAT_ROOT: run scripts/setup_mac.sh first" >&2; exit 1; }

mkdir -p "$APP/app" "$SPLAT_ROOT/scripts" "$SPLAT_ROOT/samples"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' "$REPO/queue/app/" "$APP/app/"
rsync -a "$REPO/queue/requirements.txt" "$APP/"
rsync -a "$REPO/scripts/" "$SPLAT_ROOT/scripts/"

cd "$APP"
[ -x venv/bin/python ] || "$SPLAT_ROOT/venv/bin/python" -m venv venv
./venv/bin/pip -q install -r requirements.txt
umask 077
grep -qs '^QUEUE_TOKEN=' .queue_env \
  || printf 'QUEUE_TOKEN=%s\n' "$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')" > .queue_env
set -a; . ./.queue_env; set +a

REV=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
git -C "$REPO" diff --quiet HEAD 2>/dev/null || REV="$REV-dirty"
export OSVPLAT_REVISION=$REV PYTHONUNBUFFERED=1
echo "Open: http://localhost:$PORT/?token=$QUEUE_TOKEN"
echo "  clips go in $SPLAT_ROOT/samples; the queue starts paused"
# caffeinate: a sleeping Mac stops the job it is running.
exec caffeinate -i ./venv/bin/uvicorn app.main:app --host "$BIND" --port "$PORT"
