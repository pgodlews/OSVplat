#!/bin/bash
# Container entrypoint: the queue service, plus the one-time setup deploy.sh
# does for a native install.
set -euo pipefail
: "${QUEUE_ROOT:=/data}"
PORT=${PORT:-8090}
mkdir -p "$QUEUE_ROOT" "$SPLAT_ROOT/samples" "$SPLAT_ROOT/models" 2>/dev/null || true

# Everything a job writes outside the queue's own directories -- torchvision's
# Mask R-CNN weights, pip/HF caches, LichtFeld's settings -- lands in the data
# volume too, so the container can run as an unprivileged user and a restart
# does not re-download anything.
export HOME="$QUEUE_ROOT/home" XDG_CACHE_HOME="$QUEUE_ROOT/cache_home"
export TORCH_HOME="$QUEUE_ROOT/cache_home/torch" HF_HOME="$QUEUE_ROOT/cache_home/huggingface"
mkdir -p "$HOME" "$XDG_CACHE_HOME"

# Requests arrive from the Docker bridge, never from loopback, so the service's
# "no token = loopback only" fallback would lock everyone out. Always have one:
# QUEUE_TOKEN from the environment, else one generated once and kept in the
# data volume.
if [ -z "${QUEUE_TOKEN:-}" ]; then
  f="$QUEUE_ROOT/.queue_token"
  if [ ! -s "$f" ]; then
    (umask 077; head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$f")
  fi
  QUEUE_TOKEN=$(cat "$f")
fi
export QUEUE_TOKEN

# One-off commands instead of the service:
#   docker compose run --rm queue get-weights sam3
case "${1:-}" in
  get-weights) shift; exec /opt/splat/scripts/get_mask_weights.sh "$@" ;;
  "") ;;
  *) exec "$@" ;;
esac

nvidia-smi -L >/dev/null 2>&1 || echo "WARNING: no GPU visible. Is the NVIDIA Container Toolkit installed, and is 'gpus: all' set?" >&2

echo "OSVplat: open http://<this-machine>:${PUBLIC_PORT:-$PORT}/?token=$QUEUE_TOKEN"
echo "  (the token becomes a cookie on first visit; it is kept in $QUEUE_ROOT/.queue_token)"
exec /opt/splat/queue_app/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
