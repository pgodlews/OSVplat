#!/bin/bash
# Container entrypoint: the queue service, plus the one-time setup deploy.sh
# does for a native install. On a rented GPU (docs/cloud.md) it can also start
# an SSH server and fetch the input clip; all of that is off unless its
# variables are set, so Compose users see no change.
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

die() { echo "ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------ SSH
# Keys from any of: SSH_PUBLIC_KEYS (ours), SSH_PUBLIC_KEY (what Vast injects:
# the account's keys), PUBLIC_KEY (what RunPod injects), SSH_KEYS_URL (https,
# e.g. https://github.com/<user>.keys). No key, no sshd.
#
# The host has root over this container, so: no passwords, no agent forwarding
# (a forwarded agent is usable by host root for as long as the session lasts),
# local forwarding only (for `ssh -L 8090:localhost:8090`), and host keys made
# here at first start, never baked into the image -- a shared host key would
# let anyone impersonate every container. Fingerprints go to the log, which
# Vast and RunPod show outside the container, to check the first connection.
ssh_keys() {
  local k
  for k in "${SSH_PUBLIC_KEYS:-}" "${SSH_PUBLIC_KEY:-}" "${PUBLIC_KEY:-}" "$url_keys"; do
    [ -n "$k" ] && printf '%s\n' "$k"
  done
  return 0
}

SSH_ON=0
if [ -n "${SSH_PUBLIC_KEYS:-}${SSH_PUBLIC_KEY:-}${PUBLIC_KEY:-}${SSH_KEYS_URL:-}" ]; then
  [ "$(id -u)" = 0 ] || die "SSH needs the container to run as root (no 'user:'); unset the SSH key variables otherwise"
  url_keys=""
  if [ -n "${SSH_KEYS_URL:-}" ]; then
    case "$SSH_KEYS_URL" in https://*) ;; *) die "SSH_KEYS_URL must be https://" ;; esac
    url_keys=$(curl -fsS --max-time 30 "$SSH_KEYS_URL") || die "could not fetch SSH_KEYS_URL"
  fi
  keys=$(ssh_keys | tr -d '\r' | sed 's/^[[:space:]]*//' | grep -v '^#' | grep -v '^$' || true)
  bad=$(printf '%s\n' "$keys" | grep -vE '^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(256|384|521)|sk-ssh-ed25519@openssh\.com|sk-ecdsa-sha2-nistp256@openssh\.com) ' || true)
  [ -z "$bad" ] || die "not an SSH public key: $(printf '%s' "$bad" | head -1 | cut -c1-40)..."
  [ -n "$keys" ] || die "the SSH key variables are set but hold no key"
  install -d -m 700 /root/.ssh
  (umask 077; printf '%s\n' "$keys" | sort -u > /root/.ssh/authorized_keys)
  # Made once per container: a restart keeps them, a new container gets new ones.
  ls /etc/ssh/ssh_host_*_key >/dev/null 2>&1 || ssh-keygen -A >/dev/null
  mkdir -p /run/sshd
  /usr/sbin/sshd -p "${SSH_PORT:-22}" || die "sshd did not start"
  SSH_ON=1
  echo "OSVplat: sshd on port ${SSH_PORT:-22}, $(wc -l < /root/.ssh/authorized_keys) key(s). Host key fingerprints:"
  for f in /etc/ssh/ssh_host_*_key.pub; do ssh-keygen -lf "$f"; done | sed 's/^/  /'
fi

# The UI stays inside the container when SSH is the way in: reach it with
# `ssh -L 8090:localhost:8090`. The token is still required.
BIND=${QUEUE_BIND:-$([ "$SSH_ON" = 1 ] && echo 127.0.0.1 || echo 0.0.0.0)}

# ------------------------------------------------------------- input clip
# INPUT_URL: fetch one clip into samples/ at start (a presigned GET, or any
# https URL). INPUT_SHA256 is checked when given. Kept across restarts: a clip
# already there with the right checksum is not fetched again. It runs beside the
# service, which lists the clip once it lands under its own name (.part until
# then). A failure is loud but leaves the service up, so the clip can still be
# copied in over SSH.
fetch_input() {
  local name dest
  name=${INPUT_NAME:-$(basename "${INPUT_URL%%\?*}")}
  case "$name" in ""|.*|*/*) echo "ERROR: cannot name the input from INPUT_URL; set INPUT_NAME" >&2; return 1 ;; esac
  dest="$SPLAT_ROOT/samples/$name"
  if [ -s "$dest" ] && { [ -z "${INPUT_SHA256:-}" ] || echo "$INPUT_SHA256  $dest" | sha256sum -c --status; }; then
    echo "OSVplat: input samples/$name already here"
    return 0
  fi
  echo "OSVplat: fetching input samples/$name"
  local t0=$SECONDS insecure=()
  # QUEUE_TLS_INSECURE=1: accept a self-signed certificate (docs/cloud.md).
  case "${QUEUE_TLS_INSECURE:-0}" in 1|true|yes|on)
    insecure=(-k); echo "WARNING: QUEUE_TLS_INSECURE: not checking the TLS certificate of INPUT_URL" >&2 ;; esac
  # Only scheme, host and path are printed: the query of a presigned URL is the credential.
  curl -fsS "${insecure[@]}" --connect-timeout 20 --retry 5 --retry-all-errors --max-time 7200 -o "$dest.part" "$INPUT_URL" \
    || { rm -f "$dest.part"; echo "ERROR: input download from ${INPUT_URL%%\?*} failed" >&2; return 1; }
  if [ -n "${INPUT_SHA256:-}" ] && ! echo "$INPUT_SHA256  $dest.part" | sha256sum -c --status; then
    echo "ERROR: input sha256 mismatch for $name: got $(sha256sum "$dest.part" | cut -c1-16)..., expected ${INPUT_SHA256:0:16}..." >&2
    rm -f "$dest.part"; return 1
  fi
  mv "$dest.part" "$dest"
  echo "OSVplat: input samples/$name, $(stat -c %s "$dest") bytes in $((SECONDS - t0)) s"
}
if [ -n "${INPUT_URL:-}" ]; then
  { fetch_input || echo "WARNING: no input clip from INPUT_URL; the queue runs without it" >&2; } &
fi

nvidia-smi -L >/dev/null 2>&1 || echo "WARNING: no GPU visible. Is the NVIDIA Container Toolkit installed, and is 'gpus: all' set?" >&2

if [ "$SSH_ON" = 1 ] && [ "$BIND" = 127.0.0.1 ]; then
  # Where the providers say we are reachable (docs/cloud.md).
  ip=${RUNPOD_PUBLIC_IP:-${PUBLIC_IPADDR:-<host>}}
  sp=${RUNPOD_TCP_PORT_22:-${VAST_TCP_PORT_22:-${SSH_PORT:-22}}}
  echo "OSVplat: ssh -p $sp -L 8090:localhost:$PORT root@$ip"
  echo "  then open http://localhost:8090/?token=$QUEUE_TOKEN"
else
  echo "OSVplat: open http://<this-machine>:${PUBLIC_PORT:-$PORT}/?token=$QUEUE_TOKEN"
fi
echo "  (the token becomes a cookie on first visit; it is kept in $QUEUE_ROOT/.queue_token)"
exec /opt/splat/queue_app/venv/bin/uvicorn app.main:app --host "$BIND" --port "$PORT"
