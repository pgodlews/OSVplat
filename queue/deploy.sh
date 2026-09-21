#!/bin/bash
# Install (or update) the queue service.
#   ./queue/deploy.sh              on this machine
#   ./queue/deploy.sh <ssh-host>   on another machine, over ssh (needs
#                                  passwordless sudo there for the systemd unit)
# Idempotent: rsyncs code, builds the venv if missing, installs/refreshes the
# systemd unit, restarts the service. The queue starts PAUSED, so deploying can
# never disturb a hand-launched training run.
#
# The service is access-controlled by a shared token generated here on first
# deploy and kept in $APP/.queue_env (0600). Without a token the service answers
# loopback only, so an accidental unauthenticated LAN service is not reachable.
# BIND=127.0.0.1 to keep it off the LAN entirely and reach it over ssh -L.
#
# SPLAT_DIR (default "splat", relative to the target home) and SERVICE (default
# "splat-queue") let a second install sit beside the first, e.g. for testing:
#   SPLAT_DIR=splat_test SERVICE=splat-queue-test PORT=8091 ./queue/deploy.sh
set -euo pipefail
HOST=${1:-}
# on CMD: run a shell snippet on the target. dest: rsync destination prefix.
if [ -z "$HOST" ]; then
  on() { bash -c "$1"; }
  DEST=""
  URL_HOST=$(hostname)
else
  on() { ssh "$HOST" "$1"; }
  DEST="$HOST:"
  URL_HOST=$HOST
fi
PORT=${PORT:-8090}
BIND=${BIND:-0.0.0.0}
SPLAT_DIR=${SPLAT_DIR:-splat}
SERVICE=${SERVICE:-splat-queue}
# Prometheus /metrics, off unless asked for: METRICS=1 ./queue/deploy.sh
# Accepts the spellings people actually type rather than silently staying off.
case "${METRICS:-0}" in 1|true|TRUE|yes|YES|on|ON) METRICS=1;; *) METRICS=0;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE_USER=$(on 'echo $USER')
REMOTE_HOME=$(on 'echo $HOME')
ROOT="$REMOTE_HOME/$SPLAT_DIR"
APP="$ROOT/queue_app"
echo "==> ${HOST:-this machine}  user=$REMOTE_USER  app=$APP  service=$SERVICE  port=$PORT  metrics=$METRICS"

echo "==> syncing code"
on "mkdir -p '$APP/app' '$ROOT/scripts' '$ROOT/samples'"
rsync -az --delete --exclude '__pycache__' --exclude '*.pyc' \
  "$REPO/queue/app/" "$DEST$APP/app/"
rsync -az "$REPO/queue/requirements.txt" "$DEST$APP/"
# The unit suites are run from $APP by the documented remote commands, so they
# have to be there; test_api.py is stdlib-only and runs from anywhere.
rsync -az "$REPO/queue/test_stages.py" "$REPO/queue/test_worker.py" \
  "$REPO/queue/test_api.py" "$REPO/queue/test_regressions.py" \
  "$REPO/queue/test_distance.py" "$DEST$APP/"
# summarize_sweep.py reads the queue over HTTP and is meant to be run on the
# target, where it can pick the token out of .queue_env; it was never shipped there.
rsync -az "$REPO/queue/summarize_sweep.py" "$DEST$APP/"
rsync -az "$REPO/scripts/" "$DEST$ROOT/scripts/"

echo "==> ensuring access token"
TOKEN=$(on "set -e
umask 077
f='$APP/.queue_env'
if ! grep -qs '^QUEUE_TOKEN=' \"\$f\"; then
  printf 'QUEUE_TOKEN=%s\n' \"\$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')\" > \"\$f\"
fi
chmod 600 \"\$f\"
sed -n 's/^QUEUE_TOKEN=//p' \"\$f\"")
[ -n "$TOKEN" ] || { echo "failed to establish an access token" >&2; exit 1; }

echo "==> ensuring venv"
on "set -e
cd '$APP'
[ -x venv/bin/python ] || python3 -m venv venv
./venv/bin/pip -q install --upgrade pip
./venv/bin/pip -q install -r requirements.txt
./venv/bin/python -c 'import fastapi, uvicorn; print(\"deps ok\", fastapi.__version__)'"

echo "==> installing systemd unit"
# Which commit this install runs, for telemetry records (docker: OSVPLAT_REVISION
# comes from the image build instead).
REV=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)
git -C "$(dirname "$0")/.." diff --quiet HEAD 2>/dev/null || REV="$REV-dirty"
UNIT=$(mktemp)
cat > "$UNIT" <<UNITEOF
[Unit]
Description=Splat processing queue
After=network.target

[Service]
Type=simple
User=$REMOTE_USER
WorkingDirectory=$APP
Environment=PYTHONUNBUFFERED=1
Environment=SPLAT_ROOT=$ROOT
Environment=QUEUE_METRICS=$METRICS
Environment=OSVPLAT_REVISION=$REV
# After the Environment= lines on purpose: a QUEUE_METRICS= line added to
# .queue_env then wins, so the endpoint can be toggled with a restart instead
# of a redeploy.
EnvironmentFile=$APP/.queue_env
ExecStart=$APP/venv/bin/uvicorn app.main:app --host $BIND --port $PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNITEOF
rsync -q "$UNIT" "$DEST/tmp/$SERVICE.service"
rm -f "$UNIT"
on "sudo mv /tmp/$SERVICE.service /etc/systemd/system/$SERVICE.service
sudo systemctl daemon-reload
sudo systemctl enable --now $SERVICE.service >/dev/null 2>&1
sudo systemctl restart $SERVICE.service"

echo "==> health check"
sleep 3
on "systemctl is-active $SERVICE.service || true
curl -fsS -H 'X-Queue-Token: $TOKEN' http://127.0.0.1:$PORT/api/status | head -c 500; echo"
echo
echo "Open: http://$URL_HOST:$PORT/?token=$TOKEN"
echo "  (the token is stored in $APP/.queue_env and set as a cookie"
echo "   on first visit, so later visits need only http://$URL_HOST:$PORT/)"
if [ "$METRICS" = 1 ]; then
  echo "Metrics: http://$URL_HOST:$PORT/metrics  (same token; Authorization: Bearer works)"
else
  echo "Metrics: off. Redeploy with METRICS=1, or add QUEUE_METRICS=1 to"
  echo "  $APP/.queue_env and: sudo systemctl restart $SERVICE"
fi
