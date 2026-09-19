#!/bin/bash
# Download the weights of a gated mask backend (see queue/app/mask_backends.py).
#   scripts/get_mask_weights.sh sam3 [--force]
#   docker compose run --rm queue get-weights sam3          (Docker)
#
# Needs a Hugging Face token from an account that has accepted the model's
# licence. It is taken from HF_TOKEN, else from a previous `hf auth login`;
# with neither, you are asked for one if there is a terminal to ask on.
# Access is checked before anything is downloaded, so a missing token, a bad
# token or an unaccepted licence each stop with what to do about it.
set -euo pipefail
skip=()
SPLAT_ROOT=${SPLAT_ROOT:-$HOME/splat}
MODELS=${QUEUE_MODELS_DIR:-$SPLAT_ROOT/models}
PY="$SPLAT_ROOT/venv_gs/bin/python"
HF="$SPLAT_ROOT/venv_gs/bin/hf"

name=${1:-}; force=${2:-}
case "$name" in
  # sam3.pt is Meta's original checkpoint (3.3 GB); transformers loads
  # model.safetensors, so skipping it halves the download.
  sam3) repo=facebook/sam3; skip=("--exclude" "*.pt") ;;
  # A new backend adds one line here, next to its mask_backends.py entry.
  "") echo "usage: get_mask_weights.sh <backend> [--force]   (backends: sam3)" >&2; exit 2 ;;
  *) echo "no downloadable weights for backend '$name' (known: sam3)" >&2; exit 2 ;;
esac
dest="$MODELS/$name"
page="https://huggingface.co/$repo"

[ -x "$PY" ] || { echo "$PY not found: run scripts/setup_gsplat_venv.sh first" >&2; exit 1; }

if [ -f "$dest/config.json" ] && [ "$force" != "--force" ]; then
  echo "$name weights are already in $dest (use --force to download again)."
  exit 0
fi

# Returns 0 when this token can download the repo; otherwise prints why.
check_access() {
  "$PY" - "$repo" <<'PY'
import sys
from huggingface_hub import HfApi, auth_check
from huggingface_hub.errors import (GatedRepoError, HfHubHTTPError,
                                    RepositoryNotFoundError)
repo = sys.argv[1]
page = f"https://huggingface.co/{repo}"
try:
    user = HfApi().whoami()["name"]
except HfHubHTTPError as e:
    code = getattr(getattr(e, "response", None), "status_code", "?")
    if code == 401:
        print("The Hugging Face token was rejected (401): it is invalid, expired "
              "or revoked. Create a new read token at "
              "https://huggingface.co/settings/tokens.")
        sys.exit(13)
    print(f"Could not reach Hugging Face to check the token: HTTP {code}: {e}")
    sys.exit(14)
except Exception as e:
    # LocalTokenNotFoundError and friends: no token configured at all.
    if "token" in type(e).__name__.lower() or "token" in str(e).lower():
        print("NO_TOKEN")
        sys.exit(10)
    print(f"Could not check the Hugging Face token: {type(e).__name__}: {e}")
    sys.exit(14)
try:
    auth_check(repo)
except GatedRepoError:
    print(f"The Hugging Face account '{user}' has not been granted access to "
          f"{repo}. Open {page}, accept the licence while logged in as '{user}', "
          f"wait for the approval e-mail if the page says access is pending, "
          f"then run this again.")
    sys.exit(11)
except RepositoryNotFoundError:
    print(f"{repo} was not found, or '{user}' cannot see it. Check {page}.")
    sys.exit(12)
except HfHubHTTPError as e:
    code = getattr(getattr(e, "response", None), "status_code", "?")
    if code == 401:
        print("The Hugging Face token was rejected (401): it is invalid, expired "
              "or revoked. Create a new read token at "
              "https://huggingface.co/settings/tokens.")
        sys.exit(13)
    print(f"Could not check access to {repo}: HTTP {code}: {e}")
    sys.exit(14)
print(f"OK {user}")
PY
}

set +e
msg=$(check_access); rc=$?
set -e
if [ $rc -eq 10 ]; then
  if [ -t 0 ]; then
    echo "No Hugging Face token found. Paste a read token from"
    echo "https://huggingface.co/settings/tokens (the account must have accepted $page)."
    "$HF" auth login
    set +e; msg=$(check_access); rc=$?; set -e
  else
    cat >&2 <<EOF
No Hugging Face token found, and no terminal to ask for one.
  1. Accept the licence at $page
  2. Create a read token at https://huggingface.co/settings/tokens
  3. Either put HF_TOKEN=hf_... in .env (Docker) or export it, or run
     'hf auth login' in a terminal, then run this again.
EOF
    exit 3
  fi
fi
if [ $rc -ne 0 ]; then
  [ "$msg" = "NO_TOKEN" ] && msg="Still no usable Hugging Face token."
  echo "$msg" >&2
  exit 3
fi
echo "Access to $repo confirmed (${msg#OK }). Downloading into $dest ..."

mkdir -p "$dest" || { echo "cannot create $dest: check the models folder is writable" >&2; exit 1; }
"$HF" download "$repo" --local-dir "$dest" "${skip[@]}"
[ -f "$dest/config.json" ] || { echo "download finished but $dest/config.json is missing" >&2; exit 1; }
echo "Done: $(du -sh "$dest" | cut -f1) in $dest."
echo "Restart the queue to enable '$name' (Docker: docker compose restart queue)."
