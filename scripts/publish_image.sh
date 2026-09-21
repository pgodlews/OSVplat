#!/bin/bash
# Build, check and publish the OSVplat image to GHCR (maintainers).
#
#   scripts/publish_image.sh 0.1.0            clone tag v0.1.0, build, scan. Pushes nothing.
#   scripts/publish_image.sh 0.1.0 --push     ...then push, attach SBOM + provenance, sign
#   scripts/publish_image.sh --scan IMAGE     only run the leak scan on a local image
#
# Before --push, once:  docker login ghcr.io -u <github-user>  (a token with
# write:packages), and a signing key:  cosign generate-key-pair  in the repo
# root. Commit cosign.pub; keep cosign.key private (it is git- and
# docker-ignored). --no-sign skips signing.
#
# Why each step:
#   * builds from a fresh clone of the GitHub tag, never a working copy, so
#     untracked or ignored local files cannot end up in the image;
#   * scans the built image for private data and refuses to push on a hit;
#   * pushes with an SBOM (every package inside) and build provenance (source
#     commit, build arguments) attached, and signs the digest, so users can
#     check what the image contains, where it came from and that it is yours.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/pgodlews/OSVplat.git}
# The checkout this script lives in: where cosign.key is looked for, wherever
# the script is started from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSIGN_KEY=${COSIGN_KEY:-$REPO_ROOT/cosign.key}
IMAGE_NAME=${IMAGE_NAME:-ghcr.io/pgodlews/osvplat}
CUDA_ARCH=${CUDA_ARCH:-7.5;8.0;8.6;8.9;9.0;12.0}
BUILD_JOBS=${BUILD_JOBS:-8}
BUILDER=${BUILDER:-osvplat-publish}

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- leak scan
# Private strings to look for: this machine's hostname, your home path and
# git e-mail, plus common token formats. Add your own with PRIVATE_PATTERNS.
scan_image() {
  local img=$1
  local pats=()
  # The hostname only when it is specific: "ubuntu", "server" and the like are
  # in the base image's own files (/etc/os-release) and would block every
  # publish. SCAN_HOST overrides the name, e.g. to test this.
  local h; h=${SCAN_HOST:-$(hostname -s 2>/dev/null || true)}
  case "$(printf %s "$h" | tr '[:upper:]' '[:lower:]')" in
    ""|localhost|ubuntu|debian|linux|docker*|server|worker*|runner*|build*|host|node*|desktop|workstation|pc|gpu*|nvidia*) h="" ;;
  esac
  local hostpat=""; [ ${#h} -ge 4 ] && hostpat=$h
  pats+=("/home/$USER/" "/Users/$USER/")
  local mail; mail=$(git config --get user.email 2>/dev/null || true); [ -n "$mail" ] && pats+=("$mail")
  # shellcheck disable=SC2206
  [ -n "${PRIVATE_PATTERNS:-}" ] && pats+=(${PRIVATE_PATTERNS})
  local fixed; fixed=$(printf '%s\n' "${pats[@]}")
  echo "==> scanning $img for private data"
  local out
  out=$(docker run --rm -i --entrypoint bash -e FIXED="$fixed" -e HOSTPAT="$hostpat" "$img" -s <<'SCAN'
set -u
hits=0
# Known upstream false positives, checked by hand; anything else still fails.
#   PIL/ImageFont.py: base64 font data containing "AKIA..."
#   transformers/testing_utils.py: Hugging Face's public sandbox CI token
ALLOW='/site-packages/PIL/ImageFont\.py$|/site-packages/transformers/testing_utils\.py$'
report() { echo "  $1"; hits=$((hits+1)); }
# Files that should never be in an image.
while IFS= read -r f; do report "file: $f"; done < <(find / -xdev \( \
    -name .env -o -name '.env.*' -o -name .netrc -o -name .git-credentials \
    -o -name 'id_rsa*' -o -name 'id_ed25519*' -o -name 'id_ecdsa*' -o -name cosign.key \
    -o -name authorized_keys -o -name 'authorized_keys2' -o -name 'ssh_host_*' \
    -o -name token -path '*huggingface*' -o -name stored_tokens \
    -o -iname '*.osv' -o -iname '*.lrf' -o -iname '*.mp4' -o -iname '*.insv' \
    -o -name '*.ply' -path '*/data/*' \) 2>/dev/null \
    | grep -vE '^/proc/|/site-packages/.*/(tests?|testing|data)/' )
# Anything in home directories (the base image ships empty skeletons).
while IFS= read -r f; do report "home file: $f"; done < <(find /root /home -type f \
    ! -name .bashrc ! -name .profile ! -name .bash_logout 2>/dev/null)
# Token formats, in every text file we or the build put there.
while IFS= read -r f; do report "token-like string in $f"; done < <(grep -rIlE \
    'hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----' \
    /opt/splat /etc /usr/local /root /home 2>/dev/null \
    | grep -vE '/site-packages/.*/(tests?|testing)/|/ssl/|\.pem$' \
    | grep -vE "$ALLOW" )
# Your hostname, paths and e-mail.
while IFS= read -r p; do
  [ -z "$p" ] && continue
  while IFS= read -r f; do report "'$p' in $f"; done < <(grep -rIlF -- "$p" \
      /opt/splat /etc /usr/local /root /home 2>/dev/null | head -5)
done <<< "$FIXED"
# The hostname as a whole word only, so a short name does not match a longer
# one that merely starts with it. No example name here on purpose: this file
# ships inside the image, and an example would match the scan on the machine
# it names.
if [ -n "$HOSTPAT" ]; then
  while IFS= read -r f; do report "hostname '$HOSTPAT' in $f"; done < <(grep -rIlwF -- "$HOSTPAT" \
      /opt/splat /etc /usr/local /root /home 2>/dev/null | head -5)
fi
echo "HITS=$hits"
SCAN
)
  echo "$out" | grep -v '^HITS=' || true     # a clean scan prints only HITS=0
  local hits; hits=$(echo "$out" | sed -n 's/^HITS=//p')
  # The build commands recorded in the image history.
  if docker history --no-trunc --format '{{.CreatedBy}}' "$img" \
      | grep -qE 'hf_[A-Za-z0-9]{30,}|ghp_|github_pat_|PASSWORD|SECRET'; then
    echo "  secret-like text in the image history"; hits=$((hits+1))
  fi
  if [ "${hits:-1}" != 0 ]; then
    echo "==> scan FAILED: $hits finding(s). Nothing was pushed." >&2
    return 1
  fi
  echo "==> scan clean"
}

if [ "${1:-}" = "--scan" ]; then
  [ -n "${2:-}" ] || die "usage: publish_image.sh --scan IMAGE"
  scan_image "$2"; exit
fi

# ---------------------------------------------------------------- arguments
VERSION=${1:-}; shift || true
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.][A-Za-z0-9.]+)?$ ]] \
  || die "usage: publish_image.sh <version, e.g. 0.1.0> [--push] [--no-sign]"
PUSH=0; SIGN=1
for a in "$@"; do
  case "$a" in
    --push) PUSH=1 ;;
    --no-sign) SIGN=0 ;;
    *) die "unknown option $a" ;;
  esac
done
TAG="v$VERSION"

docker buildx version >/dev/null 2>&1 || die "docker buildx is required"
if [ $PUSH = 1 ] && [ $SIGN = 1 ]; then
  command -v cosign >/dev/null || die "cosign not found (https://docs.sigstore.dev/cosign/system_config/installation/); or pass --no-sign"
  [ -f "$COSIGN_KEY" ] || die "no $COSIGN_KEY: run 'cosign generate-key-pair' in the repo root, or set COSIGN_KEY, or pass --no-sign"
fi

# ---------------------------------------------------------------- fresh clone
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
echo "==> cloning $REPO_URL at $TAG"
git clone --quiet --depth 1 --branch "$TAG" "$REPO_URL" "$WORK/src" \
  || die "could not clone tag $TAG: create and push it first (git tag $TAG && git push origin $TAG)"
REVISION=$(git -C "$WORK/src" rev-parse HEAD)
echo "    commit $REVISION"
[ $PUSH = 1 ] && [ $SIGN = 1 ] && [ ! -f "$WORK/src/cosign.pub" ] \
  && echo "WARNING: cosign.pub is not committed at $TAG; users will have no key to verify with"

# A docker-container builder is what can attach SBOM and provenance on push.
docker buildx inspect "$BUILDER" >/dev/null 2>&1 \
  || docker buildx create --name "$BUILDER" --driver docker-container >/dev/null
BUILD=(docker buildx build --builder "$BUILDER" --platform linux/amd64
       --build-arg "CUDA_ARCH=$CUDA_ARCH" --build-arg "JOBS=$BUILD_JOBS"
       --build-arg "VERSION=$VERSION" --build-arg "REVISION=$REVISION"
       -t "$IMAGE_NAME:$VERSION" -t "$IMAGE_NAME:latest")

# ---------------------------------------------------------------- build + scan
echo "==> building for CUDA_ARCH=$CUDA_ARCH (hours for the full list)"
t0=$(date +%s)
"${BUILD[@]}" --load "$WORK/src"
echo "==> built in $(( ($(date +%s)-t0)/60 )) min: $(docker image inspect -f '{{.Size}}' "$IMAGE_NAME:$VERSION" | awk '{printf "%.1f GB", $1/1e9}')"
scan_image "$IMAGE_NAME:$VERSION"

if [ $PUSH = 0 ]; then
  echo
  echo "Built and scanned $IMAGE_NAME:$VERSION ($REVISION). Nothing pushed; rerun with --push."
  exit 0
fi

# ---------------------------------------------------------------- push + sign
# Same builder and arguments, so every layer is a cache hit: this only adds
# the attestations and uploads.
echo "==> pushing with SBOM and provenance"
"${BUILD[@]}" --sbom=true --provenance=mode=max --metadata-file "$WORK/meta.json" --push "$WORK/src"
DIGEST=$(python3 -c "import json;print(json.load(open('$WORK/meta.json'))['containerimage.digest'])")
REF="$IMAGE_NAME@$DIGEST"
if [ $SIGN = 1 ]; then
  echo "==> signing $REF (asks for the key's password)"
  cosign sign --yes --key "$COSIGN_KEY" "$REF"
fi

cat <<EOF

Published. For the GitHub release notes of $TAG:

    Image:   $IMAGE_NAME:$VERSION
    Digest:  $DIGEST
    Source:  https://github.com/pgodlews/OSVplat/tree/$REVISION

    docker pull $REF
$( [ $SIGN = 1 ] && echo "    cosign verify --key cosign.pub $REF" )
    docker buildx imagetools inspect $REF --format '{{json .SBOM}}'

New packages on GHCR start private: make it public once under
https://github.com/users/pgodlews/packages/container/osvplat/settings
EOF
