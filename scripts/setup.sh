#!/bin/bash
# One-time setup of the GPU workstation: SfM venv, gsplat venv, LichtFeld
# Studio build. Run from a clone of this repo. Everything lands in $SPLAT_ROOT
# (default ~/splat). Roughly 1.5 h, most of it compiling LichtFeld.
# Each step is its own script and safe to re-run on its own.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPLAT_ROOT=${SPLAT_ROOT:-$HOME/splat}
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first" >&2; exit 1; }
[ -x /usr/local/cuda/bin/nvcc ] || { echo "/usr/local/cuda/bin/nvcc not found: install the CUDA toolkit (12.8+)" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found: sudo apt install ffmpeg" >&2; exit 1; }
command -v g++-14 >/dev/null || { echo "g++-14 not found: see the apt line in docs/install.md" >&2; exit 1; }
mkdir -p "$SPLAT_ROOT/samples"
"$HERE/setup_sfm_venv.sh"
"$HERE/setup_gsplat_venv.sh"
"$HERE/setup_lichtfeld.sh"
echo
echo "Done. Next: ./queue/deploy.sh to install the queue service,"
echo "then copy clips into $SPLAT_ROOT/samples."
