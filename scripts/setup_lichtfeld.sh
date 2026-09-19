#!/bin/bash
# LichtFeld Studio native Linux build (GPL-3, C++23, CUDA 12.8+; validated on CUDA 13.2, RTX 3090).
# Prereqs: the apt packages in docs/install.md, gcc-14, and the SfM venv
# (scripts/setup_sfm_venv.sh), which supplies cmake>=3.30. About an hour of compile.
#
# Both refs are the commits this box was actually built and validated at.
# --depth 1 on a moving head made the build unreproducible in the worst way:
# it succeeds, produces a different trainer, and every PSNR in the project notes
# quietly stops being comparable. Override to move deliberately:
#   LFS_REF=main VCPKG_REF=master ./setup_lichtfeld.sh
set -x
set -euo pipefail
LFS_REF=${LFS_REF:-04e4607bf336676cf73a5d860fccacdd26766d83}
VCPKG_REF=${VCPKG_REF:-04a9d8e5212d01ee1dd9478eadd9caade4f8b0d4}    # 2026.07.29-440
SPLAT_ROOT=${SPLAT_ROOT:-$HOME/splat}
# Compute capability of GPU 0 (8.6 for an RTX 3090, 8.9 for a 4090), or a list
# for a build without a GPU: CUDA_ARCH="7.5;8.6;12.0". CMake wants it dotless.
CUDA_ARCH=${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)}
CUDA_ARCH=$(echo "$CUDA_ARCH" | tr -d .)
export CC=gcc-14 CXX=g++-14 CUDACXX=/usr/local/cuda/bin/nvcc CUDA_HOME=/usr/local/cuda
export VCPKG_ROOT=$SPLAT_ROOT/vcpkg VCPKG_MAX_CONCURRENCY=6
export PATH=$SPLAT_ROOT/venv/bin:/usr/local/cuda/bin:$VCPKG_ROOT:$PATH
"$SPLAT_ROOT/venv/bin/pip" install -q "cmake>=3.30"
# An existing clone made with --depth 1 cannot
# check out an older commit, so unshallow it before pinning.
pin() {   # pin <dir> <ref>
  if [ "$(git -C "$1" rev-parse --is-shallow-repository)" = "true" ]; then
    git -C "$1" fetch --unshallow origin
  else
    git -C "$1" fetch origin
  fi
  git -C "$1" checkout "$2"
  git -C "$1" submodule update --init --recursive
}
# No --depth 1 on a fresh clone, for the same reason.
[ -d "$SPLAT_ROOT/LichtFeld-Studio" ] || git clone --recursive https://github.com/MrNeRF/LichtFeld-Studio.git "$SPLAT_ROOT/LichtFeld-Studio"
pin "$SPLAT_ROOT/LichtFeld-Studio" "$LFS_REF"
[ -d "$VCPKG_ROOT" ] || git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"     # NOT --depth 1
pin "$VCPKG_ROOT" "$VCPKG_REF"
echo "LichtFeld $(git -C "$SPLAT_ROOT/LichtFeld-Studio" rev-parse --short HEAD), vcpkg $(git -C "$VCPKG_ROOT" describe --tags --always)"
[ -x "$VCPKG_ROOT/vcpkg" ] || (cd "$VCPKG_ROOT" && ./bootstrap-vcpkg.sh -disableMetrics)
cd "$SPLAT_ROOT/LichtFeld-Studio"
nice -n 10 cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
  -DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++exp   # GCC 14 <stacktrace>, troubleshooting #12
# -k 0: the optional python typings step fails on a libstdc++exp stacktrace
# symbol, so the build exits non-zero even though the executable links. That is
# expected here, and `|| true` keeps `set -e` from treating it as fatal -- the
# real success gate is the --version call below, which needs a working binary.
nice -n 10 cmake --build build -j "${JOBS:-8}" -- -k 0 || true
# A Docker build has no driver, so the binary cannot start; there the image's
# own ldd check stands in for this one.
if command -v nvidia-smi >/dev/null; then
  ./build/LichtFeld-Studio --version
else
  test -x ./build/LichtFeld-Studio
fi
