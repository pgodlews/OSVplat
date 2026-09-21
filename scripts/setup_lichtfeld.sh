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
# CPU target for lfs_core, whose Release flags hard-code -march=native: code for
# whatever CPU runs the compile. That is right for a box building for itself and
# wrong for an image, which then dies with SIGILL on any CPU lacking an
# instruction set the build machine had (troubleshooting #25). The Dockerfile
# sets x86-64-v3 (AVX2 + FMA, which the rest of LichtFeld already requires).
LFS_MARCH=${LFS_MARCH:-native}
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
# Undo a previous run's -march edit first, or checking out another ref refuses.
git -C "$SPLAT_ROOT/LichtFeld-Studio" checkout -- src/core/CMakeLists.txt 2>/dev/null || true
pin "$SPLAT_ROOT/LichtFeld-Studio" "$LFS_REF"
grep -q -- '-march=native>' "$SPLAT_ROOT/LichtFeld-Studio/src/core/CMakeLists.txt" \
  || { echo "src/core/CMakeLists.txt no longer sets -march=native; recheck LFS_MARCH" >&2; exit 1; }
sed -i "s/-march=native>/-march=$LFS_MARCH>/" "$SPLAT_ROOT/LichtFeld-Studio/src/core/CMakeLists.txt"
[ -d "$VCPKG_ROOT" ] || git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"     # NOT --depth 1
pin "$VCPKG_ROOT" "$VCPKG_REF"
echo "LichtFeld $(git -C "$SPLAT_ROOT/LichtFeld-Studio" rev-parse --short HEAD), vcpkg $(git -C "$VCPKG_ROOT" describe --tags --always)"
[ -x "$VCPKG_ROOT/vcpkg" ] || (cd "$VCPKG_ROOT" && ./bootstrap-vcpkg.sh -disableMetrics)
# code.videolan.org's archive endpoint returns differently compressed bytes for
# the same x264 commit (8 downloads, 8 SHA-512s on 2026-09-21), so vcpkg rejects
# it and the build fails (troubleshooting #26). The pinned port's hash is plain
# `git archive | gzip -n` output: make that file and leave it where vcpkg looks
# before downloading.
X264=31e19f92f00c7003fa115047ce50978bc98c3a0d
X264_SHA512=707ff486677a1b5502d6d8faa588e7a03b0dee45491c5cba89341be4be23d3f2e48272c3b11d54cfc7be1b8bf4a3dfc3c3bb6d9643a6b5a2ed77539c85ecf294
x264_tgz=$VCPKG_ROOT/downloads/videolan-x264-$X264.tar.gz
if [ ! -f "$x264_tgz" ]; then
  mkdir -p "$VCPKG_ROOT/downloads"
  x264_src=$(mktemp -d)
  git clone -q --filter=blob:none https://code.videolan.org/videolan/x264.git "$x264_src"
  git -C "$x264_src" archive --format=tar --prefix="x264-$X264/" "$X264" | gzip -n > "$x264_tgz.part"
  rm -rf "$x264_src"
  [ "$(sha512sum < "$x264_tgz.part" | cut -d' ' -f1)" = "$X264_SHA512" ] \
    || { echo "rebuilt x264 tarball does not match vcpkg's SHA-512" >&2; exit 1; }
  mv "$x264_tgz.part" "$x264_tgz"
fi
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
# A portable build must not have been compiled for the build machine's CPU. The
# binary runs fine there, so check the flags actually used instead. Counting
# AVX-512 instructions does not work: builds with and without -march=native both
# carry 5,107, in code that checks the CPU before using them.
if [ "$LFS_MARCH" != native ]; then
  n=$(grep -c -E -- '-march=native|-mavx512' ./build/compile_commands.json || true)
  [ "$n" = 0 ] || { echo "LFS_MARCH=$LFS_MARCH build still compiled $n files for the build CPU" >&2; exit 1; }
fi
