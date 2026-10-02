#!/bin/bash
# Prep-only setup for an Apple silicon Mac: frames, select, mask and SfM, no
# trainer (LichtFeld and gsplat are CUDA-only). A job runs here with
# run_until: "sfm" and its handoff bundle trains on a CUDA box.
#
#   venv     conda-forge env: pycolmap built from pgodlews/colmap, branch
#            metal-sift-lanxinger (lanxinger/colmap-metal: COLMAP 4.2.0-dev with
#            selected 4.2.1 fixes and SIFT in Metal, plus a canonical feature
#            order), and OpenCV. The PyPI macOS wheel has CPU SIFT only, and its
#            VLFeat is scalar on arm64: 775 s against 85 s for 946 fisheye images
#            on an M5 Max.
#   venv_gs  torch with MPS for the stitch and Mask R-CNN. Not the Linux pin:
#            torchvision 0.24.1's roi_align takes 92 s per 1000 boxes on MPS.
#
# Needs Xcode with its Metal Toolchain, Homebrew ffmpeg, micromamba and git.
# Measurements and what differs from the CUDA path: docs/how-it-works.md,
# "Prep on Apple silicon".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLAT_ROOT=${SPLAT_ROOT:-$HOME/splat}
# Pinned by commit, not by branch: quality numbers in the docs were measured at it.
COLMAP_REPO=https://github.com/pgodlews/colmap.git
# conda-forge colmap release whose dependency set the build links against.
COLMAP_DEPS_OF=4.2.1
COLMAP_REF=0fea5683b4604ae651ede76e9b8574c2b9248a92

[ "$(uname -sm)" = "Darwin arm64" ] || { echo "setup_mac.sh is for Apple silicon Macs" >&2; exit 1; }
command -v micromamba >/dev/null || { echo "micromamba not found: brew install micromamba" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found: brew install ffmpeg" >&2; exit 1; }
ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -q videotoolbox \
  || { echo "this ffmpeg has no videotoolbox decoder" >&2; exit 1; }
xcrun -sdk macosx metal --version >/dev/null 2>&1 \
  || { echo "the Metal compiler is missing: xcodebuild -downloadComponent MetalToolchain" >&2; exit 1; }

mkdir -p "$SPLAT_ROOT/samples"
cd "$SPLAT_ROOT"
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$SPLAT_ROOT/.mamba}

# The libraries COLMAP links, as conda-forge built them for its own colmap
# package, plus the headers that package does not pull in. Validated set:
# ceres-solver 2.2.0, eigen 5.0.1, faiss 1.14.3, poselib 2.0.5, suitesparse
# 7.10.1, libboost 1.92.0, openimageio 3.1.17, llvm-openmp 23.1.2.
rm -rf venv
micromamba create -y -q -p "$SPLAT_ROOT/venv" -c conda-forge python=3.12 "colmap=$COLMAP_DEPS_OF" --only-deps
micromamba install -y -q -p "$SPLAT_ROOT/venv" -c conda-forge \
  libboost-devel suitesparse eigen glog gflags ceres-solver flann sqlite gmp lz4-c metis \
  openimageio poselib faiss libfaiss llvm-openmp libcurl openssl libblas libcblas \
  pybind11 scikit-build-core pip cmake ninja packaging

[ -d colmap_src ] || git clone "$COLMAP_REPO" colmap_src
git -C colmap_src remote set-url origin "$COLMAP_REPO"
git -C colmap_src fetch -q origin metal-sift-lanxinger
git -C colmap_src checkout -q --force "$COLMAP_REF" \
  || { echo "commit $COLMAP_REF is not on $COLMAP_REPO metal-sift-lanxinger" >&2; exit 1; }
git -C colmap_src clean -fdq

PREFIX="$SPLAT_ROOT/colmap_metal"
rm -rf colmap_src/build "$PREFIX"
"$SPLAT_ROOT/venv/bin/cmake" -S colmap_src -B colmap_src/build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$SPLAT_ROOT/venv" -DCMAKE_INSTALL_PREFIX="$PREFIX" -DOpenMP_ROOT="$SPLAT_ROOT/venv" \
  -DMETAL_ENABLED=ON -DCUDA_ENABLED=OFF -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF -DONNX_ENABLED=OFF \
  -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF -DTESTS_ENABLED=OFF -DFETCH_POSELIB=OFF -DFETCH_FAISS=OFF \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
"$SPLAT_ROOT/venv/bin/cmake" --build colmap_src/build
"$SPLAT_ROOT/venv/bin/cmake" --install colmap_src/build >/dev/null

# pycolmap from the same tree, against the install above. The env variable
# takes ':' between prefixes; the rpath is what finds the conda libraries.
CMAKE_PREFIX_PATH="$PREFIX:$SPLAT_ROOT/venv" \
CMAKE_ARGS="-DOpenMP_ROOT=$SPLAT_ROOT/venv -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_INSTALL_RPATH=$SPLAT_ROOT/venv/lib -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" \
  "$SPLAT_ROOT/venv/bin/python" -m pip install -q ./colmap_src
"$SPLAT_ROOT/venv/bin/python" -m pip install -q opencv-python-headless==5.0.0.93 numpy==2.5.2
(cd "$HERE" && "$SPLAT_ROOT/venv/bin/python" -c "
import pycolmap, cv2, sift_backend
ok = sift_backend.metal_available(pycolmap)
print('pycolmap', pycolmap.__version__, 'metal sift', ok, 'opencv', cv2.__version__)
raise SystemExit(0 if ok else 'pycolmap built, but it cannot create a Metal SIFT extractor')")

rm -rf venv_gs
# From the conda python, not the system one: macOS ships 3.9, which has no torch 2.14 wheel.
"$SPLAT_ROOT/venv/bin/python" -m venv venv_gs
./venv_gs/bin/pip install -q --upgrade pip
./venv_gs/bin/pip install -q torch==2.14.1 torchvision==0.29.1 numpy opencv-python-headless==5.0.0.93
./venv_gs/bin/python -c "
import torch, torchvision
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'mps', torch.backends.mps.is_available())
raise SystemExit(0 if torch.backends.mps.is_available() else 'torch has no MPS device here')"

echo
echo "Done: prep-only tools in $SPLAT_ROOT. Start the queue with queue/run_mac.sh;"
echo "a job needs run_until, and \"sfm\" writes the handoff bundle a CUDA box trains from."
