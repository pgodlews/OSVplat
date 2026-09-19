#!/bin/bash
# venv: pycolmap with CUDA (SIFT + matching on the GPU), OpenCV, numpy.
# Runs frame selection, SfM, the .OSV readers and the fisheye rig pipeline.
# Keep it separate from venv_gs: gsplat's example requirements install the CPU
# pycolmap under the same module name (docs/troubleshooting.md #3).
set -euo pipefail
SPLAT_ROOT=${SPLAT_ROOT:-$HOME/splat}
mkdir -p "$SPLAT_ROOT"
python3 -m venv "$SPLAT_ROOT/venv"
P="$SPLAT_ROOT/venv/bin/pip"
"$P" install -q --upgrade pip
"$P" install pycolmap-cuda12==4.2.0 opencv-python-headless==5.0.0.93 numpy==2.5.2 "cmake>=3.30"
"$SPLAT_ROOT/venv/bin/python" -c "import pycolmap, cv2; print('pycolmap', pycolmap.__version__, 'cuda', pycolmap.has_cuda, 'opencv', cv2.__version__)"
