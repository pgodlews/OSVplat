#!/bin/bash
# venv_gs: torch 2.9.1+cu130 pinned (with local tag!), example deps, gsplat built from source with nvcc 13.2.
#
# GSPLAT_REF is the commit this environment was actually built and validated at
# (gsplat 1.6.0). Cloning the moving head instead means a rebuild months later
# silently gets different CUDA kernels, and the project notes's numbers stop being
# comparable to anything you measure afterwards. Override to move deliberately:
#   GSPLAT_REF=main ./setup_gsplat_venv.sh
set -x
set -euo pipefail
GSPLAT_REF=${GSPLAT_REF:-28e794ca44a4c25ffc39175370c5ee7b38bfcc36}
SPLAT_ROOT=${SPLAT_ROOT:-$HOME/splat}
# Compute capability of GPU 0, e.g. 8.6 for an RTX 3090, 8.9 for a 4090. Or a
# list, for a build without a GPU (the Docker image): CUDA_ARCH="7.5;8.6;12.0"
CUDA_ARCH=${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)}
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" MAX_JOBS=${MAX_JOBS:-8}
mkdir -p "$SPLAT_ROOT"; cd "$SPLAT_ROOT"
[ -d gsplat_src ] || git clone --recursive https://github.com/nerfstudio-project/gsplat.git gsplat_src   # --recursive: vendored glm submodule
git -C gsplat_src fetch --tags origin
git -C gsplat_src checkout --recurse-submodules "$GSPLAT_REF"
git -C gsplat_src submodule update --init --recursive
echo "gsplat pinned at $(git -C gsplat_src rev-parse HEAD)"
rm -rf venv_gs; python3 -m venv venv_gs; P=./venv_gs/bin/pip
$P install -q --upgrade pip
$P install torch==2.9.1+cu130 torchvision==0.24.1+cu130 --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
$P install ninja numpy jaxtyping rich
$P install -r gsplat_src/examples/requirements.txt --no-build-isolation
$P install ./gsplat_src --no-build-isolation
# SAM 3 mask backend (queue/app/mask_backends.py) runs here too. This pinned
# transformers leaves torch alone; an unpinned one may pull a newer torch than
# gsplat was compiled against, which would break gsplat's CUDA kernels, so check.
torch_before=$(./venv_gs/bin/python -c "import torch; print(torch.__version__)")
$P install transformers==5.5.0
torch_after=$(./venv_gs/bin/python -c "import torch; print(torch.__version__)")
[ "$torch_before" = "$torch_after" ] || { echo "transformers changed torch $torch_before -> $torch_after" >&2; exit 1; }
# Validated build: torch 2.9.1+cu130 / CUDA 13.0 / gsplat 1.6.0 / transformers 5.5.0.
./venv_gs/bin/python -c "import torch, gsplat, transformers; print(torch.__version__, torch.version.cuda, gsplat.__version__, transformers.__version__)"
