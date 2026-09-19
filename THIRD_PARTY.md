# Third-party software

OSVplat's own code is MIT-licensed ([LICENSE](LICENSE)). The Docker image, and
a native install made with `scripts/setup*.sh`, also contain the components
below, each under its own licence. OSVplat runs LichtFeld Studio as a separate
program and imports the Python libraries; none of their code is copied into
this repository except where noted.

## Built from source in the image

| Component | Version | Licence | Source |
|---|---|---|---|
| [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) | commit `04e4607b` | GPL-3.0 | The complete source tree and `LICENSE` are in the image at `/opt/splat/LichtFeld-Studio` |
| Libraries LichtFeld links (via [vcpkg](https://github.com/microsoft/vcpkg) `04a9d8e5`) | pinned by LichtFeld's manifest | various permissive (one `copyright` file each) | `/opt/splat/LichtFeld-Studio/build/vcpkg_installed/x64-linux/share/*/copyright` |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | 1.6.0, commit `28e794ca` | Apache-2.0 | GitHub |

## Python packages (from PyPI / the PyTorch index)

| Package | Version | Licence |
|---|---|---|
| pycolmap-cuda12 ([COLMAP](https://colmap.github.io/)) | 4.2.0 | BSD-3-Clause |
| PyTorch | 2.9.1+cu130 | BSD-3-Clause |
| torchvision (Mask R-CNN code) | 0.24.1+cu130 | BSD-3-Clause |
| transformers | 5.5.0 | Apache-2.0 |
| huggingface_hub, tokenizers, safetensors | as installed | Apache-2.0 |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (the wheel bundles further libraries, listed in its `LICENSE-3RD-PARTY.txt`) |
| NumPy | 2.5 | BSD-3-Clause and bundled permissive licences |
| FastAPI, pydantic | 0.115.6, 2.x | MIT |
| uvicorn, starlette | 0.34.0, 0.41.x | BSD-3-Clause |

The PyTorch and pycolmap wheels include NVIDIA CUDA libraries under NVIDIA's
redistribution terms.

## System packages and base image

| Component | Licence |
|---|---|
| [NVIDIA CUDA runtime image](https://hub.docker.com/r/nvidia/cuda) `13.0.2-runtime-ubuntu24.04` | NVIDIA Deep Learning Container License, at `/NGC-DL-CONTAINER-LICENSE` in the image |
| Ubuntu 24.04 packages, including FFmpeg 6.1.1 | various (LGPL/GPL for FFmpeg); each package's terms are in `/usr/share/doc/<package>/copyright`, source from Ubuntu |

## Not included

| Component | Why |
|---|---|
| SAM 3 weights ([facebook/sam3](https://huggingface.co/facebook/sam3)) | Gated under Meta's SAM licence; each user downloads them after accepting it (`scripts/get_mask_weights.sh`) |
| Mask R-CNN weights | Downloaded by torchvision on first use |

## Adapted code and references

- `scripts/colmap_incremental.py` is adapted from COLMAP's
  `custom_incremental_pipeline.py` and keeps its BSD-3-Clause notice.
- Field names for DJI's `.OSV` telemetry follow AdrianEddy's
  [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser); no code
  from it is included.
