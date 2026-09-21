# OSVplat: the whole pipeline and its queue in one image.
#
#   docker compose up -d --build        build it yourself (~1 h for one GPU type, no GPU needed)
#   docker compose pull && docker compose up -d   or use the prebuilt image
#
# The build runs the same scripts/setup_*.sh as a native install, so both get
# identical pins. Tools live at /opt/splat; clips and job data are volumes.

ARG CUDA_VERSION=13.0.2
# RTX 20xx (7.5) through 50xx (12.0). Fewer entries build faster:
#   docker compose build --build-arg CUDA_ARCH="8.6"
ARG CUDA_ARCH="7.5;8.0;8.6;8.9;9.0;12.0"

############################################################ build
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ARG CUDA_ARCH
ARG JOBS=8
ENV DEBIAN_FRONTEND=noninteractive SPLAT_ROOT=/opt/splat CUDA_ARCH=${CUDA_ARCH} JOBS=${JOBS} MAX_JOBS=${JOBS}

# docs/install.md's package list, minus what only a desktop session needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates unzip zip tar pkg-config python3 python3-dev python3-venv \
      gcc-14 g++-14 ccache ninja-build nasm autoconf autoconf-archive automake libtool \
      libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev libwayland-dev libxkbcommon-dev \
      libegl-dev libdecor-0-dev libibus-1.0-dev libdbus-1-dev libsystemd-dev libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# One layer per tool, slowest and least often changed first: a change to the
# Python dependencies then rebuilds only its own venv, not LichtFeld (~1 h).
# The SfM venv comes first because LichtFeld's build takes cmake from it.
COPY scripts/setup_sfm_venv.sh /src/scripts/
RUN /src/scripts/setup_sfm_venv.sh && rm -rf /root/.cache/pip
COPY scripts/setup_lichtfeld.sh /src/scripts/
RUN LFS_MARCH=x86-64-v3 /src/scripts/setup_lichtfeld.sh \
    && cd /opt/splat/LichtFeld-Studio/build \
    && rm -rf CMakeFiles _deps/*-build _deps/*-subbuild vcpkg_installed/*/debug \
    && find . -name '*.o' -delete
COPY scripts/setup_gsplat_venv.sh /src/scripts/
RUN /src/scripts/setup_gsplat_venv.sh && rm -rf /root/.cache/pip /opt/splat/gsplat_src/.git

############################################################ runtime
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive \
    SPLAT_ROOT=/opt/splat QUEUE_ROOT=/data \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
# video: ffmpeg's -hwaccel cuda needs the driver's NVDEC library in here.

# Runtime side of the build packages: ffmpeg for decoding, and the GTK/X11/
# Wayland libraries LichtFeld links against even when run --headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg python3 python3-venv ca-certificates curl \
      libgtk-3-0t64 libglu1-mesa libegl1 libxinerama1 libxcursor1 libxkbcommon0 \
      libwayland-client0 libwayland-cursor0 libwayland-egl1 libdecor-0-0 libdbus-1-3 \
      libgomp1 libstdc++6 libjpeg-turbo8 libpng16-16t64 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*

# Same absolute paths as the build stage: the venvs and LichtFeld's RUNPATH
# point at them.
COPY --from=build /opt/splat/venv /opt/splat/venv
COPY --from=build /opt/splat/venv_gs /opt/splat/venv_gs
COPY --from=build /opt/splat/LichtFeld-Studio /opt/splat/LichtFeld-Studio

# Fail the build here, not on the first job, if a shared library is missing.
# libcuda comes from the host driver at run time, so it is allowed to be absent.
RUN missing=$(ldd /opt/splat/LichtFeld-Studio/build/LichtFeld-Studio | grep 'not found' | grep -v libcuda.so || true); \
    if [ -n "$missing" ]; then echo "LichtFeld is missing libraries:"; echo "$missing"; exit 1; fi
# ldd cannot judge the Python packages: pycolmap and torch load some CUDA
# libraries from their own pip wheels at import time, so ldd reports those as
# missing while real system gaps (pycolmap needs libSM/libICE) look the same.
# Importing is the test that matches what a job does.
RUN /opt/splat/venv/bin/python -c "import pycolmap, cv2, numpy; print('venv ok', pycolmap.__version__)" \
    && /opt/splat/venv_gs/bin/python -c "import torch, torchvision, gsplat, cv2, pycolmap, transformers; from transformers import Sam3Model; print('venv_gs ok', torch.__version__, gsplat.__version__, transformers.__version__)"

COPY queue/requirements.txt /opt/splat/queue_app/requirements.txt
RUN python3 -m venv /opt/splat/queue_app/venv \
    && /opt/splat/queue_app/venv/bin/pip install -q --no-cache-dir -r /opt/splat/queue_app/requirements.txt

COPY scripts/ /opt/splat/scripts/
COPY queue/app/ /opt/splat/queue_app/app/
COPY queue/test_*.py queue/summarize_sweep.py /opt/splat/queue_app/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY LICENSE THIRD_PARTY.md /opt/splat/
RUN chmod +x /usr/local/bin/entrypoint.sh && chmod -R a+rX /opt/splat

# Labels last, so a new version or commit does not invalidate any build layer.
# scripts/publish_image.sh fills VERSION and REVISION from the git tag.
ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="OSVplat" \
      org.opencontainers.image.description="Raw DJI .OSV dual-fisheye to Gaussian splat. No stitch." \
      org.opencontainers.image.source="https://github.com/pgodlews/OSVplat" \
      org.opencontainers.image.url="https://github.com/pgodlews/OSVplat" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

WORKDIR /opt/splat/queue_app
EXPOSE 8090
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
