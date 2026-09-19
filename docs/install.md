# Native install

The easier route is Docker: see [docker.md](docker.md). This page builds the
same tools directly on the workstation instead, which suits a machine you
already use for CUDA work or one without Docker.

Everything runs on one Linux machine with an NVIDIA GPU, called the GPU
workstation below: the tools, the queue service and its web UI. You can open
the UI from any other computer on your network, but there is nothing to install
anywhere else.

## Requirements

| | Validated on | Minimum (expected) |
|---|---|---|
| OS | Ubuntu 24.04 | a recent x86-64 Linux with systemd |
| GPU | 2× RTX 3090 24 GB | one NVIDIA GPU. A Standard run (3M splats, 3840 px) peaked at 10.1 GB; the Max preset trains at 7680 px and needs more. Less VRAM: lower `train.max_cap` or `train.max_width` |
| Driver / CUDA | driver 595, CUDA toolkit 13.2 at `/usr/local/cuda` | a driver new enough for CUDA 13.0 wheels; toolkit 12.8+ |
| RAM / disk | 62 GB / NVMe | 32 GB; ~20 GB per clip of intermediate frames |
| Python | 3.12 | 3.10+ |
| Other | ffmpeg 6.1 with `cuda` hwaccel, `sudo` for installing the systemd unit | |

## 1. System packages

```bash
sudo apt install ffmpeg rsync git curl unzip zip tar pkg-config python3 python3-dev python3-venv \
  gcc-14 g++-14 ccache ninja-build nasm autoconf autoconf-archive automake libtool \
  libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev libwayland-dev libxkbcommon-dev \
  libegl-dev libdecor-0-dev libibus-1.0-dev libdbus-1-dev libsystemd-dev libgtk-3-dev
```

Most of the second half is for building LichtFeld Studio (its
[build docs](https://github.com/MrNeRF/LichtFeld-Studio) list the same set).
The CUDA toolkit comes from NVIDIA; the scripts expect it at `/usr/local/cuda`.

## 2. Build the tools (about an hour)

```bash
git clone https://github.com/pgodlews/OSVplat.git ~/osvplat && cd ~/osvplat
scripts/setup.sh
```

That runs three scripts, each safe to re-run on its own:

| Script | Makes | Time |
|---|---|---|
| `setup_sfm_venv.sh` | `~/splat/venv`: pycolmap with CUDA, OpenCV | 2 min |
| `setup_gsplat_venv.sh` | `~/splat/venv_gs`: torch + gsplat built from source, transformers; runs the mask backends and evaluation renders | 30–45 min |
| `setup_lichtfeld.sh` | `~/splat/LichtFeld-Studio/build/LichtFeld-Studio`: the trainer | 30–60 min |

Everything goes under `$SPLAT_ROOT` (default `~/splat`). The GPU architecture
is detected from `nvidia-smi`; set `CUDA_ARCH` to override. Building LichtFeld
is CPU-heavy — `JOBS=4` lowers the parallelism if the machine becomes unresponsive.

### Optional: SAM 3 masks

Mask R-CNN (people only) works without setup. SAM 3 (anything you name) runs in
the same `venv_gs`; only its weights are extra, because they are gated behind
Meta's licence. Accept it at <https://huggingface.co/facebook/sam3>, then:

```bash
scripts/get_mask_weights.sh sam3     # 3.3 GB into ~/splat/models/sam3; asks for a HF token
```

Restart the queue (`sudo systemctl restart splat-queue`) and SAM 3 becomes
selectable in the UI.

## 3. Install the queue service

```bash
./queue/deploy.sh
```

It copies `queue/` and `scripts/` into `~/splat`, builds the service venv,
installs a `splat-queue` systemd unit (asks for your sudo password) and prints
a URL containing an access token, e.g.
`http://gpu-workstation:8090/?token=…`. Open it once, on the workstation or any
machine on your network; a cookie keeps you signed in afterwards. The queue
**starts paused**, so press **Resume queue** in the UI.

Run it again after `git pull` to update; finished jobs and the cache are kept.

Options (environment variables):

| Variable | Default | |
|---|---|---|
| `PORT` | 8090 | |
| `BIND` | 0.0.0.0 | `127.0.0.1` keeps the UI off the network |
| `METRICS` | 0 | `1` adds a Prometheus `/metrics` endpoint |
| `SPLAT_DIR`, `SERVICE` | `splat`, `splat-queue` | a second install beside the first |


## 4. Check it

```bash
~/splat/venv/bin/python ~/splat/scripts/test_fisheye.py
~/splat/queue_app/venv/bin/python ~/splat/queue_app/test_stages.py
```

Then copy a clip into `~/splat/samples/` and submit a job — see the
[README](../README.md#usage).
