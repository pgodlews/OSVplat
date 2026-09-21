# Docker

The quickest way to run the pipeline: one container holds the tools (pycolmap,
gsplat, LichtFeld Studio, ffmpeg) and the queue with its web UI. Your clips,
job data and optional model weights stay in folders on the workstation.

## Requirements

- Linux with an NVIDIA GPU and driver **580 or newer** (`nvidia-smi` shows the
  version). The image runs on RTX 20xx through 50xx (compute capability 7.5–12.0).
- An x86-64 CPU with AVX2 and FMA (Intel Haswell / AMD Zen or newer). The
  published 0.1.1 image also needed AVX-512 by accident and crashes in training
  without it ([troubleshooting #25](troubleshooting.md)).
- Docker with the Compose plugin, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Check it works:

  ```bash
  docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
  ```

- **Laptops, NUCs and eGPUs with hybrid graphics:** if `nvidia-smi` says it
  "couldn't communicate with the NVIDIA driver" although the GPU shows in
  `lspci`, check `prime-select query`. `intel` (integrated only) blacklists the
  NVIDIA driver; switch with `sudo prime-select on-demand` and reboot. The
  integrated GPU keeps driving the display.
- About 19 GB of disk for the image (CUDA runtime, two PyTorch-based
  environments, LichtFeld), plus ~20 GB per clip for job data: a 2-minute
  Standard run used 10.8 GB of decoded frames, 3.4 GB of masks, 1 GB of SfM
  and 4.5 GB of training output. The cache keeps them so later jobs on the
  same clip reuse the work; `Reclaim space` in the UI evicts the oldest.

## Start

```bash
git clone https://github.com/pgodlews/OSVplat.git ~/osvplat && cd ~/osvplat
cp .env.example .env
sed -i "s/^UID=.*/UID=$(id -u)/; s/^GID=.*/GID=$(id -g)/" .env
mkdir -p samples data models
```

Then either pull the prebuilt image (once one is published on GHCR):

```bash
docker compose pull
docker compose up -d
```

or build it yourself. No GPU is needed to build. With `CUDA_ARCH` in `.env`
set to just your card (e.g. `8.6` for an RTX 3090) it took **66 minutes** on a
16-thread Ryzen mini PC (28 min LichtFeld, 32 min gsplat, the rest downloads);
the default list of six GPU generations takes several times longer:

```bash
docker compose up -d --build
```

Open the URL from the log:

```bash
docker compose logs queue | grep token
```

`http://<this-machine>:8090/?token=…` — once is enough; a cookie remembers it.
The queue starts **paused**: press **Resume queue**. Then follow
[Usage](../README.md#usage) in the README.

## Where things are

| On the workstation | In the container | |
|---|---|---|
| `./samples/` | `/opt/splat/samples` | Put clips here (`.OSV`, `.mp4`). The UI lists them. |
| `./data/` | `/data` | Queue database, stage cache, logs, results, the access token. Keep it between upgrades. |
| `./models/` | `/opt/splat/models` | Optional mask model weights (SAM 3). Empty is fine. |

Finished splats are downloaded from each job's page. They are also on disk
under `./data/cache/export/<key>/`.

## Settings (`.env`)

| Variable | Default | |
|---|---|---|
| `IMAGE` | `ghcr.io/pgodlews/osvplat:latest` | Image to pull or to tag a local build as |
| `UID`, `GID` | 1000 | Who owns files in `./data`; set to `id -u` / `id -g` |
| `PORT` | 8090 | Port on the workstation |
| `SAMPLES_DIR`, `DATA_DIR`, `MODELS_DIR` | `./samples`, `./data`, `./models` | Anywhere with space, e.g. a big data disk |
| `HF_TOKEN` | empty | Hugging Face token, only used by `get-weights` |
| `QUEUE_TOKEN` | generated | Fix the access token instead of generating one |
| `QUEUE_GPUS` | `all` | Restrict the queue to some GPUs: `0`, `0,1` |
| `QUEUE_METRICS` | 0 | `1` enables a Prometheus `/metrics` endpoint |
| `QUEUE_TELEMETRY` | 1 | `0` stops writing per-job telemetry; it never leaves the machine unless you set an upload target ([job-telemetry.md](job-telemetry.md)) |
| `QUEUE_CPUS` | discovered | CPUs the stages are sized to; by default the container's CPU quota ([cloud.md](cloud.md#cpu-and-gpu-discovery)) |
| `SSH_PUBLIC_KEYS`, `QUEUE_BIND`, `INPUT_URL`, `OUTPUT_UPLOAD_URL` | empty | For rented GPUs: SSH into the container, fetch the clip, upload the result ([cloud.md](cloud.md)) |
| `QUEUE_WEBHOOK_URL`, `QUEUE_WEBHOOK_SECRET` | empty | POST a signed JSON event when a stage starts or finishes ([job-telemetry.md](job-telemetry.md#webhook)) |
| `CUDA_ARCH` | `7.5;8.0;8.6;8.9;9.0;12.0` | Build only: GPU architectures to compile for |
| `BUILD_JOBS` | 8 | Build only: parallel compile jobs; lower it if the machine struggles |

## Updating

```bash
git pull
docker compose pull && docker compose up -d        # prebuilt
docker compose up -d --build                       # or rebuild
```

Jobs, cache and results in `./data` are kept. A rebuild after an update reuses
the compiled layers unless a `scripts/setup_*.sh` changed, and took 73 seconds
on the same mini PC. The queue restarts paused, so press **Resume queue**;
restarting stops a running job, so update between runs (pause the queue first
and wait for the current job to finish).

## SAM 3 masks (optional)

Person masking works out of the box with Mask R-CNN (people only). The image
also includes SAM 3, which masks anything you name ("person, dog, selfie
stick"), but its weights are gated behind Meta's licence, so they are not in
the image. To enable it:

1. Accept the licence at <https://huggingface.co/facebook/sam3> and create a
   read token at <https://huggingface.co/settings/tokens>.
2. Download the weights (3.3 GB) into `./models/sam3`:

   ```bash
   docker compose run --rm queue get-weights sam3     # asks for the token
   ```

   Or put `HF_TOKEN=hf_…` in `.env` first to skip the prompt. A token pasted
   at the prompt is saved in `./data/cache_home/huggingface/`, so later
   downloads do not ask again; delete `token` and `stored_tokens` there (and
   the `HF_TOKEN` line in `.env`) once you are done with it.
3. `docker compose restart queue` (between jobs: a restart stops a running
   one). **SAM 3** then appears as a mask backend under **Quality flags**, with
   a field for what to mask.

Until the weights are there, SAM 3 is listed but disabled, and the API refuses
it with these same instructions. Future mask models are added the same way;
the list comes from `queue/app/mask_backends.py`.

## Notes

- **`pid: host`** in `compose.yaml` is deliberate. The queue refuses to schedule on
  a GPU that runs a process it did not start, and it recognises its own by PID;
  `nvidia-smi` reports host PIDs, so the container shares the host's PID
  namespace. It also means a training job you start by hand outside the queue
  keeps the queue off that GPU, as intended.
- `docker compose down` stops any running job with the container. It shows as
  failed on the next start; queue it again and every finished stage is reused
  from the cache.
- Logs of the service itself: `docker compose logs -f queue`. Logs of each
  stage are in the UI.

## Verifying the image

Every release is published with its digest, a software bill of materials and
a signature. From the release notes on GitHub:

```bash
# Pull exactly the released image; Docker checks every layer against the digest.
docker pull ghcr.io/pgodlews/osvplat@sha256:<digest from the release notes>

# Check it was signed with the project's key (cosign.pub is in the repo).
cosign verify --key cosign.pub ghcr.io/pgodlews/osvplat@sha256:<digest>

# Every package inside, and the commit and settings it was built from.
docker buildx imagetools inspect ghcr.io/pgodlews/osvplat@sha256:<digest> --format '{{json .SBOM}}'
docker buildx imagetools inspect ghcr.io/pgodlews/osvplat@sha256:<digest> --format '{{json .Provenance}}'
```

The labels on the image name the source commit too:
`docker inspect --format '{{json .Config.Labels}}' ghcr.io/pgodlews/osvplat:latest`.

None of this proves the image is harmless; it proves what is in it and that it
came from this repository unmodified. The strongest check is not to use it:
`docker compose up -d --build` builds the same image from the source you are
reading. Third-party components and their licences are listed in
[THIRD_PARTY.md](../THIRD_PARTY.md), also at `/opt/splat/THIRD_PARTY.md` in the image.

## Publishing the image (maintainers)

GitHub's hosted runners have too little disk and time for this build, so
releases are built on a workstation with `scripts/publish_image.sh`, which:

1. clones the release **tag from GitHub** into a temporary folder, so nothing
   untracked or ignored in a working copy can reach the image;
2. builds it (all GPU architectures: several hours);
3. **scans the image for private data** (your hostname, home path and git
   e-mail, `.env` files, keys, tokens, clips) and stops on any finding;
4. with `--push`: uploads it with an SBOM and build provenance attached, signs
   the digest with cosign, and prints the lines for the release notes.

One-time setup:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u pgodlews --password-stdin   # token with write:packages
cosign generate-key-pair           # in the repo root; commit cosign.pub, keep cosign.key private
```

Each release:

```bash
git tag v0.1.0 && git push origin v0.1.0
scripts/publish_image.sh 0.1.0            # build + scan only; pushes nothing
scripts/publish_image.sh 0.1.0 --push     # same build from cache, then push + sign
```

After the first push, make the package public once in its settings on GitHub,
or users cannot pull it. `scripts/publish_image.sh --scan <image>` runs just
the leak scan on any local image.
