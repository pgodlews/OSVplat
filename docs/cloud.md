# Rented GPUs (Vast.ai, RunPod)

The same image runs as a single container on a rented GPU. There is no Compose
there: the provider starts the image itself, gives it one disk, and has root
over it. Everything below is off unless its variable is set, so a Compose
install behaves as before.

What the image adds for this:

- **SSH into the container**, keys only, so the UI and files are reached through
  `ssh -L` instead of an open port.
- **`INPUT_URL`**: the container fetches the clip itself (a presigned S3 GET).
- **`OUTPUT_UPLOAD_URL`**: each finished job's splats go up as one tar (a
  presigned S3 PUT), so no results have to be pulled.
- **CPU and GPU discovery**, used on every install: stages are sized to the CPUs the container
  is allowed, not the host's cores, and GPUs the image cannot run on are
  listed but never scheduled.

Needs image **0.1.3 or newer** for everything on this page. 0.1.1 was built
with `-march=native` on an AVX-512 CPU, and its training crashes on most rented
EPYC and Ryzen hosts ([troubleshooting #25](troubleshooting.md)); 0.1.2 fixed
that but has none of the remote features.

## The trust model

The host's root can read everything in the container while it runs: the clip,
the results, the queue token, anything you type. So:

- Put nothing there that reaches further than this run. A presigned URL
  can read or write one object until it expires; an AWS key, a Hugging Face
  token (SAM 3) or a forwarded SSH agent can do much more.
- sshd refuses agent forwarding, so `ssh -A` does nothing. Don't
  forward an agent to a rented host by other means.
- Check the host key the first time you connect: the fingerprints are in the
  container log, which both providers show outside the container.
- Destroy the instance when done, don't just stop it (see below).

## Quick start

On your machine, make the two URLs (needs `pip install boto3` and your normal
AWS credentials, which stay on your machine):

```bash
scripts/presign_s3.py s3://my-bucket/osvplat/run1 --clip ~/clips/DJI_0198.OSV --expires 24
```

It uploads the clip and prints `INPUT_URL`, `INPUT_SHA256` and
`OUTPUT_UPLOAD_URL`. Start the container with those plus your SSH public key
(the provider-specific parts are below):

```bash
docker run -d --gpus all -p 2222:22 \
  -e SSH_PUBLIC_KEYS="$(cat ~/.ssh/id_ed25519.pub)" \
  -e INPUT_URL='https://…' -e INPUT_SHA256=… -e OUTPUT_UPLOAD_URL='https://…' \
  ghcr.io/pgodlews/osvplat:0.1.3
```

The log prints the connect line:

```
OSVplat: ssh -p 40178 -L 8090:localhost:8090 root@203.0.113.10
  then open http://localhost:8090/?token=…
```

Connect, open the UI, queue the clip (it appears in the input list once the
download has finished), press **Resume queue**. When the job is done its
splats are uploaded; `GET /api/jobs/<id>` shows `upload.state`, `bytes` and
`sha256`, and the log prints the same. Download the result, compare the
sha256, then destroy the instance.

## Settings

| Variable | | |
|---|---|---|
| `SSH_PUBLIC_KEYS` | one or more public keys, one per line | starts sshd |
| `SSH_PUBLIC_KEY`, `PUBLIC_KEY` | what Vast and RunPod inject from your account's keys | also start sshd |
| `SSH_KEYS_URL` | `https://` list of keys, e.g. `https://github.com/<user>.keys` | an empty list is an error |
| `SSH_PORT` | 22 | |
| `QUEUE_BIND` | `127.0.0.1` with SSH on, else `0.0.0.0` | where the UI listens |
| `INPUT_URL` | URL of one clip, fetched into `samples/` at start | name from the URL path, or `INPUT_NAME` |
| `INPUT_SHA256` | checked before the clip is used; a mismatch deletes it | recommended |
| `OUTPUT_UPLOAD_URL` | presigned PUT URL, or a JSON target (below) | upload after each job that ends done |
| `QUEUE_CPUS` | discovered | override the CPU count stages are sized to |
| `QUEUE_MIN_COMPUTE_CAP` | 7.5 | GPUs below this are listed but not scheduled |
| `QUEUE_TLS_INSECURE` | 0 | `1` accepts self-signed certificates for `INPUT_URL`, `OUTPUT_UPLOAD_URL`, telemetry and the webhook |

SSH needs the container to run as root, which is what both providers do; with
Compose's `user:` set, the SSH variables are refused with an error.

`OUTPUT_UPLOAD_URL` details:

- The tar holds the job's `.ply`/`.sog`/`.spz` and `telemetry.json` under
  `job<id>/`. A PUT carries `Content-MD5`, so S3 rejects a damaged body.
- A plain presigned URL names **one object**, so it is used once: the first
  job that finishes uploads, and later jobs are refused
  (`upload.state: refused`) rather than overwriting it. For several jobs, use a
  JSON target with a `{job}` placeholder, in the same format as
  `QUEUE_TELEMETRY_UPLOAD` ([job-telemetry.md](job-telemetry.md)); a presigned S3
  POST whose policy allows a key prefix works for that.
- Failures are loud: `upload.state: failed` with the HTTP status (an expired
  URL gives 403), a line in the service log, and the `job.uploaded` webhook
  event. The archive stays in `runs/job<id>/`, so it can still be copied off
  over SSH. The job itself stays done.
- Checked before any job runs, without using the URL up (a presigned PUT
  cannot be tried without writing the object). The service logs when the
  URL expires, as written in the URL itself or in a POST policy, and whether
  the host resolves and accepts a TCP connection on its port. A malformed
  or already expired `OUTPUT_UPLOAD_URL` makes the API refuse new jobs
  (HTTP 400), while SSH and the UI stay up. The service logs a warning for a
  job submitted when the URL has less than an hour left. The job-creation
  response carries `upload_expires_in_s`.
- A malformed telemetry setting (`QUEUE_TELEMETRY_UPLOAD`,
  `QUEUE_TELEMETRY_PLACEMENT`) never stops jobs: it is logged and ignored, and
  telemetry is still written locally.

`QUEUE_TLS_INSECURE=1` is for a private endpoint with a self-signed
certificate, on a network you trust. With it, anyone on the path can read and
change the transfers, presigned URLs included. The service logs a warning at
startup while it is on.

## CPU and GPU discovery

The service logs what it found at startup, e.g. with `--cpus=2.5`:

```
resources: 2 CPUs (cgroup quota 2.5 of 8 visible), 33 GB RAM, 2 threads per stage; …
```

The CPU count is the smaller of the CPU affinity and the cgroup quota, split
between jobs that can run at once. Stages get it as `SPLAT_THREADS` (COLMAP's
thread count) and `OMP_NUM_THREADS`. Why it matters, measured on 2026-09-21 with
the Smoke test preset on the 0198 clip: a RunPod pod showed 112 CPUs against a
23.8-CPU quota, COLMAP started 178 threads, and SfM took 3267 s. A Vast host
whose 16 visible CPUs matched its allowance did the same SfM in 1270 s. (Those
runs predate this change; the 3267 s has not been re-measured with it.)

On rented hosts SfM, which runs on the CPU, is the slow stage: training the
same Smoke test took 122 s on an L4 and 96 s on a 3090. Pick offers by CPU as
well as GPU, and by speed per core rather than core count. Smoke test SfM on
the 0198 clip, 76 frames:

| Host | CPUs used | SfM |
|---|---|---|
| Vast, Ryzen 7 7700X (2022) | 16 | 1270 s |
| Vast, EPYC 7R32 (2020) | 46 of 96 | 2855 s |
| RunPod, EPYC 7663, 178 threads on a 23.8-CPU quota (before this release) | 23 | 3267 s |

GPUs are checked against the build. The image records the architectures it
was compiled for (`OSVPLAT_CUDA_ARCH`, from the build's `CUDA_ARCH`). A card is
usable when one of them has the same major version and a minor version no
higher than the card's: code built for 8.6 runs on an L4 (8.9), not on an
RTX 5090 (12.0). Other cards, and anything below 7.5, show as unsupported with
the reason in the startup log and the UI, and are never scheduled. When no GPU
here can run the build, the API refuses new jobs (HTTP 400) instead of queueing
them. This matters for local builds trimmed to one card
(`--build-arg CUDA_ARCH=8.6`) that later run on another. The published image
covers 7.5 to 12.0.

Rented hosts can be faulty in ways `nvidia-smi` does not show. At startup the
service also checks that CUDA itself starts (`cuInit`); on a host where it does
not, the log says the host is faulty and jobs are refused
([troubleshooting #29](troubleshooting.md)). Destroy it and rent another.

## Vast.ai

- Launch in **entrypoint mode**. The SSH and Jupyter modes replace the
  image's entrypoint with Vast's own setup, and with this image that setup
  fails: Vast's sshd expects host keys baked into the image, and this image
  deliberately has none ("sshd: no hostkeys available -- exiting", seen on
  2026-09-22). Entrypoint mode is not the default everywhere: the `vastai` CLI
  and the API create SSH-mode instances unless told otherwise. With the CLI,
  end the command with an empty `--args`:

  ```bash
  vastai create instance <offer> --image ghcr.io/pgodlews/osvplat:0.1.3 --disk 100 \
    --env "-p 22:22 -e SSH_PUBLIC_KEYS='ssh-ed25519 AAAA… you@host'" --args
  ```

  With the API or SDK, pass `runtype: "args"`.
- Docker options: `-p 22:22` (plus `-e` for the variables above). Vast maps
  ports to random external ones and prints the right `ssh -p` line in our log
  from `VAST_TCP_PORT_22` and `PUBLIC_IPADDR`. Search with
  `direct_port_count>0`.
- Keys: Vast injected the account's SSH keys as `SSH_PUBLIC_KEY` in SSH mode;
  in entrypoint mode, pass `SSH_PUBLIC_KEYS` with `-e` to be sure.
- Disk is fixed at creation: the image is 19 GB, a clip's job data ~20 GB.
- Filters used: `driver_version>=580.0.0` (the image needs CUDA 13), `cpu_cores_effective>=8`,
  `inet_down>=500`, `verified=true`, `reliability>0.98`.

## RunPod

- A Pod from a custom template with this image, **TCP port 22** exposed.
  Don't expose 8090 as an HTTP port: RunPod's proxy makes it public, and
  requests over 100 s (uploads, log streams) are cut off.
- Keys: RunPod injects your account's SSH keys as `PUBLIC_KEY`, and the
  string `null` when the account has none; that is ignored, so pass your key
  as `SSH_PUBLIC_KEYS` ([troubleshooting #27](troubleshooting.md)). The log's
  connect line uses `RUNPOD_PUBLIC_IP` and `RUNPOD_TCP_PORT_22`.
- The container log, with the host key fingerprints, is only shown in
  RunPod's web console (the API has no log endpoint), so compare the
  fingerprint there on the first connection.
- Container start command: leave it empty.
- Container disk: ~60 GB. Anything outside `/workspace` is lost when a pod
  is stopped.

## When the instance dies mid-job

Rented GPUs get interrupted: interruptible/spot instances are reclaimed,
hosts go offline, a pod is terminated by mistake. What that costs:

- **The work so far is lost with the instance's disk.** The stage cache,
  logs and partial training live in `QUEUE_ROOT` (`/data`) on the instance.
  A replacement instance starts from frames again, unless `QUEUE_ROOT` is on
  storage that outlives it: a RunPod network volume, or `/workspace`, which
  survives a stop but not a terminate. On Vast the disk goes with the
  instance.
- **No half-written result reaches the bucket.** The tar is written under a
  temporary name and renamed when complete. It then goes up in a single PUT
  (or POST), which S3 stores only once the whole body has arrived and matched
  its `Content-MD5`. An upload cut off halfway leaves nothing, or the previous
  object, never a truncated file.
- **The presigned URLs outlive the instance.** They work until they expire,
  for whoever has them, and the host could read them. Keep `--expires` close
  to the run's length. Give each run its own result key (or prefix), so a
  late or repeated upload cannot overwrite another run's result.
- Nothing retries the job on another instance. Check the result exists
  (`upload.state: done` and its sha256), not just that the instance is gone.

## Stopping and destroying

Measured on 2026-09-21:

- **An exiting container does not stop billing.** Both providers restart it
  (Vast after ~60 s, RunPod after ~10 s) and keep charging for the GPU.
- **A stop can strand the data.** A stopped Vast instance had its GPU rented
  to someone else and could not restart. A stopped RunPod pod came back with a
  new SSH port and an empty container disk (clip, `/data`, host keys gone).
- Both inject a key scoped to the instance (`CONTAINER_API_KEY` on Vast,
  `RUNPOD_API_KEY` on RunPod) that can stop or destroy it from inside. The
  container does not do that on its own yet.

So: get the result (or check the upload), then **destroy** the instance from
the provider's console or CLI.

## Foreign-process guard

The queue skips a GPU that runs a process it did not start. It recognises its
own processes by PID; Compose sets `pid: host` for that. On Vast and RunPod
`nvidia-smi` reports the container's own PIDs, and the guard worked without it
(checked during a job on both).
