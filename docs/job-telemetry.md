# Job telemetry

Every job writes a record of what ran, on what hardware, and how long each
stage took. It exists so that time estimates can be compared across machines:
`estimate.py` is calibrated on one RTX 3090, and the queue's own metrics say
nothing about the machine they came from.

Not to be confused with the camera telemetry inside a `.OSV`
([Osmo 360](osmo360-telemetry.md), [Avata 360](avata360-telemetry.md)).

## What is written, where

| File | When |
|---|---|
| `QUEUE_ROOT/runs/job<id>/telemetry.json` | after every stage, and when the job ends (done, failed, cancelled, waiting for mask review) |
| `QUEUE_ROOT/runs/job<id>/samples.jsonl.gz` | every 5 s while a stage runs: one line of machine and GPU load ([Time series](#time-series)) |
| `QUEUE_ROOT/runs/job<id>/logs.tar.gz` | when the job ends: this job's stage logs, redacted, each capped at the first 256 KB and last 1 MB, and `samples.jsonl.gz` |

Telemetry never stops or fails a job, or the service. A probe, sample, write,
upload or webhook that fails prints one line in the service log (`telemetry:
...`, `job N: telemetry not written: ...`) and the job carries on; the field it
would have filled is null.

`GET /api/jobs/<id>/telemetry` returns the record (`?logs=1` for the bundle).
They stay on the machine. Purging cleared jobs
(`POST /api/jobs/clear?purge=true`) deletes them with the job.

They are not in the export folder because an export is a cache entry that
every job with the same options shares, while telemetry belongs to one job.

## Settings

| Variable | Default | |
|---|---|---|
| `QUEUE_TELEMETRY` | `1` | `0` writes nothing |
| `QUEUE_TELEMETRY_UPLOAD` | unset | JSON upload target; unset means nothing is sent anywhere |
| `QUEUE_WEBHOOK_URL` | unset | POST a JSON event per stage and job end ([Webhook](#webhook)) |
| `QUEUE_WEBHOOK_SECRET` | unset | sign webhook bodies with HMAC-SHA256 |
| `QUEUE_BENCHMARK` | `0` | `1` runs the [host benchmark](#benchmark) once at startup, before any job starts |
| `QUEUE_BENCHMARK_DISK_GB` | `2` | size of the benchmark's disk test file |
| `QUEUE_TELEMETRY_PLACEMENT` | unset | JSON object copied into each record as `placement`, e.g. `{"provider": "…", "region": "…", "price_per_hour": 0.5}`, for whoever launched the machine |

Both files are uploaded once per job, when it has ended (done, failed or
cancelled; not while it waits for mask review), so the destination gets one
complete record rather than one per stage. A job that also uploads its result
(`OUTPUT_UPLOAD_URL`) sends its telemetry after that transfer, so the record
includes it ([Transfers](#transfers)). For progress while a job runs, use the
[webhook](#webhook). Each file gets 3 attempts. A failed upload is printed in
the service log and never fails the job.

Two target forms. `{job}` becomes
`job00042`, `{file}` becomes `telemetry.json` or `logs.tar.gz`:

```json
{"method": "PUT", "url": "https://example/ingest/{job}/{file}?sig=…",
 "headers": {"Authorization": "Bearer …"}}
```

```json
{"method": "POST", "url": "https://bucket.s3.example",
 "key": "batch-1/{job}/{file}",
 "fields": {"policy": "…", "x-amz-algorithm": "…", "x-amz-credential": "…",
            "x-amz-date": "…", "x-amz-signature": "…"}}
```

The POST form is an S3 presigned POST: its policy can allow a key prefix
(`starts-with`) and cap the size (`content-length-range`), so one target
covers every job on a machine. A malformed target (or `QUEUE_TELEMETRY_PLACEMENT`)
is reported as a `WARNING: ... ignored` line in the service log and switched
off, rather than silently sending nothing; it never stops jobs from running,
and telemetry is still written locally. A self-signed endpoint needs
`QUEUE_TLS_INSECURE=1` ([cloud.md](cloud.md#settings)).

Not every S3-compatible server honours a POST policy's own expiration.
versitygw v1.8.0 answered "Invalid according to Policy: Policy expired" to a
policy about an hour after it was signed, although its `expiration` was
12 hours out (2026-09-22: 204 at +55 min, 403 at +62 min; AWS S3 honours the
expiration). Telemetry is uploaded when a job ends, so on such a server a job
that ends more than an hour after the policy was signed sends no record; the
job and its `OUTPUT_UPLOAD_URL` result (a presigned PUT, unaffected) are fine,
and the record stays on the machine.

## Webhook

Optional, separate from the record: set `QUEUE_WEBHOOK_URL` and the queue
POSTs a small JSON event when a stage starts or finishes and when a job ends.
Useful for a phone notification (ntfy), a dashboard, or whatever launched the
machine. Events are sent in order from one thread, 3 attempts each, and a
failure is only printed.

```json
{"event": "stage.finished", "job_id": 42, "sweep_id": null,
 "stage": "sfm", "state": "done", "ts": 1790000000.0}
```

| `event` | `stage` | `state` |
|---|---|---|
| `stage.started` | the stage | `running` (only for stages that run, not cache hits) |
| `stage.finished` | the stage | `done`, `cached`, `skipped` |
| `job.waiting` | null | `awaiting_review` (mask review) |
| `job.finished` | null, or the stage that was running when it failed | `done`, `failed`, `cancelled` |

Ids and states only: no names and no error text, which can quote a path. The
details are one `GET /api/jobs/<id>/telemetry` away. `placement` is included
when `QUEUE_TELEMETRY_PLACEMENT` is set. With `QUEUE_WEBHOOK_SECRET`, each
request carries `X-OSVplat-Signature: sha256=<HMAC-SHA256 of the body>`.

## Fields (`osvplat.telemetry/1`)

| Field | Content |
|---|---|
| `job` | id, state, sweep id, created/started/ended, first 500 chars of the error (redacted like the logs), input size in bytes, the job config **without** the job name and clip file name (the extension and content hash stay) |
| `plan` | the per-stage estimate frozen at submit time, with its assumptions (panoramas, iterations, it/s) |
| `stages[]` | stage, state (done, cached, skipped, failed…), cache key, start/end, `wall_s`, `planned_s`, the numeric results the stage reported (`info`), and `resources` |
| `stages[].info` of a `.OSV` sfm stage with `sfm.upright` | `upright_requested`, `upright`, `metric` (flags); `upright_residual_deg`, `upright_scale`, and with GPS `gps_rms_m`, `gps_yaw_correction_deg`, `gravity_vs_gps_deg`. No coordinates: the GPS reference stays in the job's own `sfm/alignment.json` |
| `stages[].resources` | sampled every 5 s over the stage's process tree: CPU seconds, average cores busy, peak RSS, GPU utilisation p50/p95/mean and peak GPU memory (whole device), GPU health ([below](#gpu-health)), and `host`: the machine over the whole stage ([Machine load](#machine-load)) |
| `metrics` | the job's metrics table: PSNR, SSIM, splats, peak VRAM, train seconds… |
| `host` | OS, whether in a container; CPU model, logical CPUs, physical cores, effective CPUs (affinity and cgroup quota), AVX/AVX2/FMA/AVX-512; RAM, cgroup memory limit, `/dev/shm` size; `disk`: used/free/total under `QUEUE_ROOT` when the record is written, and the job's peak ([Disk space](#disk-space)); per GPU: name, memory, compute capability, driver, max PCIe gen/width, power limit, the card's default and max power limit, max SM and memory clocks |
| `software` | image version and git revision (`OSVPLAT_VERSION`, `OSVPLAT_REVISION`; the Dockerfile and `deploy.sh` set them), Python version |
| `transfers` | `input`: the `INPUT_URL` download of this job's clip; `output`: the `OUTPUT_UPLOAD_URL` upload ([below](#transfers)) |
| `placement` | `QUEUE_TELEMETRY_PLACEMENT`, or null |
| `timeline` | host boot time, service start time |

CPU seconds undercount processes that start and exit between two samples; the
stage scripts and LichtFeld are long-lived, so the figure holds for them.
Resource figures are null where there is no `/proc` (macOS).

### GPU health

Read from `nvidia-smi` in the same 5 s samples as utilisation. They tell a slow
host from a slow stage: a lowered power cap, a card on a narrow PCIe slot or
riser, or one that throttles hot shows here and not in utilisation. Compare
them with the card's limits under `host.gpus`.

| Field | Content |
|---|---|
| `gpu_busy_samples` | samples with utilisation at or above 50%; the `_busy_p50` figures are the median over these, because an idle GPU lowers its clocks on purpose |
| `gpu_power_w_busy_p50`, `gpu_power_w_max` | power draw, W |
| `gpu_sm_mhz_busy_p50`, `gpu_mem_mhz_busy_p50` | SM and memory clocks, MHz |
| `gpu_temp_c_max` | highest temperature, °C |
| `gpu_pcie_gen_max`, `gpu_pcie_width_max` | the highest PCIe link seen. The link trains down when idle, so only a stage that used the GPU says what the slot can do; below `host.gpus[].pcie_gen`/`pcie_width` there means a narrower slot or a riser |
| `gpu_ecc_uncorrected` | uncorrected ECC errors since the driver loaded; null on cards without ECC |
| `gpu_clock_reasons` | for each reason that held the clocks down, the share of samples it did: `gpu_idle`, `sw_power_cap`, `hw_slowdown`, `sw_thermal`, `hw_thermal`, `hw_power_brake`, `applications_clocks`, `sync_boost`, `display_clocks` |
| `gpu_sw_power_cap_busy` | share of busy samples in which the power cap held the clocks down. Use this, not `gpu_clock_reasons.sw_power_cap`: an idle card can report the cap too |
| `gpu_power_limit_w_min`, `gpu_power_limit_w_max` | the power limit the card enforced (`enforced.power.limit`), lowest and highest over the stage. Below `host.gpus[].power_default_w` means the host capped the card; `host.gpus[].power_limit_w` is read once, at service start |
| `gpu_power_limit_changes` | each change of that limit, `{t, from_w, to_w}`, at most 20 per stage. Compared with the last value seen on that GPU, so a change between two stages lands in the later one. Each is also a line in the service log. Null where the driver cannot report the limit |

One unknown field fails a whole `nvidia-smi` query. The sampler tries the
newer name (`clocks_event_reasons`), then the older one
(`clocks_throttle_reasons`), each first with `enforced.power.limit` and then
without, then utilisation and memory only, and keeps the first that works. A
driver with none of these gives null health fields, not a failed job. Xid errors are not recorded: `nvidia-smi` cannot query them, and a
container usually cannot read the kernel log where they appear.

### Disk space

The filesystem under `QUEUE_ROOT`, read in the same 5 s samples (`statvfs`,
cheap). Used counts every file on it: the clip, the cache, runs and, in a
container, the image's writable layer. So the peak is the disk a machine needs
for this job, which is what a rented container disk is sized on.

| Field | Content |
|---|---|
| stage `disk_used_bytes_start`, `disk_used_bytes_peak` | used at the stage's first sample, and the most during it |
| stage `disk_free_bytes_min` | the least free space during the stage |
| `host.disk.used_peak_bytes`, `host.disk.free_min_bytes` | the same over the whole job |
| `host.disk.job_growth_peak_bytes` | peak used minus used at the start of the stage that ran first: what the job itself wrote on top of the clip and what was already there. Cached stages write little, so compare jobs that ran every stage |

### Machine load

`stages[].resources.host`, each line of the [time series](#time-series), and the
benchmark's `resources.host` describe the machine over an interval, from
counters the kernel keeps. They tell a slow host from a slow stage: a
neighbour on the same box, a CPU quota the job keeps hitting, a slow disk.

| Field | Content |
|---|---|
| `seconds` | the interval |
| `cpu_busy`, `cpu_iowait`, `cpu_steal` | share of all CPU time on the machine (`/proc/stat`, not limited to this container). Steal is time a hypervisor gave to other guests: well above zero means the cores are shared. Busy leaves steal out, so busy + idle + iowait + steal = 1 |
| `core_busy_pct` | per core, 0–100, in the time series only |
| `psi` | pressure stall information for the whole machine: the share of the interval in which some (`_some`) or all (`_full`) tasks waited for CPU, memory or IO |
| `cgroup_psi` | the same for this container's cgroup alone |
| `cgroup_throttled`, `cgroup_throttled_s` | the share of CPU-quota periods in which this container was throttled, and the time it lost. High means the job wants more CPUs than the host gives it |
| `disk_read_mb_s`, `disk_write_mb_s` | all disks, summed (partitions, loop and device-mapper devices left out so nothing is counted twice) |
| `net_rx_mb_s`, `net_tx_mb_s` | all interfaces but `lo`; in a container, its own traffic |

A source the kernel does not have (no PSI, macOS) is null. No device or
interface names are recorded.

### Time series

`samples.jsonl.gz`: one JSON line per 5 s sample, while a stage's process
runs. Each line is its own gzip member, so a service killed mid-write loses at
most that line. The first sample of each stage only sets the baseline, so a
stage shorter than about 5 s adds no line. Writing stops at 32 MB, with a line in
the service log.

| Field | Content |
|---|---|
| `t`, `stage` | wall-clock time and the running stage |
| `cores_busy`, `rss_mb` | the stage's process tree: cores' worth of CPU over the interval, resident memory |
| `disk_used_mb`, `disk_free_mb` | the filesystem under `QUEUE_ROOT` ([Disk space](#disk-space)) |
| `host` | [machine load](#machine-load) over the interval, with `core_busy_pct` |
| `gpu` | this job's GPU at the sample: `util`, `mem_mib`, `power_w`, `sm_mhz`, `mem_mhz`, `temp_c`, `pcie_gen`, `pcie_width`, `clock_reasons` (the reasons active), `power_limit_w` (the enforced limit); null without a GPU |

### Transfers

Sizes and times of the transfers the machine already does, so a slow link
shows next to the stage timings. Nothing extra is sent to measure it.

| Field | Content |
|---|---|
| `input` | `bytes`, `seconds`, `mb_s` (10⁶ bytes/s), `first_byte_s`, `ended`, from the entrypoint's curl (`QUEUE_ROOT/runs/input_fetch.json`); timings are of the last attempt if curl retried. Null when the clip came another way or was already there |
| `output` | `state`, `bytes`, `seconds` (the successful request alone), `mb_s`, `attempts`, `pack_s` (building the archive), `ended`. Null when there is no `OUTPUT_UPLOAD_URL`. The upload ends after the job does, so `telemetry.json` is written once more when it finishes, and only then uploaded |

Neither carries the URL, the bucket or the clip name.

## Benchmark

A fixed workload, so that machines compare: per-job timings do not, because
every clip is different. `POST /api/benchmark` starts it and returns
`{"state": "running", ...}`; `GET /api/benchmark` returns the running state or
the last result, which is also kept in `QUEUE_ROOT/benchmark.json`. It takes
one to two minutes and holds the whole machine: it is refused (HTTP 409) while
a job or compare render runs, no job starts until it ends, and `/api/status`
shows `benchmark_running`. `QUEUE_BENCHMARK=1` runs it at startup.

Every input is generated from a fixed seed; no clip is read. The record holds
raw scores only. Deciding whether a machine is good enough is left to whoever
launched it.

| Field | Content |
|---|---|
| `state` | `running`, `done`, or `failed` when any part failed (the other parts still ran and have scores) |
| `errors` | per part, what went wrong. A GPU that is busy with a foreign process is `"gpu": "no free GPU: ..."` and fails the run |
| `cpu` | a synthetic 3840×1920 frame: JPEG decode and encode per second on one thread; SIFT extract (1920×960, about 7500 features) and match per second on one thread; decode and SIFT per second with one process per allowed CPU (`processes`, the effective CPUs in `host`), and `decode_scaling` / `sift_scaling`: how many single cores that was worth |
| `memory` | `copy_gb_s`: single-thread copy of a 512 MiB buffer |
| `disk` | `write_fsync_mb_s` and `read_mb_s` of a `QUEUE_BENCHMARK_DISK_GB` file under `QUEUE_ROOT`, read back with the page cache dropped (`cache_dropped`; false where the OS cannot, such as macOS) |
| `gpu` | `device`; `torch_import_s`, `cuda_init_s`; `matmul_fp32_tflops` (TF32 off) and `matmul_fp16_tflops` at 8192²; `d2d_copy_gb_s`; `h2d_pinned_gb_s` and `d2h_pinned_gb_s` for 1 GiB (a card on a narrow slot or riser shows here); `gsplat_it_s`: forward + backward of 1 M Gaussians at 1920×1080 |
| `resources` | the [stage sampler](#gpu-health) over the run, every 2 s: GPU power, clocks, clock reasons and PCIe link under load, CPU seconds and cores busy, and `host`: steal, pressure and throttling while it ran ([Machine load](#machine-load)). A benchmark on a shared or quota-limited host says so here |
| `durations_s`, `wall_s`, `started`, `ended`, `gpu_index` | timing, and which GPU ran the GPU part |
| `host`, `software`, `placement` | as in a job record |

The log is `QUEUE_ROOT/logs/benchmark.log`.

## Prometheus metrics

With `QUEUE_METRICS=1`, `/metrics` also carries the machine and GPU figures
above, as the kernel's own counters: `rate()` them for shares and speeds.

| Metric | |
|---|---|
| `splatqueue_host_cpu_seconds_total{mode}` | `busy`, `idle`, `iowait`, `steal`, whole machine |
| `splatqueue_host_pressure_stalled_seconds_total{scope,resource,kind}` | PSI stall time; `scope` is `system` or `cgroup` |
| `splatqueue_cgroup_cpu_periods_total`, `…_throttled_periods_total`, `…_throttled_seconds_total` | CPU-quota throttling of this container |
| `splatqueue_host_disk_read_bytes_total`, `…_written_bytes_total` | all disks |
| `splatqueue_queue_root_used_bytes`, `…_free_bytes` | the filesystem under `QUEUE_ROOT`, gauges at scrape time |
| `splatqueue_host_network_receive_bytes_total`, `…_transmit_bytes_total` | all interfaces but `lo` |
| `splatqueue_gpu_power_watts`, `…_sm_clock_hertz`, `…_memory_clock_hertz`, `…_temperature_celsius`, `…_pcie_link_generation`, `…_pcie_link_width`, `…_ecc_uncorrected_errors` | per `gpu`, gauges at scrape time |
| `splatqueue_gpu_clock_event_reason{gpu,reason}` | 1 while that reason holds the clocks down |
| `splatqueue_gpu_power_limit_watts` | per `gpu`, the power limit the card enforces now; below the card's default is a host cap |

## What is left out

No hostnames, user names, file paths, clip or job names, GPU UUIDs or serial
numbers. Stage info keeps numbers and booleans only, because it also carries
paths. In the log bundle, `SPLAT_ROOT`, `QUEUE_ROOT` and the home directory
become `$SPLAT_ROOT` / `$QUEUE_ROOT` / `$HOME`, the hostname becomes `<host>`,
the clip name `<clip>` (when it is at least 6 characters and not all digits,
so frame numbers survive), and `serial`/`sn` values and `lat`/`lon` numbers
are replaced. Logs of stages reused from the cache are left out: they belong
to the job that built them.

Redaction is pattern-based. Read a bundle before sharing it if your logs
might carry something else.
