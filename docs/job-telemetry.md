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
| `QUEUE_ROOT/runs/job<id>/logs.tar.gz` | when the job ends: this job's stage logs, redacted, each capped at the first 256 KB and last 1 MB |

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
| `QUEUE_TELEMETRY_PLACEMENT` | unset | JSON object copied into each record as `placement`, e.g. `{"provider": "…", "region": "…", "price_per_hour": 0.5}`, for whoever launched the machine |

Two target forms. Files are uploaded after every write, with 3 attempts. A failed
upload is printed in the service log and never fails the job. `{job}` becomes
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
expiration). Telemetry is uploaded after every stage, so on such a server the
records of a job longer than an hour stop arriving; the job and its
`OUTPUT_UPLOAD_URL` result (a presigned PUT, unaffected) are fine.

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
| `stages[].resources` | sampled every 5 s over the stage's process tree: CPU seconds, average cores busy, peak RSS, GPU utilisation p50/p95/mean and peak GPU memory (whole device) |
| `metrics` | the job's metrics table: PSNR, SSIM, splats, peak VRAM, train seconds… |
| `host` | OS, whether in a container; CPU model, logical CPUs, physical cores, effective CPUs (affinity and cgroup quota), AVX/AVX2/FMA/AVX-512; RAM, cgroup memory limit, `/dev/shm` size; free/total disk under `QUEUE_ROOT`; per GPU: name, memory, compute capability, driver, PCIe gen/width, power limit |
| `software` | image version and git revision (`OSVPLAT_VERSION`, `OSVPLAT_REVISION`; the Dockerfile and `deploy.sh` set them), Python version |
| `placement` | `QUEUE_TELEMETRY_PLACEMENT`, or null |
| `timeline` | host boot time, service start time |

CPU seconds undercount processes that start and exit between two samples; the
stage scripts and LichtFeld are long-lived, so the figure holds for them.
Resource figures are null where there is no `/proc` (macOS).

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
