# How it works

What the pipeline does with a clip, why each stage is built the way it is, and
the measurements behind the defaults. For setup see [install.md](install.md);
for things that go wrong see [troubleshooting.md](troubleshooting.md).

## Architecture

Everything runs on one Linux workstation with an NVIDIA GPU: NVDEC decode,
frame selection, person masks, COLMAP SfM on the GPU and LichtFeld Studio
training, driven by a queue service with a web UI. The finished `.sog` can be
hosted on any static web server with a splat viewer; no server code.

## Two input paths

```
Raw DJI .OSV (Osmo 360, Avata 360)                      ← recommended
  frames ─► select ─► [mask] ─► fisheye rig SfM ─► LichtFeld (--gut, fisheye) ─► PLY / SOG / SPZ

Stitched equirectangular video (a graded DJI Studio export, or any 2:1 360° video)
  frames ─► select ─► [mask] ─► spherical SfM ─► LichtFeld (--gut, equirect) ─► PLY / SOG / SPZ
```

The queue picks the path from the file extension. The six stage names, the
cache, the review gate and the UI are shared; each stage dispatches on the
input. Per-stage detail is in [queue/README.md](../queue/README.md).

| Stage | What it does |
|---|---|
| `frames` | ffmpeg decode (CUDA) at `frames.fps` (default 10). For `.OSV`: both lens streams, plus `calibration.json` read out of the file. |
| `select` | Keep the sharpest frame of every `select.window` (Laplacian variance; for a rig, scored on the blurrier lens). Optional gyro veto, below. |
| `mask` | Optional person masks, below. |
| `sfm` | COLMAP 4.2 via pycolmap-cuda: GPU SIFT, sequential matching, incremental mapping. Picks the largest model, drops stray cameras, and gates on registration rate and reprojection error. |
| `train` | LichtFeld Studio headless, MRNF strategy, 3M splat cap, SH degree 1, 30k iterations, 3DGUT rasterizer (`--gut`), which is the only way it trains fisheye and equirect cameras. |
| `export` | Collects `.ply`, `.sog` and `.spz`. |

Every stage is cached by a hash of the options that affect it. Changing a
training option reuses the frames, selection, masks and SfM.

## Fisheye rig (no stitch)

Both DJI 360 cameras write a per-unit lens calibration into every `.OSV`
([osmo360-telemetry.md](osmo360-telemetry.md), [avata360-telemetry.md](avata360-telemetry.md)).
Instead of stitching and running SfM on the panorama, each instant becomes one
frame of a **two-camera rig**, both cameras `OPENCV_FISHEYE` (the same
Kannala–Brandt model DJI stores), seeded from the file's values. No DJI Studio,
no seam.

On a 31 s handheld Osmo 360 clip, same 104 instants for every arm:

| Arm | Points | Median angular error |
|---|---|---|
| Stitched equirect (calibration as first read), spherical SfM | 47,080 | 0.0627° |
| Stitched equirect with the refined lens + rig | 57,398 | 0.0485° |
| **Fisheye rig, stored calibration read in full, held fixed** | **109,137** | **0.0439°** |

(Angular error, not pixels: a pixel means different things in a fisheye and an
equirect. `83_sfm_angular_error.py` unprojects every observation to a ray.)

Training both to the same budget and scoring the same held-out fisheye pixels:
the rig model won PSNR in 20 of 26 views (+0.29 dB mean) and SSIM in 25 of 26,
in about 60% of the training time. Most of the margin is at the lens rims,
where the stitched model smears. The splat-stage margin is small; the SfM-stage
margin is not. **For `.OSV` input, use the fisheye path** — the queue does so
automatically.

`scripts/osv_meta.py <clip> <outdir>` writes the calibration and a telemetry
CSV for any `.OSV` or `.LRF` on its own, if you want to look.

## Stitched video input (graded or D-Log footage)

The raw `.OSV` path reads the file straight off the camera, so it trains on the
colours the camera recorded. If you want to grade D-Log M footage, denoise, or
otherwise preprocess first, export a stitched equirectangular video instead and
queue that. It takes the stitched path: spherical SfM on the panoramas and
LichtFeld trained on the equirect frames.

- **Format:** `.mp4`, `.mov` or `.mkv`, 2:1 equirectangular, e.g. DJI Studio's
  *Panoramic Video* export at full resolution (7680×3840). Anything ffmpeg can
  decode works; H.265 decodes on the GPU.
- **Turn stabilization and horizon levelling off** in the export. They warp
  each frame differently, which SfM reads as the scene moving.
- **Colour:** the splat reproduces the pixels it is given, so grade before
  export if you want a graded splat. Ungraded D-Log reconstructs fine but
  gives an equally flat-looking splat.
- Put it in `~/splat/samples/` like any clip; the UI shows it as "stitched".

What the stitched path gives up, compared with the raw `.OSV`:

| | Raw `.OSV` | Stitched video |
|---|---|---|
| SfM | fisheye rig, ~2× the points (above) | spherical, one projection centre for both lenses |
| Seams | none | whatever the stitcher left |
| Gyro blur veto (`select.imu`) | yes | refused: no orientation stream |
| Distance-based selection (Avata 360) | yes | no: needs the flight telemetry |
| Person masking, review gate, presets, sweeps | yes | yes |
| Training resolution | full 3840² per lens | LichtFeld's `--max-width` (default 3840, i.e. 3840×1920 per panorama) |

## Person masking

On a handheld or selfie-stick 360 capture the operator is **rigidly attached
to the camera**: the same solid angle in every frame, never seen from a second
viewpoint. The trainer bakes them in as a smear along the camera path. No
trainer setting fixes that; the fix is to stop supervising those pixels.

`scripts/70_person_masks.py` has two backends:

| | `maskrcnn` (default) | `sam3` |
|---|---|---|
| Model | torchvision Mask R-CNN v2, COCO `person` | SAM 3, open-vocabulary text prompts |
| Inferences per panorama | 13 tangent views | 2 (equirect + nadir) |
| Time per frame (RTX 3090) | 0.70 s | 0.53 s |
| Setup | none, weights auto-download | one download of gated weights ([Docker](docker.md#sam-3-masks-optional), [native](install.md#optional-sam-3-masks)) |
| Licence | BSD-3-Clause | Meta SAM licence |

Evaluated over identical pixels, the two produce equivalent splats (scene-pixel
PSNR 21.33 vs 21.41 dB, against 20.91 unmasked). Pick on licence and setup, not
quality. SAM 3 can mask other things too (`"prompts": ["person", "dog"]`).

For `.OSV` input the masker runs on a calibrated stitch and the masks are
carried back into each fisheye through the same geometry. With
`"mask": {"review": true}` the job pauses after masking and releases its GPU
until you approve a contact sheet of overlays in the UI.

A masked run's PSNR is not comparable to an unmasked one — see
[troubleshooting #22](troubleshooting.md).

## Gyro blur veto

An `.OSV` carries the camera's orientation at 1 kHz (Osmo 360) or 4 kHz
(Avata 360), and every frame's exposure time. So each candidate frame's
rotational motion blur can be *predicted*:

**smear ≈ fx × mean |ω| over the exposure × exposure time**

With `"select": {"imu": true}`, candidates predicted to smear more than 0.5 px
beyond the stillest in their window are vetoed before the sharpness pick. On a
99 s indoor clip it cut frames predicted over 4 px of smear from 101 to 85, and
the swapped frames are visibly sharper. But the final splat came out the same
(+0.04 dB over 64 held-out views, p = 0.53), and in daylight exposures are too
short for it to change anything. It is **off by default**; try it for indoor
or dusk footage.

## Capturing for a good splat

From aerial and handheld runs on both cameras, two things dominate quality and
neither is a trainer setting:

1. **How much of the sphere is static, textured surface.** A low pass over
   land gave the richest reconstruction; a lake filling the lower hemisphere
   gave haze.
2. **How close the subject is, and whether you orbit it.** A low, close orbit
   gave the finest detail even with worse SfM numbers.

So: **fly or walk low and slow, orbit what you care about, stay over land.**
Water, sky, and moving people, cars and boats contribute nothing and pull the
far field into haze. Handheld: keep the stick long and turn on person masking.
Keep the shutter fast (daylight, or a short exposure setting) — blur costs
more than resolution.

Measured on one RTX 3090 (a 3090 over OcuLink in a mini PC; a desktop 3090
was within a minute of it on the Draft run):

| Preset | Clip | Frames | frames+select | mask | sfm | train | total |
|---|---|---|---|---|---|---|---|
| Smoke test | 30 s of an Osmo walk | 100 rig | 1.0 min | 1.9 min | 4.1 min | 1.7 min | **9.5 min** |
| Draft | 2.5 min Osmo walk | 302 rig | 6.6 min | 5.2 min | 10.8 min | 12.0 min | **35 min** |
| Standard | 2.2 min Osmo walk | 267 rig | 5.9 min | 4.7 min | 9.1 min | 105.7 min | **125 min** |

Training dominates, and it is the one stage a preset changes most: Smoke and
Draft cut iterations and the splat cap, not frame density. Masking adds about
1 s per frame and only runs when people are in shot. The Standard run peaked
at **10.1 GB of VRAM** (3M splats, 3840 px training width).

## Output formats

| Format | Typical size (2M splats) | For |
|---|---|---|
| `.ply` | ~200 MB | Lossless master; every tool reads it; the only one to retrain or re-decimate from. |
| `.sog` | ~27 MB | Web viewers ([SuperSplat](https://superspl.at/editor), PlayCanvas). |
| `.spz` | ~40 MB | Compact interchange (v4, zstd). |

## Pinned toolchain

The setup scripts pin the exact commits this pipeline was validated at, so a
rebuild months later does not silently produce a different trainer:

| Component | Pin | Override |
|---|---|---|
| LichtFeld Studio | `04e4607b` | `LFS_REF` |
| vcpkg | `04a9d8e5` (2026.07.29) | `VCPKG_REF` |
| gsplat (renders and evaluation only) | `28e794ca` (1.6.0), torch 2.9.1+cu130 | `GSPLAT_REF` |
| pycolmap-cuda12 | 4.2.0 | edit `setup_sfm_venv.sh` |
