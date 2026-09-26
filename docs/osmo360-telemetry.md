# DJI Osmo 360 telemetry format (`.OSV` / `.LRF`)

Status: **decoded 2026-09-11** from one clip, `CAM_20260813121712_0040_D` (Osmo 360, firmware 10.00.25.29, 63 s at 50 fps). Nothing here comes from DJI documentation; every value was read out of that file. Field paths use ExifTool's `dvtm_oq101` notation (`3-2-3-1` = top-level field 3 → 2 → 3 → 1). Status column: **✓** agrees with ExifTool 13.55's `lib/Image/ExifTool/DJI.pm`; **new** is not in it; **?** is present but not understood. Other firmware or recording modes may differ. **Orientation-stream timing and the quaternion convention were measured the same day** with [`scripts/osv_imu.py`](../scripts/osv_imu.py) (§6–7).

Related: extractors [`scripts/osv_meta.py`](../scripts/osv_meta.py) and [`scripts/osv_imu.py`](../scripts/osv_imu.py); the fisheye SfM that uses the calibration, [how-it-works: fisheye rig](how-it-works.md#fisheye-rig-no-stitch).

---

## 1. Where it lives

Both files are ordinary ISO-BMFF (MP4).

| Track | `.OSV` | `.LRF` |
|---|---|---|
| video | 2 × `hvc1` 3840×3840 @ 50 fps, one per lens, unstitched | 1 × `avc1` 2048×1024 @ 25 fps, both fisheye circles side by side, unstitched |
| audio | `mp4a` 48 kHz | `mp4a` 48 kHz (byte-identical) |
| `djmd` (handler "CAM meta") | 2 tracks, 2,022,401 B + 451,036 B | 2 tracks, 1,736,918 B + 228,659 B |
| `dbgi` (handler "CAM dbgi") | 2 tracks, ~12.45 MB each (~3.9 kB/frame) — **not decoded** | absent |

- **Trailing top-level `camd` box** (OSV: 2,525,384 B at the end of the file): strip its 8-byte header and the rest is a complete, valid MP4 — `ftyp`, `free`, `mdat`, and a `moov` holding the **same two `djmd` tracks**. It exists so a tool can read the telemetry without demuxing gigabytes of HEVC. Its `moov/udta/Xtra` repeats the schema as a UTF-16 string: `pb_file:dvtm_oq101.proto;model_name:OQ001;pb_version:2.0.8;pb_lib_version:02.01.15;`.
- `moov/udta/meta/ilst` holds two 688×344 **stitched equirect** JPEG thumbnails, `covr` and a DJI-private `snal`, same layout (`data` child, type 13).
- A 4 kB index at `free@36` lists `covr`/`snal`/`camd` offsets, but in a finished file those point into video data and padding. Ignore it; walk the box tree.

## 2. Framing

A `djmd` sample, and equally the `camd` box's `mdat`, is a flat run of **length-delimited protobuf fields** at the top level. No DJI framing around them.

| Top-level field | Meaning |
|---|---|
| 1 | header: device identity and global values |
| 2 | config: stream description and, once, the lens calibration |
| 3 | one per-frame record |

Order in the `camd` `mdat` (the two tracks' samples interleaved):

```
[1 header 177 B][2 config 5,622 B][3 rich 1,087 B]    <- track 1, sample 0
[1 header 124 B][2 config    40 B][3 light  137 B]    <- track 2, sample 0
[3 rich ~636 B][3 light ~140 B]                       <- every later instant, rich first
```

So each track's first sample opens with its own header and config, and **only track 1's carries the calibration**. There are exactly 2 records per instant (OSV: 6,308 records for 3,154 frames), and both share the same timestamp.

Decoding without the `.proto`: only wire types 0 (varint), 2 (length-delimited) and 5 (32-bit, always a little-endian `float` here) occur. Several fields are written **present but zero-length** — the light record carries an empty `3-2-9` — so treat "present and empty" as absent.

## 3. Header (field 1)

| Path | Value in this clip | Meaning | Status |
|---|---|---|---|
| `1-1-1` | `dvtm_oq101.proto` | schema name; identifies the camera family | ✓ |
| `1-1-2` | `02.01.15` | protobuf library version (per `Xtra`) | new |
| `1-1-3` | `2.0.8` | schema version (per `Xtra`) | new |
| `1-1-5` | (omitted) | serial number | ✓ |
| `1-1-6` | `10.00.25.29` | firmware version | new |
| `1-1-9` | 184053920 | timestamp of the first frame, µs (= `3-1-2` of record 0) | new |
| `1-1-10` | `Osmo 360` | model | ✓ |
| `1-2-1`, `1-2-2` | 1, 1 | — | ? |
| `1-3-1` | 4 × f32: 0.15513, 0.13714, −0.09386, 0.00417 | — | ? |
| `1-4-1` | 21299040 | — | ? |
| `1-5-1` | 4 | — | ? |
| `1-8-1` | f32 1061.58 | — (magnitude of a focal length in px) | ? |
| `1-10-1` | 1000 | probably the nominal IMU rate in Hz; the orientation stream measures 1,001.24 Hz (§6) | ? |
| `1-11-1` | f32 49.9985 | frame rate; the frame timestamps run on exactly this grid (§6) | new |
| `1-12-1` | varint 18446744073709551448, i.e. **−168** as int64 | — | ? |
| `1-13-1` | 4226 | — | ? |
| `1-14-1`, `1-14-2` | 3840, 3840 | **frame size**; ExifTool reads `1-14-1` as FNumber (a rational), which here would print f/1.0 | ✗ in ExifTool |
| `1-15-1` | bytes `13 0a` | — | ? |

Track 2's header carries only `1-1-*`, `1-2-*`, `1-11-1` and `1-15-1`; the rest are empty. The LRF's track-1 header is **byte-identical** to the OSV's, including the 3840×3840 frame size its own video does not have.

## 4. Config (field 2) and the lens calibration

| Path | Value | Meaning | Status |
|---|---|---|---|
| `2-1-3` | `video` | stream type | new |
| `2-3-1`, `2-3-2` | 3840, 3840 | frame size | new |
| `2-3-3` | f32 50 | frame rate | new |
| `2-3-4`, `2-3-6`, `2-3-8` | 1, 4, 1 | — | ? |
| `2-3-5` | 10 | possibly bit depth | ? |
| `2-5-1` | 1 | — | ? |
| `2-6-n` | 24 repeated slots, **track 1 only** | **lens calibration** | new |
| `2-8-1` | 7 | only in track 2's config | ? |

### Calibration slots (`2-6-n`)

Slots 1 and 2 are the active pair for this recording mode: **slot 1 describes video stream 0 (rear lens), slot 2 stream 1 (front lens)**. Slots 3–10 are zero-filled placeholders. Slots 11–24 are populated with similar but not identical values, presumably for other resolution/FOV modes (unverified). Each populated slot is 264 B.

| Sub-field | Slot 1 (rear) | Slot 2 (front) | Meaning | Status |
|---|---|---|---|---|
| 1, 2 | 1043.0132, 1042.7195 | 1051.8203, 1051.5679 | fx, fy (px) | new |
| 3, 4 | 1911.4990, 1919.9227 | 1908.6526, 1920.6356 | cx, cy (px) | new |
| 5–8 | 0.0659303, −0.00934961, 0.00795592, −0.00577805 | 0.0649202, −0.0121472, 0.00925767, −0.00635162 | k1–k4, Kannala–Brandt: r = f·θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸) — the same polynomial as COLMAP's `OPENCV_FISHEYE` | new |
| 10, 11 | 3840, 3840 | 3840, 3840 | calibrated image size | new |
| 12, 13, 14 | −179.8096, 90.1301, −0.7701 | −0.0577, 90.3230, −0.1016 | yaw, pitch, roll (°); pitch ≈ 90 puts both optical axes in the horizontal plane, yaws ~180° apart | new |
| 15 | f32 0.000818 | f32 0.000953 | — | ? |
| 20 (packed 2 × f32), 27 (same values) | −0.00011, 0.00083 | 0.00015, 0.00023 | — | ? |
| 21 (packed 4 × f32), 28 (the same four as sub-fields 1–4) | −0.003584, 0.005923, −0.706294, 0.707884 | 0.705110, 0.709097, −0.000984, −0.000268 | the lens rotation as a unit quaternion; component order not established | new |
| 22, 23 (14 × f32 each) | x: 1920, 317.9, 518.1 … 3522.1; y: 3735, 2845, 3096 … 2845 | same | boundary polyline in px: element 0 is (1920, 3735), then 13 points from 150° to 30° in 10° steps at radius ≈ 1815 px, widening to 1830/1850 px at ±40°/±30° | new |
| 24, 25 | f32 8, f32 −1000 | same | — | ? |

**How good the calibration is**, measured by letting COLMAP bundle adjustment refine it on 104 rig frames of this clip (docs/how-it-works.md, "Fisheye rig"):

- **The focal length is right. The distortion rolls off too fast past ~70° off-axis.** Radius in px for the rear lens, stored → refined: 60° 1161 → 1162; 80° 1556 → 1586; 85° 1623 → 1685; 90° 1657 → 1775; 95° **1636** → 1850. The stored polynomial turns back on itself past ~91°, so it cannot reach the image circle its own boundary polyline describes; the refined one does. Both lenses refine to fx ≈ 1046.
- **The lens-to-lens rotation built from the stored yaw/pitch/roll is 1.65° off** from the refined rig. Stitching with the stored values gives ~150 px smeared seams; with the refined ones the seams disappear. This is the stored angles *as interpreted* in [`scripts/osmo_fisheye.py`](../scripts/osmo_fisheye.py) — the construction there is validated only up to that 1.65°, so the misreading could be ours.

## 5. Per-frame records (field 3)

Two records per instant, one per video track:

- **rich** — track 1, ~636 B (LRF ~1,094 B): everything, including orientation and the high-rate block.
- **light** — track 2, ~140 B: timestamp, exposure, model code.

Both carry their own ISO, shutter, colour temperature and exposure block, and at some instants **those values differ between the two** — consistent with per-lens exposure, not established.

| Path | In | Value / range over the 63 s clip | Meaning | Status |
|---|---|---|---|---|
| `3-1-1` | both | OSV: 0, 2, 4 … 6306 (**+2 per instant**); LRF: +1 per instant | ExifTool calls it FrameNumber; that fits the LRF but not the OSV | ✓ path, semantics unclear |
| `3-1-2` | both | 184,053,920 → 247,115,770; step 20,000–20,001 (LRF 40,001–40,002) | timestamp, **µs on the camera's clock** (since boot; no absolute time), on an exact 20,000.587 µs grid; the container PTS assume a nominal 20,000 µs (§6) | ✓ |
| `3-2-3-1` | both | f32, 626–5,853, median 795 | ISO | ✓ |
| `3-2-4-1` | both | packed varints `(1, N)`, N = 100–5,454 | shutter 1/N s | ✓ |
| `3-2-5-1` | both | always 1.0 | — | ? |
| `3-2-6-1` | both | 5,363–5,884 | colour temperature, K | ✓ |
| `3-2-9-1…4` | rich | unit quaternion (w, x, y, z); all 3,154 have norm 1.000000 | camera orientation at the frame: Hamilton, body → world, world z down. **Bit-identical to sample index 5** of the same record's `3-3-2-1-3` block, at every exposure (§6) | new (ExifTool's Action 4 note guesses `3-2-9-1` is an accelerometer axis) |
| `3-2-10-2/3/4` | rich | X −2.87…1.85, Y −1.72…1.58, Z −2.55…0.80 (median −0.99); \|a\| median 1.005 | accelerometer, **g**, one sample per frame (50 Hz) | ✓ |
| `3-2-15-1` | both | f32 7.89–14.28, median 13.35 | — | ? |
| `3-2-15-2` | both | f32 99.0–248.8, median 120.7 | — | ? |
| `3-2-15-3/4/5` | both | f32 1.0–7.70, always equal to each other | — (looks like one gain applied to three channels) | ? |
| `3-2-15-6` | both | f32 6.44–8.04 | — | ? |
| `3-2-15-7` | both | f32 1.53–2.00 | — | ? |
| `3-2-15-8` | both | always 1 | — | ? |
| `3-2-16-1` | both | f32 24–35, median 31, 12 distinct values | — (range fits a temperature in °C) | ? |
| `3-2-17-1` | both | packed varints `(100, 1000)`, constant | — | ? |
| `3-3-2-1-1` | rich | µs, 978 before to 19 after `3-1-2` (median 478 before) | block start: **the true time of the block's first sample** — steps between starts are whole IMU periods to 1.2 µs | new |
| `3-3-2-1-2` | rich | 680 → 3,833, **+1 per instant**, no gaps (LRF: +2, the numbers of the OSV blocks it keeps) | block sequence number | new |
| `3-3-2-1-3` | rich | repeated, 20 per instant (21 in 80 of 3,154), each 4 × f32 sub-fields | **orientation samples at 1,001.24 Hz**: unit quaternions, w, x, y, z, Hamilton, body → world, world z down (§6) | new |
| `3-3-2-1-4` | rich | f32 2.50–7.41 | = 7.50 − 0.497 × exposure in ms (R² 0.990; the LRF's copy fits 7.5 − exposure/2 exactly): a timing term referenced to mid-exposure; its zero point is not identified | new |
| `3-3-2-2` | rich, first OSV record only | a second block, same layout: sequence 679 (one before the first `3-3-2-1-2`), same start time | start-up leftover; ignore | new |
| `3-4-1-1`, `3-4-1-2` | both* | 1, 1 | — | ? |
| `3-4-1-4` | both* | `Osmo OQ001` | internal model code | ✓ (ExifTool: "model code?") |
| `3-4-1-5` | both* | f32 50 (LRF: 25) | frame rate | new |
| `3-4-2-3` | both | 1 | the only field in the GPS subtree: **no `3-4-2-1` GPSInfo, `3-4-2-2` altitude or `3-4-2-6-1` time** — the camera has no GNSS. Read as `GpsBasic.gps_status` = invalid, the message the Avata 360 fills at `3-4-4` ([avata360-telemetry.md](avata360-telemetry.md) §6); [`scripts/upright.py`](../scripts/upright.py) finds that message by content, so a clip that carries a fix here is used for scale without a code change (not yet seen) | ✓ path, no fix |

\* field 4 exists in both records; the values shown were read from the light record.

## 6. The ~1 kHz orientation stream

This is the data RockSteady/HorizonSteady and rolling-shutter correction need, and what aligning the camera with another IMU runs on. Measured on this clip; [`scripts/osv_imu.py`](../scripts/osv_imu.py) prints most of these figures when run on it.

### Timing

- **Block starts are true sample times.** `3-3-2-1-1` is the time of the block's first sample on the camera clock: every step between block starts is a whole number of IMU periods, to 1.2 µs.
- **In the OSV the blocks tile time.** 3,153 of 3,153 steps lose or repeat no sample, and `3-3-2-1-2` has no gaps. 20 samples per block, 21 in 80 of 3,154.
- **The IMU runs at 1,001.24 Hz**, mean over the clip. (An earlier reading here gave 1,001.6 Hz by dividing 63,160 samples by the first-to-last *frame* span, which leaves out the last block.)
- **The frame clock is an exact grid.** `3-1-2` steps 20,000.587 µs — 49.99853 fps, header `1-11-1` — to ±0.7 µs, +29 ppm from nominal. The container's PTS assume exactly 20,000 µs, which is 35 ms wrong over a 20-minute clip: time frames from `3-1-2`.
- **The IMU's rate is not constant against that clock.** Over 10 s windows it moved smoothly from −102 to +76 ppm, so a single best-fit rate is off by up to 1.24 ms within the minute. Time each sample from its own block start and the local period, as `osv_imu.py` does, never from a fixed rate. The likeliest reading is the IMU's own oscillator warming up against a crystal frame clock; not established.
- **The per-frame quaternion `3-2-9` is sample index 5**, 4.51 ms after the frame timestamp on average, in 2,276 of 2,276 rotating frames — at every exposure from 0.18 to 10 ms. A fixed pick, not mid-exposure: for colouring or sync, interpolate the stream at your own image time instead.
- **`3-3-2-1-4` = 7.50 − 0.497 × exposure in ms** (R² 0.990). It moves by half the exposure, so it is referenced to mid-exposure, presumably for the stabiliser; what instant it counts from is not identified.

### The LRF keeps every other block

*Correction to an earlier reading here, which took the LRF block for a ~500 Hz stream spanning 40 ms.* All 1,577 LRF blocks have an OSV block with the identical start time and sequence number, consecutive LRF blocks are exactly two OSV blocks apart, and all 31,579 LRF samples are bit-identical to the OSV sample at the same index. So the LRF stream is 1,001.24 Hz samples in ~20 ms bursts every 40 ms, with the 20–21 samples of the skipped block missing between bursts. Spread over 40 ms, a block halves every rate and puts a false 1 ms jump at each boundary. Use the OSV for timing work; `osv_imu.py` times LRF samples correctly and reports the gaps.

### The signal

- **Convention:** w, x, y, z; Hamilton; body → world; world z pointing down. Checked against the 50 Hz accelerometer `3-2-10`: the gravity direction this predicts in the body frame agrees with the measured one at a mean cosine of 0.984, where the next-best convention scores 0.867.
- **Fused, but not smoothed away.** [Gyroflow documents](https://docs.gyroflow.xyz/app/getting-started/supported-cameras/dji) that DJI cameras store the final computed orientation rather than raw IMU data, and the Osmo 360 is consistent with that. Body rates differentiated from it are still clean: no repeated samples, and 99.9% of the rate power below 70 Hz on this handheld walk. Raw gyro rates are not needed to align with another IMU.
- **Excitation on this handheld walk:** median 21 °/s, p90 79 °/s, p99 210 °/s.
- **No raw gyro rates** appear anywhere in `djmd`, and the accelerometer is only 50 Hz. The undecoded `dbgi` tracks are the likeliest home for raw IMU data.

## 7. Aligning with another IMU (e.g. a Livox Mid-360)

The camera side is measured above; the other side has not been tried. How it goes, and what it does not give you:

1. **Why it works.** Every point of a rigid body turns at the same angular velocity, so a camera clamped to another IMU measures the same ω in different axes: ω_osmo(t) = R·ω_other(t + δ(t)). The lever arm between them does not enter the time sync at all.
2. **Body rates.** `osv_imu.py` writes them to `imu.csv` (rad/s) from **central** differences on the true sample times; a forward difference sits half a sample, 0.5 ms, late. Low-pass both signals to their shared band before correlating: a Mid-360's IMU samples at 200 Hz, and 99.9% of this stream's rate power is below 70 Hz.
3. **Time offset.** Cross-correlate |ω| first; it needs no knowledge of R. The 3-axis vectors correlate more sharply once R is known: here their autocorrelation is 84 ms wide at half height, and no side lobe from walking rhythm exceeds 0.09. Fit δ in windows (say 30 s) and model **offset plus drift**: the camera clock has no absolute reference and runs 29 ppm off nominal on this clip.
4. **Rotation between the sensors.** Kabsch/SVD on the time-aligned rate vectors, over a capture that turns about all three axes — a wiggle at the start and end. So a mount's rotation need not be trusted, which matters: 1° of it is 17 cm of colour error at 10 m.
5. **Lever arm.** A gyro cannot observe it and a 50 Hz accelerometer is too weak. Take it from the mount's geometry, or from aligning the COLMAP reconstruction to the LiDAR map. An error in it maps 1:1 onto the surface, so it matters far less than rotation.

### What the sync does not give you: when each image was exposed

δ aligns the Osmo's *IMU* clock. Colouring needs the *image* time on that clock, and three parts of it are unknown: which instant `3-1-2` marks, the rolling-shutter readout time (not published), and any lag the fusion filter adds to the quaternions. All are constant per recording mode, so one calibration covers them:

- **In this repo, with no other sensor:** run the fisheye rig SfM (docs/how-it-works.md, "Fisheye rig") densely over a few hundred consecutive frames of fast rotation, then fit a time offset, together with the IMU-to-lens rotation, between the SfM camera rotations and the orientation stream. That also tests whether DJI's per-lens yaw/pitch/roll are expressed in the stream's body frame, which is unverified.
- **With a GNSS receiver:** film a white card lit by an LED on the receiver's PPS output. Each rising edge lands on one row of one frame, giving image time against GPS time, and the readout time from how that row moves across several seconds.
- **Against a LiDAR map:** photo-consistency. A point should get the same colour from every frame that sees it; tune the offset and lens rotation until it does.

**Why milliseconds.** Surface error ≈ range × rotation rate × timing error; at walking pace translation is negligible, 1.4 mm per ms. At 10 m and this clip's rates, 1 ms costs 4 mm at the median and 14 mm at p90, and 5 ms at p90 costs 7 cm. Rolling shutter follows the same formula with the readout time in place of the timing error, which is the other reason to colour from low-rotation frames when you can.

**Not measured yet:** no joint capture with another IMU exists. Everything past this camera's own data is a prediction.

## 8. Extracting it

```bash
python3 scripts/osv_meta.py CAM_20260813121712_0040_D.OSV out/   # works on .LRF too
```

- `calibration.json` — device identity, plus every populated calibration slot: fx, fy, cx, cy, k1–k4, width/height, yaw/pitch/roll, packed quaternion, boundary polyline.
- `telemetry.csv` — one row per record, rich and light alternating: `frame` (`3-1-1`), `t_sec` (`3-1-2` / 1e6), `iso`, `color_temp`, `qw,qx,qy,qz` (`3-2-9` sub-fields 1–4 in storage order, which is w, x, y, z — §6), `ax,ay,az` (`3-2-10-2/3/4`), `imu_samples` (block size). Light rows leave the quaternion and accelerometer blank.
- **Not exported yet:** the `3-2-15` exposure block.

```bash
python3 scripts/osv_imu.py CAM_20260813121712_0040_D.OSV out/   # needs numpy; works on .LRF too
```

- `imu.csv` — one row per orientation sample: `t_us` (camera clock: block start plus the local IMU period), `qw,qx,qy,qz` as stored, `wx,wy,wz` body rates in rad/s from central differences.
- `frames.csv` — one row per rich record: `record`, `frame` (`3-1-1`), `t_us` (`3-1-2`), `exposure_us` (`3-2-4-1`), `iso`, `block_t_us`, `block_seq`, `samples`, `timing` (`3-3-2-1-4`).
- It also prints the §6 checks for the clip: contiguity or gaps, rate stability, the frame grid, the `3-2-9` sample index, the convention against gravity, rotation rates and signal bandwidth.

ExifTool extracts the ✓ rows: `exiftool -ee -G3 file.OSV`. Running it on the `camd` sidecar (the box minus its 8-byte header) avoids scanning the video.

## 9. Open questions

- The `dbgi` tracks (~12.5 MB per track per minute): raw IMU? ISP state?
- The `3-2-15` exposure block, `3-2-16-1` (temperature?), `3-2-17-1`, header `1-3-1` (four floats) and `1-8-1` (1061.58).
- Calibration sub-fields 15, 20/27, 24, 25; what slots 11–24 are for; the quaternion component order.
- Image exposure time on the IMU clock: which instant `3-1-2` marks, the rolling-shutter readout time, any fusion-filter lag (§7).
- The zero point of `3-3-2-1-4`, and why the IMU's rate wanders ~180 ppm within a minute (oscillator warm-up?).
- The frame of the calibration quaternion (sub-field 21), and whether the per-lens yaw/pitch/roll are expressed in the orientation stream's body frame. `scripts/upright.py` solves lens 0's rotation from the body frame on every levelled job (`lens_from_body` in `sfm/alignment.json`), which is the data to compare them against.
- Why `3-1-1` steps by 2 in the OSV but 1 in the LRF.
- Whether the Avata 360 (`DJI_*.OSV`) uses the same schema.
