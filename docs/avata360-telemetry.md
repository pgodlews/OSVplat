# DJI Avata 360 telemetry format (`.OSV` / `.LRF` / `.SRT`)

Status: **decoded 2026-09-11** from one clip, `DJI_20260621181643_0010_D`: an Avata 360 flight, camera firmware 01.00.02.00, 71 s at 50 fps, 3000×3000 per lens. Every value was read out of that file. No Avata 360 product `.proto` is published, so field names come from telemetry-parser's shared [`dvtm_library.proto`](https://github.com/AdrianEddy/telemetry-parser/tree/master/src/dji) wherever a message's type and values match. Field paths use ExifTool's notation (`3-4-4-2` = top-level field 3 → 4 → 4 → 2). Status column: **named** comes from the library type; **measured** was checked against independent data; **?** is present but not understood. This file deliberately carries no coordinates and no serial numbers.

The Osmo 360's format is in [osmo360-telemetry.md](osmo360-telemetry.md). This page is written as the differences from it.

---

## 1. The three files

| File | Contents |
|---|---|
| `.OSV` | 2 × `hvc1` **3000×3000, 10-bit** (`yuv420p10le`), 50 fps, one per lens, **unstitched**. On this aircraft lens 0 looks up and lens 1 down. 2 × `djmd` ("CAM meta", 8,172,922 B each), 2 × `dbgi` (224 MB each, ~63 kB/frame, not decoded), 1 × **`camd` track** (2 samples, 16.4 MB), a 960×540 MJPEG thumbnail. No audio. |
| `.LRF` | 1920×960 HEVC at 25 fps: **both fisheye circles side by side, not stitched**. 2 × `djmd` and the `camd` track. |
| `.SRT` | One entry per frame (3,550): frame counter, local time, ISO, shutter, f-number, EV, colour temperature, colour mode, focal length, digital zoom, **latitude, longitude, relative and absolute altitude**, gimbal yaw/pitch/roll, EIS state, and `pp_*` stabiliser fields. |

## 2. Where the protobuf lives, and what differs from the Osmo 360

- **`camd` is a track, not a trailing box.** Its two samples (10,486,338 + 5,917,799 B) concatenate into the same nested MP4 the Osmo stores as a box: `ftyp`, `free`, `mdat` holding the protobuf stream, `moov`. `scripts/osv_meta.py` finds either.
- **Schema `dvtm_AVATA360.proto`**, library 02.01.20, product proto 02.00.11. ExifTool 13.55 reports "Unknown protocol" and extracts nothing from it.
- **Both `djmd` tracks are full records.** The first sample of each is header (field 1) + config (field 2) + record (field 3); later samples hold one record. Unlike the Osmo's rich/light pair, both tracks here carry the orientation stream, and their timestamps differ by at most 17 µs.

## 3. Header (field 1, `ClipMeta`)

The numbering matches the Osmo up to field 7, then sits **two lower**. Types are inferred from the Osmo's `ClipMeta` order and each value's match.

| Path | Value | Meaning | Osmo 360 path |
|---|---|---|---|
| `1-1-1/2/3/6/10` | `dvtm_AVATA360.proto`, `02.01.20`, `02.00.11`, `01.00.02.00`, `DJI Avata360` | `ClipMetaHeader`: proto name, library and product proto versions, firmware, product name (1-1-5 is the serial) | same |
| `1-3-1` | 4 × 0.0 | `LensDistortionCoefficients`, unset | set |
| `1-4-1` | 21268800 | `SensorFrameReadOutTime` | `1-4-1` |
| `1-7-1` | 1 | ? | — |
| `1-8-1` | **4000** | `IMUSamplingRate`; the stream measures 4,001 Hz (§7) | `1-10-1` = 1000 |
| `1-9-1` | f32 49.993 | `SensorFrameRate` | `1-11-1` |
| `1-10-1` | −663 (int32) | `ITDValue` | `1-12-1` = −168 |
| `1-11-1` | 4220 | `LROValue` | `1-13-1` |
| `1-12-1/2` | 3840, 3840 | `SensorRes`: the **calibrated** frame, not the 3000² stream | `1-14-1/2` |

## 4. Config (field 2, `StreamMeta`) and the lens calibration

| Path | Value | Meaning | Status |
|---|---|---|---|
| `2-1-3` | `video` | `StreamMetaHeader.stream_name` | named |
| `2-2-1` | device 1, physical, camera body, `DJI FCA188`, 50 | `MetaHeaderOfDevice` of the camera | named |
| `2-2-2-1/2` | ` AC Ver.03`, `10.00.06.90` | aircraft version strings | ? |
| `2-2-3-1` | a second serial | aircraft serial | ? |
| `2-3-1…8` | 3000, 3000, 50, true, 10, 4, –, 1 | `VideoStreamMeta`: width, height, fps, bit depth valid, **bit depth 10**, YUV420, codec H.265 | named, matches ffprobe |
| `2-5-n` | 24 slots | **`PanoDewarpParams`** (the Osmo's is `2-6`) | named |

### Calibration slots (`2-5-n`)

Slots **3 and 4** carry the calibration: `native_refine_far_slave` drives stream 0 and `native_refine_far_master` stream 1. That is the far-focus pair, which suits a drone; the handheld Osmo 360 uses `native_refine`, slots 1–2. On the Avata, slots 1 and 2 hold only a temperature and the rest are zero.

| `DewarpParams` field | Slot 3 (stream 0) | Slot 4 (stream 1) | Note |
|---|---|---|---|
| `fx`, `fy` | 1037.00, 1037.23 | 1040.72, 1041.04 | px **in the 3840² frame**; ×0.78125 → 810.16 / 813.06 in the stream |
| `cx`, `cy` | 1913.53, 1922.41 | 1926.87, 1923.27 | → 1494.94, 1501.88 / 1505.37, 1502.56 |
| `k1`–`k4` | 0.07841, −0.02802, 0.01796, −0.00820 | 0.07627, −0.02748, 0.01932, −0.00898 | Kannala–Brandt, angular: do not scale |
| `k5` (field 15) | 0.00104743 | 0.00116565 | the fifth radial term; COLMAP has no k5, so `osmo_fisheye.kb4_fit` refits k1–k4 |
| `width`, `height` | 3840, 3840 | 3840, 3840 | the calibrated frame |
| `yaw`, `pitch`, `roll` | −179.956, 91.190, 0.106 | −0.114, 91.164, 0.474 | degrees; opposite-facing lenses |
| `lens_model` (24) | 8 | 8 | same as the Osmo 360 |
| `temperature` (25) | 38 | 35 | °C (the Osmo writes −1000) |
| `cam_imu_extri_q` (26) | 0.0049, −0.0013, −0.0015, −0.99999 | 0.99997, −0.0047, 0.0020, 0.0060 | camera↔IMU extrinsic, w x y z |
| `cam_extri_q` (28) | absent | absent | the Osmo stores it; here the rig rotation starts from yaw/pitch/roll and bundle adjustment refines it |
| `occlusion_pt_x/y` (22/23) | absent | absent | |
| fields 29–33 | 1, 1, 6.65e-5, 49, 50 | same | `cam_imu_cali_enable`, `temp_compen_enable`, `temp_compen_k`, `gimbal_yaw_h1/h2` |

**Scaling.** The calibration is stated for a 3840² frame and the stream is 3000². Each lens's whole image circle sits inside the 3000² frame (radius ~1440–1450 px), which a centre crop of a calibration at this focal length could not produce. So the stream is treated as a uniform resize: pixel quantities scale by 0.78125, and the angular distortion coefficients do not. `osv_meta.py` applies this and records `calib_width` and `scale`. A rig SfM on this clip is the test that confirms it.

## 5. Per-frame records (field 3)

| Path | Value / range over the flight | Meaning | Status |
|---|---|---|---|
| `3-1-1` | +1 per frame | `FrameMetaHeader.frame_seq_num` (the Osmo steps +2) | named |
| `3-1-2` | step 20,000 µs | `frame_timestamp`, µs since power-up | named |
| `3-2-1` | device, `DJI FCA188`, 50, µs | camera `MetaHeaderOfDevice` | named |
| `3-2-3-1` | f32 | `ISO`; equals the SRT's | named, measured |
| `3-2-4-1` | varint pair (1, N) | `ExposureTime`, 1/N s | named |
| `3-2-5-1` | 1.0 | `DigitalZoomRatio` | named |
| `3-2-6-1` | K | `WhiteBalanceCCT`; equals the SRT's `ct` | named, measured |
| `3-2-9-1` | varint (5956 at frame 1) | not the Osmo's quaternion | ? |
| `3-2-11-1` | f32 m | `AbsoluteAltitude`; equals `3-4-4-2` to 1 mm | named, measured |
| `3-2-12-1` | f32 26 | ? | ? |
| `3-2-10`, `-13`, `-14`, `-16`, `-18`, `-19` | bytes, varint pairs, 20-byte blobs, 4 floats, an exposure block | not decoded | ? |
| `3-3-2-1-1` | µs | `DeviceAttitude.timestamp` of the block's first sample | named |
| `3-3-2-1-2` | +1 per frame | `DeviceAttitude.vsync` | named |
| `3-3-2-1-3` | 79–81 quaternions per frame | `DeviceAttitude.attitude`: **fused** orientation, w x y z, 4,001 Hz | named, measured |
| `3-4-1` | device 1, physical, **drone** (sub-type 2), 10, µs | the aircraft's `MetaHeaderOfDevice` | named |
| `3-4-2-1…3` | f32 m/s, 7.1–17.9 / −16.6–8.8 / −2.0–6.6 in magnitude | **`Velocity`: North, East, Down**. Correlation with GPS-derived velocity 0.997 / 0.999 / −0.995; median speed 18.0 m/s, as GPS | named, measured |
| `3-4-3-1…3` | int, ÷10: −48…−24, −40…+31, ±180 | aircraft pitch, roll, yaw in 0.1°. The yaw tracks the SRT's gimbal yaw at a near-constant ~25° offset | ?, probable |
| `3-4-4-1-2/3` | f64 degrees | **`GpsBasic.gps_coordinates`: latitude, longitude** (WGS-84; the unit field is left unset) | named, measured |
| `3-4-4-2` | int32 mm | `gps_altitude_mm` | named, measured |
| `3-4-4-4` | 1 | `gps_altitude_type` = GPS–barometer fusion (ellipsoidal); `gps_status` absent = normal | named |
| `3-4-5-1/2` | f32 mm, true | **`RelativeAltitude`**, valid flag | named, measured |
| `3-4-14-1…` | 127, 3 floats (m), 3 floats, … | matches `FlightPoseNavi` (imu_flag, position, velocity, quaternion) | ?, structure only |

## 6. GPS

**Yes, in two places, and they agree.**

- **Embedded, every record:** `3-4-4` (`GpsBasic`) and `3-4-5` (`RelativeAltitude`). Checked frame by frame against the SRT over all 3,550 frames: latitude/longitude within 5×10⁻⁷°, absolute altitude within 1 mm, relative altitude exact.
- **SRT sidecar,** in plain text.
- **Update rate:** 711 distinct fixes over 71 s, about 10 Hz, each held for five frames. Altitude is GPS–barometer fusion.
- **Extraction:** ExifTool cannot read the embedded copy. `scripts/osv_meta.py` writes `lat`, `lon`, `abs_alt_m` and `rel_alt_m` per record to `telemetry.csv`.
- **The Osmo 360 has no fix:** its `GpsBasic` carries only `gps_status` = invalid.

## 7. The orientation stream

- **4,001 Hz,** 79–81 samples per 20 ms block; block starts step 20,000 µs; header `1-8-1` = 4000.
- **Stored twice:** identical in both `djmd` tracks, sample for sample, at all 3,550 frames. Count it once.
- **`scripts/osv_imu.py`** (written for the Osmo 360) reads the nominal rate from `1-10-1`. On the Avata that field is `ITDValue`; the rate is `1-8-1`.

## 8. Using it for the fisheye rig

- **Relative rotation only:** the rig SfM needs just the lenses' relative rotation, so the up/down mount does not matter to it.
- **Seeds:** intrinsics scaled ×0.78125; k5 folded into a k1–k4 refit; features inside an 88° circle, which is 1,353 and 1,358 px here.
- **Masking:** off for drone footage — nobody rides along.

## 9. Open questions

- `3-2-9`, `-10`, `-12`, `-13`, `-16`, `-18`, `-19`; whether `3-4-3` is the aircraft's attitude or the gimbal's; `3-4-14`.
- The `dbgi` tracks.
- Slots 1–2 holding only a temperature.
- Confirming the resize-not-crop scaling with a rig SfM on this clip.
