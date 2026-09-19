"""Avata 360 velocity-based candidate spacing (stdlib only, also used by the queue).

Distances are integrals of aircraft NED speed, not GPS chord lengths or
double-integrated acceleration. Positions are relative dead reckoning, intended
for selection diagnostics, never authoritative camera poses.
"""
import math
import statistics
from functools import lru_cache
from pathlib import Path

try:
    from osv_meta import camd_payload, f32, find_camd, get, records
except ImportError:
    from scripts.osv_meta import camd_payload, f32, find_camd, get, records

VIDEO_RATES = (24000 / 1001, 24., 25., 30000 / 1001, 30., 48000 / 1001,
               48., 50., 60000 / 1001, 60., 100., 120000 / 1001, 120.)
SPEED_FLOOR_MPS = 0.15  # suppress low-speed hover noise; also ignores very slow travel
DUPLICATE_TOL_MPS = 0.25  # per axis, between the two lens tracks' copies of one sample


def decode_records(raw):
    """Read one velocity sample per video frame; refuse gaps/unknown schemas."""
    header = None
    samples = []
    for fn, body in records(raw):
        if fn == 1 and header is None:
            header = body
            ident = b" ".join(v for v in (get(header, 1, 1), get(header, 1, 10))
                              if isinstance(v, bytes))
            if b"avata" not in ident.lower():
                raise ValueError("Distance selection supports raw Avata 360 recordings only")
        elif fn == 3:
            seq, ts = get(body, 1, 1), get(body, 1, 2)
            # Protobuf omits zero-valued scalars, including frame_seq_num=0.
            seq = 0 if seq is None else seq
            if not isinstance(seq, int) or not isinstance(ts, int):
                raise ValueError("Avata distance selection: missing frame timestamps")
            velocity_msg = get(body, 4, 2)
            if not isinstance(velocity_msg, bytes):
                raise ValueError("Avata distance selection requires complete NED velocity telemetry")
            values = [get(velocity_msg, axis) for axis in (1, 2, 3)]
            if any(v is not None and (not isinstance(v, bytes) or len(v) != 4) for v in values):
                raise ValueError("Avata distance selection: malformed velocity telemetry")
            velocity = tuple(0. if v is None else f32(v) for v in values)
            if not all(math.isfinite(v) for v in velocity) or math.hypot(*velocity) > 100:
                raise ValueError("Avata distance selection: invalid velocity telemetry")
            # Both lens metadata tracks carry the same aircraft velocity. Their
            # camera timestamps differ by ~17 us; use the first track's clock.
            # The 0.1 m/s-quantised velocity can update between the two tracks
            # (1 of 4251 pairs in DJI_..._0141), so average small disagreements.
            if samples and seq == samples[-1][0]:
                prev = samples[-1][2]
                if (abs(ts - samples[-1][1]) > 1000
                        or max(abs(a - b) for a, b in zip(velocity, prev)) > DUPLICATE_TOL_MPS):
                    raise ValueError("Avata distance selection: conflicting duplicate telemetry")
                samples[-1] = (seq, samples[-1][1], tuple((a + b) / 2 for a, b in zip(velocity, prev)))
                continue
            samples.append((seq, ts, velocity))
    if header is None:
        raise ValueError("Distance selection supports raw Avata 360 recordings only")
    if len(samples) < 2 or samples[0][0] != 0:
        raise ValueError("Avata distance selection: insufficient velocity samples")
    steps = [(b[1] - a[1]) / 1e6 for a, b in zip(samples, samples[1:])]
    if any(b[0] != a[0] + 1 for a, b in zip(samples, samples[1:])) or min(steps) <= 0:
        raise ValueError("Avata distance selection: missing or duplicated telemetry frames")
    measured = 1 / statistics.median(steps)
    fps = min(VIDEO_RATES, key=lambda r: abs(r - measured))
    if abs(measured / fps - 1) > .01 or any(abs(dt * fps - 1) > .1 for dt in steps):
        raise ValueError("Avata distance selection: irregular telemetry timing")
    return {"fps": fps, "t": [(s[1] - samples[0][1]) / 1e6 for s in samples],
            "velocity": [s[2] for s in samples]}


@lru_cache(maxsize=4)
def _load(path, size, mtime_ns):
    try:
        return decode_records(camd_payload(find_camd(path)))
    except (SystemExit, IndexError, TypeError) as exc:
        raise ValueError("Cannot read Avata velocity telemetry from this recording") from exc


def load(path):
    p = Path(path).resolve()
    st = p.stat()
    return _load(str(p), st.st_size, st.st_mtime_ns)


def integrate(t, velocity, speed_floor=SPEED_FLOOR_MPS):
    """Trapezoidal integration at the original telemetry rate, before subsampling."""
    vel = [v if math.hypot(*v) >= speed_floor else (0., 0., 0.) for v in velocity]
    speed = [math.hypot(*v) for v in vel]
    distance, position = [0.], [(0., 0., 0.)]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        distance.append(distance[-1] + .5 * (speed[i - 1] + speed[i]) * dt)
        position.append(tuple(position[-1][a] + .5 * (vel[i - 1][a] + vel[i][a]) * dt
                              for a in range(3)))
    return distance, position, speed


def candidates(motion, n, fps, start=0.):
    """Match ffmpeg fps-filter candidates to source video frames, including trims."""
    vf = motion["fps"]
    if n < 1 or fps <= 0 or fps > vf or start < 0:
        raise ValueError("Distance selection needs candidates at or below the source frame rate")
    indices = [math.ceil(vf * (start + (k + .5) / fps) - 1e-9) - 1 for k in range(n)]
    if indices[-1] >= len(motion["t"]) or indices[0] < 0:
        raise ValueError("Candidate frames extend beyond Avata telemetry; check trim and candidate fps")
    distance, position, speed = integrate(motion["t"], motion["velocity"])
    origin = position[indices[0]]
    return {"video_frame": indices, "t_sec": [i / vf for i in indices],
            "distance_m": [distance[i] - distance[indices[0]] for i in indices],
            "position_ned_m": [tuple(position[i][a] - origin[a] for a in range(3)) for i in indices],
            "speed_mps": [speed[i] for i in indices]}


def groups(path, spacing_m, max_gap_s):
    """Distance bins split into spans <= max_gap/2, so adjacent picks stay connected.

    Later sharpness selection prefers candidates within 20% of the spacing of
    the one nearest each bin centre.
    Splitting also retains temporal coverage while hovering. This is a time
    guard, not a guarantee of visual overlap. Empty bins are never fabricated.
    """
    if not math.isfinite(spacing_m) or spacing_m <= 0 or not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("Distance spacing and maximum time gap must be positive and finite")
    times = path["t_sec"]
    step = max((b - a for a, b in zip(times, times[1:])), default=0.)
    span = (max_gap_s - step) / 2
    if span <= 0:
        raise ValueError("Candidate fps is too low for the maximum time gap")
    out = []
    prev_bin = None
    for i, (d, t) in enumerate(zip(path["distance_m"], path["t_sec"])):
        bin_id = math.floor(d / spacing_m + 1e-9)
        if bin_id != prev_bin or t - path["t_sec"][out[-1][0]] > span:
            out.append([])
        out[-1].append(i)
        prev_bin = bin_id
    return out


def plan(src, n, fps, start, spacing_m, max_gap_s):
    path = candidates(load(src), n, fps, start)
    if fps * max_gap_s < 2:
        raise ValueError("Maximum time gap must span at least two candidate intervals; increase candidate fps")
    return path, groups(path, spacing_m, max_gap_s)
