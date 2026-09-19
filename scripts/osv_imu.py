#!/usr/bin/env python3
"""Decode the Osmo 360's ~1 kHz orientation stream with per-sample timestamps.

osv_meta.py counts the high-rate block in each rich record; this unpacks it.
Layout, measured on CAM_20260813121712_0040_D, firmware 10.00.25.29
(docs/osmo360-telemetry.md, sections 5-7):

  3-3-2-1-1  block start, µs on the camera clock (the clock of 3-1-2)
  3-3-2-1-2  block sequence number, +1 per instant
  3-3-2-1-3  orientation samples, 4 x f32 (w, x, y, z): Hamilton, body -> world,
             world z pointing down
  3-3-2-1-4  f32 = 7.50 - 0.497 x exposure_ms: an exposure-centred timing term
             whose reference instant is not identified

Block starts are the true times of each block's first sample, and every step
between them is a whole number of IMU periods. In the .OSV the blocks tile time.
The IMU's rate is not constant against the camera clock (~180 ppm of wander in
that 63 s clip), so each sample is timed from its own block's start and the
local IMU period, never from a fixed rate.

usage: osv_imu.py <file.OSV|.LRF> <out_dir>
  imu.csv     t_us, qw, qx, qy, qz, wx, wy, wz   (body rates, rad/s)
  frames.csv  record, frame (3-1-1), t_us (3-1-2), exposure_us, iso,
              block_t_us, block_seq, samples, timing (3-3-2-1-4)
and prints the checks that decide whether a clip's stream can be trusted for
aligning with another IMU.
"""
import csv
import os
import sys

import numpy as np

try:
    from osv_meta import camd_payload, f32, fields, find_camd, get, msg4f, records, rv
except ImportError:
    from scripts.osv_meta import camd_payload, f32, fields, find_camd, get, msg4f, records, rv


def children(buf, want):
    return [v for fn, wt, v in fields(buf) if fn == want and wt == 2]


def block(record):
    """The 3-3-2-1 block of a rich record, or None for a light record.

    A clip's first record can also carry a 3-3-2-2 block (seen once: the previous
    sequence number with the same start time); it is ignored.
    """
    for c3 in children(record, 3):
        for c2 in children(c3, 2):
            for c1 in children(c2, 1):
                b = dict(t=None, seq=None, timing=None, q=[])
                for fn, wt, v in fields(c1):
                    if fn == 1 and wt == 0:
                        b["t"] = v
                    elif fn == 2 and wt == 0:
                        b["seq"] = v
                    elif fn == 3 and wt == 2 and len(v) == 20:
                        b["q"].append(msg4f(v))
                    elif fn == 4 and wt == 5:
                        b["timing"] = f32(v)
                if b["q"] and b["t"] is not None:
                    return b
    return None


def exposure_us(record):
    """3-2-4-1 is two bare varints (1, N) for a 1/N s shutter, not protobuf fields."""
    raw = get(record, 2, 4, 1)
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        return np.nan
    num, i = rv(raw, 0)
    den, _ = rv(raw, i)
    return 1e6 * num / den if den else np.nan


def iso_of(record):
    raw = get(record, 2, 3, 1)
    return f32(raw) if isinstance(raw, (bytes, bytearray)) and len(raw) == 4 else np.nan


def sample_times(starts, counts, nominal_hz, half_window=12):
    """Time every sample from its block start and the local IMU period.

    Each step between block starts is a whole number of IMU periods. Rounding a
    step against the nominal rate gives that count unambiguously at these block
    lengths; the step over its count is a period, and a running median follows
    the IMU's slow rate wander. Returns (sample times µs, period per block µs,
    IMU periods between consecutive block starts).
    """
    steps = np.diff(starts)
    per = steps / np.rint(steps * nominal_hz / 1e6)
    per = np.r_[per, per[-1]]
    period = np.array([np.median(per[max(0, k - half_window):k + half_window + 1])
                       for k in range(len(starts))])
    periods = np.rint(steps / period[:-1]).astype(int)
    t = np.concatenate([s + p * np.arange(n) for s, p, n in zip(starts, period, counts)])
    return t, period, periods


def body_rates(t_us, q):
    """Body-frame angular velocity, rad/s, at each sample from its neighbours.

    Central differences keep the estimate on the sample's own time; a forward
    difference sits half a sample (~0.5 ms) late, which matters when the point
    of the exercise is timing.
    """
    q = q.copy()
    for i in range(1, len(q)):
        if q[i - 1] @ q[i] < 0:
            q[i] = -q[i]
    n = len(q)
    prev = np.r_[0, np.arange(n - 1)]
    nxt = np.r_[np.arange(1, n), n - 1]
    a, b = q[prev], q[nxt]
    dt = (t_us[nxt] - t_us[prev]) / 1e6
    aw, ax, ay, az = a[:, 0], -a[:, 1], -a[:, 2], -a[:, 3]  # conjugate of the earlier sample
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    rw = aw * bw - ax * bx - ay * by - az * bz
    v = np.stack([aw * bx + ax * bw + ay * bz - az * by,
                  aw * by - ax * bz + ay * bw + az * bx,
                  aw * bz + ax * by - ay * bx + az * bw], axis=1)
    v *= np.where(rw < 0, -1.0, 1.0)[:, None]
    s = np.linalg.norm(v, axis=1)
    angle = 2 * np.arctan2(s, np.abs(rw))
    scale = np.divide(angle, s, out=np.full_like(s, 2.0), where=s > 0)
    return v * (scale / dt)[:, None]


def moving_mean(x, k):
    c = np.cumsum(np.r_[0.0, x])
    i = np.arange(len(x))
    lo = np.clip(i - k // 2, 0, len(x))
    hi = np.clip(i + k - k // 2, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def report(path, recs, frame_t, starts, counts, period, periods, q, w, exposure, timing):
    G = np.concatenate([[0], np.cumsum(counts)[:-1]])
    P = np.r_[0, np.cumsum(periods)]  # IMU periods elapsed since the first block start
    mean_p = (starts[-1] - starts[0]) / P[-1]
    print(f"{os.path.basename(path)}: {len(starts)} blocks, {len(q)} samples, IMU at "
          f"{1e6 / mean_p:.3f} Hz (mean over the clip, from block starts)")

    gaps = periods - counts[:-1]  # IMU periods skipped between one block's last sample and the next block
    resid = np.diff(starts) - periods * period[:-1]
    if gaps.any():
        vals, cnts = np.unique(gaps, return_counts=True)
        tiling = "samples missing between blocks: " + ", ".join(f"{v} x{c}" for v, c in zip(vals, cnts))
    else:
        tiling = f"blocks tile time, no sample missing or repeated in {len(gaps)} steps"
    print(f"  contiguity: {tiling} (residual max {np.abs(resid).max():.1f} µs)")

    slope, icpt = np.polyfit(P, starts, 1)
    fixed = np.abs(starts - (slope * P + icpt)).max()
    win = max(2, int(round(10e6 / np.median(np.diff(frame_t)))))
    ppm = [((starts[s + win] - starts[s]) / (P[s + win] - P[s]) / mean_p - 1) * 1e6
           for s in range(0, len(starts) - win, win)]
    wander = (f"{min(ppm):+.0f} to {max(ppm):+.0f} ppm over 10 s windows" if len(ppm) > 1
              else "clip too short for 10 s windows")
    print(f"  rate stability: {wander}; a best-fit fixed rate is off by up to {fixed / 1e3:.2f} ms")

    fp = (frame_t[-1] - frame_t[0]) / (len(frame_t) - 1)
    nominal = 1e6 / round(1e6 / fp)
    grid = frame_t - (frame_t[0] + fp * np.arange(len(frame_t)))
    print(f"  frame clock (3-1-2): {fp:.3f} µs = {1e6 / fp:.5f} fps, on that grid to ±{np.abs(grid).max():.1f} µs; "
          f"a nominal {nominal:.0f} µs period drifts {abs(fp / nominal - 1) * 1200e3:.1f} ms per 20 min")
    lead = starts - frame_t
    print(f"  block start minus frame time: {lead.min() / 1e3:+.3f} to {lead.max() / 1e3:+.3f} ms")

    speed = np.degrees(np.linalg.norm(w, axis=1))
    raw_pf = [get(r, 2, 9) for r in recs]
    idx = []
    for k, raw in enumerate(raw_pf):
        if not isinstance(raw, (bytes, bytearray)) or len(raw) != 20:
            continue
        a0, a1 = int(G[k]), int(G[k] + counts[k])
        if speed[a0:a1].mean() < 8:
            continue
        pf = np.array(msg4f(raw), dtype=float)
        seg = q[a0:a1]
        d = np.minimum(np.linalg.norm(seg - pf, axis=1), np.linalg.norm(seg + pf, axis=1))
        if d.min() < 1e-6:
            idx.append(int(np.argmin(d)))
    if idx:
        vals, cnts = np.unique(idx, return_counts=True)
        top = int(vals[np.argmax(cnts)])
        print(f"  per-frame quaternion 3-2-9: bit-identical to sample {top} in {cnts.max()} of {len(idx)} rotating "
              f"frames, {(lead.mean() + top * mean_p) / 1e3:.2f} ms after the frame time on average")

    pairs = []
    for r, raw in zip(recs, raw_pf):
        acc = get(r, 2, 10)
        if isinstance(raw, (bytes, bytearray)) and len(raw) == 20 and isinstance(acc, (bytes, bytearray)):
            av = {fn: f32(v) for fn, wt, v in fields(acc) if wt == 5}
            if all(k in av for k in (2, 3, 4)):
                pairs.append((msg4f(raw), (av[2], av[3], av[4])))
    if pairs:
        qa = np.array([p[0] for p in pairs], dtype=float)
        acc = np.array([p[1] for p in pairs], dtype=float)
        acc /= np.linalg.norm(acc, axis=1, keepdims=True)
        qw, qx, qy, qz = qa.T
        # world up is -z; the accelerometer sees it in the body frame as R(q)^T (0, 0, -1)
        up = -np.stack([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)], axis=1)
        cos = float(np.mean(np.sum(up * acc, axis=1)))
        flag = "" if cos > 0.95 else "  <-- the convention does NOT hold for this clip"
        print(f"  convention (w, x, y, z; body -> world; world z down): gravity agreement {cos:.3f}{flag}")

    print(f"  rotation: median {np.median(speed):.1f}, p90 {np.percentile(speed, 90):.1f}, "
          f"p99 {np.percentile(speed, 99):.0f} deg/s")
    if gaps.any():
        print("  bandwidth and correlation width skipped: the stream has gaps, so it is not evenly sampled")
    else:
        dt_s = mean_p / 1e6
        x = w - w.mean(axis=0)
        han = np.hanning(len(x))
        power = sum(np.abs(np.fft.rfft(x[:, c] * han)) ** 2 for c in range(3))
        freq = np.fft.rfftfreq(len(x), d=dt_s)
        cum = np.cumsum(power) / power.sum()
        k3 = max(3, int(round(3 / dt_s)))
        acf = np.zeros(len(x))
        for c in range(3):
            xs = w[:, c] - moving_mean(w[:, c], k3)
            xs -= xs.mean()
            acf += np.fft.ifft(np.abs(np.fft.fft(xs, n=2 * len(xs))) ** 2).real[: len(xs)]
        acf /= acf[0]
        lo, hi = int(0.2 / dt_s), int(2.0 / dt_s)
        side = f", largest side lobe from 0.2 to 2 s {acf[lo:hi].max():.2f}" if hi < len(acf) else ""
        print(f"  signal: 99.9% of rate power below {freq[min(int(np.searchsorted(cum, 0.999)), len(freq) - 1)]:.0f} Hz; "
              f"3-axis rate autocorrelation half-width {np.argmax(acf < 0.5) * dt_s * 1e3:.0f} ms{side}")

    m = ~np.isnan(exposure) & ~np.isnan(timing)
    if m.sum() > 10 and np.ptp(exposure[m]) > 0:
        slope_t, icpt_t = np.polyfit(exposure[m] / 1e3, timing[m], 1)
        res = timing[m] - (icpt_t + slope_t * exposure[m] / 1e3)
        print(f"  3-3-2-1-4 = {icpt_t:.3f} {slope_t:+.3f} x exposure_ms (R^2 {1 - res.var() / timing[m].var():.3f}); "
              f"exposure {np.nanmin(exposure) / 1e3:.2f}-{np.nanmax(exposure) / 1e3:.2f} ms")


def exposure_rates(t_us, w, frame_t_us, exposure_us, min_span_us=3000.0):
    """Mean body rate |omega|, rad/s, over each frame's exposure.

    The window is [frame time, frame time + exposure], widened about its middle
    to min_span_us so a 1/5000 s frame still averages a few samples. Which
    instant 3-1-2 marks is not established (docs/osmo360-telemetry.md §7), and
    for this it hardly matters: body rates change over tens to hundreds of ms
    (rate autocorrelation half-width 84 ms on a handheld walk, 250 ms indoors),
    while an exposure lasts 10 ms at most.
    """
    speed = np.linalg.norm(w, axis=1)
    cum = np.r_[0.0, np.cumsum(speed)]
    exposure_us = np.asarray(exposure_us, dtype=float)
    mid = np.asarray(frame_t_us, dtype=float) + exposure_us / 2
    half = np.maximum(exposure_us, min_span_us) / 2
    lo = np.searchsorted(t_us, mid - half)
    hi = np.clip(np.maximum(np.searchsorted(t_us, mid + half), lo + 1), 1, len(speed))
    lo = np.minimum(lo, hi - 1)
    return (cum[hi] - cum[lo]) / (hi - lo)


def decode(path):
    """The timed orientation stream and one row per rich record, in memory.

    Exactly what main() writes to imu.csv and frames.csv, for callers that only
    want the numbers (80_fisheye_frames.py ranks candidates by body rate).
    """
    rich, nominal_hz = [], None
    for fn, rec in records(camd_payload(find_camd(path))):
        if fn == 1 and nominal_hz is None:
            # Osmo 360 stores nominal IMU rate at 1-10-1 (1000 Hz).
            # Avata 360 stores it at 1-8-1 (4000 Hz), while 1-10-1 is ITDValue (-663).
            r8 = get(rec, 8, 1)
            r10 = get(rec, 10, 1)
            if isinstance(r8, int) and 500 <= r8 <= 10000:
                nominal_hz = float(r8)
            elif isinstance(r10, int) and 500 <= r10 <= 10000:
                nominal_hz = float(r10)
        elif fn == 3:
            b = block(rec)
            if b:
                # Deduplicate: on Avata 360 both djmd tracks carry identical orientation blocks
                if not rich or abs(b["t"] - rich[-1][1]["t"]) > 1000:
                    rich.append((rec, b))
    nominal_hz = nominal_hz or 1000.0
    if len(rich) < 3:
        sys.exit(f"{path}: fewer than 3 records carry an orientation block")
    recs = [r for r, _ in rich]
    blks = [b for _, b in rich]
    frame_t = np.array([get(r, 1, 2) for r in recs], dtype=float)
    starts = np.array([b["t"] for b in blks], dtype=float)
    counts = np.array([len(b["q"]) for b in blks])
    q = np.array([s for b in blks for s in b["q"]], dtype=float)
    t, period, periods = sample_times(starts, counts, nominal_hz)
    w = body_rates(t, q)
    exposure = np.array([exposure_us(r) for r in recs])
    iso = np.array([iso_of(r) for r in recs])
    timing = np.array([np.nan if b["timing"] is None else b["timing"] for b in blks])
    return dict(recs=recs, blocks=blks, frame_t=frame_t, starts=starts, counts=counts,
                t=t, q=q, w=w, period=period, periods=periods,
                exposure=exposure, iso=iso, timing=timing)


def main(path, out):
    os.makedirs(out, exist_ok=True)
    s = decode(path)
    recs, blks, frame_t, t, q, w = s["recs"], s["blocks"], s["frame_t"], s["t"], s["q"], s["w"]
    exposure, iso = s["exposure"], s["iso"]

    with open(os.path.join(out, "imu.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_us", "qw", "qx", "qy", "qz", "wx", "wy", "wz"])
        for ti, qi, wi in zip(t, q, w):
            wr.writerow([f"{ti:.1f}", *(f"{v:.7f}" for v in qi), *(f"{v:.6f}" for v in wi)])
    with open(os.path.join(out, "frames.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["record", "frame", "t_us", "exposure_us", "iso", "block_t_us", "block_seq", "samples", "timing"])
        for k, (r, b) in enumerate(zip(recs, blks)):
            fr = get(r, 1, 1)  # a zero varint is omitted from the wire, so absent means 0
            wr.writerow([k, fr if isinstance(fr, int) else 0, int(frame_t[k]), f"{exposure[k]:.1f}",
                         f"{iso[k]:.0f}", b["t"], b["seq"], len(b["q"]),
                         "" if b["timing"] is None else f"{b['timing']:.6f}"])

    report(path, recs, frame_t, s["starts"], s["counts"], s["period"], s["periods"], q, w, exposure, s["timing"])
    print(f"  wrote {os.path.join(out, 'imu.csv')} and frames.csv")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
