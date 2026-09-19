#!/usr/bin/env python3
"""Extract DJI Osmo 360 / Avata 360 (.OSV/.LRF) calibration + telemetry.

The telemetry is a stream of protobuf messages (`dvtm_oq101.proto` on the Osmo
360, `dvtm_AVATA360.proto` on the Avata 360) inside a self-contained MP4 called
`camd`. Where that MP4 lives differs by camera:

  Osmo 360   a trailing top-level `camd` box
  Avata 360  the samples of a track whose sample description is `camd`,
             split in two -- concatenated, they are the same nested MP4

Field paths below match ExifTool's dvtm numbering (e.g. 3-2-3-1 = ISO). The lens
calibration is found by content rather than by field number, because the two
schemas put it in different places (config 2-6 on the Osmo, 2-5 on the Avata).
Full field reference: docs/osmo360-telemetry.md, docs/avata360-telemetry.md.

usage: osv_meta.py <file.OSV|.LRF> [out_dir]  -> calibration.json, telemetry.csv
"""
import csv
import json
import os
import struct
import sys


def rv(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7f) << s; s += 7
        if not c & 0x80:
            return r, i


def fields(b):
    i = 0
    while i < len(b):
        k, i = rv(b, i); fn, wt = k >> 3, k & 7
        if wt == 0: v, i = rv(b, i)
        elif wt == 1: v = b[i:i + 8]; i += 8
        elif wt == 2:
            n, i = rv(b, i); v = b[i:i + n]; i += n
        elif wt == 5: v = b[i:i + 4]; i += 4
        else: raise ValueError('wiretype %d' % wt)
        yield fn, wt, v


def get(b, *path):
    """Walk a field path; returns raw payload or None."""
    cur = b
    for p in path:
        found = None
        for fn, wt, v in fields(cur):
            if fn == p: found = v; break
        if found is None: return None
        cur = found
    return cur


f32 = lambda v: struct.unpack('<f', v)[0]
f64 = lambda v: struct.unpack('<d', v)[0]


def floats(v): return list(struct.unpack('<%df' % (len(v) // 4), v))


def msg4f(v):
    """A 20-byte sub-message of four f32 sub-fields (1..4) -> [a,b,c,d]."""
    out = [None] * 4
    for fn, wt, x in fields(v):
        if wt == 5 and 1 <= fn <= 4: out[fn - 1] = f32(x)
    return out


# ------------------------------------------------------------ container

def _boxes(f, start, end):
    pos = start
    while pos < end - 7:
        f.seek(pos); h = f.read(8)
        if len(h) < 8: break
        n = struct.unpack('>I', h[:4])[0]; t = h[4:8]; hl = 8
        if n == 1: n = struct.unpack('>Q', f.read(8))[0]; hl = 16
        elif n == 0: n = end - pos
        if n < hl or pos + n > end: break
        yield pos, n, t, hl
        pos += n


def _child(f, start, end, want):
    for pos, n, t, hl in _boxes(f, start, end):
        if t == want:
            s = pos + hl
            if t == b'stsd': s += 8
            return s, pos + n
    return None


def _track_samples(f, trak):
    """(offset, size) of every sample in a trak, from stsz/stco|co64/stsc."""
    mdia = _child(f, *trak, b'mdia'); minf = _child(f, *mdia, b'minf')
    stbl = _child(f, *minf, b'stbl')
    f.seek(stbl[0]); stsd = _child(f, *stbl, b'stsd')
    f.seek(stsd[0] + 4); fmt = f.read(4)
    s, e = _child(f, *stbl, b'stsz'); f.seek(s); z = f.read(e - s)
    ssz, cnt = struct.unpack('>II', z[4:12])
    sizes = [ssz] * cnt if ssz else list(struct.unpack('>%dI' % cnt, z[12:12 + 4 * cnt]))
    co, big = _child(f, *stbl, b'stco'), False
    if co is None:
        co, big = _child(f, *stbl, b'co64'), True
    f.seek(co[0]); cb = f.read(co[1] - co[0]); nco = struct.unpack('>I', cb[4:8])[0]
    offs = struct.unpack('>%d%s' % (nco, 'Q' if big else 'I'), cb[8:8 + nco * (8 if big else 4)])
    s, e = _child(f, *stbl, b'stsc'); f.seek(s); sc = f.read(e - s)
    runs = [struct.unpack('>III', sc[8 + 12 * i:20 + 12 * i]) for i in range(struct.unpack('>I', sc[4:8])[0])]
    out, si = [], 0
    for ci, off in enumerate(offs, start=1):
        spc = [r[1] for r in runs if r[0] <= ci][-1]
        for _ in range(spc):
            if si >= len(sizes): break
            out.append((off, sizes[si])); off += sizes[si]; si += 1
    return fmt, out


def find_camd(path):
    """The camd MP4 as bytes (starting at its ftyp), wherever the camera put it."""
    sz = os.path.getsize(path)
    with open(path, 'rb') as f:
        moov = None
        for pos, n, t, hl in _boxes(f, 0, sz):
            if t == b'camd':                          # Osmo 360: trailing box
                f.seek(pos + hl); return f.read(n - hl)
            if t == b'moov':
                moov = (pos + hl, pos + n)
        if moov:                                       # Avata 360: a camd track
            for pos, n, t, hl in _boxes(f, *moov):
                if t != b'trak': continue
                fmt, samples = _track_samples(f, (pos + hl, pos + n))
                if fmt == b'camd' and samples:
                    parts = []
                    for off, size in samples:
                        f.seek(off); parts.append(f.read(size))
                    return b''.join(parts)
    raise SystemExit('no camd box or camd track in %s' % path)


def camd_payload(camd):
    """camd holds a nested MP4; return its mdat payload (the protobuf stream)."""
    pos = 0
    while pos < len(camd) - 7:
        n = struct.unpack('>I', camd[pos:pos + 4])[0]; t = camd[pos + 4:pos + 8]
        if n == 1: n = struct.unpack('>Q', camd[pos + 8:pos + 16])[0]; hl = 16
        else: hl = 8
        if t == b'mdat': return camd[pos + hl:pos + n]
        if n < hl: break
        pos += n
    raise SystemExit('no mdat inside camd')


def records(buf):
    i = 0
    while i < len(buf):
        try:
            k, j = rv(buf, i); fn, wt = k >> 3, k & 7
            if wt != 2: return
            n, j = rv(buf, j)
            if j + n > len(buf): return
        except Exception: return
        yield fn, buf[j:j + n]; i = j + n


# ----------------------------------------------------------- calibration

LENS_KEYS = ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4']

# DewarpParams, named as in telemetry-parser's dvtm_library.proto
# (github.com/AdrianEddy/telemetry-parser/tree/master/src/dji).
DEWARP_FLOAT = {1: 'fx', 2: 'fy', 3: 'cx', 4: 'cy', 5: 'k1', 6: 'k2', 7: 'k3', 8: 'k4', 9: 'xi',
                10: 'width', 11: 'height', 12: 'yaw_deg', 13: 'pitch_deg', 14: 'roll_deg',
                15: 'k5', 16: 'k6', 17: 'k7', 18: 'k8', 19: 'k9', 24: 'lens_model', 25: 'temperature',
                31: 'temp_compen_k', 32: 'gimbal_yaw_h1', 33: 'gimbal_yaw_h2'}
DEWARP_REPEATED = {20: 'p', 21: 'q', 22: 'occlusion_pt_x', 23: 'occlusion_pt_y', 27: 'tangent_coeff'}
DEWARP_QUAT = {26: 'cam_imu_extri_q', 28: 'cam_extri_q'}     # Quaternion: w, x, y, z
DEWARP_BOOL = {29: 'cam_imu_cali_enable', 30: 'temp_compen_enable'}
PIXEL_FIELDS = ('fx', 'fy', 'cx', 'cy')


def parse_lens(m):
    d = {}
    for fn, wt, v in fields(m):
        if fn in DEWARP_FLOAT and wt == 5:
            d[DEWARP_FLOAT[fn]] = round(f32(v), 8)
        elif fn in DEWARP_REPEATED and wt == 2 and v and len(v) % 4 == 0 and any(floats(v)):
            d[DEWARP_REPEATED[fn]] = [round(x, 8) for x in floats(v)]
        elif fn in DEWARP_QUAT and wt == 2:
            q = msg4f(v)
            if any(x is not None for x in q):
                d[DEWARP_QUAT[fn]] = [round(0.0 if x is None else x, 8) for x in q]
        elif fn in DEWARP_BOOL and wt == 0:
            d[DEWARP_BOOL[fn]] = bool(v)
    return d


def find_calibration(cfg):
    """(config field, [lens slots]) for the repeated field that holds lens calibrations.

    Found by content, not number: the Osmo 360 keeps it at 2-6 with its active
    pair in slots 1-2, the Avata 360 at 2-5 with its pair in slots 3-4.
    """
    best = None
    for fn, wt, v in fields(cfg):
        if wt != 2 or not v: continue
        slots = []
        try:
            for sfn, swt, sv in fields(v):
                if swt != 2: continue
                try:
                    lens = parse_lens(sv)
                except Exception:
                    continue
                if all(k in lens for k in LENS_KEYS) and lens['fx'] > 0:
                    slots.append(dict(slot=sfn, **lens))
        except Exception:
            continue
        if slots and (best is None or len(slots) > len(best[1])):
            best = (fn, slots)
    return best


def detect_camera(path):
    """Fast probe: detect whether a file is an Osmo 360, Avata 360, or generic video.

    Returns dict:
      camera: 'osmo360' | 'avata360' | 'unknown'
      camera_name: human-readable name
      model: camera model string
      proto: protobuf schema name
      serial: device serial
      firmware: device firmware version
      is_fisheye: bool
      recommended_mask: bool (True for osmo360, False for avata360 / stitched)
      imu_rate_hz: int or None (1000 for osmo360, 4000 for avata360)
      has_gps: bool
    """
    res = {
        "camera": "unknown",
        "camera_name": "Generic / Stitched",
        "model": None,
        "proto": None,
        "serial": None,
        "firmware": None,
        "is_fisheye": False,
        "recommended_mask": False,
        "imu_rate_hz": None,
        "has_gps": False,
    }
    try:
        raw = camd_payload(find_camd(str(path)))
    except (SystemExit, Exception):
        return res

    res["is_fisheye"] = True
    hdr = None
    for fn, body in records(raw):
        if fn == 1:
            hdr = body
            break
    if not hdr:
        return res

    h1 = get(hdr, 1)
    if h1:
        for fn, wt, v in fields(h1):
            if wt == 2:
                try:
                    field_name = {1: 'proto', 5: 'serial', 6: 'firmware', 10: 'model'}.get(fn)
                    if field_name:
                        res[field_name] = v.decode()
                except (KeyError, UnicodeDecodeError):
                    pass

    model = res.get("model") or ""
    proto = res.get("proto") or ""
    ident = f"{model} {proto}".lower()

    if "avata" in ident:
        res["camera"] = "avata360"
        res["camera_name"] = "DJI Avata 360"
        res["model"] = res["model"] or "DJI Avata360"
        res["recommended_mask"] = False  # Aerial drone; enable only if flying near people
        res["imu_rate_hz"] = 4000
        res["has_gps"] = True
    elif "osmo" in ident or "oq101" in ident:
        res["camera"] = "osmo360"
        res["camera_name"] = "DJI Osmo 360"
        res["model"] = res["model"] or "Osmo 360"
        res["recommended_mask"] = True  # Handheld/stick; operator always in frame
        res["imu_rate_hz"] = 1000
        res["has_gps"] = False
    else:
        # Fallback inspection on IMU rate fields in header
        r8 = get(hdr, 8, 1)
        r10 = get(hdr, 10, 1)
        if isinstance(r8, int) and 3000 <= r8 <= 5000:
            res["camera"] = "avata360"
            res["camera_name"] = "DJI Avata 360"
            res["recommended_mask"] = False
            res["imu_rate_hz"] = r8
            res["has_gps"] = True
        elif isinstance(r10, int) and 800 <= r10 <= 1200:
            res["camera"] = "osmo360"
            res["camera_name"] = "DJI Osmo 360"
            res["recommended_mask"] = True
            res["imu_rate_hz"] = r10
            res["has_gps"] = False
        else:
            res["camera_name"] = model or "DJI Dual Fisheye"

    return res


def main(path, outdir):
    raw = camd_payload(find_camd(path))
    hdr = cfg = None; rows = []
    for fn, body in records(raw):
        if fn == 1 and hdr is None: hdr = body
        elif fn == 2 and cfg is None: cfg = body
        elif fn == 3: rows.append(body)
    if hdr is None or cfg is None:
        raise SystemExit('camd stream has no header/config')

    info = {}
    h1 = get(hdr, 1)
    if h1:
        for fn, wt, v in fields(h1):
            if wt == 2:
                try: info[{1: 'proto', 2: 'pb_lib_version', 3: 'pb_version', 5: 'serial', 6: 'firmware', 10: 'model'}[fn]] = v.decode()
                except (KeyError, UnicodeDecodeError): pass
    vid = get(cfg, 3)
    video_w = get(vid, 1) if vid else None
    video_h = get(vid, 2) if vid else None
    info['video_width'], info['video_height'] = video_w, video_h

    found = find_calibration(cfg)
    lenses = []
    if found:
        info['calibration_field'] = f"2-{found[0]}"
        for i, lens in enumerate(found[1]):
            # The first two populated slots are the active pair: stream 0, then
            # stream 1. Calibrations are stated for their own frame size (3840 on
            # both cameras so far), which is not the stream size on the Avata 360
            # (3000). The stream is a uniform resize of that frame -- its whole
            # image circle fits inside it -- so pixel quantities scale and the
            # angular distortion coefficients do not.
            lens['active'] = i < 2
            if i < 2:
                lens['stream'] = i
            cw = lens.get('width')
            if video_w and cw and abs(video_w - cw) > 0.5:
                s = video_w / cw
                lens.update(calib_width=cw, calib_height=lens.get('height'), scale=s,
                            width=float(video_w), height=float(video_h or video_w))
                for k in PIXEL_FIELDS:
                    lens[k] = round(lens[k] * s, 6)
                for k in ('occlusion_pt_x', 'occlusion_pt_y'):
                    if k in lens:
                        lens[k] = [round(x * s, 3) for x in lens[k]]
            lenses.append(lens)
    cal = dict(device=info, lenses=lenses)
    os.makedirs(outdir, exist_ok=True)
    json.dump(cal, open(os.path.join(outdir, 'calibration.json'), 'w'), indent=1)

    csvf = open(os.path.join(outdir, 'telemetry.csv'), 'w', newline='')
    w = csv.writer(csvf)
    w.writerow(['frame', 't_sec', 'iso', 'color_temp', 'qw', 'qx', 'qy', 'qz', 'ax', 'ay', 'az',
                'imu_samples', 'lat', 'lon', 'abs_alt_m', 'rel_alt_m'])
    n_imu = n_gps = 0
    for body in rows:
        fr = get(body, 1, 1); ts = get(body, 1, 2)
        iso = get(body, 2, 3, 1); ct = get(body, 2, 6, 1)
        q = get(body, 2, 9); a = get(body, 2, 10)
        qs = [('' if x is None else round(x, 6)) for x in msg4f(q)] if q else [''] * 4
        if a:
            av = {fn: round(f32(v), 6) for fn, wt, v in fields(a) if wt == 5}
            acc = [av.get(2, ''), av.get(3, ''), av.get(4, '')]
        else:
            acc = [''] * 3
        blk = get(body, 3, 2, 1); cnt = 0
        if blk:
            cnt = sum(1 for fn, wt, v in fields(blk) if fn == 3 and wt == 2 and len(v) == 20)
        n_imu += cnt
        # Avata 360: 3-4-4-1-2/3 latitude/longitude (f64 degrees), 3-4-4-2 absolute
        # altitude (mm), 3-4-5-1 relative altitude (f32 mm). Absent on the Osmo 360.
        gps = get(body, 4, 4, 1)
        lat = get(gps, 2) if gps else None
        lon = get(gps, 3) if gps else None
        alt = get(body, 4, 4, 2)
        rel = get(body, 4, 5, 1)
        latv = f64(lat) if lat is not None and len(lat) == 8 else ''
        lonv = f64(lon) if lon is not None and len(lon) == 8 else ''
        if latv != '' and lonv != '' and (latv or lonv):
            n_gps += 1
        w.writerow([fr if isinstance(fr, int) else '',
                    round(ts / 1e6, 6) if isinstance(ts, int) else '',
                    round(f32(iso), 1) if iso and len(iso) == 4 else '',
                    ct if isinstance(ct, int) else '',
                    *qs, *acc, cnt,
                    round(latv, 8) if latv != '' else '', round(lonv, 8) if lonv != '' else '',
                    round(alt / 1000, 3) if isinstance(alt, int) else '',
                    round(f32(rel) / 1000, 3) if rel is not None and len(rel) == 4 else ''])
    csvf.close()
    print(f'{os.path.basename(path)}: {len(rows)} records, {n_imu} high-rate IMU samples, {n_gps} with a GPS fix')
    print(f'  device: {info}')
    print(f'  lens calibrations: {[(l["slot"], "active" if l["active"] else "") for l in lenses]}')
    return cal


if __name__ == '__main__':
    out = sys.argv[2] if len(sys.argv) > 2 else '.'
    main(sys.argv[1], out)
