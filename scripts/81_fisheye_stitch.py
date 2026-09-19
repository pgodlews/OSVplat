#!/usr/bin/env python3
"""DJI dual fisheye <-> equirect with the camera's own calibration.

Two jobs sharing one geometry, so the round trip is exact by construction:

  stitch  rig frames (images/lens0, lens1) -> 2W x W equirect pano_NNNN.jpg, for
          a W-pixel fisheye (7680x3840 on the Osmo 360, 6000x3000 on the Avata
          360). The stitched arm of the fisheye-vs-stitch A/B, and what
          70_person_masks.py already knows how to segment.
  masks   equirect person masks (70_person_masks.py, LichtFeld layout
          pano_NNNN.png) -> per-lens fisheye masks in COLMAP's layout,
          lens0/frame_NNNN.jpg.png, optionally AND-ed with a valid-image circle.

Projection is Kannala-Brandt, i.e. COLMAP's OPENCV_FISHEYE:
    px = cx + s*fx * theta_d * cos(phi),  theta_d = theta(1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)
with s = --fscale; the lens rotations come from osmo_fisheye.world_to_cams (lens 0
sets the world frame, lens 1 follows by the stored module extrinsic).
Image sizes come from the calibration (osv_meta.py scales them to the stream).

--valid-radius is a PIXEL radius, not an angle, on purpose: it has to mean the
same pixels whichever focal turns out to be right. 0 disables it (the queue's
mask stage writes person-only masks and lets the SfM choose its own circle).

usage (venv_gs):
  81_fisheye_stitch.py stitch --calib calibration.json --images DIR/images --out DIR/erp [--fscale 1.0]
  81_fisheye_stitch.py masks  --calib calibration.json --images DIR/images \
                              --erp-masks DIR/masks_erp --out DIR/fmasks_colmap [--valid-radius 1620]
  common: [--fscale 1.0] [--limit N]
"""
import argparse
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from osmo_fisheye import lenses, theta_d, world_to_cams


def size_of(l):
    w = int(round(l.get("width", 3840)))
    return w, int(round(l.get("height", w)))


def erp_grid(l, M_np, fscale, dev, erp_w, erp_h, thmax_deg=92.0, blend_deg=8.0):
    """For every equirect pixel: where to sample this lens, and how much to trust it."""
    W, H = size_of(l)
    u = (torch.arange(erp_w, device=dev, dtype=torch.float64) + 0.5) / erp_w * 2 * math.pi - math.pi
    v = (torch.arange(erp_h, device=dev, dtype=torch.float64) + 0.5) / erp_h * math.pi
    lat = math.pi / 2 - v
    M = torch.tensor(M_np, device=dev)
    cl, sl = torch.cos(lat)[:, None], torch.sin(lat)[:, None]
    dx, dy, dz = cl * torch.cos(u)[None, :], cl * torch.sin(u)[None, :], sl.expand(erp_h, erp_w)
    cx = M[0, 0] * dx + M[0, 1] * dy + M[0, 2] * dz
    cy = M[1, 0] * dx + M[1, 1] * dy + M[1, 2] * dz
    cz = M[2, 0] * dx + M[2, 1] * dy + M[2, 2] * dz
    th = torch.atan2(torch.sqrt(cx * cx + cy * cy), cz)
    ph = torch.atan2(cy, cx)
    td = theta_d(th, l)
    px = l["cx"] + fscale * l["fx"] * td * torch.cos(ph)
    py = l["cy"] + fscale * l["fy"] * td * torch.sin(ph)
    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    w = torch.clamp((math.radians(thmax_deg) - th) / math.radians(blend_deg), 0, 1) * inside
    grid = torch.stack([px / W * 2 - 1, py / H * 2 - 1], -1).float()[None]
    return grid, w.float()[None, None]


def fish_grid(l, M_np, fscale, dev, valid_radius):
    """For every fisheye pixel: where it lands in the equirect, and whether it is usable."""
    W, H = size_of(l)
    X = (torch.arange(W, device=dev, dtype=torch.float64) + 0.5)[None, :].expand(H, W)
    Y = (torch.arange(H, device=dev, dtype=torch.float64) + 0.5)[:, None].expand(H, W)
    ex, ey = (X - l["cx"]) / (fscale * l["fx"]), (Y - l["cy"]) / (fscale * l["fy"])
    td = torch.sqrt(ex * ex + ey * ey)
    ph = torch.atan2(ey, ex)
    # theta_d(theta) is monotonic only up to its fold; invert on that branch.
    th_lut = torch.linspace(0, math.radians(110), 20001, device=dev, dtype=torch.float64)
    td_lut = theta_d(th_lut, l)
    falling = td_lut[1:] <= td_lut[:-1]
    fold = int(torch.nonzero(falling)[0]) if bool(falling.any()) else len(td_lut) - 1
    th_lut, td_lut = th_lut[: fold + 1], td_lut[: fold + 1]
    idx = torch.clamp(torch.searchsorted(td_lut, td.reshape(-1)), 1, len(td_lut) - 1)
    lo, hi = td_lut[idx - 1], td_lut[idx]
    frac = (td.reshape(-1) - lo) / (hi - lo)
    th = (th_lut[idx - 1] + frac * (th_lut[idx] - th_lut[idx - 1])).reshape(H, W)
    st = torch.sin(th)
    c = torch.stack([st * torch.cos(ph), st * torch.sin(ph), torch.cos(th)], -1)
    M = torch.tensor(M_np, device=dev)
    d = c @ M  # M is orthogonal, so d = M^T c
    lon, lat = torch.atan2(d[..., 1], d[..., 0]), torch.asin(torch.clamp(d[..., 2], -1, 1))
    gx = (lon + math.pi) / (2 * math.pi) * 2 - 1
    gy = (math.pi / 2 - lat) / math.pi * 2 - 1
    valid = td <= td_lut[-1]
    if valid_radius > 0:
        valid &= torch.sqrt((X - l["cx"]) ** 2 + (Y - l["cy"]) ** 2) <= valid_radius
    return torch.stack([gx, gy], -1).float()[None], valid


def to_tensor(path, dev, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(str(path), flags)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    t = torch.from_numpy(img).to(dev)
    t = t.permute(2, 0, 1) if t.ndim == 3 else t[None]
    return t[None].float()


def frame_number(name):
    # The last run of digits before the extension, so "frame_0001.jpg" and
    # "lens0_frame_0001.jpg" both give "0001".
    m = re.search(r"(\d+)\.[A-Za-z0-9]+$", name)
    if not m:
        raise ValueError(f"no frame number in {name!r}")
    return m.group(1)


def stitch(a, L, dev):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W, _ = size_of(L[0])
    erp_w, erp_h = 2 * W, W
    grids = [erp_grid(l, M, a.fscale, dev, erp_w, erp_h) for l, M in zip(L, world_to_cams(L))]
    wsum = (grids[0][1] + grids[1][1]).clamp_min(1e-6)
    names = sorted(p.name for p in (Path(a.images) / "lens0").glob("*.jpg"))[: a.limit or None]
    pool, pending = ThreadPoolExecutor(6), []
    t = time.time()
    for name in names:
        acc = None
        for i, (grid, w) in enumerate(grids):
            s = F.grid_sample(to_tensor(Path(a.images) / f"lens{i}" / name, dev), grid,
                              mode="bilinear", padding_mode="zeros", align_corners=False) * w
            acc = s if acc is None else acc + s
        pano = (acc / wsum).clamp(0, 255).byte()[0].permute(1, 2, 0).cpu().numpy()
        dst = out / f"pano_{frame_number(name)}.jpg"
        pending.append(pool.submit(cv2.imwrite, str(dst), pano, [cv2.IMWRITE_JPEG_QUALITY, 95]))
    ok = all(p.result() for p in pending)
    print(f"stitch: {len(names)} panoramas {erp_w}x{erp_h} (fscale {a.fscale}) in {time.time() - t:.1f}s"
          + ("" if ok else " -- SOME WRITES FAILED"), flush=True)
    if not ok:
        raise SystemExit(1)


def masks(a, L, dev):
    grids = [fish_grid(l, M, a.fscale, dev, a.valid_radius) for l, M in zip(L, world_to_cams(L))]
    names = sorted(p.name for p in (Path(a.images) / "lens0").glob("*.jpg"))[: a.limit or None]
    for i in (0, 1):
        (Path(a.out) / f"lens{i}").mkdir(parents=True, exist_ok=True)
    t = time.time()
    keep = [0.0, 0.0]
    for name in names:
        src = Path(a.erp_masks) / f"pano_{frame_number(name)}.png"
        erp = to_tensor(src, dev, cv2.IMREAD_GRAYSCALE)
        for i, (grid, valid) in enumerate(grids):
            m = F.grid_sample(erp, grid, mode="nearest", padding_mode="border", align_corners=False)[0, 0]
            m = ((m >= 128) & valid).byte() * 255
            keep[i] += float(m.float().mean() / 255)
            if not cv2.imwrite(str(Path(a.out) / f"lens{i}" / f"{name}.png"), m.cpu().numpy()):
                raise SystemExit(f"could not write mask for lens{i}/{name}")
    n = max(len(names), 1)
    circle = (f"valid circle alone: {math.pi * a.valid_radius ** 2 / (size_of(L[0])[0] * size_of(L[0])[1]):.3f}"
              if a.valid_radius > 0 else "no valid circle")
    print(f"masks: {len(names)} frames x 2 lenses in {time.time() - t:.1f}s; kept fraction of the "
          f"frame: lens0 {keep[0] / n:.3f}, lens1 {keep[1] / n:.3f} ({circle})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("stitch", "masks"))
    ap.add_argument("--calib", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--erp-masks", default="")
    ap.add_argument("--fscale", type=float, default=1.0)
    ap.add_argument("--valid-radius", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    L = lenses(a.calib)
    with torch.no_grad():
        (stitch if a.cmd == "stitch" else masks)(a, L, dev)


if __name__ == "__main__":
    main()
