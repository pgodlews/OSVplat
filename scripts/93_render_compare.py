#!/usr/bin/env python3
"""Render several trained models from the SAME held-out poses into one contact sheet.

This is the comparison the project notes has been assembling by hand five times
(check_run*_render_vs_photo.jpg). Rows are poses, columns are models, with the
matching source-panorama crop in the last column.

Renders a 90-degree pinhole view along each camera's +z. The photo column shows
the identical region: reprojected off the sphere for an EQUIRECTANGULAR
reconstruction, and warped through the source camera's own intrinsics for the
perspective SfM modes, which register generated pinhole views rather than the
panoramas. Which one is chosen comes from the reconstruction's camera model.

usage:
  93_render_compare.py --dataset DIR --images DIR --out sheet.jpg \
      --poses pano_0000.jpg,pano_0060.jpg \
      --model "baseline=/path/a.ply" --model "sh3=/path/b.ply"

Run with venv_gs (needs torch + gsplat + opencv + pycolmap).

Note: venv_gs holds the CPU build of pycolmap (docs/troubleshooting.md #3 -- gsplat's
example requirements overwrote pycolmap-cuda12 there). That is fine and
intentional here: this script only READS a reconstruction, it never runs SfM.
Do not "fix" it by installing pycolmap-cuda12 into venv_gs.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from gsplat import rasterization

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="dir containing sparse/0")
ap.add_argument("--images", required=True,
                help="the directory the reconstruction's image names refer to "
                     "-- panoramas for spherical SfM, generated pinhole views "
                     "for the perspective modes (see stages.images_dir)")
ap.add_argument("--out", required=True)
ap.add_argument("--poses", default="", help="comma-separated pano filenames")
ap.add_argument("--n-poses", type=int, default=4,
                help="if --poses is empty, sample this many evenly")
ap.add_argument("--test-every", type=int, default=8,
                help="the trainer's held-out split: image i is a validation "
                     "view when i %% N == 0 (LichtFeld's --test-every, default "
                     "8). 0 or 1 means the models saw every image, so poses "
                     "are sampled from all of them and labelled as training "
                     "views.")
ap.add_argument("--model", action="append", default=[],
                help="LABEL=path.ply, repeatable. Split on the LAST '=', so a "
                     "sweep label like 'sh_degree=3' is safe.")
ap.add_argument("--tile", type=int, default=512)
ap.add_argument("--view-yaw", type=float, default=0.0,
                help="rotate the comparison view left/right, degrees")
ap.add_argument("--view-pitch", type=float, default=0.0,
                help="rotate the comparison view DOWN by this many degrees "
                     "(positive looks down). The default +z view is the "
                     "equirect centre, which on a walking capture points along "
                     "the path -- useless for inspecting anything at the "
                     "camera's feet, such as whether a person was masked out.")
args = ap.parse_args()

rec = pycolmap.Reconstruction(str(Path(args.dataset) / "sparse" / "0"))
by_name = {im.name: im for im in rec.images.values()}
if not by_name:
    sys.exit("no images in reconstruction")

ordered = sorted(by_name)
# The point of this sheet is to compare models on views none of them trained
# on. Sampling evenly across ALL registered images returns training views most
# of the time (7 in 8 at the default split), which flatters every model equally
# and hides exactly the overfitting the comparison exists to expose.
if args.test_every > 1:
    held_out = [n for i, n in enumerate(ordered) if i % args.test_every == 0]
else:
    held_out = []
pool = held_out or ordered
split = (f"held-out (every {args.test_every}th of {len(ordered)})"
         if held_out else "ALL images — these are TRAINING views")

if args.poses.strip():
    names = [n.strip() for n in args.poses.split(",") if n.strip()]
    missing = [n for n in names if n not in by_name]
    if missing:
        sys.exit(f"poses not in reconstruction: {missing}")
    trained_on = [n for n in names if held_out and n not in held_out]
    if trained_on:
        print(f"warning: {len(trained_on)} of the requested poses are training "
              f"views, not held out: {', '.join(trained_on[:4])}", flush=True)
else:
    step = max(1, len(pool) // max(1, args.n_poses))
    names = pool[::step][:args.n_poses]
print(f"pose pool: {len(pool)} {split}; using {len(names)}", flush=True)

models = []
for spec in args.model:
    # rsplit, not split: sweep labels are built from the varied parameter and
    # routinely contain '=' ("sh_degree=3"), which split-on-first turned into a
    # truncated label and a path starting mid-name.
    label, sep, path = spec.rpartition("=")
    if not sep or not label:
        sys.exit(f"--model needs LABEL=path, got {spec!r}")
    if not Path(path).is_file():
        sys.exit(f"no such ply: {path}")
    models.append((label, path))
if not models:
    sys.exit("at least one --model is required")

# gsplat's rasterizer is CUDA-only; say so up front instead of failing deep
# inside it with CPU tensors.
if not torch.cuda.is_available():
    sys.exit("93_render_compare.py needs a CUDA GPU (gsplat has no CPU rasterizer)")
dev = "cuda"
T = lambda a: torch.from_numpy(np.ascontiguousarray(a).astype(np.float32)).to(dev)
S = args.tile
K = np.array([[S / 2, 0, S / 2], [0, S / 2, S / 2], [0, 0, 1]], np.float32)


def _view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rays in the rotated view -> rays in the camera's own frame.

    COLMAP camera axes are x right, y DOWN, z forward, so "look down" means
    tilting the forward axis toward +y. Written out explicitly with that sign
    because the textbook Rx tilts it the other way, and getting it backwards
    silently renders the sky -- which is how the first version of the person
    masking looked like it had found nobody.
    """
    yw, pi_ = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    ry = np.array([[np.cos(yw), 0, np.sin(yw)],
                   [0, 1, 0],
                   [-np.sin(yw), 0, np.cos(yw)]], np.float32)
    rx = np.array([[1, 0, 0],
                   [0, np.cos(pi_), np.sin(pi_)],
                   [0, -np.sin(pi_), np.cos(pi_)]], np.float32)
    return (ry @ rx).astype(np.float32)


VIEW_R = _view_rotation(args.view_yaw, args.view_pitch)


# PLY scalar types -> numpy. Lists (faces) never occur in a splat file.
_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
              "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply_vertices(path: str) -> np.ndarray:
    """The vertex element of a binary PLY as a numpy structured array.

    A dozen lines instead of the plyfile package, which is GPL-3.0: every
    splat trainer here writes one binary vertex element, which is all this
    needs to read.
    """
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, elements, cur = None, [], None
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: header has no end_header")
            parts = line.decode("ascii", "replace").split()
            if not parts or parts[0] in ("comment", "obj_info"):
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                cur = [parts[1], int(parts[2]), []]
                elements.append(cur)
            elif parts[0] == "property":
                if cur is None:
                    raise ValueError(f"{path}: property before any element")
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties are not supported")
                if parts[1] not in _PLY_TYPES:
                    raise ValueError(f"{path}: unsupported property type {parts[1]!r}")
                cur[2].append((parts[2], _PLY_TYPES[parts[1]]))
            elif parts[0] == "end_header":
                break
        order = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(fmt)
        if order is None:
            raise ValueError(f"{path}: PLY format {fmt!r} is not supported (binary only)")
        for name, count, props in elements:
            dtype = np.dtype([(p, order + t) for p, t in props])
            data = np.fromfile(fh, dtype=dtype, count=count)
            if name == "vertex":
                if len(data) != count:
                    raise ValueError(f"{path}: truncated, {len(data)} of {count} vertices")
                return data
    raise ValueError(f"{path}: no vertex element")


def load_ply(path: str):
    v = read_ply_vertices(path)
    g = lambda k: np.asarray(v[k], dtype=np.float32)
    means = np.stack([g("x"), g("y"), g("z")], 1)
    scales = np.exp(np.stack([g(f"scale_{i}") for i in range(3)], 1))
    quats = np.stack([g(f"rot_{i}") for i in range(4)], 1)
    opac = 1 / (1 + np.exp(-g("opacity")))
    dc = np.stack([g(f"f_dc_{i}") for i in range(3)], 1)
    nrest = len([n for n in v.dtype.names if n.startswith("f_rest_")])
    if nrest:
        ksh = nrest // 3
        rest = np.stack([g(f"f_rest_{i}") for i in range(nrest)], 1)
        rest = rest.reshape(-1, 3, ksh).transpose(0, 2, 1)
    else:
        rest = np.zeros((len(means), 0, 3), np.float32)
    sh = np.concatenate([dc[:, None, :], rest], 1)
    deg = int(np.sqrt(sh.shape[1]) - 1)
    return means, quats, scales, opac, sh, deg


def render(model, name):
    means, quats, scales, opac, sh, deg = model
    im = by_name[name]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = im.cam_from_world().matrix()
    # Rotating the view means rotating the camera in its own frame, which is a
    # left-multiply of the world->camera matrix by the inverse.
    w2c[:3, :] = VIEW_R.T @ w2c[:3, :]
    img, _, _ = rasterization(T(means), T(quats), T(scales), T(opac), T(sh),
                              T(w2c)[None], T(K)[None], S, S,
                              sh_degree=deg, packed=True)
    return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]


# Reprojection map from the equirectangular source into the SAME pinhole camera
# the models are rendered with. A plain centre-crop of the equirect does not
# work: the render is rectilinear and the crop is not, so the two disagree
# increasingly toward the edges and fine detail cannot be compared at all.
_MAP_CACHE: dict[tuple, tuple] = {}


def _erp_to_pinhole_maps(pano_w, pano_h):
    key = (pano_w, pano_h, S, args.view_yaw, args.view_pitch)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    j, i = np.meshgrid(np.arange(S, dtype=np.float32),
                       np.arange(S, dtype=np.float32))
    # Camera rays through each pixel, looking down +z (matches the render).
    x = (j - K[0, 2]) / K[0, 0]
    y = (i - K[1, 2]) / K[1, 1]
    z = np.ones_like(x)
    n = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    # Same rotation as the render, so the ground-truth column keeps showing the
    # identical patch of sphere.
    d = np.stack([x, y, z], -1) @ VIEW_R.T
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lon = np.arctan2(x, z)              # 0 at +z, matching the ERP centre
    lat = np.arcsin(np.clip(y, -1, 1))
    mx = ((lon / (2 * np.pi)) + 0.5) * pano_w
    my = (0.5 + (lat / np.pi)) * pano_h
    maps = (mx.astype(np.float32), my.astype(np.float32))
    _MAP_CACHE[key] = maps
    return maps


# Which projection the ground-truth images are actually in. Only the spherical
# SfM path registers the panoramas themselves; the two perspective modes
# register generated PINHOLE views, and stages.images_dir hands those to both
# the trainer and this script. Remapping one of them as though it were an
# equirect warps the photo column into something that reads as reconstruction
# error. Decide it from the reconstruction's own camera model rather than from a
# flag nobody would remember to set.
CAM_MODELS = sorted({c.model.name for c in rec.cameras.values()})
GT_IS_ERP = any("EQUIRECTANGULAR" in m for m in CAM_MODELS)
print(f"cameras: {', '.join(CAM_MODELS) or 'unknown'}; reading ground truth as "
      f"{'equirectangular' if GT_IS_ERP else 'pinhole'}", flush=True)
_warned = set()


def _pinhole_gt(pano, name):
    """Reproject a PINHOLE ground-truth image into the comparison camera.

    The two share a centre and differ only by VIEW_R and by intrinsics, so this
    is a plain homography and it is exact, not an approximation. cv2 wants the
    source->destination map, and a comparison pixel u traces the ray
    VIEW_R @ K^-1 u into the source camera -- so the map is the inverse of
    Ks @ VIEW_R @ K^-1.
    """
    try:
        cam = rec.cameras[by_name[name].camera_id]
        ks = np.asarray(cam.calibration_matrix(), np.float32)
        hom = K @ VIEW_R.T @ np.linalg.inv(ks)
    except Exception as exc:                                   # noqa: BLE001
        if "K" not in _warned:
            _warned.add("K")
            print(f"warning: no linear intrinsics for these cameras ({exc}); "
                  f"the photo column is a plain resize and will not line up "
                  f"with the renders", flush=True)
        return None
    return cv2.warpPerspective(pano, hom, (S, S), flags=cv2.INTER_CUBIC)


def gt_crop(name):
    pano = cv2.imread(str(Path(args.images) / name))
    if pano is None:
        return np.zeros((S, S, 3), np.uint8)
    if not GT_IS_ERP:
        out = _pinhole_gt(pano, name)
        return out if out is not None else cv2.resize(pano, (S, S))
    h, w = pano.shape[:2]
    mx, my = _erp_to_pinhole_maps(w, h)
    return cv2.remap(pano, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


def label_tile(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (S, 22), (0, 0, 0), -1)
    cv2.putText(out, text[:44], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


rows = []
loaded = []
for label, path in models:
    print(f"loading {label}: {path}", flush=True)
    loaded.append((label, load_ply(path)))

for name in names:
    tiles = []
    for label, model in loaded:
        tiles.append(label_tile(render(model, name), f"{label} · {name[:-4]}"))
    tiles.append(label_tile(gt_crop(name), f"photo · {name[:-4]}"))
    rows.append(np.hstack(tiles))
    print(f"rendered {name}", flush=True)

sheet = np.vstack(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"wrote {args.out} ({sheet.shape[1]}x{sheet.shape[0]}, "
      f"{len(loaded)} models x {len(names)} poses)")
