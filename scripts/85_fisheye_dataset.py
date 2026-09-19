#!/usr/bin/env python3
"""LichtFeld dataset from a fisheye rig model or a stitched equirect model, holding out the same instants.

  * Every Nth instant (frame_NNNN / pano_NNNN with NNNN % N == 0) is
    deregistered, so neither arm of the A/B trains on those moments and
    86_fisheye_eval.py can render both at them. LichtFeld's own --eval split is
    not used: it holds out by position in its own image order, which is not
    capture order (docs/troubleshooting.md #23), so two arms would not hold out the same moments.
  * Fisheye only: "lens0/frame_0001.jpg" is flattened to "lens0_frame_0001.jpg".
    LichtFeld matches masks by file stem across the whole dataset and refuses a
    stem that appears twice -- and a rig names both lenses of an instant alike.

A model that registers under 90% of the instants is refused -- COLMAP's
sparse/0 can be a fragment while the real model sits in sparse/1.

The written images.bin is read back with a minimal parser (what LichtFeld sees,
not what pycolmap thinks it wrote) and every pose is checked against the model,
because a rig model's image pose is composed from its frame and sensor.

Writes <out>/sparse/0, <out>/images/ and <out>/masks/ (symlinks; masks in
LichtFeld's <stem>.png layout, white = keep).

usage (venv):
  85_fisheye_dataset.py fisheye <model_dir> <images_root> <masks_colmap_root> <out> [--holdout-every 8]
  85_fisheye_dataset.py erp     <model_dir> <panos_dir>   <masks_lichtfeld_dir> <out> [--holdout-every 8]
"""
import argparse
import json
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pycolmap


def call(x):
    return x() if callable(x) else x


def instant(name):
    # \d+, not \d{4}: frame_{k:04d} grows a fifth digit at candidate 10000
    # (17 min at 10 fps), and four digits would fold it onto frame 0000.
    m = re.search(r"(\d+)\.jpg$", name)
    if not m:
        raise SystemExit(f"cannot read an instant index from {name}")
    return int(m.group(1))


def read_images_bin(path):
    d = open(path, "rb").read()
    n, off, out = struct.unpack_from("<Q", d, 0)[0], 8, {}
    for _ in range(n):
        off += 4  # image_id
        q = np.array(struct.unpack_from("<4d", d, off)); off += 32
        t = np.array(struct.unpack_from("<3d", d, off)); off += 24
        off += 4  # camera_id
        end = d.index(b"\0", off)
        name = d[off:end].decode(); off = end + 1
        npts = struct.unpack_from("<Q", d, off)[0]; off += 8 + npts * 24
        out[name] = (q, t)
    if off != len(d):
        raise SystemExit(f"images.bin parse ended at {off} of {len(d)} bytes; format is not the one LichtFeld reads")
    return out


def qmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("fisheye", "erp"))
    ap.add_argument("model")
    ap.add_argument("images")
    ap.add_argument("masks")
    ap.add_argument("out")
    ap.add_argument("--holdout-every", type=int, default=8,
                    help="hold out every Nth instant; 0 holds out nothing (the queue uses LichtFeld's own --eval)")
    ap.add_argument("--holdout-instants", default="",
                    help="comma-separated instants to hold out instead of every Nth, so two arms whose "
                         "selections differ can hold out only the moments both kept (94_fisheye_ab_eval.py)")
    a = ap.parse_args()
    listed = {int(x) for x in a.holdout_instants.split(",") if x.strip()}

    rec = pycolmap.Reconstruction(a.model)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for d in ("sparse/0", "images", "masks"):
        (out / d).mkdir(parents=True)

    reg = list(rec.reg_image_ids())
    # COLMAP can emit several models and sparse/0 is not necessarily the real one
    # (docs/troubleshooting.md #18; it bit this script's first stitched run, which trained on a
    # 2-frame fragment). Refuse anything that misses a tenth of the instants.
    reg_instants = {instant(rec.images[i].name) for i in reg}
    avail = {instant(p.name) for p in (Path(a.images) / ("lens0" if a.kind == "fisheye" else "")).glob("*.jpg")}
    if len(reg_instants) < 0.9 * len(avail):
        siblings = [(p.name, pycolmap.Reconstruction(str(p)).num_reg_images())
                    for p in sorted(Path(a.model).parent.iterdir()) if (p / "images.bin").exists()]
        raise SystemExit(f"{a.model} registers {len(reg_instants)} of {len(avail)} instants: a fragment. "
                         f"Models next to it (name, registered images): {siblings}")
    if listed - reg_instants:
        raise SystemExit(f"--holdout-instants lists {sorted(listed - reg_instants)[:5]}, "
                         f"which {a.model} does not register")

    def held_out(name):
        k = instant(name)
        return k in listed if listed else a.holdout_every > 0 and k % a.holdout_every == 0
    held_frames = {rec.images[i].frame_id for i in reg if held_out(rec.images[i].name)}
    held = sorted({instant(rec.images[i].name) for i in reg if held_out(rec.images[i].name)})
    for fid in held_frames:
        rec.deregister_frame(fid)

    kept = []
    for iid in rec.reg_image_ids():
        img = rec.images[iid]
        src = img.name
        new = src.replace("/", "_") if a.kind == "fisheye" else src
        img_src = Path(a.images) / src
        mask_src = Path(a.masks) / (f"{src}.png" if a.kind == "fisheye" else f"{Path(src).stem}.png")
        if not img_src.exists() or not mask_src.exists():
            raise SystemExit(f"missing {img_src} or {mask_src}")
        img.name = new
        (out / "images" / new).symlink_to(img_src.resolve())
        (out / "masks" / f"{Path(new).stem}.png").symlink_to(mask_src.resolve())
        kept.append(new)

    rec.write_binary(str(out / "sparse" / "0"))
    written = read_images_bin(out / "sparse" / "0" / "images.bin")
    worst_r = worst_t = 0.0
    missing = []
    for iid in rec.reg_image_ids():
        img = rec.images[iid]
        if img.name not in written:
            missing.append(img.name)
            continue
        M = np.asarray(call(img.cam_from_world).matrix())
        q, t = written[img.name]
        worst_r = max(worst_r, float(np.degrees(np.arccos(np.clip((np.trace(qmat(q) @ M[:, :3].T) - 1) / 2, -1, 1)))))
        worst_t = max(worst_t, float(np.linalg.norm(t - M[:, 3])))
    extra = sorted(set(written) - set(kept))

    centres = np.array([call(rec.images[i].projection_center) for i in rec.reg_image_ids()])
    med = np.median(centres, 0)
    rad = np.linalg.norm(centres - med, axis=1)
    summary = {"kind": a.kind, "model": a.model, "train_images": len(kept),
               "held_out_instants": held, "held_out_frames": len(held_frames),
               "points3D": rec.num_points3D(), "images_bin_entries": len(written),
               "images_bin_extra_entries": extra[:5], "n_extra": len(extra), "missing": missing[:5],
               "pose_check_worst_rot_deg": worst_r, "pose_check_worst_t": worst_t,
               "camera_radius_max_over_median": float(rad.max() / max(np.median(rad), 1e-9))}
    json.dump(summary, open(out / "dataset.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if missing or worst_r > 1e-3 or worst_t > 1e-5:
        sys.exit("images.bin does not carry the model's composed poses")
    if extra:
        sys.exit(f"images.bin still lists {len(extra)} images that are not in the training set")


if __name__ == "__main__":
    main()
