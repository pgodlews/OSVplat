#!/usr/bin/env python3
"""Turn a refined fisheye rig reconstruction back into a calibration.json.

82_fisheye_sfm.py --refine-intrinsics lets bundle adjustment move focal, k1-k4
and the lens1-from-lens0 rotation. This writes those values in the shape of
osv_meta.py's calibration.json, so 81_fisheye_stitch.py can stitch with the
refined lens and 82 can start a wider-radius run from it -- DJI's own polynomial
folds back on itself near 1658 px and cannot seed one.

lens0 keeps DJI's orientation, which defines the stitch's world frame; lens1 gets
the refined relative rotation applied to it, stored as an explicit world_to_cam
matrix that osmo_fisheye.world_to_cam honours.

usage (venv): 84_refined_calib.py <model_dir> <calibration.json> <out.json>
"""
import json
import math
import sys

import numpy as np
import pycolmap

from osmo_fisheye import lenses, rig_rotation, theta_d, world_to_cam


def call(x):
    return x() if callable(x) else x


def rot_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


model, calib, out = sys.argv[1:4]
rec = pycolmap.Reconstruction(model)
cal = json.load(open(calib))
L = lenses(calib)
cam_of, poses = {}, {}
for iid in rec.reg_image_ids():
    img = rec.images[iid]
    lens = img.name.split("/")[0]
    cam_of[lens] = img.camera_id
    poses.setdefault(img.frame_id, {})[lens] = np.asarray(call(img.cam_from_world).matrix())
both = [p for p in poses.values() if "lens0" in p and "lens1" in p]
if not both or set(cam_of) != {"lens0", "lens1"}:
    sys.exit("model has no frame with both lenses registered")
R10s = [p["lens1"][:, :3] @ p["lens0"][:, :3].T for p in both]
R10 = R10s[0]
spread = max(rot_deg(R @ R10.T) for R in R10s)  # one rig, so ~0
M0 = world_to_cam(L[0])
R10_before = rig_rotation(L)

refined = []
for i, lens in enumerate(("lens0", "lens1")):
    p = [float(x) for x in rec.cameras[cam_of[lens]].params]
    old, l = dict(L[i]), dict(L[i])
    l.update(fx=p[0], fy=p[1], cx=p[2], cy=p[3], k1=p[4], k2=p[5], k3=p[6], k4=p[7])
    # The refined model is COLMAP's four-coefficient one. Carrying DJI's k5.. over
    # would add it on top of k1..k4 that already absorbed it.
    for extra in ("k5", "k6", "k7", "k8", "k9", "xi", "tangent_coeff"):
        l.pop(extra, None)
    l["world_to_cam"] = (M0 if i == 0 else R10 @ M0).tolist()
    l["refined_from"] = str(model)
    radii = {deg: (round(old["fx"] * theta_d(math.radians(deg), old), 1),
                   round(l["fx"] * theta_d(math.radians(deg), l), 1))
             for deg in (30, 60, 75, 80, 85, 88, 90, 95)}
    print(f"{lens}: fx {old['fx']:.2f} -> {l['fx']:.2f}, fy {old['fy']:.2f} -> {l['fy']:.2f}; "
          f"radius px (before, after) by off-axis deg: {radii}")
    refined.append(l)
print(f"lens1_from_lens0 moved {rot_deg(R10 @ R10_before.T):.3f} deg; spread across frames {spread:.5f} deg")
out_cal = dict(cal)
out_cal["lenses"] = refined
json.dump(out_cal, open(out, "w"), indent=1)
print(f"wrote {out}")
