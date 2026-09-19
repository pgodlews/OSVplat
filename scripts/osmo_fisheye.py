"""Osmo 360 lens geometry from the camd calibration, shared by 81_fisheye_stitch.py
and 82_fisheye_sfm.py so the stitch, the mask warp and the rig SfM can never
disagree about where a lens points. No torch or pycolmap import on purpose: the
two scripts run in different venvs.

calibration.json comes from osv_meta.py (config field 2-6 of the camd protobuf).
"""
import json
import math

import numpy as np


def lenses(path):
    """[stream 0 lens, stream 1 lens].

    osv_meta.py marks the active pair with `stream`: slots 1-2 on the Osmo 360,
    3-4 on the Avata 360. Files written before that field existed are Osmo 360
    ones, where slot 1 drives stream 0.
    """
    all_lenses = json.load(open(path))["lenses"]
    by_stream = {l["stream"]: l for l in all_lenses if "stream" in l}
    if 0 in by_stream and 1 in by_stream:
        return [by_stream[0], by_stream[1]]
    by_slot = {l["slot"]: l for l in all_lenses}
    return [by_slot[1], by_slot[2]]


def world_to_cam(l):
    """World (x fwd, y left, z up) -> OpenCV camera (x right, y down, z optical axis).

    Built from the stored yaw/pitch/roll; validated by reprojecting both lenses
    and matching the camera's own stitched thumbnail. The final x flip makes the
    matrix improper on its own, but both lenses carry it, so the relative
    rotation between them, M1 @ M0.T, is a proper rotation.

    A lens dict carrying an explicit "world_to_cam" matrix (written by
    84_refined_calib.py after bundle adjustment) uses that instead.
    """
    if "world_to_cam" in l:
        return np.asarray(l["world_to_cam"], dtype=float)
    az = math.radians(l["yaw_deg"])
    el = math.radians(90.0 - l["pitch_deg"])
    rl = math.radians(l["roll_deg"])
    z = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    x = np.array([math.sin(az), -math.cos(az), 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    xr = math.cos(rl) * x + math.sin(rl) * y
    yr = -math.sin(rl) * x + math.cos(rl) * y
    return np.diag([-1.0, 1.0, 1.0]) @ np.stack([xr, yr, z])


def quat_matrix(q):
    """Rotation matrix of a DJI Quaternion message: w, x, y, z (Hamilton)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rig_rotation(L):
    """lens1_from_lens0 in OpenCV camera frames, from the best source the calibration holds.

    1. explicit world_to_cam matrices -- a refined calibration (84_refined_calib.py)
    2. DewarpParams.cam_extri_q, DJI's stored extrinsic between camera modules.
       On the Osmo 360 garden clip Q1 Q0^T lands 0.15 deg from the bundle-adjusted
       rig, about the same axis, with no change of axes between module and camera
    3. the yaw/pitch/roll reading in world_to_cam, 1.65 deg off on that clip; the
       Avata 360 stores no cam_extri_q, so it starts here and bundle adjustment
       refines the rig
    """
    if "world_to_cam" in L[0] and "world_to_cam" in L[1]:
        return world_to_cam(L[1]) @ world_to_cam(L[0]).T
    if "cam_extri_q" in L[0] and "cam_extri_q" in L[1]:
        return quat_matrix(L[1]["cam_extri_q"]) @ quat_matrix(L[0]["cam_extri_q"]).T
    return world_to_cam(L[1]) @ world_to_cam(L[0]).T


def world_to_cams(L):
    """[world->cam0, world->cam1]: lens 0 defines the world frame, lens 1 hangs off it by rig_rotation."""
    M0 = world_to_cam(L[0])
    return [M0, rig_rotation(L) @ M0]


KB_EXTRA = ("k5", "k6", "k7", "k8", "k9")


def theta_d(theta, l):
    """DJI's Kannala-Brandt distorted angle, with every coefficient the calibration stores.

    theta_d = theta (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8 + k5 t^10 + ... + k9 t^18).
    DewarpParams carries k5..k9 in fields 15-19 (dvtm_library.proto) and the
    Osmo 360 fills k5. Leaving it out under-predicts the radius by ~115 px at
    90 deg off-axis, which an earlier reading here mistook for DJI's calibration
    rolling off too fast. Works on floats, numpy arrays and torch tensors.
    """
    t2 = theta * theta
    acc = 1 + l["k1"] * t2 + l["k2"] * t2 ** 2 + l["k3"] * t2 ** 3 + l["k4"] * t2 ** 4
    power = t2 ** 4
    for name in KB_EXTRA:
        power = power * t2
        c = l.get(name, 0.0)
        if c:
            acc = acc + c * power
    return theta * acc


def kb4_fit(l, max_deg=95.0, samples=4000):
    """k1..k4 reproducing this lens's full polynomial, for COLMAP's OPENCV_FISHEYE.

    COLMAP's model stops at k4. Dropping k5 instead of refitting throws away the
    rim; a least-squares fit over the angles the lens actually uses keeps it.
    """
    if not any(l.get(k) for k in KB_EXTRA):
        return [l["k1"], l["k2"], l["k3"], l["k4"]]
    th = np.linspace(1e-4, math.radians(max_deg), samples)
    A = np.stack([th ** 3, th ** 5, th ** 7, th ** 9], 1)
    k, *_ = np.linalg.lstsq(A, theta_d(th, l) - th, rcond=None)
    return [float(x) for x in k]


def colmap_params(l, fscale=1.0, max_deg=95.0):
    """OPENCV_FISHEYE parameters fx, fy, cx, cy, k1..k4 for this lens."""
    return [fscale * l["fx"], fscale * l["fy"], l["cx"], l["cy"], *kb4_fit(l, max_deg)]
