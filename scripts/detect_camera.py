#!/usr/bin/env python3
"""Detect whether a 360 video file is an Osmo 360, Avata 360, or generic video.

Usage:
  scripts/detect_camera.py <file1> [file2 ...] [--json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add repo root to import path if needed
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.osv_meta import detect_camera


def format_human(path: str, info: dict) -> str:
    lines = [f"File: {os.path.basename(path)}"]
    cam = info.get("camera", "unknown")
    if cam == "osmo360":
        lines.append(f"  Camera:           {info.get('camera_name')} (Handheld / Stick)")
        lines.append(f"  Model / Proto:    {info.get('model')} ({info.get('proto')})")
        lines.append(f"  IMU Rate:         ~{info.get('imu_rate_hz')} Hz")
        lines.append(f"  GPS:              No")
        lines.append(f"  Recommended Mask: ON (operator in frame)")
    elif cam == "avata360":
        lines.append(f"  Camera:           {info.get('camera_name')} (Aerial Drone)")
        lines.append(f"  Model / Proto:    {info.get('model')} ({info.get('proto')})")
        lines.append(f"  IMU Rate:         ~{info.get('imu_rate_hz')} Hz")
        lines.append(f"  GPS:              Yes (embedded fixes + .SRT)")
        lines.append(f"  Recommended Mask: OFF (drone; enable if flying near people)")
    else:
        lines.append(f"  Camera:           {info.get('camera_name', 'Unknown / Generic')}")
        lines.append(f"  Is Fisheye:       {info.get('is_fisheye', False)}")
        lines.append(f"  Recommended Mask: OFF")
    if info.get("firmware"):
        lines.append(f"  Firmware:         {info.get('firmware')}")
    if info.get("serial"):
        lines.append(f"  Serial:           {info.get('serial')}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Detect camera type for 360 footage")
    parser.add_argument("files", nargs="+", help="Video files (.OSV, .LRF, .mp4, etc.)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    results = {}
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"Error: file not found: {f}", file=sys.stderr)
            continue
        info = detect_camera(p)
        results[str(p)] = info

    if args.json:
        if len(args.files) == 1 and len(results) == 1:
            first_val = next(iter(results.values()), {})
            print(json.dumps(first_val, indent=2))
        else:
            print(json.dumps(results, indent=2))
    else:
        for p, info in results.items():
            print(format_human(p, info))
            print()


if __name__ == "__main__":
    main()
