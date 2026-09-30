"""ffprobe of an input clip, cached: the API (plans, the input list) and the
worker (space estimates for the prep stages) both need it."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

# Keyed by path and mtime, so it never serves a stale probe -- but every new
# clip (and every re-encode of one) adds an entry that is never read again.
_probe_cache: dict[str, dict] = {}
PROBE_CACHE_MAX = 64


def ffprobe(path: Path) -> dict:
    key = f"{path}:{path.stat().st_mtime_ns}"
    if key in _probe_cache:
        return _probe_cache[key]
    info: dict[str, Any] = {"duration": None, "width": None, "height": None,
                            "fps": None, "codec": None}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=width,height,codec_name,avg_frame_rate:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            j = json.loads(out.stdout)
            st = (j.get("streams") or [{}])[0]
            info["width"], info["height"] = st.get("width"), st.get("height")
            info["codec"] = st.get("codec_name")
            fr = st.get("avg_frame_rate", "0/1")
            if "/" in fr:
                n, d = fr.split("/")
                info["fps"] = round(float(n) / float(d), 3) if float(d) else None
            dur = (j.get("format") or {}).get("duration")
            info["duration"] = round(float(dur), 2) if dur else None
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError):
        pass
    if len(_probe_cache) >= PROBE_CACHE_MAX:
        _probe_cache.clear()
    _probe_cache[key] = info
    return info
