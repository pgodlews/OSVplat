"""The .OSV readers in scripts/ that the API calls itself.

osv_meta.detect_camera (which camera made a clip, for the masking default) and
avata_motion (Avata 360 velocity, for distance selection) are stdlib-only, so
they run in the service's own venv; AGENTS.md's rule keeps numpy and OpenCV
out of it, not these.

They are found in SPLAT_ROOT/scripts, where deploy.sh, run_mac.sh and the
image put the pipeline scripts, or else in the checkout this package sits in.
A failed import used to be silent: Osmo clips quietly stopped defaulting to
masks and a distance-selection submit became a 500. Now ERROR says why, the
service prints it at startup, and the routes that need them answer 400.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .config import SCRIPTS

CANDIDATES = (SCRIPTS, Path(__file__).resolve().parent.parent.parent / "scripts")

ERROR: str | None = None
detect_camera = None
avata_motion = None

_dir = next((d for d in CANDIDATES if (d / "osv_meta.py").is_file()), None)
if _dir is None:
    ERROR = (f"the .OSV readers (osv_meta.py, avata_motion.py) are in neither "
             f"{' nor '.join(str(d) for d in CANDIDATES)}")
else:
    # Appended, not prepended: nothing in scripts/ may shadow a module the
    # service imports by the same name.
    if str(_dir) not in sys.path:
        sys.path.append(str(_dir))
    try:
        import avata_motion  # noqa: E402
        import osv_meta as _osv_meta  # noqa: E402
        detect_camera = _osv_meta.detect_camera
    except Exception as exc:                                  # noqa: BLE001
        avata_motion = None
        ERROR = f"cannot import the .OSV readers from {_dir}: {type(exc).__name__}: {exc}"


def need_avata_motion():
    """The module, or ValueError with why it is missing."""
    if avata_motion is None:
        raise ValueError(f"distance selection is unavailable: {ERROR}")
    return avata_motion
