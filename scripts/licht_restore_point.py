#!/usr/bin/env python3
"""One-checkpoint restore point from a training run's project.licht (issue #10).

    licht_restore_point.py <project.licht> <out.licht>

A training project keeps every snapshot's checkpoint live, so it grows by about
half a GB per snapshot at 3M splats. LichtFeld's own clean_project_file keeps
only the bound (latest) checkpoint and compacts: 1.56 GB -> 467 MB measured.
It takes a writer lock on its input, and the trainer is still writing the live
file, so it works on a copy. The result has to verify, or nothing is written.

Runs under the python3 the trainer's `lichtfeld` module was built for, with
PYTHONPATH=<build>/src/python and LD_LIBRARY_PATH=<build>; the module links
libcuda, so a GPU driver has to be visible (no GPU work is done). Prints one
JSON line on success; exits non-zero with a message otherwise.
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def remove(*paths: Path) -> None:
    for p in paths:
        for q in (p, Path(str(p) + ".lock")):
            try:
                q.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        import lichtfeld.io as io
    except ImportError as exc:
        print(f"cannot import the trainer's lichtfeld module: {exc}", file=sys.stderr)
        return 1
    if not src.is_file():
        print(f"no project at {src}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.with_name(out.name + ".src.licht")
    tmp = out.with_name(out.name + ".part.licht")
    remove(work, tmp)
    try:
        t0 = time.monotonic()
        shutil.copyfile(src, work)
        copy_s = time.monotonic() - t0
        t0 = time.monotonic()
        io.clean_project_file(str(work), str(tmp))
        clean_s = time.monotonic() - t0
        t0 = time.monotonic()
        v = io.verify_project_file(str(tmp))
        verify_s = time.monotonic() - t0
        status = str(getattr(v, "status", v))
        if not status.endswith("VERIFIED"):
            print(f"restore point does not verify: {status}, first mismatch "
                  f"{getattr(v, 'first_mismatch', None)}", file=sys.stderr)
            return 1
        source_bytes = work.stat().st_size
        remove(work)
        os.replace(tmp, out)
    except Exception as exc:                                  # noqa: BLE001
        print(f"restore point failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        remove(work, tmp)
    print(json.dumps({"bytes": out.stat().st_size, "sha256": sha256(out),
                      "source_bytes": source_bytes,
                      "chunks": getattr(v, "verified_chunks", None),
                      "copy_s": round(copy_s, 2), "clean_s": round(clean_s, 2),
                      "verify_s": round(verify_s, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
