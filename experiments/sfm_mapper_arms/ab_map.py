#!/usr/bin/env python3
"""Mapping-only CPU arms; --db --images --out --arm, matching ab_global.py."""
import argparse
import hashlib
import sqlite3
from time import perf_counter
from pathlib import Path

import pycolmap
from common import (ARMS, mirror, model_stats, new_output, options_for,
                    threads_default, write_json)
from instrumentation import NativeLog, instrument, timing_summary


def run(db, images, out, arm, threads, *, instrumented=True, refine_intrinsics=True):
    db, images = Path(db).resolve(), Path(images).resolve()
    if not db.is_file() or not images.is_dir():
        raise ValueError("--db must exist and --images must be a directory")
    opts = options_for(arm, threads, refine_intrinsics)
    out = new_output(out)
    # Consistent snapshot, including committed WAL content, without modifying source.
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(out / "database.db") as dest:
            source.backup(dest)
    database_hash = hashlib.sha256()
    with (out / "database.db").open("rb") as snapshot:
        for chunk in iter(lambda: snapshot.read(1024 * 1024), b""):
            database_hash.update(chunk)
    local_opts = opts.get_local_bundle_adjustment()
    if arm == "localmt":
        local_opts.ceres.min_num_residuals_for_cpu_multi_threading = 5000
    events = {"global": [], "local": []}
    report = {"arm": arm, "threads": threads, "instrumented": instrumented,
              "refine_intrinsics": refine_intrinsics,
              "pycolmap_version": pycolmap.__version__, "build": pycolmap.COLMAP_build,
              "database_snapshot_sha256": database_hash.hexdigest(),
              "mapper_sha256": hashlib.sha256(Path(mirror.__file__).read_bytes()).hexdigest(),
              "pipeline_options": opts.todict(), "local_ba_options": local_opts.todict(),
              "global_ba_options": mirror.global_bundle_adjustment_options(
                  opts, mirror.DIRECT_SOLVER_MAX_IMAGES).todict()}
    # pybind enums/paths become strings; retain full settings for audit/replay.
    import json
    report = json.loads(json.dumps(report, default=str))
    try:
        with NativeLog(out / "mapping.log") as log:
            start = perf_counter()
            if instrumented:
                with instrument(arm, log, events):
                    recs = mirror.incremental_mapping(out / "database.db", images,
                                                      out / "sparse", opts)
            else:
                if arm == "localmt":
                    raise ValueError("localmt exists only in the instrumented loops; no native run")
                recs = mirror.incremental_mapping(out / "database.db", images,
                                                  out / "sparse", opts)
            elapsed = perf_counter() - start
        if not recs or max(r.num_reg_frames() for r in recs.values()) < 3:
            raise RuntimeError("Mapping produced no useful model (need >=3 rig frames)")
        if any(r.num_points3D() == 0 for r in recs.values()):
            raise RuntimeError("Mapping produced an empty point cloud")
        report.update({"status": "ok", "map_s": elapsed,
                       "models": [dict(name=str(i), **model_stats(r)) for i, r in
                                  sorted(recs.items(), key=lambda pair: -pair[1].num_reg_images())]})
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["timings"] = timing_summary(events)
        report["passes"] = events
        write_json(out / "report.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="New directory; existing output is refused")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--threads", type=int, default=threads_default())
    ap.add_argument("--native", action="store_true",
                    help="no instrumentation: native COLMAP refinement loops, as production runs them")
    ap.add_argument("--fixed-intrinsics", action="store_true",
                    help="82's standalone default; production (88) refines focal + k1-k4")
    a = ap.parse_args()
    report = run(a.db, a.images, a.out, a.arm, a.threads, instrumented=not a.native,
                 refine_intrinsics=not a.fixed_intrinsics)
    print(f"STAGE map {report['map_s']:.3f}s; report: {Path(a.out) / 'report.json'}")


if __name__ == "__main__":
    main()
