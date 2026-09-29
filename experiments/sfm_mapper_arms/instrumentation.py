"""Instrumentation scoped to this executable; shipped mapper is never edited.

The two refinement loops follow COLMAP 4.2.0 incremental_mapper.cc exactly
(except logging and cancellation, which the 4.2 Python options do not expose).
All registration, BA, filtering and triangulation remain native calls.
Upstream: https://github.com/colmap/colmap/blob/4.2.0/src/colmap/sfm/incremental_mapper.cc

Adapted from COLMAP under the following licence:
Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of ETH Zurich and UNC Chapel Hill nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from contextlib import contextmanager
from copy import deepcopy
import os
import re
import sys
from time import perf_counter

import pycolmap
from common import mirror


class NativeLog:
    """Capture native fd 2, including VLOG(1)'s exact redundant-point counts.

    File offsets delimit each call; pread does not disturb the writer's offset.
    Single-process CLI only: this is intentionally not thread-safe/reentrant.
    """
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.file = open(self.path, "w+b", buffering=0)
        sys.stderr.flush()
        self.saved_fd = os.dup(2)
        self.settings = {k: getattr(pycolmap.logging, k) for k in
                         ("verbose_level", "logtostderr", "logtostdout", "minloglevel")}
        os.dup2(self.file.fileno(), 2)
        pycolmap.logging.logtostderr = True
        pycolmap.logging.logtostdout = False
        pycolmap.logging.minloglevel = 0
        pycolmap.logging.verbose_level = 1
        return self

    def position(self):
        return os.fstat(self.file.fileno()).st_size

    def since(self, offset):
        return os.pread(self.file.fileno(), self.position() - offset, offset).decode(
            "utf-8", errors="replace")

    def __exit__(self, *exc):
        sys.stderr.flush()
        os.dup2(self.saved_fd, 2)
        os.close(self.saved_fd)
        for k, v in self.settings.items():
            setattr(pycolmap.logging, k, v)
        self.file.close()


def solver_reports(log):
    # Native summary time is rounded by COLMAP, separate from our wall timer.
    return [{"residuals_reduced": int(n), "solver_reported_s": float(t)}
            for n, t in re.findall(
                r"Residuals\s*:\s*(\d+).*?Time\s*:\s*([\d.eE+-]+)\s*\[s\]", log, re.S)]


class TimedMapper:
    def __init__(self, native, arm, log, events):
        self.native, self.arm, self.log, self.events = native, arm, log, events

    def __getattr__(self, key):
        return getattr(self.native, key)

    def adjust_global_bundle(self, options, ba_options):
        rec = self.reconstruction
        event = {"reg_frames": rec.num_reg_frames(), "reg_images": rec.num_reg_images(),
                 "points_before": rec.num_points3D()}
        pos = self.log.position()
        start = perf_counter()
        usable = self.native.adjust_global_bundle(options, ba_options)
        event["wall_s"] = perf_counter() - start
        log = self.log.since(pos)
        redundant = re.search(r"Ignoring (\d+) / (\d+) redundant 3D points", log)
        active = options.ba_global_ignore_redundant_points3D and event["reg_frames"] >= 10
        if active and redundant is None:
            raise RuntimeError("Missing redundant-point VLOG; cannot report this arm reliably")
        event.update({"ignored_redundant_points": int(redundant[1]) if redundant else 0,
                      "points_at_selection": int(redundant[2]) if redundant else None,
                      "points_after": rec.num_points3D(), "solution_usable": usable,
                      "solves": solver_reports(log)})
        self.events["global"].append(event)
        return usable

    def iterative_global_refinement(self, max_refinements, max_change, options,
                                    ba_options, tri_options, normalize):
        self.complete_and_merge_tracks(tri_options)
        self.retriangulate(tri_options)
        for _ in range(max_refinements):
            num_obs = self.reconstruction.compute_num_observations()
            self.adjust_global_bundle(options, ba_options)
            # All arms use 82's use_prior_position=False. That mapper flag
            # is not exposed on IncrementalMapperOptions in the 4.2 binding.
            if normalize:
                self.reconstruction.normalize()
            changed = self.complete_and_merge_tracks(tri_options)
            changed += self.filter_points(options)
            if (changed / num_obs if num_obs else 0) < max_change:
                break
        self.clear_modified_points3D()

    def local_residual_estimate(self, options, image_id, modified_points):
        # Read-only counterpart of AdjustLocalBundle's problem membership.
        # Count both lenses of every selected frame, plus observations outside
        # the bundle of new / <=15-observation variable points. The temporary
        # config is never used to solve or modify the reconstruction.
        bundle = self.find_local_bundle(options, image_id)
        if not bundle:
            return 0
        config = pycolmap.BundleAdjustmentConfig()
        for iid in [image_id, *bundle]:
            for data in self.reconstruction.images[iid].frame.image_ids:
                config.add_image(data.id)
        for pid in modified_points:
            point = self.reconstruction.points3D[pid]
            if not point.has_error() or point.track.length() <= 15:
                config.add_variable_point(pid)
        return config.num_residuals(self.reconstruction)

    def iterative_local_refinement(self, max_refinements, max_change, options,
                                   ba_options, tri_options, image_id):
        ba_options = deepcopy(ba_options)
        if self.arm == "localmt":
            ba_options.ceres.min_num_residuals_for_cpu_multi_threading = 5000
        for iteration in range(max_refinements):
            points = self.get_modified_points3D()
            estimate_start = perf_counter()
            residuals = self.local_residual_estimate(options, image_id, points)
            estimate_s = perf_counter() - estimate_start
            pos = self.log.position()
            start = perf_counter()
            report = self.native.adjust_local_bundle(options, ba_options, tri_options,
                                                     image_id, points)
            wall = perf_counter() - start
            self.events["local"].append({
                "image_id": image_id, "iteration": iteration, "wall_s": wall,
                "residuals_estimate": residuals, "estimate_wall_s": estimate_s,
                "residuals_reduced": 2 * report.num_adjusted_observations,
                "threading_threshold": ba_options.ceres.min_num_residuals_for_cpu_multi_threading,
                "solves": solver_reports(self.log.since(pos)),
            })
            changed = (report.num_merged_observations + report.num_completed_observations
                       + report.num_filtered_observations)
            if (changed / report.num_adjusted_observations
                    if report.num_adjusted_observations else 0) < max_change:
                break
            ba_options.ceres.loss_function_type = pycolmap.LossFunctionType.TRIVIAL
        self.clear_modified_points3D()


@contextmanager
def instrument(arm, log, events):
    original = mirror.IncrementalMapper
    mirror.IncrementalMapper = lambda cache: TimedMapper(original(cache), arm, log, events)
    try:
        yield
    finally:
        mirror.IncrementalMapper = original


def timing_summary(events):
    local = events["local"]
    bins = [0, 5000, 10000, 25000, 50000, 100000]
    histogram = []
    for i, low in enumerate(bins):
        high = bins[i + 1] if i + 1 < len(bins) else None
        histogram.append({"min_inclusive": low, "max_exclusive": high,
                          "calls": sum(e["residuals_estimate"] >= low and
                                       (high is None or e["residuals_estimate"] < high) for e in local)})
    return {"global_call_count": len(events["global"]),
            "global_wall_s": sum(e["wall_s"] for e in events["global"]),
            "local_call_count": len(local),
            "local_wall_s": sum(e["wall_s"] for e in local),
            "local_residual_estimation_s": sum(e["estimate_wall_s"] for e in local),
            "local_residual_histogram": histogram,
            "local_share_under_50000": (sum(e["residuals_estimate"] < 50000 for e in local) / len(local)
                                        if local else None)}
