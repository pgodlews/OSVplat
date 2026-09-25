#!/usr/bin/env python3
"""Machine load counters (hoststats), the per-job time series, and the
/metrics families built from them. Fake /proc and cgroup trees; no GPU.

    python3 queue/test_hoststats.py
"""
import gzip
import json
import os
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

TMP = tempfile.TemporaryDirectory(prefix="queue_hoststats_")
# Assigned before any app import: config.py reads these at import time.
os.environ["SPLAT_ROOT"] = TMP.name
os.environ["QUEUE_ROOT"] = str(Path(TMP.name) / "queue")
os.environ["QUEUE_GPUS"] = ""
os.environ.pop("QUEUE_TELEMETRY", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import db, hoststats, metrics, telemetry             # noqa: E402

_READ = hoststats.read              # the real one, for when read() is patched


class FakeHost:
    """A /proc, /sys/fs/cgroup and /sys/block the counters can be moved in."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(dir=TMP.name))
        self.proc, self.cg, self.block = (self.root / "proc", self.root / "cgroup",
                                          self.root / "block")
        for d in (self.proc / "pressure", self.proc / "net", self.cg, self.block):
            d.mkdir(parents=True)
        for dev in ("nvme0n1", "sda", "loop0", "dm-0"):
            (self.block / dev).mkdir()
        self.set(0)

    def set(self, k: int) -> None:
        """Counters after k ticks: each tick is 100 jiffies per core."""
        # Two cores. Per tick, core 0: 60 user, 20 idle, 10 iowait, 10 steal;
        # core 1: 20 user, 80 idle.
        c0 = (60 * k, 0, 0, 20 * k, 10 * k, 0, 0, 10 * k)
        c1 = (20 * k, 0, 0, 80 * k, 0, 0, 0, 0)
        tot = tuple(a + b for a, b in zip(c0, c1))
        fmt = lambda n, v: f"{n} " + " ".join(map(str, v)) + " 0 0"      # noqa: E731
        (self.proc / "stat").write_text("\n".join(
            [fmt("cpu", tot), fmt("cpu0", c0), fmt("cpu1", c1), "intr 1 2 3", ""]))
        # 0.25 s of "some" CPU stall per tick, 0.5 s of IO.
        for r, us in (("cpu", 250_000), ("memory", 0), ("io", 500_000)):
            (self.proc / "pressure" / r).write_text(
                f"some avg10=1.00 avg60=0.00 avg300=0.00 total={us * k}\n"
                f"full avg10=0.00 avg60=0.00 avg300=0.00 total={us * k // 2}\n")
            (self.cg / f"{r}.pressure").write_text(
                f"some avg10=0.00 avg60=0.00 avg300=0.00 total={us * k // 10}\n")
        (self.cg / "cpu.stat").write_text(
            f"usage_usec {k}\nnr_periods {10 * k}\nnr_throttled {3 * k}\n"
            f"throttled_usec {200_000 * k}\n")
        # 1 MB read and 2 MB written per tick on each real disk; the
        # partition, loop and device-mapper lines must not be added.
        sec_r, sec_w = 1_000_000 // 512 * k, 2_000_000 // 512 * k
        line = lambda n: f"   259 0 {n} 1 0 {sec_r} 0 1 0 {sec_w} 0 0 0 0"  # noqa: E731
        (self.proc / "diskstats").write_text("\n".join(
            line(n) for n in ("nvme0n1", "nvme0n1p1", "sda", "loop0", "dm-0")) + "\n")
        (self.proc / "net" / "dev").write_text(
            "Inter-|   Receive\n face |bytes packets\n"
            f"    lo: {9_000_000 * k} 1 0 0 0 0 0 0 {9_000_000 * k} 1 0 0 0 0 0 0\n"
            f"  eth0: {3_000_000 * k} 1 0 0 0 0 0 0 {1_000_000 * k} 1 0 0 0 0 0 0\n")

    def read(self, t: float) -> dict:
        snap = _READ(self.proc, self.cg, self.block)
        snap["t"] = t
        return snap


class Counters(unittest.TestCase):
    def test_rates_between_two_snapshots(self):
        h = FakeHost()
        a = h.read(0.0)
        h.set(10)
        b = h.read(10.0)
        r = hoststats.rates(a, b, per_core=True)
        # 2000 jiffies: busy 800, idle 1000, iowait 100, steal 100.
        self.assertEqual((r["cpu_busy"], r["cpu_iowait"], r["cpu_steal"]), (0.4, 0.05, 0.05))
        self.assertEqual(r["core_busy_pct"], [60, 20])
        self.assertEqual(r["psi"], {"cpu_some": 0.25, "cpu_full": 0.125,
                                    "memory_some": 0.0, "memory_full": 0.0,
                                    "io_some": 0.5, "io_full": 0.25})
        self.assertEqual(r["cgroup_psi"]["io_some"], 0.05)
        self.assertEqual((r["cgroup_throttled"], r["cgroup_throttled_s"]), (0.3, 2.0))
        # nvme0n1 and sda only: 2 disks x 1 MB/tick, 10 ticks, 10 s.
        self.assertAlmostEqual(r["disk_read_mb_s"], 2.0, places=1)
        self.assertAlmostEqual(r["disk_write_mb_s"], 4.0, places=1)
        self.assertEqual((r["net_rx_mb_s"], r["net_tx_mb_s"]), (3.0, 1.0))   # lo left out
        self.assertNotIn("core_busy_pct", hoststats.rates(a, b))

    def test_missing_sources_are_none_not_errors(self):
        empty = Path(tempfile.mkdtemp(dir=TMP.name))
        snap = hoststats.read(empty, empty, empty)
        self.assertEqual({k for k, v in snap.items() if v and k != "t"},
                         {"psi", "cgroup_psi"})      # dicts of None per resource
        r = hoststats.rates(dict(snap, t=0), dict(snap, t=5))
        self.assertEqual(r, {"seconds": 5, "psi": None, "cgroup_psi": None})
        self.assertIsNone(hoststats.rates(None, snap))

    def test_cgroup_v1_throttling(self):
        root = Path(tempfile.mkdtemp(dir=TMP.name))
        (root / "cpu").mkdir()
        (root / "cpu" / "cpu.stat").write_text(
            "nr_periods 100\nnr_throttled 25\nthrottled_time 3000000000\n")
        self.assertEqual(hoststats._cgroup_cpu(root),
                         {"periods": 100, "throttled": 25, "throttled_us": 3_000_000})


class Series(unittest.TestCase):
    def setUp(self):
        db.init()
        db.conn().execute("DELETE FROM jobs")

    def test_lines_rates_and_bundle(self):
        h = FakeHost()
        jid = db.create_job("x", {"name": "x", "input": {"file": "samples/x.mp4"}})
        ticks = iter(range(1, 1000))
        clock = iter(float(x) for x in range(0, 5000, 5))

        def fake_read():
            h.set(next(ticks))
            return h.read(next(clock))
        s = telemetry.ResourceSampler(None, os.getpid(), interval=0.05,
                                      job_id=jid, stage="sfm")
        with patch.object(telemetry.hoststats, "read", fake_read):
            for _ in range(4):
                s.sample()
        lines = [json.loads(l) for l in gzip.open(telemetry.samples_path(jid)).read().splitlines()]
        self.assertEqual(len(lines), 3)              # the first sample is the baseline
        ln = lines[0]
        self.assertEqual((ln["stage"], ln["host"]["seconds"]), ("sfm", 5.0))
        self.assertEqual(ln["host"]["cpu_steal"], 0.05)
        self.assertEqual(ln["host"]["core_busy_pct"], [60, 20])
        self.assertIsNone(ln["gpu"])
        if Path("/proc").is_dir():
            self.assertIsNotNone(ln["cores_busy"])
        # The stage summary: the machine over the whole stage, no per-core.
        # Each tick here is 5 s of clock, so 0.5 s of IO stall is 0.1 of it.
        host = s.summary(15.0)["host"]
        self.assertEqual((host["seconds"], host["cpu_busy"], host["psi"]["io_some"]),
                         (15.0, 0.4, 0.1))
        self.assertNotIn("core_busy_pct", host)
        # Into the log bundle as it is.
        telemetry.write(jid, final=True)
        with tarfile.open(telemetry.logs_path(jid)) as tar:
            got = tar.extractfile("samples.jsonl.gz").read()
        self.assertEqual(len(gzip.decompress(got).splitlines()), 3)

    def test_a_torn_last_line_leaves_the_rest_readable(self):
        jid = db.create_job("y", {"name": "y", "input": {"file": "samples/y.mp4"}})
        s = telemetry.ResourceSampler(None, os.getpid(), job_id=jid, stage="train")
        s._host_prev = {"t": time.monotonic() - 5}
        s._series(0, {"t": time.monotonic()})
        p = telemetry.samples_path(jid)
        with open(p, "ab") as f:                      # a write killed half-way
            f.write(gzip.compress(b'{"t": 1}\n')[:12])
        good = []
        with gzip.open(p) as f:
            try:
                for line in f:
                    good.append(json.loads(line))
            except EOFError:
                pass
        self.assertEqual(len(good), 1)

    def test_no_series_without_a_job_or_with_telemetry_off(self):
        s = telemetry.ResourceSampler(None, os.getpid())
        s.sample()
        s.sample()
        self.assertTrue(s._series_off)
        jid = db.create_job("z", {"name": "z", "input": {"file": "samples/z.mp4"}})
        with patch.object(telemetry, "TELEMETRY_ENABLED", False):
            s = telemetry.ResourceSampler(None, os.getpid(), job_id=jid, stage="sfm")
        s.sample()
        s.sample()
        self.assertFalse(telemetry.samples_path(jid).exists())

    def test_a_write_that_fails_is_logged_once_and_stops(self):
        jid = db.create_job("w", {"name": "w", "input": {"file": "samples/w.mp4"}})
        s = telemetry.ResourceSampler(None, os.getpid(), job_id=jid, stage="sfm")
        with patch.object(telemetry.gzip, "open", side_effect=OSError("read-only")):
            for _ in range(4):
                s.sample()
        self.assertTrue(s._series_off)


class Metrics(unittest.TestCase):
    def setUp(self):
        db.init()

    def test_host_counters_and_gpu_health(self):
        h = FakeHost()
        h.set(10)
        row = ["0", "97", "4000", "301.5", "1890", "9501", "71", "4", "4",
               "[N/A]", "0x0000000000000004"]
        with patch.object(metrics.hoststats, "read", lambda: h.read(0)), \
                patch.object(metrics.telemetry, "_gpu_sample_rows", lambda: [row]):
            out = metrics.render([])
        want = [
            'splatqueue_host_cpu_seconds_total{mode="steal"} 1',     # 100 jiffies / 100 Hz
            'splatqueue_host_pressure_stalled_seconds_total{scope="system",resource="io",kind="some"} 5',
            "splatqueue_cgroup_cpu_throttled_periods_total 30",
            "splatqueue_cgroup_cpu_throttled_seconds_total 2",
            "splatqueue_host_network_receive_bytes_total 30000000",
            'splatqueue_gpu_power_watts{gpu="0"} 301.5',
            'splatqueue_gpu_sm_clock_hertz{gpu="0"} 1890000000',
            'splatqueue_gpu_pcie_link_width{gpu="0"} 4',
            'splatqueue_gpu_clock_event_reason{gpu="0",reason="sw_power_cap"} 1',
            'splatqueue_gpu_clock_event_reason{gpu="0",reason="hw_slowdown"} 0',
        ]
        if os.sysconf("SC_CLK_TCK") != 100:
            want = want[1:]
        for w in want:
            self.assertIn(w, out)
        self.assertNotIn("gpu_ecc_uncorrected_errors", out)    # [N/A]: absent
        self.assertNotIn("nvme", out)
        self.assertNotIn("eth0", out)

    def test_power_limit_gauge(self):
        row = ["0", "97", "4000", "149.5", "510", "9501", "60", "4", "16",
               "[N/A]", "0x0000000000000004"]
        with patch.object(metrics.telemetry, "_gpu_sample_rows", lambda: [row]):
            self.assertNotIn("gpu_power_limit_watts", metrics.render([]))   # driver without it
        with patch.object(metrics.telemetry, "_gpu_sample_rows", lambda: [row + ["150.00"]]):
            self.assertIn('splatqueue_gpu_power_limit_watts{gpu="0"} 150', metrics.render([]))

    def test_disk_space_gauges(self):
        with patch.object(metrics.telemetry, "_disk_space", lambda: (6_000, 44_000, 50_000)):
            out = metrics.render([])
        self.assertIn("splatqueue_queue_root_used_bytes 6000", out)
        self.assertIn("splatqueue_queue_root_free_bytes 44000", out)

    def test_old_driver_rows_are_skipped(self):
        with patch.object(metrics.telemetry, "_gpu_sample_rows", lambda: [["0", "50", "100"]]):
            out = metrics.render([])
        self.assertNotIn("gpu_power_watts", out)


if __name__ == "__main__":
    unittest.main(verbosity=1)
