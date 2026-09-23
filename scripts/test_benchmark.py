#!/usr/bin/env python3
"""scripts/benchmark.py at --quick sizes: it runs, and says what it measured.

Needs numpy and OpenCV (venv_gs has them); the GPU part runs only where
torch sees a CUDA device and gsplat is installed, and is skipped otherwise.
The scores at --quick sizes mean nothing; only their shape is checked.

    ~/splat/venv_gs/bin/python scripts/test_benchmark.py
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def has_cuda() -> bool:
    try:
        import gsplat  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except Exception:                                         # noqa: BLE001
        return False


class Quick(unittest.TestCase):
    def run_bench(self, *extra):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            p = subprocess.run([sys.executable, str(HERE / "benchmark.py"), "--quick",
                                "--out", str(out), "--scratch", d, "--seconds", "0.2",
                                "--disk-gb", "0.02", "--cpus", "2", *extra],
                               capture_output=True, text=True, timeout=300)
            rec = json.loads(out.read_text())
            left = [x.name for x in Path(d).iterdir() if x.name != "r.json"]
        return p, rec, left

    def test_cpu_memory_disk(self):
        p, rec, left = self.run_bench("--no-gpu")
        self.assertEqual(p.returncode, 0, p.stdout[-2000:])
        self.assertEqual(rec["errors"], {})
        cpu = rec["cpu"]
        for k in ("jpeg_decode_1t_per_s", "jpeg_decode_all_per_s", "jpeg_encode_1t_per_s",
                  "sift_1t_per_s", "sift_all_per_s", "sift_match_1t_per_s"):
            self.assertGreater(cpu[k], 0, k)
        self.assertEqual(cpu["processes"], 2)
        self.assertGreater(cpu["sift_features"], 100)
        self.assertGreater(rec["memory"]["copy_gb_s"], 0)
        self.assertGreater(rec["disk"]["write_fsync_mb_s"], 0)
        self.assertGreater(rec["disk"]["read_mb_s"], 0)
        self.assertNotIn("gpu", rec)
        self.assertEqual(left, [])                  # the disk test cleans up

    def test_disk_refuses_without_space(self):
        # Asking for more than any test machine has: the part fails loudly,
        # the others still run, and the exit code says so.
        p, rec, _ = self.run_bench("--no-gpu", "--disk-gb", "1000000")
        self.assertEqual(p.returncode, 1)
        self.assertIn("free under the scratch directory", rec["errors"]["disk"])
        self.assertIsNone(rec["disk"])
        self.assertGreater(rec["memory"]["copy_gb_s"], 0)

    @unittest.skipUnless(has_cuda(), "no CUDA device or no gsplat here")
    def test_gpu(self):
        p, rec, _ = self.run_bench()
        self.assertEqual(p.returncode, 0, p.stdout[-2000:])
        g = rec["gpu"]
        for k in ("matmul_fp32_tflops", "matmul_fp16_tflops", "d2d_copy_gb_s",
                  "h2d_pinned_gb_s", "d2h_pinned_gb_s", "gsplat_it_s"):
            self.assertGreater(g[k], 0, k)
        self.assertTrue(g["device"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
