#!/usr/bin/env python3
"""Which SIFT backend a pycolmap build gets (no pycolmap needed: a stand-in is enough).

Runs anywhere:  python3 scripts/test_sift_backend.py
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sift_backend  # noqa: E402


class FakePycolmap:
    """The four things sift_backend touches."""

    class Device:
        cuda, cpu = "cuda", "cpu"

    def __init__(self, has_cuda, metal):
        self.has_cuda, self._metal = has_cuda, metal

    def Sift(self, device):
        if not self._metal:
            raise RuntimeError("no GPU extractor in this build")

    def FeatureExtractionOptions(self, **kw):
        return kw


class Choose(unittest.TestCase):
    def test_cuda_wins_then_metal_then_cpu(self):
        self.assertEqual(sift_backend.choose(True, False), "cuda")
        self.assertEqual(sift_backend.choose(True, True), "cuda")
        self.assertEqual(sift_backend.choose(False, True), "metal")
        self.assertEqual(sift_backend.choose(False, False), "cpu")

    def test_forced_backend_must_exist(self):
        self.assertEqual(sift_backend.choose(True, False, "cpu"), "cpu")
        self.assertEqual(sift_backend.choose(False, True, " Metal "), "metal")
        for has_cuda, has_metal, forced in ((False, True, "cuda"), (True, False, "metal"),
                                            (True, True, "opengl")):
            with self.assertRaises(SystemExit):
                sift_backend.choose(has_cuda, has_metal, forced)


class Options(unittest.TestCase):
    def options(self, has_cuda, metal, platform, forced=""):
        with mock.patch.object(sys, "platform", platform), \
                mock.patch.dict(os.environ, {"SPLAT_SIFT": forced}):
            return sift_backend.options(FakePycolmap(has_cuda, metal), 8)

    def test_cuda_call_is_what_it_always_was(self):
        # No device argument: Device.auto already means CUDA on that build, and
        # the Linux reconstruction must not change under the fisheye-sfm key.
        backend, extraction, kwargs, matching = self.options(True, False, "linux")
        self.assertEqual((backend, kwargs), ("cuda", {}))
        self.assertEqual(extraction, {"use_gpu": True, "num_threads": 8})
        self.assertEqual(matching, {"use_gpu": True, "num_threads": 8})

    def test_metal_extracts_on_the_gpu_and_matches_on_the_cpu(self):
        backend, extraction, kwargs, matching = self.options(False, True, "darwin")
        self.assertEqual(backend, "metal")
        self.assertTrue(extraction["use_gpu"])
        self.assertEqual(kwargs, {"device": "cuda"})   # the only way pycolmap leaves use_gpu on
        self.assertFalse(matching["use_gpu"])

    def test_stock_mac_wheel_is_cpu(self):
        backend, extraction, kwargs, matching = self.options(False, False, "darwin")
        self.assertEqual((backend, kwargs), ("cpu", {"device": "cpu"}))
        self.assertFalse(extraction["use_gpu"] or matching["use_gpu"])

    def test_metal_is_never_probed_off_a_mac(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertFalse(sift_backend.metal_available(FakePycolmap(False, True)))


if __name__ == "__main__":
    unittest.main()
