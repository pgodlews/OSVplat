#!/usr/bin/env python3
"""Which SIFT implementation this machine's pycolmap can run.

Three builds exist (docs/how-it-works.md, "Prep on Apple silicon"):
  cuda   pycolmap-cuda12 on Linux: SiftGPU extracts and matches on the GPU.
  metal  pycolmap built by scripts/setup_mac.sh: extraction in Metal compute
         shaders, matching on the CPU (FAISS). The build has a Metal matcher
         too; on a 946-image clip it took 309 s where the CPU took 236 s, for
         the same model (camera centres within 0.003 % of the path length).
  cpu    any other build, e.g. the PyPI macOS wheel: VLFeat, which is scalar on
         arm64. Same quality, about five times slower to extract.

They do not produce the same features, so the backend is part of what a
reconstruction is. SPLAT_SIFT=cuda|metal|cpu overrides the choice; asking for
one the build does not have is an error rather than a silent fallback.
"""
import os
import sys

BACKENDS = ("cuda", "metal", "cpu")


def choose(has_cuda, has_metal, forced=""):
    """The backend name for a build with these capabilities."""
    forced = (forced or "").strip().lower()
    if forced:
        if forced not in BACKENDS:
            raise SystemExit(f"SPLAT_SIFT={forced!r}: expected one of {', '.join(BACKENDS)}")
        if (forced == "cuda" and not has_cuda) or (forced == "metal" and not has_metal):
            raise SystemExit(f"SPLAT_SIFT={forced} but this pycolmap build has no {forced} SIFT")
        return forced
    return "cuda" if has_cuda else "metal" if has_metal else "cpu"


def metal_available(pycolmap):
    """True when this pycolmap can create a Metal SIFT extractor.

    COLMAP_build does not say (it reports CUDA and OpenGL only), so ask for an
    extractor: a build without Metal has no GPU extractor to hand back.
    """
    if sys.platform != "darwin" or pycolmap.has_cuda:
        return False
    try:
        pycolmap.Sift(device=pycolmap.Device.cuda)
    except Exception:  # noqa: BLE001  pycolmap raises its own types here
        return False
    return True


def options(pycolmap, threads):
    """(backend, extraction options, extract_features kwargs, matching options kwargs).

    pycolmap decides use_gpu from `device`, and Device.auto means "CUDA if the
    build has it": on a Metal build use_gpu=True alone is quietly turned off,
    so the device is passed explicitly there.
    """
    backend = choose(pycolmap.has_cuda, metal_available(pycolmap), os.environ.get("SPLAT_SIFT"))
    extraction = pycolmap.FeatureExtractionOptions(use_gpu=backend != "cpu", num_threads=threads)
    extract_kwargs = {"metal": {"device": pycolmap.Device.cuda},
                      "cpu": {"device": pycolmap.Device.cpu}}.get(backend, {})
    matching = {"use_gpu": backend == "cuda", "num_threads": threads}
    return backend, extraction, extract_kwargs, matching
