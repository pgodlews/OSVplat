#!/usr/bin/env python3
"""COLMAP 4.2 panorama SfM via pycolmap-cuda12 (venv/).

usage: 30_run_sfm.py [perspective_overlapping|perspective_non_overlapping|spherical]
                     [incremental|global] [panos_dir] [out_root] [--masks DIR]

perspective_overlapping = 12 x 90-deg pinhole views per panorama (4 yaw x pitch
-35/0/35) locked as a rig; GPU SIFT; sequential matching.
spherical = features on the equirect image itself with the EQUIRECTANGULAR
camera model (faster, less accurate).

Outputs: <out>/images/, masks/, sparse/0 (rig, PINHOLE) and
sparse_equirectangular/0 (same poses, one EQUIRECTANGULAR camera).

--masks DIR excludes regions from feature extraction (spherical only). Masks are
named COLMAP-style, `<image name>.png` -- so `pano_0000.jpg.png`, NOT
`pano_0000.png`, which is what LichtFeld wants. 70_person_masks.py writes both
layouts into separate directories for exactly this reason; a mask COLMAP cannot
find is silently ignored and the run looks normal.
"""
import argparse
import logging
import time
from pathlib import Path

import pycolmap
from pycolmap.panorama import (Mapper, Matcher, PanoRenderType,
                               PanoramaReconstructionOptions, reconstruct,
                               run_matcher)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

ap = argparse.ArgumentParser()
ap.add_argument("render", nargs="?", default="perspective_overlapping")
ap.add_argument("mapper", nargs="?", default="incremental")
ap.add_argument("panos", nargs="?", default="run1/panos")
ap.add_argument("out_root", nargs="?", default="run1")
ap.add_argument("--masks", default="",
                help="directory of COLMAP-named masks (<image name>.png), "
                     "black = ignore. Spherical render only.")
args = ap.parse_args()

render, mapper = args.render, args.mapper
panos = Path(args.panos).expanduser()
out = Path(args.out_root).expanduser() / f'sfm_{render}_{mapper}'
out.mkdir(parents=True, exist_ok=True)

masks = Path(args.masks).expanduser() if args.masks else None
if masks is not None:
    if render != "spherical":
        # The perspective path generates its OWN masks, one per virtual camera,
        # to stop a pano pixel being extracted in more than one view. Pointing
        # mask_path elsewhere would overwrite that and break the rig.
        raise SystemExit("--masks is only supported for the spherical render")
    if not masks.is_dir():
        raise SystemExit(f"--masks directory does not exist: {masks}")
    n_masks = len(list(masks.glob("*.png")))
    n_panos = len(list(panos.glob("*.jpg")))
    if n_masks < n_panos:
        # COLMAP treats a missing mask as "extract everywhere". Half a masked
        # run is not a masked run, and nothing downstream would report it.
        raise SystemExit(
            f"only {n_masks} masks for {n_panos} panoramas in {masks}; "
            f"refusing to run a partially masked reconstruction")

options = PanoramaReconstructionOptions(
    matcher=Matcher.SEQUENTIAL, mapper=Mapper(mapper),
    render_type=PanoRenderType(render),
    gpu_index='0', use_gpu=True, num_threads=-1)

logging.info(f'START render={render} mapper={mapper} '
             f'panos={len(list(panos.glob("*.jpg")))} cuda={pycolmap.has_cuda} '
             f'masks={masks or "none"}')
t = time.time()


def run_spherical_masked(input_image_path, opts, database_path, rec_path,
                         mask_dir):
    """pycolmap.panorama.run_spherical, with masks fed to feature extraction.

    Kept as a near-copy rather than a wrapper because the upstream function
    builds ImageReaderOptions itself and exposes no way to add a mask path;
    PanoramaReconstructionOptions has no mask field either. Everything else --
    the single shared EQUIRECTANGULAR camera, no rig, the matcher, the mapper
    options -- is deliberately identical to upstream so masked and unmasked runs
    stay comparable. Re-check this against panorama.py after a pycolmap upgrade.
    """
    pycolmap.set_random_seed(opts.random_seed)
    reader_options = pycolmap.ImageReaderOptions(
        camera_model="EQUIRECTANGULAR", mask_path=str(mask_dir))
    extraction_options = pycolmap.FeatureExtractionOptions(
        use_gpu=opts.use_gpu, gpu_index=opts.gpu_index,
        num_threads=opts.num_threads)
    pycolmap.extract_features(
        database_path, input_image_path,
        reader_options=reader_options,
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=extraction_options)

    run_matcher(opts, database_path, pycolmap.FeatureMatchingOptions())

    if opts.mapper == Mapper.INCREMENTAL:
        recs = pycolmap.incremental_mapping(
            database_path, input_image_path, rec_path,
            pycolmap.IncrementalPipelineOptions(
                num_threads=opts.num_threads, random_seed=opts.random_seed))
    elif opts.mapper == Mapper.GLOBAL:
        recs = pycolmap.global_mapping(
            database_path, input_image_path, rec_path,
            pycolmap.GlobalPipelineOptions(
                num_threads=opts.num_threads, random_seed=opts.random_seed))
    else:
        raise ValueError(f"Unknown mapper: {opts.mapper}")
    return recs


if masks is not None:
    database_path = out / "database.db"
    if database_path.exists():
        database_path.unlink()
    rec_path = out / "sparse"
    rec_path.mkdir(exist_ok=True, parents=True)
    recs = run_spherical_masked(panos, options, database_path, rec_path, masks)
else:
    recs = reconstruct(panos, out, options)

logging.info(f'TOTAL {time.time()-t:.0f}s')
for i, r in recs.items():
    logging.info(f'model {i}: {r.summary()}')
