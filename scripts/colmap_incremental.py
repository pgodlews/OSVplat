"""COLMAP's incremental mapper with a configurable direct-solver limit.

COLMAP's IncrementalPipelineOptions never sets
CeresBundleAdjustmentOptions.max_num_images_direct_sparse_cpu_solver (1000), so
global bundle adjustment switches from SPARSE_SCHUR to ITERATIVE_SCHUR once the
model holds 1000 images -- 500 frames of a two-lens rig. On queue job 211 (1404
Osmo 360 rig frames) every global pass past that point took about two hours
(109, 123, 138, 124 min) and the mapper was on course for more than a day. With
the limit lifted the same database mapped all 1404 frames in 4 h 57 min. On the
finished model one global bundle adjustment from the same perturbed start took
6.6 min direct (converged, cost 0.816539) against 56 min iterative (stopped at
the iteration cap, cost 0.816784): faster and no worse.

This is COLMAP 4.2.0's python/examples/custom_incremental_pipeline.py, the
Python mirror of IncrementalPipeline::Run, with three changes: local and global
refinement call the C++ IncrementalMapper methods instead of the example's
Python bundle adjustment, every global bundle adjustment gets the lifted limit,
and snapshots and the progress bar are left out. On a 67-frame clip it
reproduced the C++ mapper (49,221 vs 49,230 points, 0.8533 vs 0.8535 px) in the
same time.

The mirror follows the pycolmap API it was copied from. On any other pycolmap
version incremental_mapping() falls back to pycolmap.incremental_mapping, cliff
and all, and says so in the log.

Adapted from COLMAP (https://github.com/colmap/colmap), under its licence:

    Copyright (c), ETH Zurich and UNC Chapel Hill.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.

        * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
          its contributors may be used to endorse or promote products derived
          from this software without specific prior written permission.

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
from pathlib import Path

import pycolmap
from pycolmap import (
    IncrementalMapper,
    IncrementalMapperOptions,
    IncrementalPipeline,
    IncrementalPipelineCallback,
    IncrementalPipelineOptions,
    IncrementalPipelineStatus,
    Reconstruction,
    ReconstructionManager,
    logging,
)

MIRRORED_VERSION = "4.2."
# Large enough that a clip never reaches it; memory is what bounds the direct
# solver, and 2808 images peaked at 12.7 GB for the whole mapper.
DIRECT_SOLVER_MAX_IMAGES = 1_000_000


def global_bundle_adjustment_options(options: IncrementalPipelineOptions,
                                     max_images: int) -> pycolmap.BundleAdjustmentOptions:
    ba_options = options.get_global_bundle_adjustment()
    ba_options.ceres.max_num_images_direct_sparse_cpu_solver = max_images
    return ba_options


def has_unknown_sensor_from_rig(reconstruction: Reconstruction) -> bool:
    parameterized_rig_ids = set()
    for image in reconstruction.images.values():
        parameterized_rig_ids.add(image.frame.rig_id)
    for rig_id in parameterized_rig_ids:
        rig = reconstruction.rig(rig_id)
        for sensor_id, sensor_from_rig in rig.non_ref_sensors.items():
            if sensor_id.type == pycolmap.SensorType.CAMERA and sensor_from_rig is None:
                return True
    return False


def iterative_global_refinement(options: IncrementalPipelineOptions,
                                mapper_options: IncrementalMapperOptions,
                                mapper: IncrementalMapper, max_images: int) -> None:
    logging.info("Retriangulation and Global bundle adjustment")
    mapper.iterative_global_refinement(
        options.ba_global_max_refinements,
        options.ba_global_max_refinement_change,
        mapper_options,
        global_bundle_adjustment_options(options, max_images),
        options.get_triangulation(),
        True,
    )
    mapper.filter_frames(mapper_options)


def initialize_reconstruction(controller: IncrementalPipeline, mapper: IncrementalMapper,
                              mapper_options: IncrementalMapperOptions,
                              reconstruction: Reconstruction,
                              max_images: int) -> IncrementalPipelineStatus:
    options = controller.options
    init_pair = (options.init_image_id1, options.init_image_id2)

    if not options.is_initial_pair_provided():
        logging.info("Finding good initial image pair")
        ret = mapper.find_initial_image_pair(mapper_options, *init_pair)
        if ret is None:
            logging.info("No good initial image pair found.")
            return IncrementalPipelineStatus.NO_INITIAL_PAIR
        init_pair, init_cam2_from_cam1 = ret
    else:
        if not all(reconstruction.exists_image(i) for i in init_pair):
            logging.info(f"=> Initial image pair {init_pair} does not exist.")
            return IncrementalPipelineStatus.NO_INITIAL_PAIR
        maybe_init_cam2_from_cam1 = mapper.estimate_initial_two_view_geometry(
            mapper_options, *init_pair)
        if maybe_init_cam2_from_cam1 is None:
            logging.info("Provided pair is unsuitable for initialization")
            return IncrementalPipelineStatus.BAD_INITIAL_PAIR
        init_cam2_from_cam1 = maybe_init_cam2_from_cam1
    logging.info(f"Registering initial image pair #{init_pair[0]} and #{init_pair[1]}")
    mapper.register_initial_image_pair(mapper_options, *init_pair, init_cam2_from_cam1)

    tri_options = options.get_triangulation()
    tri_options.min_angle = mapper_options.init_min_tri_angle
    for image_id in init_pair:
        image = reconstruction.images[image_id]
        assert image.frame is not None
        for data_id in image.frame.image_ids:
            mapper.triangulate_image(tri_options, data_id.id)

    logging.info("Global bundle adjustment")
    mapper.adjust_global_bundle(mapper_options,
                                global_bundle_adjustment_options(options, max_images))
    reconstruction.normalize()
    mapper.filter_points(mapper_options)
    mapper.filter_frames(mapper_options)

    if reconstruction.num_reg_frames() == 0 or reconstruction.num_points3D() == 0:
        return IncrementalPipelineStatus.BAD_INITIAL_PAIR
    if options.extract_colors:
        reconstruction.extract_colors_for_all_images(options.image_path)
    return IncrementalPipelineStatus.SUCCESS


def reconstruct_sub_model(controller: IncrementalPipeline, mapper: IncrementalMapper,
                          mapper_options: IncrementalMapperOptions,
                          reconstruction: Reconstruction,
                          max_images: int) -> IncrementalPipelineStatus:
    mapper.begin_reconstruction(reconstruction)

    if has_unknown_sensor_from_rig(reconstruction):
        return IncrementalPipelineStatus.UNKNOWN_SENSOR_FROM_RIG

    if reconstruction.num_reg_frames() == 0:
        init_status = initialize_reconstruction(
            controller, mapper, mapper_options, reconstruction, max_images)
        if init_status != IncrementalPipelineStatus.SUCCESS:
            return init_status
    controller.callback(IncrementalPipelineCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK)

    options = controller.options

    if options.structure_less_registration_only:
        structure_less_flags = [True]
    elif options.structure_less_registration_fallback:
        structure_less_flags = [False, True]
    else:
        structure_less_flags = [False]

    ba_prev_num_reg_frames = reconstruction.num_reg_frames()
    ba_prev_num_points = reconstruction.num_points3D()
    reg_next_success, prev_reg_next_success = True, True
    while True:
        if not (reg_next_success or prev_reg_next_success):
            break
        if controller.check_reached_max_runtime():
            break
        prev_reg_next_success = reg_next_success
        reg_next_success = False
        next_image_id = None
        for structure_less in structure_less_flags:
            next_images = mapper.find_next_images(mapper_options, structure_less=structure_less)
            for reg_trial, next_image_id in enumerate(next_images):
                logging.info(f"Registering image #{next_image_id} "
                             f"(num_reg_frames={reconstruction.num_reg_frames()})")
                if structure_less:
                    logging.info("Registering image with structure-less fallback")
                    num_vis = mapper.observation_manager.num_visible_correspondences(next_image_id)
                    num_corrs = mapper.observation_manager.num_correspondences(next_image_id)
                    logging.info(f"=> Image sees {num_vis} / {num_corrs} correspondences")
                    reg_next_success = mapper.register_next_structure_less_image(
                        mapper_options, next_image_id)
                else:
                    num_vis = mapper.observation_manager.num_visible_points3D(next_image_id)
                    num_obs = mapper.observation_manager.num_observations(next_image_id)
                    logging.info(f"=> Image sees {num_vis} / {num_obs} points")
                    reg_next_success = mapper.register_next_image(mapper_options, next_image_id)
                if reg_next_success:
                    break
                logging.info("=> Could not register, trying another image.")
                # If the initial pair fails to continue for some time, abort and
                # try a different initial pair.
                kMinNumInitialRegTrials = 30
                if (reg_trial >= kMinNumInitialRegTrials
                        and reconstruction.num_reg_images() < options.min_model_size):
                    break
            if reg_next_success:
                break
        if reg_next_success and next_image_id is not None:
            image = reconstruction.images[next_image_id]
            assert image.frame is not None
            for data_id in image.frame.image_ids:
                mapper.triangulate_image(options.get_triangulation(), data_id.id)
            mapper.iterative_local_refinement(
                options.ba_local_max_refinements,
                options.ba_local_max_refinement_change,
                mapper_options,
                options.get_local_bundle_adjustment(),
                options.get_triangulation(),
                next_image_id,
            )
            if controller.check_run_global_refinement(
                    reconstruction, ba_prev_num_reg_frames, ba_prev_num_points):
                iterative_global_refinement(options, mapper_options, mapper, max_images)
                ba_prev_num_points = reconstruction.num_points3D()
                ba_prev_num_reg_frames = reconstruction.num_reg_frames()
            if options.extract_colors:
                for data_id in image.frame.image_ids:
                    if not reconstruction.extract_colors_for_image(data_id.id, options.image_path):
                        logging.warning(f"Could not read image "
                                        f"{reconstruction.images[data_id.id].name} "
                                        f"at path {options.image_path}")
            controller.callback(IncrementalPipelineCallback.NEXT_IMAGE_REG_CALLBACK)
        if mapper.num_shared_reg_images() >= int(options.max_model_overlap):
            break
        if (not reg_next_success) and prev_reg_next_success:
            iterative_global_refinement(options, mapper_options, mapper, max_images)

    if controller.check_reached_max_runtime():
        return IncrementalPipelineStatus.INTERRUPTED

    # Only run the final global BA if the last incremental BA was not global.
    if (reconstruction.num_reg_frames() > 0
            and reconstruction.num_reg_frames() != ba_prev_num_reg_frames
            and reconstruction.num_points3D() != ba_prev_num_points):
        iterative_global_refinement(options, mapper_options, mapper, max_images)
    return IncrementalPipelineStatus.SUCCESS


def reconstruct(controller: IncrementalPipeline, mapper: IncrementalMapper,
                mapper_options: IncrementalMapperOptions, continue_reconstruction: bool,
                max_images: int) -> IncrementalPipelineStatus:
    options = controller.options
    database_cache = controller.database_cache
    reconstruction_manager = controller.reconstruction_manager

    for num_trials in range(options.init_num_trials):
        if controller.check_reached_max_runtime():
            break
        if not continue_reconstruction or num_trials > 0:
            reconstruction_idx = reconstruction_manager.add()
        else:
            reconstruction_idx = 0

        reconstruction = reconstruction_manager.get(reconstruction_idx)
        status = reconstruct_sub_model(controller, mapper, mapper_options, reconstruction,
                                       max_images)
        if status == IncrementalPipelineStatus.INTERRUPTED:
            reconstruction.update_point_3d_errors()
            logging.info("Keeping reconstruction due to interrupt")
            mapper.end_reconstruction(False)
            pycolmap.align_reconstruction_to_orig_rig_scales(database_cache.rigs, reconstruction)
        elif status == IncrementalPipelineStatus.UNKNOWN_SENSOR_FROM_RIG:
            logging.error(
                "Discarding reconstruction due to unknown sensor_from_rig "
                "poses. Either explicitly define the poses by configuring the "
                "rigs or first run reconstruction without configured rigs and "
                "then derive the poses from the initial reconstruction for a "
                "subsequent reconstruction with rig constraints.")
            mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
            return IncrementalPipelineStatus.STOP
        elif status in (IncrementalPipelineStatus.BAD_INITIAL_PAIR,
                        IncrementalPipelineStatus.NO_INITIAL_PAIR):
            reason = "bad" if status == IncrementalPipelineStatus.BAD_INITIAL_PAIR else "no"
            logging.info(f"Discarding reconstruction due to {reason} initial pair")
            mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
            if status == IncrementalPipelineStatus.NO_INITIAL_PAIR:
                return IncrementalPipelineStatus.CONTINUE
        elif status == IncrementalPipelineStatus.SUCCESS:
            num_reg_images = reconstruction.num_reg_images()
            total_num_reg_images = mapper.num_total_reg_images()
            if ((options.multiple_models and reconstruction_manager.size() > 1
                    and num_reg_images < options.min_model_size) or num_reg_images == 0):
                logging.info("Discarding reconstruction due to insufficient size")
                mapper.end_reconstruction(True)
                reconstruction_manager.delete(reconstruction_idx)
            else:
                reconstruction.update_point_3d_errors()
                logging.info("Keeping successful reconstruction")
                mapper.end_reconstruction(False)
                pycolmap.align_reconstruction_to_orig_rig_scales(database_cache.rigs,
                                                                 reconstruction)
            controller.callback(IncrementalPipelineCallback.LAST_IMAGE_REG_CALLBACK)
            if (not options.multiple_models
                    or reconstruction_manager.size() >= options.max_num_models
                    or total_num_reg_images >= database_cache.num_images() - 1):
                return IncrementalPipelineStatus.STOP
        else:
            logging.fatal(f"Unknown reconstruction status: {status}")

    return IncrementalPipelineStatus.CONTINUE


def main_incremental_mapper(controller: IncrementalPipeline, max_images: int) -> None:
    timer = pycolmap.Timer()
    timer.start()
    database_cache = controller.database_cache

    if database_cache.num_images() == 0:
        logging.warning("No images with matches found in the database")
        return
    if controller.options.use_prior_position and database_cache.num_pose_priors() == 0:
        logging.warning("No pose priors")
        return

    reconstruction_manager = controller.reconstruction_manager
    continue_reconstruction = reconstruction_manager.size() > 0
    if reconstruction_manager.size() > 1:
        logging.fatal("Can only resume from a single reconstruction, but multiple are given")

    num_images = database_cache.num_images()
    mapper = IncrementalMapper(database_cache)
    mapper_options = controller.options.get_mapper()
    if reconstruct(controller, mapper, mapper_options, continue_reconstruction,
                   max_images) == IncrementalPipelineStatus.STOP:
        return

    def should_stop():
        return (mapper.num_total_reg_images() == num_images
                or controller.check_reached_max_runtime())

    for _ in range(2):  # number of relaxations
        if should_stop():
            break
        logging.info("=> Relaxing the initialization constraints")
        mapper_options.init_min_num_inliers = int(mapper_options.init_min_num_inliers / 2)
        mapper.reset_initialization_stats()
        if reconstruct(controller, mapper, mapper_options, False,
                       max_images) == IncrementalPipelineStatus.STOP:
            return
        if should_stop():
            break
        logging.info("=> Relaxing the initialization constraints")
        mapper_options.init_min_tri_angle /= 2
        mapper.reset_initialization_stats()
        if reconstruct(controller, mapper, mapper_options, False,
                       max_images) == IncrementalPipelineStatus.STOP:
            return
    timer.print_minutes()


def incremental_mapping(database_path, image_path, output_path,
                        options: IncrementalPipelineOptions | None = None,
                        direct_solver_max_images: int = DIRECT_SOLVER_MAX_IMAGES,
                        ) -> dict[int, Reconstruction]:
    """Drop-in for pycolmap.incremental_mapping with the direct-solver limit lifted."""
    if options is None:
        options = IncrementalPipelineOptions()
    if not pycolmap.__version__.startswith(MIRRORED_VERSION):
        logging.warning(
            f"colmap_incremental mirrors pycolmap {MIRRORED_VERSION}x but found "
            f"{pycolmap.__version__}; falling back to pycolmap.incremental_mapping, "
            f"whose global bundle adjustment turns iterative past 1000 images")
        return pycolmap.incremental_mapping(database_path, image_path, output_path, options)

    database_path, image_path, output_path = Path(database_path), Path(image_path), Path(output_path)
    if not database_path.exists():
        logging.fatal(f"Database path does not exist: {database_path}")
    if not image_path.exists():
        logging.fatal(f"Image path does not exist: {image_path}")
    options.image_path = image_path
    output_path.mkdir(exist_ok=True, parents=True)

    reconstruction_manager = ReconstructionManager()
    with pycolmap.Database.open(database_path) as database:
        controller = IncrementalPipeline(options, database, reconstruction_manager)
        main_incremental_mapper(controller, direct_solver_max_images)
    reconstruction_manager.write(output_path)
    return {i: reconstruction_manager.get(i) for i in range(reconstruction_manager.size())}
