"""Job config schema, canonical hashing, and the cache-key chain."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .mask_backends import DEFAULT_BACKEND, MASK_BACKENDS


# ---------------------------------------------------------------- hashing

def canon(obj) -> str:
    """Stable JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def key_of(*parts) -> str:
    return hashlib.sha256(canon(parts).encode()).hexdigest()[:16]


CHUNK = 8 << 20


def quick_hash(path: Path) -> str:
    """Identity for a large video file: size + head, middle and tail.

    A full sha256 of a 7 GB clip costs ~30 s and buys nothing here. But sampling
    has to actually sample: the previous version read the tail only when the
    file exceeded 2 chunks, so every file between 8 and 16 MiB was identified by
    its first 8 MiB alone --- two different 9 MiB clips hashed the same, and the
    second one silently inherited the first one's entire cached pipeline.
    Anything up to 3 chunks is now hashed whole, and larger files contribute a
    middle chunk so a re-encode that keeps the container header and size cannot
    pass as the same input.
    """
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with path.open("rb") as f:
        if size <= 3 * CHUNK:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        else:
            h.update(f.read(CHUNK))
            f.seek((size - CHUNK) // 2)
            h.update(f.read(CHUNK))
            f.seek(-CHUNK, os.SEEK_END)
            h.update(f.read(CHUNK))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- schema

# Reject unknown fields everywhere rather than ignoring them. A misspelt
# "sh_degrees" or a field set at the wrong nesting level used to validate
# cleanly, hash into the cache key as if it mattered, and train the default.
class Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputCfg(Cfg):
    file: str                                   # relative to SPLAT_ROOT
    quick_hash: str = ""                        # always recomputed server-side
    trim_start: Optional[float] = None          # seconds
    trim_end: Optional[float] = None

    @model_validator(mode="after")
    def _check_trim(self):
        if self.trim_start is not None and self.trim_start < 0:
            raise ValueError("trim_start must be >= 0")
        if self.trim_end is not None and self.trim_end <= 0:
            # Without a trim_start to compare against, a negative or zero
            # trim_end used to sail through and become a negative ffmpeg -t,
            # which extracts nothing.
            raise ValueError("trim_end must be > 0")
        if self.trim_start is not None and self.trim_end is not None:
            if self.trim_end <= self.trim_start:
                raise ValueError("trim_end must be greater than trim_start")
        return self


class FramesCfg(Cfg):
    # fps 0 or negative produces an ffmpeg filter that yields no frames; the
    # upper bound is well past any sane candidate rate for a 30 fps source.
    fps: float = Field(default=10.0, gt=0, le=120)
    jpeg_q: int = Field(default=2, ge=1, le=31)


class SelectCfg(Cfg):
    mode: Literal["window", "target", "distance"] = "window"
    window: int = Field(default=5, ge=1)
    target_panos: int = Field(default=300, ge=10)
    distance_m: float = Field(default=5., gt=0, allow_inf_nan=False)
    max_gap_s: float = Field(default=2., gt=0, allow_inf_nan=False)
    # Raw .OSV only: the camera's orientation stream vetoes candidates it
    # predicts to be motion-blurred before the Laplacian ranks the rest
    # (80_fisheye_frames.py --imu, docs/how-it-works.md, "Gyro blur veto"). Stitched input has no stream,
    # so JobConfig refuses it there. The API fills it in for .OSV input when a
    # request leaves it out (main._prepare).
    imu: bool = False


class SfmCfg(Cfg):
    render: Literal["spherical", "perspective_overlapping",
                    "perspective_non_overlapping"] = "spherical"
    mapper: Literal["incremental", "global"] = "incremental"
    # Raw .OSV only: level the reconstruction with the clip's orientation
    # stream, and scale it to metres and turn it to north when the clip carries
    # GPS (scripts/upright.py, docs/how-it-works.md, "Upright and in metres").
    # Fails safe: without GPS the model is levelled only; when levelling cannot
    # be trusted it is left as SfM made it, with a warning. Stitched input has
    # no stream, so JobConfig refuses it there. The API fills it in for .OSV
    # input when a request leaves it out (main._prepare, UPRIGHT_DEFAULT).
    upright: bool = False
    # Raw .OSV only, off by default: each global bundle adjustment of the rig
    # SfM leaves out the 3D points that add no coverage and refines them after
    # with the poses fixed (COLMAP's ba_global_ignore_redundant_points3D,
    # 82_fisheye_sfm.py --skip-redundant-points). Clip 0005 on nuc3: mapping
    # 42.0 -> 34.6 min, the same cameras within 0.016 % of the path, PSNR
    # +0.03 dB and SSIM -0.002 after training -- but the point count swung
    # 436k-729k between two near-identical databases (issue #19), so it stays
    # opt-in until a second clip and repeat trainings say otherwise.
    skip_redundant_points: bool = False


class MaskCfg(Cfg):
    """Per-frame masks excluding people from the photometric loss.

    Off by default, and the switch is the capture type rather than the clip:
    ON for Osmo 360 footage, which is handheld or stick-mounted so the operator
    is always in shot and always directly behind the camera; OFF for Avata 360
    drone footage, where nobody rides along and the aircraft body is a separate
    problem this does not address.
    """
    enabled: bool = False
    # One of mask_backends.MASK_BACKENDS. maskrcnn is BSD-licensed with nothing
    # to download or accept, which makes it the safe default; sam3 is faster,
    # tighter and open-vocabulary, but carries Meta's SAM licence and needs the
    # weights staged locally.
    backend: str = DEFAULT_BACKEND
    # Prompt-capable backends only. Each entry is a concept segmented exhaustively; anything not
    # named is simply not masked, which is how "people and cats but not dogs" is
    # expressed -- there is no textual negation.
    prompts: list[str] = Field(default_factory=lambda: ["person"])
    nadir_view: bool = True
    # Masks also suppress SIFT features on people. They sit at near-zero
    # parallax (rigidly attached, or keeping station with the camera), which is
    # geometry consistent with a point at infinity -- exactly the kind of match
    # that can drag a pose. Turn off to compare training-side masking alone
    # against an identical reconstruction.
    use_for_sfm: bool = True
    # Hold the job after masking and wait for a human to look at the contact
    # sheet before spending a GPU-hour on training.
    review: bool = False

    @model_validator(mode="after")
    def _check_backend(self):
        if self.backend not in MASK_BACKENDS:
            raise ValueError(f"unknown mask backend {self.backend!r}; "
                             f"one of {sorted(MASK_BACKENDS)}")
        if not MASK_BACKENDS[self.backend]["prompts"] and self.prompts != ["person"]:
            with_prompts = sorted(n for n, m in MASK_BACKENDS.items() if m["prompts"])
            raise ValueError(
                f"backend {self.backend!r} segments people only; prompts "
                f"{self.prompts} need one of {with_prompts}. Ignoring them "
                "would mask people under a config claiming otherwise, and the "
                "cache key would describe the wrong run")
        if not self.prompts:
            raise ValueError("at least one prompt is required")
        return self
    # None means "the backend's own default" (0.5 maskrcnn, 0.3 sam3). Stored as
    # None rather than resolved, so the two backends cannot collide on one key.
    score: Optional[float] = Field(default=None, ge=0.05, le=0.95)
    # Segmentation edges cut through the colour halo around a person; an
    # unmasked rim is still a camera-fixed object for the trainer to bake in.
    dilate: int = Field(default=9, ge=0, le=129)
    work_width: int = Field(default=1920, ge=640, le=7680)


# Fields that change what the mask stage PRODUCES. `review` is a workflow gate
# and `use_for_sfm` decides who consumes the result -- neither changes a single
# pixel, and hashing them would make toggling either one recompute every mask.
# `use_for_sfm` still reaches k_sfm through JobConfig._sfm_mask_term.
MASK_OUTPUT_FIELDS = {"enabled", "backend", "prompts", "nadir_view",
                      "score", "dilate", "work_width"}


# Flags the queue constructs itself. Passing one through extra_args means two
# copies on the command line, which for --iter/--steps-scaler silently changes
# the training budget the cache key claims to describe.
MANAGED_TRAIN_FLAGS = {
    "-d", "--images", "-o", "--headless", "--strategy", "--max-cap",
    "--sh-degree", "--iter", "--steps-scaler", "--max-width", "--gut",
    "--eval", "--no-save-eval-images", "--enable-mip", "--background-improvements",
    "--exposure-correction", "--bilateral-grid", "--min-opacity",
    "--max-screen-share", "--export", "--mask-mode", "--invert-masks", "--ppisp",
}


class TrainCfg(Cfg):
    trainer: Literal["lichtfeld"] = "lichtfeld"
    strategy: Literal["mrnf", "mcmc", "igs+"] = "mrnf"

    # Training budget. steps_scaler scales iterations AND every schedule
    # (fill_pacing_iter, sh_degree_interval, refine_every, eval_steps, ...).
    # iter is the unscaled base; see the guard in _check_budget.
    iter: int = Field(default=30000, ge=100)
    steps_scaler: float = Field(default=1.0, gt=0, le=10)

    max_cap: int = Field(default=3_000_000, ge=10_000)
    # 3: +0.12 dB over 1 on 0005 for ~6 % more training time; the .sog barely
    # grows (docs/how-it-works.md, "Trainer options, measured").
    sh_degree: int = Field(default=3, ge=0, le=3)
    max_width: int = Field(default=3840, ge=0)
    gut: Optional[bool] = None                  # None = auto from camera model
    eval: bool = True

    enable_mip: bool = False
    background_improvements: bool = False
    exposure_correction: bool = False
    bilateral_grid: bool = False
    ppisp: bool = False
    min_opacity: Optional[float] = None
    max_screen_share: Optional[float] = None

    extra_args: str = ""

    @model_validator(mode="after")
    def _check_extra_args(self):
        try:
            toks = shlex.split(self.extra_args)
        except ValueError as exc:
            # e.g. an unbalanced quote. str.split() used to hand the halves to
            # the trainer as separate arguments.
            raise ValueError(f"extra_args is not a valid command line: {exc}")
        clashes = sorted({t.split("=", 1)[0] for t in toks
                          if t.startswith("-")} & MANAGED_TRAIN_FLAGS)
        if clashes:
            raise ValueError(
                f"extra_args must not repeat flags the queue already sets: "
                f"{', '.join(clashes)} -- use the matching config field, or the "
                f"command line carries two copies and the cache key describes "
                f"the wrong run")
        return self

    @model_validator(mode="after")
    def _check_budget(self):
        # LichtFeld applies steps_scaler AFTER the CLI assignment, so passing
        # both --iter and --steps-scaler double-scales: --iter 15000
        # --steps-scaler 0.5 yields 7500 iterations. Refuse it.
        if self.steps_scaler != 1.0 and self.iter != 30000:
            raise ValueError(
                "set either steps_scaler (scales iterations and all schedules "
                "together) or a bare iter, not both: LichtFeld applies "
                "steps_scaler after --iter, so the two multiply")
        # LichtFeld refuses this pair at startup, after the whole SfM has run:
        # "exposure correction replaces the standalone bilateral grid".
        if self.exposure_correction and (self.bilateral_grid or self.ppisp):
            raise ValueError(
                "exposure_correction cannot be combined with bilateral_grid or "
                "ppisp: LichtFeld's exposure correction replaces both. Pick one")
        return self

    @property
    def effective_iters(self) -> int:
        return round(self.iter * self.steps_scaler)


# Formats LichtFeld can emit via --export=. Anything else is a typo that would
# otherwise be passed through to the trainer and produce a run with no usable
# output, discovered 95 minutes later.
EXPORT_FORMATS = ("ply", "sog", "spz")


class ExportCfg(Cfg):
    # All three by default. They are not redundant: PLY is the lossless master
    # every tool reads, SOG is what the web viewer streams, SPZ is the compact
    # interchange container (v4/zstd from both LichtFeld and splat-transform).
    # Exporting them together costs seconds at the end of a 75-minute run;
    # regenerating one later means finding the PLY again.
    formats: list[str] = Field(
        default_factory=lambda: ["ply", "sog", "spz"])

    @model_validator(mode="after")
    def _check_formats(self):
        seen, cleaned = set(), []
        for f in self.formats:
            f = f.strip().lower()
            if f not in EXPORT_FORMATS:
                raise ValueError(
                    f"unknown export format {f!r}; expected one of "
                    f"{', '.join(EXPORT_FORMATS)}")
            if f not in seen:
                seen.add(f)
                cleaned.append(f)
        if not cleaned:
            raise ValueError("at least one export format is required")
        # Canonical order, so ["sog","ply"] and ["ply","sog"] share a cache key.
        self.formats = [f for f in EXPORT_FORMATS if f in seen]
        return self


# Camera-native dual fisheye that the pipeline reconstructs WITHOUT stitching:
# both lenses go to SfM as a calibrated two-camera rig (docs/how-it-works.md, "Fisheye rig"). Insta360's
# containers stay refused -- their calibration has not been decoded.
FISHEYE_EXTS = (".osv",)

# Participates in every fisheye cache key and in none of the stitched ones, so
# changing a fisheye script can invalidate fisheye caches alone, and every
# cache entry the stitched path already holds keeps its key.
FISHEYE_PIPELINE = "fisheye-rig-1"

# Joins the select key -- and so every key after it -- only while select.imu is
# on: off selects exactly what the fisheye path did before the option existed,
# so those cache entries keep their keys. Bump it after changing how
# 80_fisheye_frames.py scores candidates with the orientation stream.
IMU_SELECT = "imu-select-1"

# Joins the sfm key -- and so train and export -- of fisheye jobs only. Bump it
# after changing what 82/88_fisheye_sfm.py or colmap_incremental.py reconstruct
# from the same selection and masks: frames, selection and masks keep their
# cache entries, where a FISHEYE_PIPELINE bump would redo all three. 2: global
# bundle adjustment every 40 % of growth, 2 refinements (was 10 %, 5). 3: every
# 100 % of growth; local bundle adjustment threaded from 5,000 residuals.
# 4: pycolmap-cuda12 4.2.0 -> 4.2.1 (track-merge colours, rig and two-view
# fixes that change extraction, matching and mapping).
FISHEYE_SFM = "fisheye-sfm-4"
# What the API sets for .OSV input when a request leaves select.imu out.
IMU_SELECT_DEFAULT = False

# Joins the sfm key -- and so train and export -- only while sfm.upright is on,
# so a job with it off keeps every key it had before the option existed. Bump
# it after changing what scripts/upright.py does to a model.
UPRIGHT = "upright-1"

# Joins the sfm key -- and so train and export -- only while
# sfm.skip_redundant_points is on, so a job with it off keeps every key it had
# before the option existed. Bump it after changing what the option does in
# 82_fisheye_sfm.py.
REDUNDANT_POINTS = "redundant-points-1"
# Joins every train key -- and so export -- and nothing upstream of it: a new
# trainer build retrains from the frames, masks and SfM already cached, where a
# config_version bump would redo all of them. Moves with LFS_REF in
# scripts/setup_lichtfeld.sh.
TRAINER = "lichtfeld-3067e9e0"

# What the API sets for .OSV input when a request leaves sfm.upright out. On:
# it only moves and scales the model, and falls back on its own when the clip
# cannot support it.
UPRIGHT_DEFAULT = True


class JobConfig(Cfg):
    # Bumping this invalidates every cache entry for jobs submitted afterwards,
    # which is how a change to a pipeline script is made to take effect (a
    # trainer upgrade moves TRAINER instead). It was previously stored and
    # never hashed, so it did nothing. Version 2 is the first that
    # participates; it also marks the quick_hash sampling fix, which changes
    # every input identity anyway.
    config_version: int = Field(default=2, ge=1)
    name: str
    input: InputCfg
    frames: FramesCfg = Field(default_factory=FramesCfg)
    select: SelectCfg = Field(default_factory=SelectCfg)
    sfm: SfmCfg = Field(default_factory=SfmCfg)
    mask: MaskCfg = Field(default_factory=MaskCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)
    # Stop after this stage: the job ends done, the later stages skipped. With
    # "sfm" it also writes a handoff bundle for another machine to train from
    # (handoff.py). Where a job stops changes no stage's output, so it is in no
    # cache key; a prep job and an all-in-one job share every upstream entry.
    run_until: Optional[Literal["frames", "select", "mask", "sfm"]] = None

    @property
    def is_fisheye(self) -> bool:
        """A raw dual-fisheye input: the stages run the rig pipeline instead."""
        return Path(self.input.file).suffix.lower() in FISHEYE_EXTS

    @model_validator(mode="after")
    def _check_pipeline(self):
        # sfm.render and sfm.mapper choose between the STITCHED reconstructions.
        # A raw .OSV has exactly one: the two-pass incremental rig. Refuse a
        # config that asks for something else rather than run the rig under a
        # config -- and a cache key -- that claims otherwise. The defaults are
        # what every client sends, so they pass.
        if self.is_fisheye and (self.sfm.render != "spherical"
                                or self.sfm.mapper != "incremental"):
            raise ValueError(
                f"sfm.render/sfm.mapper ({self.sfm.render}/{self.sfm.mapper}) "
                f"apply to stitched equirect input; {Path(self.input.file).name} "
                f"is raw dual fisheye and always runs the incremental fisheye "
                f"rig reconstruction. Leave both at their defaults")
        if self.sfm.upright and not self.is_fisheye:
            raise ValueError(
                f"sfm.upright reads the orientation stream and GPS inside a "
                f"raw DJI .OSV; {Path(self.input.file).name} is stitched and "
                f"carries neither. Leave sfm.upright off")
        if self.sfm.skip_redundant_points and not self.is_fisheye:
            raise ValueError(
                f"sfm.skip_redundant_points applies to the fisheye rig "
                f"reconstruction of a raw DJI .OSV; {Path(self.input.file).name} "
                f"is stitched. Leave sfm.skip_redundant_points off")
        if self.select.imu and not self.is_fisheye:
            raise ValueError(
                f"select.imu reads the orientation stream inside a raw DJI "
                f".OSV; {Path(self.input.file).name} is stitched and carries "
                f"none. Leave select.imu off")
        # 30_run_sfm.py refuses --masks for the perspective renders (they make
        # their own per-virtual-camera masks), so this failed only after frames,
        # selection and masking had run.
        if (self.mask.enabled and self.mask.use_for_sfm and not self.is_fisheye
                and self.sfm.render != "spherical"):
            raise ValueError(
                f"sfm.render={self.sfm.render} cannot take person masks during "
                f"SfM; set mask.use_for_sfm=false (masks still apply to "
                f"training) or use sfm.render=spherical")
        # LichtFeld: "GUT and igs+ strategy cannot be used together", and every
        # 360 camera model here (the fisheye rig, a spherical reconstruction)
        # trains only through GUT. Only the pinhole-render SfM modes leave it off.
        if self.train.strategy == "igs+" and (
                self.is_fisheye or self.sfm.render == "spherical" or self.train.gut):
            raise ValueError(
                "strategy igs+ cannot train through GUT, which every .OSV job and "
                "every stitched job with sfm.render=spherical needs. Use mrnf or "
                "mcmc, or a perspective_* sfm.render for a stitched clip")
        if self.select.mode == "distance":
            if not self.is_fisheye:
                raise ValueError("select.mode=distance requires a raw Avata 360 .OSV")
            if self.frames.fps * self.select.max_gap_s < 2:
                raise ValueError("select.max_gap_s must span at least two candidate intervals")
        return self

    def _pipeline_terms(self) -> tuple:
        return (FISHEYE_PIPELINE,) if self.is_fisheye else ()

    # ------------------------------------------------------- cache keys
    # Each key covers its own params plus the key of everything upstream, so a
    # change to fps invalidates select/sfm/train while a change to sh_degree
    # invalidates only train and export.

    def k_frames(self) -> str:
        # Every downstream key chains from this one, so the pipeline term here
        # separates the whole fisheye chain. Stitched configs add nothing, which
        # leaves their keys exactly what they were.
        return key_of("frames", self.config_version, self.input.quick_hash,
                      self.input.trim_start, self.input.trim_end,
                      self.frames.model_dump(), *self._pipeline_terms())

    def k_select(self) -> str:
        # target mode resolves to a window at runtime; both modes hash the same
        # way once resolved, so keep the raw config here and let the resolved
        # window land in the stage record. select.imu is a term, and only while
        # on (IMU_SELECT), so every entry cached before the option keeps its key.
        sel = self.select.model_dump(exclude={"imu", "distance_m", "max_gap_s"})
        if self.select.mode == "distance":
            sel = {"mode": "distance", "distance_m": self.select.distance_m,
                   "max_gap_s": self.select.max_gap_s, "algorithm": "avata_velocity_v1"}
        return key_of("select", self.k_frames(), sel,
                      *((IMU_SELECT,) if self.select.imu else ()))

    def _sfm_mask_term(self) -> str:
        """What the reconstruction itself saw of the masks.

        Always contributes a term, never a conditionally-omitted field: a key
        that sometimes hashes an input and sometimes does not is how two
        different configs come to share one cache entry. Unmasked configs all
        agree on the sentinel, so they still share a reconstruction.
        """
        if self.mask.enabled and self.mask.use_for_sfm:
            return self.k_mask()
        return "no-sfm-mask"

    def k_sfm(self) -> str:
        # sfm.upright and sfm.skip_redundant_points are terms, and only while
        # on (UPRIGHT, REDUNDANT_POINTS), so every entry cached before the
        # options keeps its key.
        return key_of("sfm", self.k_select(),
                      self.sfm.model_dump(exclude={"upright", "skip_redundant_points"}),
                      self._sfm_mask_term(),
                      *((FISHEYE_SFM,) if self.is_fisheye else ()),
                      *((UPRIGHT,) if self.sfm.upright else ()),
                      *((REDUNDANT_POINTS,) if self.sfm.skip_redundant_points else ()))

    def k_mask(self) -> str:
        # A disabled mask stage produces nothing, so every disabled config has
        # to hash the same way. It used to hash its own backend, prompts and
        # dilate even when switched off, so editing a dormant field forked
        # k_train and bought a 95-minute retrain for a bit-identical model.
        # Normalise to the default disabled form rather than inventing a fresh
        # sentinel: that collapses those configs onto one key WITHOUT moving the
        # key every already-cached unmasked run is stored under.
        m = self.mask if self.mask.enabled else MaskCfg()
        return key_of("mask", self.k_select(),
                      m.model_dump(include=MASK_OUTPUT_FIELDS))

    def k_train(self) -> str:
        # Masks change what the trainer optimises, so they belong in the train
        # key. Included even when disabled: hashing a field only conditionally
        # is how two different configs end up sharing one cache entry. k_mask
        # normalises the disabled case, so every unmasked config still agrees
        # here rather than forking on fields that changed no pixel.
        return key_of("train", TRAINER, self.k_sfm(), self.k_mask(),
                      self.train.model_dump())

    def k_export(self) -> str:
        return key_of("export", self.k_train(), self.export.model_dump())

    def keys(self) -> dict[str, str]:
        return {"frames": self.k_frames(), "select": self.k_select(),
                "mask": self.k_mask(), "sfm": self.k_sfm(),
                "train": self.k_train(), "export": self.k_export()}
