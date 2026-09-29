#!/usr/bin/env python3
"""Handoff bundles (#8): prep on one machine, train on another.

CPU only, no clip, no GPU. The cache entries are built by hand in the shapes
the real stages leave: absolute symlinks from the fisheye dataset into the
select dir and the valid-circle masks, hardlinked masks, a stitched
dataset/sparse/0 that is a link to the chosen model.

    python3 queue/test_handoff.py
"""
import hashlib
import http.server
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

TMP = tempfile.mkdtemp(prefix="queue_handoff_test_")
# Assigned, never setdefault (AGENTS.md, "Tests").
os.environ["SPLAT_ROOT"] = TMP
os.environ["QUEUE_ROOT"] = str(Path(TMP) / "q")
os.environ["QUEUE_GPUS"] = ""
os.environ["QUEUE_TELEMETRY"] = "1"
# Most tests import one bundle several times; Cleanup tests the default.
os.environ["QUEUE_HANDOFF_KEEP"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config, db, handoff, resources, telemetry, worker  # noqa: E402
from app.jobs import JobConfig                                # noqa: E402
from app.stages import ORDER, STAGES, is_cached, read_done    # noqa: E402

CACHE = config.CACHE_ROOT
db.init()

OSV = {"name": "rig", "input": {"file": "samples/c.OSV", "quick_hash": "q1"},
       "mask": {"enabled": True}}
MP4 = {"name": "pano", "input": {"file": "samples/c.mp4", "quick_hash": "q2"}}
PERSP = {"name": "persp", "input": {"file": "samples/c.mp4", "quick_hash": "q3"},
         "sfm": {"render": "perspective_overlapping"}}
N = 10


def _w(p: Path, data: bytes = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if data is not None else os.urandom(64))
    return p


def _done(d: Path, info: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / ".done").write_text(json.dumps(info))


def build_cache(cfg_d: dict) -> JobConfig:
    """Cache entries for frames..sfm, shaped like the real stages leave them."""
    cfg = JobConfig.model_validate(cfg_d)
    k = cfg.keys()
    fr, se, ma, sf = (CACHE / s / k[s] for s in ("frames", "select", "mask", "sfm"))
    for d in (fr, se, ma, sf):
        shutil.rmtree(d, ignore_errors=True)
    if cfg.is_fisheye:
        for i in (0, 1):
            for n in range(1, 2 * N + 1):
                _w(fr / f"lens{i}" / f"{n:06d}.jpg")
        _w(fr / "calibration.json", b"{}")
        _done(fr, {"candidates": 2 * N})
        for i in (0, 1):
            for n in range(1, N + 1):                      # hardlinks, like 80_fisheye_frames.py
                dst = se / "images" / f"lens{i}" / f"frame_{n:04d}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(fr / f"lens{i}" / f"{2 * n:06d}.jpg", dst)
        _w(se / "selection.json", b'{"mode": "window"}')
        _done(se, {"panos": N, "rig_frames": N, "window": 2})
        for n in range(1, N + 1):
            _w(ma / "masks" / f"pano_{n:04d}.png")
            _w(ma / "overlay" / f"pano_{n:04d}.jpg")          # review only: not shipped
            for i in (0, 1):
                _w(ma / "fisheye_masks" / f"lens{i}" / f"frame_{n:04d}.jpg.png")
        _w(ma / "summary.json", b'{"coverage_solid_angle_mean": 0.01}')
        _done(ma, {"masks": N, "fisheye_masks": N, "coverage_solid_angle_mean": 0.01})
        _w(sf / "rig" / "database.db", os.urandom(4096))     # not shipped
        for i in (0, 1):
            circle = _w(sf / "valid_masks" / f"lens{i}" / ".circle.png")
            for n in range(1, N + 1):
                os.link(circle, sf / "valid_masks" / f"lens{i}" / f"frame_{n:04d}.jpg.png")
        ds = sf / "dataset"
        for f in ("cameras.bin", "images.bin", "points3D.bin"):
            _w(ds / "sparse" / "0" / f)
        (ds / "images").mkdir(parents=True)
        (ds / "masks").mkdir(parents=True)
        for i in (0, 1):
            for n in range(1, N + 1):                      # absolute links, like 85_fisheye_dataset.py
                (ds / "images" / f"lens{i}_frame_{n:04d}.jpg").symlink_to(
                    (se / "images" / f"lens{i}" / f"frame_{n:04d}.jpg").resolve())
                (ds / "masks" / f"lens{i}_frame_{n:04d}.png").symlink_to(
                    (sf / "valid_masks" / f"lens{i}" / f"frame_{n:04d}.jpg.png").resolve())
        _w(sf / "summary.json", json.dumps({"dataset": str(ds)}).encode())
        _done(sf, {"num_reg_frames": N, "registration_pct": 100.0, "needs_gut": True,
                   "images_dir": str(ds / "images"), "warnings": []})
    else:
        for n in range(1, 3 * N + 1):
            _w(fr / f"{n:06d}.jpg")
        _done(fr, {"candidates": 3 * N})
        for n in range(1, N + 1):
            se.mkdir(parents=True, exist_ok=True)
            os.link(fr / f"{3 * n:06d}.jpg", se / f"pano_{n:04d}.jpg")
        _done(se, {"panos": N, "window": 3})
        run = sf / f"sfm_{cfg.sfm.render}_{cfg.sfm.mapper}"
        _w(run / "database.db", os.urandom(4096))
        for f in ("cameras.bin", "images.bin", "points3D.bin"):
            _w(run / "sparse" / "0" / f)
        if cfg.sfm.render != "spherical":
            for n in range(1, 4 * N + 1):
                _w(run / "images" / f"view_{n:04d}.jpg")
        (sf / "dataset" / "sparse").mkdir(parents=True)
        (sf / "dataset" / "sparse" / "0").symlink_to((run / "sparse" / "0").resolve(),
                                                      target_is_directory=True)
        imgs = se if cfg.sfm.render == "spherical" else run / "images"
        _done(sf, {"num_reg_frames": N, "registration_pct": 100.0, "needs_gut": False,
                   "images_dir": str(imgs), "warnings": []})
    return cfg


def prep_job(cfg: JobConfig, review="approved") -> int:
    """A job whose upstream stages are done, as a prep run leaves it."""
    jid = db.create_job(cfg.name, cfg.model_dump())
    db.set_review(jid, review)
    for st, key in cfg.keys().items():
        state = ("skipped" if st in ("train", "export") or (st == "mask" and not cfg.mask.enabled)
                 else "done")
        db.upsert_stage(jid, st, key, state, path=str(CACHE / st / key),
                        progress=json.dumps(read_done(CACHE / st / key)))
    return jid


def snapshot(cfg: JobConfig) -> dict:
    """What training would read: resolved path -> sha256, per shipped stage."""
    out = {}
    k = cfg.keys()
    for st, rels in handoff.shipped_paths(cfg).items():
        base = CACHE / st / k[st]
        for rel in rels:
            root = base / rel if rel else base
            if not root.exists():
                continue
            walk = [root] if root.is_file() else sorted(
                Path(dp) / f for dp, _, fs in os.walk(root, followlinks=True) for f in fs)
            for p in walk:
                if p.is_file() and p.name not in handoff.PRIVATE:
                    out[str(p.relative_to(CACHE))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def wipe(cfg: JobConfig) -> None:
    """The train machine: none of this job's cache entries."""
    for st, key in cfg.keys().items():
        shutil.rmtree(CACHE / st / key, ignore_errors=True)
        db.conn().execute("DELETE FROM cache WHERE cache_key=?", (key,))


def to_handoffs(job_id: int) -> str:
    src = handoff.bundle_path(job_id)
    dst = config.HANDOFF_ROOT / f"b{job_id}-{uuid.uuid4().hex[:4]}.tar"
    shutil.copy(src, dst)
    return dst.name


def rewrite(name: str, edit_manifest=None, extra: list = None, drop_manifest=False) -> str:
    """A copy of a bundle with its manifest edited or members added."""
    src = config.HANDOFF_ROOT / name
    dst = config.HANDOFF_ROOT / f"x-{uuid.uuid4().hex[:6]}.tar"
    with tarfile.open(src) as tin, tarfile.open(dst, "w", format=tarfile.PAX_FORMAT) as tout:
        manifest = None
        for m in tin:
            if m.name == handoff.MANIFEST:
                manifest = json.loads(tin.extractfile(m).read())
                continue
            tout.addfile(m, tin.extractfile(m) if m.isreg() else None)
        for ti, data in extra or []:
            tout.addfile(ti, io.BytesIO(data) if data is not None else None)
        if edit_manifest:
            edit_manifest(manifest)
        if not drop_manifest:
            data = json.dumps(manifest).encode()
            ti = tarfile.TarInfo(handoff.MANIFEST)
            ti.size = len(data)
            tout.addfile(ti, io.BytesIO(data))
    return dst.name


class RoundTrip(unittest.TestCase):
    def roundtrip(self, cfg_d):
        cfg = build_cache(cfg_d)
        before = snapshot(cfg)
        jid = prep_job(cfg)
        st = handoff.export(jid)
        self.assertEqual(st["state"], "built")
        uuid.UUID(st["id"])                                  # a real UUID
        name = to_handoffs(jid)
        wipe(cfg)
        out = handoff.import_bundle(handoff.bundle_file(name))
        self.assertEqual(out["handoff"], st["id"])
        self.assertEqual(snapshot(cfg), before)              # same bytes, same names
        return cfg, st, out

    def test_fisheye_masked(self):
        cfg, st, out = self.roundtrip(OSV)
        k = cfg.keys()
        self.assertEqual(sorted(out["installed"]), ["mask", "select", "sfm"])
        ds = CACHE / "sfm" / k["sfm"] / "dataset"
        # Links into select come back as relative links that resolve here.
        link = ds / "images" / "lens0_frame_0001.jpg"
        self.assertTrue(link.is_symlink())
        self.assertFalse(os.path.isabs(os.readlink(link)))
        self.assertEqual(link.resolve(),
                         (CACHE / "select" / k["select"] / "images/lens0/frame_0001.jpg").resolve())
        # Links out of what ships are replaced by their content.
        self.assertTrue((ds / "masks" / "lens0_frame_0001.png").is_file())
        self.assertFalse((ds / "masks" / "lens0_frame_0001.png").is_symlink())
        self.assertFalse((CACHE / "sfm" / k["sfm"] / "rig").exists())       # not shipped
        self.assertFalse((CACHE / "mask" / k["mask"] / "overlay").exists())
        self.assertFalse((CACHE / "frames" / k["frames"]).exists())
        # The .done is this machine's, with this machine's paths.
        self.assertEqual(read_done(CACHE / "sfm" / k["sfm"])["images_dir"], str(ds / "images"))
        # The queued job starts at train.
        row = db.get_job(out["id"])
        self.assertIsNone(json.loads(row["config"])["run_until"])
        states = {s["stage"]: s["state"] for s in db.job_stages(out["id"])}
        self.assertEqual(states, {"frames": "imported", "select": "imported",
                                  "mask": "imported", "sfm": "imported",
                                  "train": "pending", "export": "pending"})
        self.assertEqual(db.job_handoff(row)["id"], st["id"])

    def test_manifest_names_no_path_of_this_machine(self):
        cfg = build_cache(OSV)
        jid = prep_job(cfg)
        handoff.export(jid)
        manifest = handoff.read_manifest(handoff.bundle_path(jid))
        text = json.dumps(manifest["stages"])
        self.assertNotIn(str(CACHE), text)
        self.assertIn(handoff.CACHE_TOKEN, text)
        with tarfile.open(handoff.bundle_path(jid)) as tar:
            names = tar.getnames()
            self.assertEqual(names[-1], handoff.MANIFEST)     # last, so a cut bundle has none
            self.assertTrue(all(not os.path.isabs(m.linkname) for m in tar if m.issym()))
        # No telemetry of the prep job travels (samples, logs, host).
        self.assertFalse([n for n in names if "telemetry" in n or n.endswith(".log")])

    def test_stitched_spherical(self):
        cfg, _, out = self.roundtrip(MP4)
        self.assertEqual(sorted(out["installed"]), ["select", "sfm"])
        sp = CACHE / "sfm" / cfg.keys()["sfm"] / "dataset" / "sparse" / "0"
        self.assertTrue((sp / "images.bin").is_file())
        states = {s["stage"]: s["state"] for s in db.job_stages(out["id"])}
        self.assertEqual(states["mask"], "skipped")

    def test_stitched_perspective_ships_the_pinhole_views(self):
        cfg, _, _ = self.roundtrip(PERSP)
        run = CACHE / "sfm" / cfg.keys()["sfm"] / "sfm_perspective_overlapping_incremental"
        self.assertEqual(len(list((run / "images").glob("*.jpg"))), 4 * N)
        self.assertFalse((run / "database.db").exists())

    def test_second_import_reuses_what_is_cached(self):
        cfg = build_cache(MP4)
        jid = prep_job(cfg)
        handoff.export(jid)
        name = to_handoffs(jid)
        out = handoff.import_bundle(handoff.bundle_file(name))
        self.assertEqual(sorted(out["already_cached"]), ["select", "sfm"])
        self.assertEqual(out["installed"], [])

    def test_train_override_keeps_upstream_keys(self):
        cfg = build_cache(MP4)
        jid = prep_job(cfg)
        handoff.export(jid)
        name = to_handoffs(jid)
        wipe(cfg)
        out = handoff.import_bundle(handoff.bundle_file(name), train={"iter": 40000})
        self.assertEqual({s: out["keys"][s] for s in handoff.UPSTREAM},
                         {s: cfg.keys()[s] for s in handoff.UPSTREAM})
        self.assertNotEqual(out["keys"]["train"], cfg.keys()["train"])
        with self.assertRaises(handoff.HandoffError):
            handoff.import_bundle(handoff.bundle_file(name), train={"sh_degrees": 3})

    def test_export_refuses_a_stage_that_does_not_verify(self):
        cfg = build_cache(OSV)
        jid = prep_job(cfg)
        shutil.rmtree(CACHE / "sfm" / cfg.keys()["sfm"] / "dataset" / "images")
        with self.assertRaisesRegex(handoff.HandoffError, "sfm .* does not verify"):
            handoff.export(jid)
        self.assertFalse(handoff.bundle_path(jid).exists())


class Cleanup(unittest.TestCase):
    """A verified import deletes its tar: kept, it doubled the train side's
    input on disk (2026-09-29, 5.45 GB tar next to 5.47 GB unpacked)."""

    def bundle(self, name):
        cfg = build_cache({**MP4, "name": name})
        jid = prep_job(cfg)
        self.assertEqual(handoff.export(jid)["state"], "built")
        return cfg, to_handoffs(jid)

    def test_deleted_after_a_verified_import(self):
        cfg, name = self.bundle("clean")
        wipe(cfg)
        with patch.object(handoff, "KEEP_BUNDLES", False):
            out = handoff.import_bundle(handoff.bundle_file(name))
        self.assertFalse((config.HANDOFF_ROOT / name).exists())
        self.assertTrue(out["status"]["bundle_removed"])
        self.assertGreater(out["status"]["bytes"], 0)       # recorded before deleting
        self.assertTrue(out["installed"])
        self.assertTrue(all(is_cached(CACHE / st / out["keys"][st]) for st in out["installed"]))

    def test_keep_keeps_it(self):
        cfg, name = self.bundle("keep")
        wipe(cfg)
        with patch.object(handoff, "KEEP_BUNDLES", False):
            out = handoff.import_bundle(handoff.bundle_file(name), keep=True)
        self.assertTrue((config.HANDOFF_ROOT / name).is_file())
        self.assertFalse(out["status"]["bundle_removed"])

    def test_a_refused_import_never_deletes(self):
        cfg, name = self.bundle("refused")
        wipe(cfg)
        with patch.object(handoff, "KEEP_BUNDLES", False):
            with self.assertRaises(handoff.HandoffError):
                handoff.import_bundle(handoff.bundle_file(name), sha256="0" * 64)
        self.assertTrue((config.HANDOFF_ROOT / name).is_file())


class Refusals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = build_cache(OSV)
        jid = prep_job(cls.cfg)
        handoff.export(jid)
        cls.name = to_handoffs(jid)
        cls.jid = jid

    def refused(self, name, pattern, **kw):
        n_jobs = db.conn().execute("SELECT COUNT(*) n FROM jobs").fetchone()["n"]
        with self.assertRaisesRegex(handoff.HandoffError, pattern):
            handoff.import_bundle(handoff.bundle_file(name), **kw)
        self.assertEqual(db.conn().execute("SELECT COUNT(*) n FROM jobs").fetchone()["n"],
                         n_jobs, "a refused bundle queued a job")
        self.assertFalse(list(CACHE.glob(".handoff-*")), "staging left behind")

    def test_wrong_sha256(self):
        self.refused(self.name, "sha256", sha256="0" * 64)

    def test_right_sha256(self):
        good = handoff.file_sha256(config.HANDOFF_ROOT / self.name)
        self.assertEqual(good, handoff.status(self.jid)["sha256"])
        handoff.import_bundle(handoff.bundle_file(self.name), sha256=good.upper())

    def test_truncated(self):
        src = config.HANDOFF_ROOT / self.name
        data = src.read_bytes()
        with tarfile.open(src) as tar:
            m = tar.getmember(handoff.MANIFEST)
        # Half way; just before the manifest's header; inside its data. (Cutting
        # only the zero padding after the last member loses nothing.)
        for i, cut in enumerate((len(data) // 2, m.offset - 1, m.offset_data + m.size // 2)):
            dst = config.HANDOFF_ROOT / f"cut{i}.tar"
            dst.write_bytes(data[:cut])
            self.refused(dst.name, "truncated or not a handoff bundle")

    def test_not_a_tar(self):
        (config.HANDOFF_ROOT / "junk.tar").write_bytes(b"hello")
        self.refused("junk.tar", "truncated or not a handoff bundle")

    def test_damaged_file(self):
        src = config.HANDOFF_ROOT / self.name
        with tarfile.open(src) as tar:
            m = next(m for m in tar if m.isreg() and m.name.endswith(".jpg"))
        data = bytearray(src.read_bytes())
        data[m.offset_data] ^= 0xFF
        dst = config.HANDOFF_ROOT / "damaged.tar"
        dst.write_bytes(bytes(data))
        self.refused(dst.name, "sha256 does not match")

    def test_version_term_mismatch(self):
        with patch.object(handoff, "FISHEYE_SFM", "fisheye-sfm-99"):
            self.refused(self.name, "FISHEYE_SFM: bundle 'fisheye-sfm-4', here 'fisheye-sfm-99'")

    def test_config_version_mismatch(self):
        def edit(m):
            m["versions"]["config_version"] = 1
        self.refused(rewrite(self.name, edit), "config_version")

    def test_key_mismatch(self):
        def edit(m):
            m["keys"]["sfm"] = m["stages"]["sfm"]["key"] = "0" * 16
        self.refused(rewrite(self.name, edit), "sfm key")

    def test_config_edited_under_the_same_keys(self):
        def edit(m):
            m["config"]["frames"]["fps"] = 5.0          # keys now describe other frames
        self.refused(rewrite(self.name, edit), "frames key")

    def test_member_not_in_manifest(self):
        k = self.cfg.keys()
        ti = tarfile.TarInfo(f"cache/sfm/{k['sfm']}/dataset/extra.bin")
        ti.size = 3
        self.refused(rewrite(self.name, extra=[(ti, b"abc")]), "not in the manifest")

    def test_missing_member(self):
        def edit(m):
            m["files"].append({"path": m["files"][0]["path"] + ".gone", "type": "file",
                               "size": 1, "sha256": "0" * 64})
        self.refused(rewrite(self.name, edit), "not in the bundle")

    def test_link_out_of_the_cache(self):
        k = self.cfg.keys()
        ti = tarfile.TarInfo(f"cache/sfm/{k['sfm']}/dataset/evil")
        ti.type, ti.linkname = tarfile.SYMTYPE, "../../../../../../etc/passwd"

        def edit(m):
            m["files"].append({"path": ti.name, "type": "symlink", "target": ti.linkname})
        self.refused(rewrite(self.name, edit, extra=[(ti, None)]), "outside the stages")

    def test_link_chain_out_of_the_cache(self):
        # Each link is harmless on paper; followed for real, the second climbs
        # out of the shallow directory the first one lands in.
        k = self.cfg.keys()
        deep = f"cache/sfm/{k['sfm']}/a/b/c/d/e"
        l1 = tarfile.TarInfo(f"{deep}/l1")
        l1.type, l1.linkname = tarfile.SYMTYPE, "../../../../../dataset"
        l2 = tarfile.TarInfo(f"{deep}/l2")
        l2.type, l2.linkname = tarfile.SYMTYPE, "l1/../../../../../../etc"

        def edit(m):
            for ti in (l2, l1):
                m["files"].append({"path": ti.name, "type": "symlink", "target": ti.linkname})
        self.refused(rewrite(self.name, edit, extra=[(l2, None), (l1, None)]),
                     "resolves outside")

    def test_write_through_a_link(self):
        k = self.cfg.keys()
        link = tarfile.TarInfo(f"cache/sfm/{k['sfm']}/alias")
        link.type, link.linkname = tarfile.SYMTYPE, f"../../select/{k['select']}"
        f = tarfile.TarInfo(f"cache/sfm/{k['sfm']}/alias/images/lens0/frame_0001.jpg")
        f.size = 1

        def edit(m):
            m["files"].append({"path": link.name, "type": "symlink", "target": link.linkname})
            m["files"].append({"path": f.name, "type": "file", "size": 1,
                               "sha256": hashlib.sha256(b"x").hexdigest()})
        self.refused(rewrite(self.name, edit, extra=[(link, None), (f, b"x")]),
                     "through a link")

    def test_member_named_as_a_stage_dir(self):
        k = self.cfg.keys()
        ti = tarfile.TarInfo(f"cache/sfm/{k['sfm']}")
        ti.size = 1

        def edit(m):
            m["files"].append({"path": ti.name, "type": "file", "size": 1,
                               "sha256": hashlib.sha256(b"x").hexdigest()})
        self.refused(rewrite(self.name, edit, extra=[(ti, b"x")]), "outside the stages")

    def test_path_out_of_the_cache(self):
        ti = tarfile.TarInfo("cache/frames/x/../../../evil")
        ti.size = 1

        def edit(m):
            m["files"].append({"path": ti.name, "type": "file", "size": 1,
                               "sha256": hashlib.sha256(b"x").hexdigest()})
        self.refused(rewrite(self.name, edit, extra=[(ti, b"x")]), "outside the stages")

    def test_unapproved_review(self):
        cfg = build_cache({**OSV, "name": "rev", "mask": {"enabled": True, "review": True}})
        jid = prep_job(cfg, review="pending")
        handoff.export(jid)
        self.refused(to_handoffs(jid), "never approved")

    def test_bundle_name_cannot_leave_the_handoffs_dir(self):
        for bad in ("../queue.db", "/etc/passwd", ".hidden", "nope.tar"):
            with self.assertRaises(handoff.HandoffError):
                handoff.bundle_file(bad)


def _fake(stage, calls):
    def argv(ctx):
        calls.append(stage)
        return [sys.executable, "-c", "pass"]
    return {"argv": argv, "finalize": lambda ctx: {"n": 1}, "parse": None,
            "prepare": None, "gpu": False, "verify": None}


def _explode(stage):
    def argv(ctx):
        raise AssertionError(f"{stage} must not run")
    return {"argv": argv, "finalize": None, "parse": None, "prepare": None,
            "gpu": False, "verify": None}


class _Sink(http.server.BaseHTTPRequestHandler):
    got = []

    def do_PUT(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        _Sink.got.append((self.path, body))
        self.send_response(200)
        self.send_header("ETag", '"%s"' % hashlib.md5(body).hexdigest())
        self.end_headers()

    def log_message(self, *a):
        pass


class Worker(unittest.TestCase):
    def run_job(self, jid, fakes):
        with patch.dict(STAGES, fakes), \
                patch.object(worker.retention, "ensure_space", lambda *a, **k: None):
            worker.run_job(jid, -1)
        return db.get_job(jid)

    def test_imported_job_starts_at_train_without_the_clip(self):
        cfg = build_cache(OSV)
        prep = prep_job(cfg)
        st = handoff.export(prep)
        name = to_handoffs(prep)
        wipe(cfg)
        jid = handoff.import_bundle(handoff.bundle_file(name))["id"]
        self.assertFalse((Path(TMP) / cfg.input.file).exists())
        calls = []
        fakes = {s: _explode(s) for s in ("frames", "select", "mask", "sfm")}
        fakes.update(train=_fake("train", calls), export=_fake("export", calls))
        # Only argv is replaced upstream: the real verify still checks the
        # imported directories, as a cache hit does.
        for s in ("select", "mask", "sfm"):
            fakes[s]["verify"] = STAGES[s]["verify"]
        row = self.run_job(jid, fakes)
        self.assertEqual(row["state"], "done", row["error"])
        self.assertEqual(calls, ["train", "export"])
        stages = {s["stage"]: s for s in db.job_stages(jid)}
        for s in handoff.UPSTREAM:
            self.assertEqual(stages[s]["state"], "imported")
            self.assertIsNone(stages[s]["started"])
            self.assertIsNone(stages[s]["ended"])
            self.assertEqual(json.loads(stages[s]["progress"])["imported_from"], st["id"])
        self.assertEqual(stages["train"]["state"], "done")
        # Telemetry: the same id, role train, and nothing measured for stages
        # that ran elsewhere.
        rec = json.loads(telemetry.telemetry_path(jid).read_text())
        self.assertEqual(rec["handoff"]["id"], st["id"])
        self.assertEqual(rec["handoff"]["role"], "train")
        for s in rec["stages"]:
            if s["stage"] in handoff.UPSTREAM:
                self.assertEqual(s["state"], "imported")
                self.assertEqual(s["handoff_id"], st["id"])
                self.assertEqual(s["cache_key"], cfg.keys()[s["stage"]])
                self.assertIsNone(s["resources"])
                self.assertIsNone(s["wall_s"])
                self.assertEqual(s["info"], {})
        self.assertIsNotNone(next(s for s in rec["stages"] if s["stage"] == "train")["wall_s"])

    def test_evicted_import_fails_loudly(self):
        cfg = build_cache(MP4)
        prep = prep_job(cfg)
        handoff.export(prep)
        name = to_handoffs(prep)
        wipe(cfg)
        jid = handoff.import_bundle(handoff.bundle_file(name))["id"]
        shutil.rmtree(CACHE / "sfm" / cfg.keys()["sfm"])
        fakes = {s: _explode(s) for s in ORDER}
        fakes["select"]["verify"] = STAGES["select"]["verify"]
        fakes["sfm"]["verify"] = STAGES["sfm"]["verify"]
        with patch.object(worker.debugdump, "on_failure", lambda j: None):
            row = self.run_job(jid, fakes)
        self.assertEqual(row["state"], "failed")
        self.assertIn("import", row["error"])
        self.assertIn(name, row["error"])

    def test_an_ordinary_job_still_needs_its_clip(self):
        jid = db.create_job("t", {"name": "t", "input": {"file": "samples/gone.mp4"}})
        with patch.object(worker.debugdump, "on_failure", lambda j: None):
            row = self.run_job(jid, {s: _explode(s) for s in ORDER})
        self.assertEqual(row["state"], "failed")
        self.assertIn("gone since this job was queued", row["error"])

    def test_run_until_sfm_writes_and_uploads_a_bundle(self):
        cfg_d = {**OSV, "name": "prep", "run_until": "sfm"}
        cfg = build_cache(cfg_d)
        self.assertEqual(cfg.keys(), JobConfig.model_validate(OSV).keys(),
                         "run_until must not enter any cache key")
        (Path(TMP) / "samples").mkdir(exist_ok=True)
        clip = Path(TMP) / cfg.input.file
        clip.write_bytes(b"clip")
        from app.jobs import quick_hash
        cfg.input.quick_hash = quick_hash(clip)
        # The fixture was built under the old hash; rebuild under the new keys.
        cfg = build_cache(cfg.model_dump())
        jid = db.create_job(cfg.name, cfg.model_dump())
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Sink)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        _Sink.got.clear()
        target = {"method": "PUT", "url": f"http://127.0.0.1:{srv.server_port}/h/{{file}}"}
        fakes = {s: _explode(s) for s in ORDER}
        for s in ("frames", "select", "mask", "sfm"):
            fakes[s]["verify"] = STAGES[s]["verify"]
        try:
            with patch.object(handoff, "HANDOFF_UPLOAD", target):
                row = self.run_job(jid, fakes)
                deadline = time.time() + 15
                while (handoff.status(jid) or {}).get("upload") is None and time.time() < deadline:
                    time.sleep(0.05)
        finally:
            srv.shutdown()
        self.assertEqual(row["state"], "done", row["error"])
        states = {s["stage"]: s["state"] for s in db.job_stages(jid)}
        self.assertEqual((states["train"], states["export"]), ("skipped", "skipped"))
        st = handoff.status(jid)
        self.assertEqual(st["upload"]["state"], "done", st["upload"])
        self.assertEqual(_Sink.got[0][0], f"/h/{handoff.bundle_name(jid)}")
        self.assertEqual(hashlib.sha256(_Sink.got[0][1]).hexdigest(), st["sha256"])
        # The id: a random UUID, in the manifest and in this job's telemetry.
        manifest = handoff.read_manifest(handoff.bundle_path(jid))
        self.assertEqual(manifest["id"], st["id"])
        self.assertEqual(str(uuid.UUID(st["id"])), st["id"])
        self.assertNotIn("c.OSV", st["id"])
        deadline = time.time() + 10
        rec = {}
        while time.time() < deadline:
            rec = json.loads(telemetry.telemetry_path(jid).read_text())
            if (rec["transfers"].get("handoff") or {}).get("state") == "done":
                break
            time.sleep(0.05)
        self.assertEqual(rec["handoff"]["id"], st["id"])
        self.assertEqual(rec["handoff"]["role"], "prep")
        self.assertEqual(rec["transfers"]["handoff"]["bytes"], st["bytes"])
        self.assertNotIn("127.0.0.1", json.dumps(rec))

    def test_ids_are_random(self):
        cfg = build_cache(MP4)
        a, b = prep_job(cfg), prep_job(cfg)
        ia, ib = handoff.export(a)["id"], handoff.export(b)["id"]
        self.assertNotEqual(ia, ib)
        self.assertEqual(uuid.UUID(ia).version, 4)

    def test_run_until_select_stops_without_a_bundle(self):
        cfg = build_cache(MP4)
        (Path(TMP) / "samples").mkdir(exist_ok=True)
        clip = Path(TMP) / cfg.input.file
        clip.write_bytes(b"mp4")
        from app.jobs import quick_hash
        d = cfg.model_dump()
        d["input"]["quick_hash"] = quick_hash(clip)
        d["run_until"] = "select"
        cfg = build_cache(d)
        jid = db.create_job(cfg.name, cfg.model_dump())
        fakes = {s: _explode(s) for s in ORDER}
        fakes["frames"]["verify"] = STAGES["frames"]["verify"]
        fakes["select"]["verify"] = STAGES["select"]["verify"]
        row = self.run_job(jid, fakes)
        self.assertEqual(row["state"], "done", row["error"])
        states = {s["stage"]: s["state"] for s in db.job_stages(jid)}
        self.assertEqual([states[s] for s in ORDER],
                         ["cached", "cached", "skipped", "skipped", "skipped", "skipped"])
        self.assertIsNone(handoff.status(jid))


class Api(unittest.TestCase):
    """The routes, called as functions: no server, no httpx."""

    @classmethod
    def setUpClass(cls):
        from app import main
        cls.main = main

    def test_image_variants_refuse_what_they_cannot_finish(self):
        from fastapi import HTTPException
        m = self.main
        full = JobConfig.model_validate(MP4)
        prep = JobConfig.model_validate({**MP4, "run_until": "sfm"})
        with patch.object(m, "IMAGE_VARIANT", "train"):
            with self.assertRaisesRegex(HTTPException, "train image"):
                m._refuse_for_image()
        with patch.object(m, "IMAGE_VARIANT", "prep"):
            m._refuse_for_image()
            with self.assertRaises(HTTPException):
                m._refuse_for_image(full)
            m._refuse_for_image(prep)
            with self.assertRaisesRegex(HTTPException, "prep image"):
                m.api_handoff_import(m.HandoffImportReq(bundle="x.tar"))
        with patch.object(m, "IMAGE_VARIANT", "all"):
            m._refuse_for_image(full)
            with patch.object(handoff, "HANDOFF_UPLOAD_ERROR", "HANDOFF_UPLOAD_URL: bad"):
                with self.assertRaisesRegex(HTTPException, "HANDOFF_UPLOAD_URL"):
                    m._refuse_for_image(prep)
                m._refuse_for_image(JobConfig.model_validate({**MP4, "run_until": "select"}))

    def test_a_cpu_below_the_build_refuses_training(self):
        from fastapi import HTTPException
        m = self.main
        full = JobConfig.model_validate(MP4)
        prep = JobConfig.model_validate({**MP4, "run_until": "sfm"})
        with patch.object(resources, "CPU_ERROR", "Xeon E5-2697 v2 lacks avx2, fma"), \
                patch.object(m, "IMAGE_VARIANT", "all"):
            with self.assertRaisesRegex(HTTPException, "cannot train"):
                m._refuse_for_image(full)
            m._refuse_for_image(prep)                        # prep still runs here
            with self.assertRaisesRegex(HTTPException, "cannot train"):
                m.api_handoff_import(m.HandoffImportReq(bundle="x.tar"))
            self.assertIn("E5-2697", m.api_status()["cpu_error"])

    def test_stage_rows_skip_past_run_until(self):
        cfg = JobConfig.model_validate({**MP4, "name": "rows", "run_until": "mask"})
        jid = db.create_job(cfg.name, cfg.model_dump())
        self.main._stage_rows(jid, cfg)
        states = {s["stage"]: s["state"] for s in db.job_stages(jid)}
        self.assertEqual((states["sfm"], states["train"], states["export"]),
                         ("skipped", "skipped", "skipped"))
        self.assertNotEqual(states["mask"], "skipped")

    def test_import_route(self):
        from fastapi import HTTPException
        m = self.main
        cfg = build_cache({**MP4, "name": "route"})
        jid = prep_job(cfg)
        m.api_handoff_build(jid, send=False)
        name = to_handoffs(jid)
        self.assertIn(name, [b["name"] for b in m.api_handoff_bundles()])
        wipe(cfg)
        out = m.api_handoff_import(m.HandoffImportReq(bundle=name, name="trained"))
        self.assertEqual(db.get_job(out["id"])["name"], "trained")
        self.assertEqual(m.api_job(out["id"])["handoff"]["role"], "train")
        with self.assertRaisesRegex(HTTPException, "400"):
            try:
                m.api_handoff_import(m.HandoffImportReq(bundle="../queue.db"))
            except HTTPException as exc:
                raise HTTPException(exc.status_code, f"{exc.status_code}") from None


if __name__ == "__main__":
    try:
        unittest.main(verbosity=1)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
