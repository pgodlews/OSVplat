"""Handoff bundles: prep on one machine, train on another (docs/cloud.md, "Split pipeline").

On a rented GPU, frames, selection, masks and above all SfM mapping are billed
at the GPU's hourly rate while the GPU mostly waits. A job with
run_until="sfm" stops after SfM, and export() packs exactly what training and
export read (the selected images, the masks, the SfM dataset) into
runs/job<id>/job<id>-handoff.tar. The manifest, handoff.json, is the tar's LAST
member. It holds the job config, every cache key, the version terms from
jobs.py, the image version and a sha256 per file.

import_bundle() on the train side verifies all of that, installs the entries in
its own cache under the same keys, and queues a job whose upstream stages are
already there. The worker starts that job at train, and never asks for the clip.

Import refuses rather than guesses. These are all import errors, never a job
trained on something other than what its keys describe:
  * keys this build computes differently (another FISHEYE_SFM, config_version);
  * a file whose sha256 or size differs;
  * a member the manifest does not list, or one it lists that is missing;
  * a truncated tar. The manifest is written last, so a cut-off bundle has none.

Symlinks: the cache links across stage directories (the fisheye dataset's
images point into the select dir). A link whose target is shipped too goes in
as a relative link, which resolves the same under any CACHE_ROOT. A link to
anything else (the chosen sparse model, the valid-circle masks) is replaced by
what it points to. Files hardlinked to each other go in once.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from . import db, outputs, telemetry
from .config import (CACHE_ROOT, HANDOFF_ROOT, HANDOFF_UPLOAD, HANDOFF_UPLOAD_ERROR,
                     IMAGE_VARIANT, RUNS_ROOT, SPLAT_ROOT, ssl_context)
from .jobs import (FISHEYE_PIPELINE, FISHEYE_SFM, IMU_SELECT, REDUNDANT_POINTS, UPRIGHT,
                   JobConfig)
from .stages import (ORDER, STAGES, Ctx, dir_bytes, is_cached, lock_holder_alive,
                     mark_done, read_done, read_lock, release_lock, reset_stage_dir,
                     take_lock)

SCHEMA = "osvplat.handoff/1"
MANIFEST = "handoff.json"
# The stages a handoff job takes as given, and the ones that ship directories.
# Frames are not shipped: training never reads them, and at 5.5 GB for a
# 946-image clip they are the largest thing in the cache.
UPSTREAM = ORDER[:ORDER.index("train")]
SHIPPED = ("select", "mask", "sfm")
# Written into stage info in place of this machine's cache path.
CACHE_TOKEN = "{cache}"
# Names the cache itself uses inside a stage dir; never shipped.
PRIVATE = {".lock", ".done", ".done.tmp"}
# Entrypoint record of HANDOFF_URL's download, like input_fetch.json.
FETCH_RECORD = RUNS_ROOT / "handoff_fetch.json"
# A bundle in HANDOFF_ROOT is deleted once its import has verified and installed
# every file: kept, it doubles the train side's input on disk (a 5.45 GB tar
# next to the 5.47 GB it unpacked to, 2026-09-29). QUEUE_HANDOFF_KEEP=1, or
# keep=true on the import, keeps it to import again.
KEEP_BUNDLES = os.environ.get("QUEUE_HANDOFF_KEEP", "0").strip().lower() in ("1", "true", "yes", "on")

_lock = threading.Lock()          # one upload at a time; they share the uplink
# One build at a time, and never behind an upload: the worker writes the bundle
# while its job still holds a GPU.
_build_lock = threading.Lock()


class HandoffError(RuntimeError):
    """A bundle that cannot be built, or is refused on import."""


def version_terms() -> dict:
    """What, besides the config, decides this build's cache keys."""
    return {"config_version": JobConfig.model_fields["config_version"].default,
            "FISHEYE_PIPELINE": FISHEYE_PIPELINE, "FISHEYE_SFM": FISHEYE_SFM,
            "IMU_SELECT": IMU_SELECT, "UPRIGHT": UPRIGHT,
            "REDUNDANT_POINTS": REDUNDANT_POINTS}


def bundle_name(job_id: int) -> str:
    return f"job{job_id:05d}-handoff.tar"


def bundle_path(job_id: int) -> Path:
    return telemetry.run_dir(job_id) / bundle_name(job_id)


def status_path(job_id: int) -> Path:
    return telemetry.run_dir(job_id) / "handoff.json"


def status(job_id: int) -> Optional[dict]:
    try:
        return json.loads(status_path(job_id).read_text())
    except (OSError, ValueError):
        return None


def _set_status(job_id: int, **st) -> dict:
    p = status_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, p)
    return st


def problems(now: Optional[float] = None) -> list[str]:
    """Why HANDOFF_UPLOAD_URL cannot work; jobs that need it are refused."""
    if HANDOFF_UPLOAD_ERROR:
        return [HANDOFF_UPLOAD_ERROR]
    if not HANDOFF_UPLOAD:
        return []
    exp = outputs.expires_at(HANDOFF_UPLOAD)
    now = time.time() if now is None else now
    if exp is not None and exp <= now:
        return [f"HANDOFF_UPLOAD_URL expired "
                f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(exp))}"]
    return []


# ------------------------------------------------------------ path tokens

def _portable(obj):
    """Stage info with this machine's cache path replaced by CACHE_TOKEN."""
    roots = sorted({str(CACHE_ROOT), str(CACHE_ROOT.resolve())}, key=len, reverse=True)
    if isinstance(obj, str):
        for r in roots:
            if obj == r or obj.startswith(r + "/"):
                return CACHE_TOKEN + obj[len(r):]
        return obj
    if isinstance(obj, dict):
        return {k: _portable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_portable(v) for v in obj]
    return obj


def _localize(obj):
    if isinstance(obj, str):
        return str(CACHE_ROOT) + obj[len(CACHE_TOKEN):] \
            if obj.startswith(CACHE_TOKEN) else obj
    if isinstance(obj, dict):
        return {k: _localize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_localize(v) for v in obj]
    return obj


# ------------------------------------------------------------------ export

def shipped_paths(cfg: JobConfig) -> dict[str, list[str]]:
    """Per stage, the paths inside its cache dir that training and export read.

    "" is the whole directory. The select dir goes whole: it is the stitched
    spherical path's --images and the fisheye dataset's link target. Of the
    mask dir, the masks only (not the overlays and contact sheet made for
    review). Of the SfM dir, the dataset LichtFeld reads, plus the perspective
    renders' pinhole images, which are a stitched job's --images there.
    """
    out = {"select": [""]}
    if cfg.mask.enabled:
        out["mask"] = ["masks", "fisheye_masks", "summary.json"]
    sfm = ["dataset", "summary.json"]
    if not cfg.is_fisheye and cfg.sfm.render != "spherical":
        sfm.append(f"sfm_{cfg.sfm.render}_{cfg.sfm.mapper}/images")
    out["sfm"] = sfm
    return out


class _HashingWriter:
    """The bundle file, hashed as it is written: no second 2 GB read."""

    def __init__(self, f):
        self.f, self.sha, self.md5, self.n = f, hashlib.sha256(), hashlib.md5(), 0

    def write(self, b) -> int:
        self.sha.update(b)
        self.md5.update(b)
        self.n += len(b)
        return self.f.write(b)


class _HashingReader:
    def __init__(self, f):
        self.f, self.sha, self.n = f, hashlib.sha256(), 0

    def read(self, size: int = -1) -> bytes:
        b = self.f.read(size)
        self.sha.update(b)
        self.n += len(b)
        return b


class _Packer:
    def __init__(self, tar: tarfile.TarFile, roots: list[tuple[Path, str]]):
        self.tar = tar
        # (real path, name in the bundle), longest first so the most specific wins.
        self.roots = sorted(((p.resolve(), arc) for p, arc in roots),
                            key=lambda r: len(str(r[0])), reverse=True)
        self.files: list[dict] = []
        self.inodes: dict[tuple[int, int], str] = {}
        self.payload = 0

    def arc_of(self, real: Path) -> Optional[str]:
        for root, arc in self.roots:
            if real == root:
                return arc
            if root in real.parents:
                return f"{arc}/{real.relative_to(root).as_posix()}"
        return None

    def add(self, path: Path, arc: str, seen: frozenset = frozenset()) -> None:
        if path.is_symlink():
            try:
                real = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise HandoffError(f"{arc} is a dangling link: {exc}") from None
            target = self.arc_of(real)
            if target is not None:
                ti = tarfile.TarInfo(arc)
                ti.type, ti.mode, ti.mtime = tarfile.SYMTYPE, 0o777, int(time.time())
                ti.linkname = posixpath.relpath(target, posixpath.dirname(arc))
                self.tar.addfile(ti)
                self.files.append({"path": arc, "type": "symlink", "target": ti.linkname})
                return
            path = real                     # outside what ships: take what it points to
        if path.is_dir():
            real = path.resolve()
            if real in seen:
                raise HandoffError(f"{arc} loops back into {real}")
            for child in sorted(path.iterdir()):
                if child.name not in PRIVATE:
                    self.add(child, f"{arc}/{child.name}", seen | {real})
            return
        if path.is_file():
            self._file(path, arc)

    def _file(self, path: Path, arc: str) -> None:
        st = path.stat()
        first = self.inodes.get((st.st_dev, st.st_ino))
        if first is not None:
            ti = tarfile.TarInfo(arc)
            ti.type, ti.mode, ti.mtime, ti.linkname = tarfile.LNKTYPE, 0o644, int(st.st_mtime), first
            self.tar.addfile(ti)
            self.files.append({"path": arc, "type": "hardlink", "target": first})
            return
        self.inodes[(st.st_dev, st.st_ino)] = arc
        ti = tarfile.TarInfo(arc)
        ti.size, ti.mode, ti.mtime = st.st_size, 0o644, int(st.st_mtime)
        with path.open("rb") as f:
            r = _HashingReader(f)
            self.tar.addfile(ti, r)
        if r.n != st.st_size:
            raise HandoffError(f"{arc} changed size while it was packed")
        self.payload += st.st_size
        self.files.append({"path": arc, "type": "file", "size": st.st_size,
                           "sha256": r.sha.hexdigest()})


def _clip_seconds(cfg: JobConfig) -> Optional[float]:
    """The clip's duration, so the train side can estimate without it."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(SPLAT_ROOT / cfg.input.file)],
            capture_output=True, text=True, timeout=60)
        return round(float(out.stdout.strip()), 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _verify(ctx: Ctx, stage: str, info: dict) -> None:
    spec = STAGES[stage]
    if spec.get("verify"):
        spec["verify"](ctx, info)


def export(job_id: int) -> dict:
    """Pack this job's upstream stages as a handoff bundle. Raises HandoffError.

    Every shipped stage has to be a valid cache entry right now, checked
    with the same verify a cache hit runs: a bundle is a promise that the
    train side can skip these stages.
    """
    row = db.get_job(job_id)
    if row is None:
        raise HandoffError(f"no job {job_id}")
    cfg = JobConfig.model_validate(json.loads(row["config"]))
    keys = cfg.keys()
    rows = {s["stage"]: s for s in db.job_stages(job_id)}
    ctx = Ctx(job_id=job_id, cfg=cfg, gpu=-1, keys=keys)
    stages, roots = {}, []
    paths = shipped_paths(cfg)
    for st in UPSTREAM:
        r = rows.get(st)
        if r is None or r["state"] not in ("done", "cached", "skipped"):
            raise HandoffError(f"job {job_id}'s {st} stage has not finished "
                               f"({r['state'] if r else 'no row'})")
        d = CACHE_ROOT / st / keys[st]
        if st == "mask" and not cfg.mask.enabled:
            stages[st] = {"key": keys[st], "shipped": False, "skipped": True, "info": {}}
            continue
        if st in paths:
            if not is_cached(d):
                raise HandoffError(f"{st} {keys[st]} is no longer in the cache")
            info = read_done(d)
            try:
                _verify(ctx, st, info)
            except Exception as exc:                          # noqa: BLE001
                raise HandoffError(f"{st} {keys[st]} does not verify: {exc}") from None
            for rel in paths[st]:
                p = d / rel if rel else d
                if p.exists():
                    # A root that is a link is walked as what it points to; packed
                    # as a link it would point at itself.
                    roots.append((p.resolve() if p.is_symlink() else p,
                                  f"cache/{st}/{keys[st]}" + (f"/{rel}" if rel else "")))
        else:
            # Frames: only their record travels.
            info = read_done(d) if is_cached(d) else json.loads(r["progress"] or "{}")
        stages[st] = {"key": keys[st], "shipped": st in paths, "info": _portable(info)}

    # Random, never derived from the clip, a path or the host: telemetry
    # carries it on both sides and it must name nothing (docs/job-telemetry.md).
    hid = str(uuid.uuid4())
    with _build_lock:
        return _write(job_id, row, cfg, keys, stages, roots, hid)


def _write(job_id: int, row, cfg: JobConfig, keys: dict, stages: dict,
           roots: list, hid: str) -> dict:
    out = bundle_path(job_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    t0 = time.time()
    try:
        with tmp.open("wb") as f:
            w = _HashingWriter(f)
            with tarfile.open(fileobj=w, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                packer = _Packer(tar, roots)
                for p, arc in roots:
                    packer.add(p, arc)
                manifest = {
                    "schema": SCHEMA, "id": hid, "created": round(time.time(), 1),
                    "image": {"version": os.environ.get("OSVPLAT_VERSION") or None,
                              "revision": os.environ.get("OSVPLAT_REVISION") or None,
                              "variant": IMAGE_VARIANT},
                    "versions": version_terms(),
                    "job": {"id": job_id, "name": row["name"],
                            "review_state": row["review_state"]},
                    "clip_seconds": _clip_seconds(cfg),
                    "config": cfg.model_dump(),
                    "keys": keys,
                    "stages": stages,
                    "payload_bytes": packer.payload,
                    "files": packer.files,
                }
                data = json.dumps(manifest, indent=1).encode()
                ti = tarfile.TarInfo(MANIFEST)
                ti.size, ti.mode, ti.mtime = len(data), 0o644, int(time.time())
                tar.addfile(ti, io.BytesIO(data))
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    st = _set_status(
        job_id, state="built", role="prep", id=hid, file=out.name, bytes=w.n,
        sha256=w.sha.hexdigest(), md5=w.md5.hexdigest(),
        files=len(packer.files), payload_bytes=packer.payload,
        stages=[s for s, v in stages.items() if v["shipped"]],
        pack_s=round(time.time() - t0, 1), created=manifest["created"])
    print(f"job {job_id}: handoff bundle {out.name} ({w.n / 2**20:.1f} MiB, "
          f"{len(packer.files)} files, sha256 {st['sha256']})")
    return st


# ------------------------------------------------------------------ upload

def _owner(dest: str, job_id: int) -> Optional[int]:
    """Another job whose bundle already went to this same fixed object."""
    for d in sorted(RUNS_ROOT.glob("job*")):
        if not d.name[3:].isdigit() or int(d.name[3:]) == job_id:
            continue
        st = status(int(d.name[3:]))
        if st and (st.get("upload") or {}).get("state") == "done" \
                and (st.get("upload") or {}).get("url") == dest:
            return int(d.name[3:])
    return None


def _request(target: dict, job_id: int, name: str, f, size: int) -> urllib.request.Request:
    """A streaming PUT (bundles run to GBs), or outputs' builder for a POST."""
    if (target.get("method") or "PUT").upper() == "POST":
        return outputs.request(target, job_id, name, f.read())
    url = target["url"].replace("{job}", f"job{job_id:05d}").replace("{file}", name)
    headers = {"Content-Type": "application/x-tar", **(target.get("headers") or {}),
               "Content-Length": str(size)}
    return urllib.request.Request(url, data=f, headers=headers, method="PUT")


def upload(job_id: int, target: Optional[dict] = None, attempts: int = 3) -> dict:
    """Send the built bundle to HANDOFF_UPLOAD_URL. Records the outcome in
    handoff.json; the bundle stays on disk either way."""
    target = target or HANDOFF_UPLOAD
    st = status(job_id) or {}
    if not target or st.get("role") != "prep" or not bundle_path(job_id).is_file():
        return st
    name = bundle_name(job_id)
    dest = outputs.safe_url(target["url"].replace("{job}", f"job{job_id:05d}")
                            .replace("{file}", name))
    with _lock:
        fixed = "{job}" not in target["url"] and "{job}" not in (target.get("key") or "")
        owner = _owner(dest, job_id) if fixed else None
        if owner is not None:
            st["upload"] = {"state": "refused", "url": dest,
                            "error": f"HANDOFF_UPLOAD_URL names one object and job "
                                     f"{owner}'s bundle is already there; use a {{job}} "
                                     f"placeholder or a new URL per run"}
            print(f"job {job_id}: handoff upload refused: {st['upload']['error']}")
            return _set_status(job_id, **st)
        path = bundle_path(job_id)
        size = path.stat().st_size
        err, tries, transfer_s = None, 0, None
        for attempt in range(attempts):
            tries += 1
            try:
                with path.open("rb") as f:
                    req = _request(target, job_id, name, f, size)
                    t0 = time.monotonic()
                    with urllib.request.urlopen(req, timeout=3600, context=ssl_context()) as r:
                        r.read()
                        etag = r.headers.get("ETag")
                transfer_s = round(time.monotonic() - t0, 2)
                err = outputs.etag_problem(etag, st.get("md5", ""))
                if err is None:
                    break
            except Exception as exc:                          # noqa: BLE001
                err = (f"HTTP {exc.code} {exc.reason}" if hasattr(exc, "code")
                       else f"{type(exc).__name__}: {exc}")
            if attempt < attempts - 1:
                time.sleep(2 ** (attempt + 1))
        st["upload"] = {"state": "failed" if err else "done", "url": dest, "bytes": size,
                        "attempts": tries, "transfer_s": None if err else transfer_s,
                        "error": err, "ended": round(time.time(), 1)}
        print(f"job {job_id}: handoff bundle upload to {dest} "
              + (f"FAILED: {err} (kept at {path})" if err
                 else f"done ({size} bytes, sha256 {st.get('sha256')})"))
        return _set_status(job_id, **st)


def upload_async(job_id: int) -> None:
    """Upload in the background if HANDOFF_UPLOAD_URL is set; then the job's
    one telemetry upload, so the record carries transfers.handoff."""
    if not HANDOFF_UPLOAD:
        return

    def run():
        try:
            upload(job_id)
        except Exception as exc:                              # noqa: BLE001
            print(f"job {job_id}: handoff upload FAILED: {exc}")
        telemetry.write(job_id, upload=True)
    threading.Thread(target=run, daemon=True, name=f"handoff{job_id}").start()


# ------------------------------------------------------------------ import

def bundle_file(name: str) -> Path:
    """A bundle in HANDOFF_ROOT, by file name only: the import route cannot be
    pointed anywhere else on disk."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HandoffError("bundle must be a file name in the handoffs directory")
    p = HANDOFF_ROOT / name
    if not p.is_file():
        raise HandoffError(f"no bundle {name} in the handoffs directory")
    return p


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> dict:
    """The manifest, or HandoffError. It is the last member, so a bundle cut
    short anywhere has none: tarfile reads a short archive as a shorter one."""
    try:
        with tarfile.open(path, "r:") as tar:
            m = tar.getmember(MANIFEST)
            data = tar.extractfile(m).read()
        if len(data) != m.size:
            raise EOFError("manifest cut short")
        manifest = json.loads(data)
    except (tarfile.TarError, KeyError, OSError, EOFError, ValueError) as exc:
        raise HandoffError(f"{path.name} is truncated or not a handoff bundle "
                           f"({type(exc).__name__}: {exc})") from None
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise HandoffError(f"{path.name}: unknown manifest schema "
                           f"{manifest.get('schema') if isinstance(manifest, dict) else None!r}"
                           f"; this build reads {SCHEMA}")
    return manifest


def check_manifest(manifest: dict) -> JobConfig:
    """The bundle's config, if this build computes the same upstream keys."""
    mine, theirs = version_terms(), manifest.get("versions") or {}
    diff = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    if diff:
        raise HandoffError(
            "the bundle was made by a build with different version terms ("
            + ", ".join(f"{k}: bundle {theirs.get(k)!r}, here {mine.get(k)!r}" for k in diff)
            + "); its cache keys mean something else here. Prep it again with this image's version")
    try:
        cfg = JobConfig.model_validate(manifest.get("config"))
    except ValidationError as exc:
        raise HandoffError(f"the bundle's config does not validate here: {exc}") from None
    keys, stages = cfg.keys(), manifest.get("stages") or {}
    for st in UPSTREAM:
        claimed = {(manifest.get("keys") or {}).get(st), (stages.get(st) or {}).get("key")}
        if claimed != {keys[st]}:
            raise HandoffError(
                f"{st} key: the bundle says {' / '.join(sorted(map(str, claimed)))}, this "
                f"build computes {keys[st]} from the bundle's own config. Refusing: the "
                f"cache entry would be stored under a key that does not describe it")
    if cfg.mask.enabled and cfg.mask.review \
            and (manifest.get("job") or {}).get("review_state") != "approved":
        raise HandoffError("the bundle's masks were never approved in review")
    return cfg


def _safe_name(name: str, prefixes: tuple[str, ...], root_ok: bool = False) -> None:
    """Inside a shipped stage dir; with root_ok (a link's target), or the dir itself."""
    norm = posixpath.normpath(name)
    if name.startswith("/") or norm != name or norm.startswith("..") \
            or not (norm + "/" if root_ok else norm).startswith(prefixes):
        raise HandoffError(f"member {name!r} is outside the stages the manifest ships")


def _extract(path: Path, manifest: dict, staging: Path) -> int:
    """Unpack into staging, checking every member against the manifest.

    Members are written one by one rather than with extractall: a symlink,
    hardlink or path that leaves the shipped stage dirs is refused, whatever
    tarfile version this Python has. Nothing is written through a link (the
    packer only ever walks real directories, so a real bundle never asks to),
    nothing overwrites a member already written, and once all are out every
    link is resolved for real, which catches a chain of links that looked
    harmless one at a time.
    """
    listed = {f["path"]: f for f in manifest.get("files") or []}
    keys = manifest["keys"]
    prefixes = tuple(f"cache/{st}/{keys[st]}/" for st in SHIPPED
                     if (manifest["stages"].get(st) or {}).get("shipped"))
    done: set[str] = set()
    nbytes = 0
    real_staging = os.path.realpath(staging)
    with tarfile.open(path, "r:") as tar:
        for m in tar:
            if m.name == MANIFEST:
                continue
            want = listed.get(m.name)
            if want is None:
                raise HandoffError(f"member {m.name!r} is not in the manifest")
            if m.name in done:
                raise HandoffError(f"member {m.name!r} appears twice")
            _safe_name(m.name, prefixes)
            dest = staging / m.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if os.path.realpath(dest.parent) != os.path.join(real_staging,
                                                             posixpath.dirname(m.name)):
                raise HandoffError(f"member {m.name!r} would be written through a link")
            if os.path.lexists(dest):
                raise HandoffError(f"member {m.name!r} would overwrite another")
            kind = want.get("type")
            if kind == "file":
                if not m.isreg() or m.size != want.get("size"):
                    raise HandoffError(f"{m.name}: not the {want.get('size')}-byte file "
                                       f"the manifest lists")
                h, n = hashlib.sha256(), 0
                src = tar.extractfile(m)
                with dest.open("xb") as out:
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        h.update(chunk)
                        n += len(chunk)
                        out.write(chunk)
                if n != m.size or h.hexdigest() != want.get("sha256"):
                    raise HandoffError(f"{m.name}: sha256 does not match the manifest "
                                       f"(damaged or altered bundle)")
                nbytes += n
            elif kind == "symlink":
                if not m.issym() or m.linkname != want.get("target"):
                    raise HandoffError(f"{m.name}: not the link the manifest lists")
                _safe_name(posixpath.normpath(posixpath.join(posixpath.dirname(m.name),
                                                             m.linkname)), prefixes,
                           root_ok=True)
                os.symlink(m.linkname, dest)
            elif kind == "hardlink":
                if not m.islnk() or m.linkname != want.get("target") \
                        or listed.get(m.linkname, {}).get("type") != "file" \
                        or m.linkname not in done:
                    raise HandoffError(f"{m.name}: not the hardlink the manifest lists")
                os.link(staging / m.linkname, dest)
            else:
                raise HandoffError(f"{m.name}: unknown manifest type {kind!r}")
            done.add(m.name)
    allowed = tuple(os.path.join(real_staging, pre) for pre in prefixes)
    for name in done:
        if listed[name]["type"] == "symlink":
            real = os.path.realpath(staging / name)
            if not (real + "/").startswith(allowed):
                raise HandoffError(f"link {name!r} resolves outside the stages the "
                                   f"manifest ships")
    missing = sorted(set(listed) - done)
    if missing:
        raise HandoffError(f"{len(missing)} file(s) the manifest lists are not in the "
                           f"bundle, e.g. {missing[0]}")
    return nbytes


def _install(cfg: JobConfig, manifest: dict, staging: Path) -> tuple[list[str], list[str]]:
    """Move the staged stage dirs into the cache. (installed, already present)."""
    keys = cfg.keys()
    ctx = Ctx(job_id=0, cfg=cfg, gpu=-1, keys=keys)
    installed, present = [], []
    for st in SHIPPED:
        ent = manifest["stages"].get(st) or {}
        if not ent.get("shipped"):
            continue
        d = CACHE_ROOT / st / keys[st]
        if is_cached(d):
            try:
                _verify(ctx, st, read_done(d))
                present.append(st)
                continue
            except Exception:                                 # noqa: BLE001
                pass                                          # rebuild it from the bundle
        if not take_lock(d, os.getpid()):
            lk = read_lock(d)
            if lock_holder_alive(lk):
                raise HandoffError(f"{st} {keys[st]} is being built here by pid "
                                   f"{lk.get('pid')}; import again when it is done")
            release_lock(d)
            if not take_lock(d, os.getpid()):
                raise HandoffError(f"could not lock {st} {keys[st]}")
        try:
            reset_stage_dir(d)
            src = staging / "cache" / st / keys[st]
            for child in src.iterdir():
                os.replace(child, d / child.name)
            mark_done(d, _localize(ent.get("info") or {}))
        finally:
            release_lock(d)
        db.cache_put(keys[st], st, str(d), dir_bytes(d))
        installed.append(st)
    # After every stage is in place: the fisheye dataset links into select.
    for st in installed + present:
        d = CACHE_ROOT / st / keys[st]
        try:
            _verify(ctx, st, read_done(d))
        except Exception as exc:                              # noqa: BLE001
            for s in installed:
                (CACHE_ROOT / s / keys[s] / ".done").unlink(missing_ok=True)
            raise HandoffError(f"imported {st} does not pass its cache check: {exc}") from None
    return installed, present


def _fetch_record(name: str) -> Optional[dict]:
    """How the entrypoint's HANDOFF_URL download of this bundle went, if it did."""
    try:
        rec = json.loads(FETCH_RECORD.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or rec.get("file") != name:
        return None
    return {k: rec.get(k) for k in ("bytes", "seconds", "first_byte_s", "ended")}


def import_bundle(path: Path, sha256: Optional[str] = None, name: Optional[str] = None,
                  train: Optional[dict] = None, export: Optional[dict] = None,
                  priority: int = 0, keep: Optional[bool] = None) -> dict:
    """Verify a bundle, install its stages, and queue a job that starts at train.

    train/export replace those sections of the bundle's config: a different
    training budget trains from the same SfM. Everything upstream has to stay
    as the bundle made it, which the key check enforces. A bundle in
    HANDOFF_ROOT is deleted afterwards unless keep (default KEEP_BUNDLES).
    """
    t0 = time.time()
    if sha256:
        got = file_sha256(path)
        if got != sha256.strip().lower():
            raise HandoffError(f"{path.name}: sha256 {got[:16]}... is not the expected "
                               f"{sha256.strip()[:16]}...")
    manifest = read_manifest(path)
    cfg = check_manifest(manifest)
    d = cfg.model_dump()
    d["run_until"] = None
    if name:
        d["name"] = name
    if train is not None:
        d["train"] = train
    if export is not None:
        d["export"] = export
    try:
        job_cfg = JobConfig.model_validate(d)
    except ValidationError as exc:
        raise HandoffError(f"config for the train job: {exc}") from None
    keys = job_cfg.keys()
    if any(keys[st] != manifest["keys"][st] for st in UPSTREAM):
        raise HandoffError("the train/export overrides changed an upstream key")

    need = sum(f.get("size") or 0 for f in manifest.get("files") or [])
    free = shutil.disk_usage(CACHE_ROOT).free
    if free < need * 1.05 + (1 << 30):
        raise HandoffError(f"{need / 1e9:.1f} GB to unpack and {free / 1e9:.1f} GB free "
                           f"in the cache")

    staging = CACHE_ROOT / f".handoff-{manifest['id'][:12]}-{uuid.uuid4().hex[:6]}"
    staging.mkdir(parents=True)
    try:
        nbytes = _extract(path, manifest, staging)
        installed, present = _install(job_cfg, manifest, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    stages = {st: {"key": manifest["stages"][st]["key"],
                   "shipped": bool(manifest["stages"][st].get("shipped")),
                   "skipped": bool(manifest["stages"][st].get("skipped")),
                   "info": _localize(manifest["stages"][st].get("info") or {})}
              for st in UPSTREAM}
    jid = db.create_job(job_cfg.name, job_cfg.model_dump(), priority=priority)
    db.set_handoff(jid, {"id": manifest["id"], "bundle": path.name,
                         "source_job": (manifest.get("job") or {}).get("id"),
                         "image": manifest.get("image"),
                         "clip_seconds": manifest.get("clip_seconds"),
                         "stages": stages})
    if job_cfg.mask.enabled and job_cfg.mask.review:
        db.set_review(jid, "approved", f"approved before handoff {manifest['id'][:12]}")
    for st, key in keys.items():
        sd = CACHE_ROOT / st / key
        if st in UPSTREAM:
            state = "skipped" if stages[st]["skipped"] else "imported"
            db.upsert_stage(jid, st, key, state,
                            path=str(sd) if stages[st]["shipped"] else None)
        else:
            db.upsert_stage(jid, st, key, "cached" if is_cached(sd) else "pending",
                            path=str(sd))
    size = path.stat().st_size
    removed = False
    if not (KEEP_BUNDLES if keep is None else keep) and path.parent.resolve() == HANDOFF_ROOT.resolve():
        try:
            path.unlink()
            removed = True
        except OSError as exc:
            print(f"job {jid}: could not delete {path.name} after import: {exc}")
    st = _set_status(jid, state="imported", role="train", id=manifest["id"],
                     file=path.name, bytes=size, files=len(manifest["files"]),
                     payload_bytes=nbytes, stages=installed + present,
                     already_cached=present, import_s=round(time.time() - t0, 1),
                     bundle_removed=removed,
                     source_image=manifest.get("image"), fetch=_fetch_record(path.name))
    print(f"job {jid}: imported handoff {manifest['id'][:12]} from {path.name} "
          f"({', '.join(installed) or 'nothing new'} installed"
          + (f", {', '.join(present)} already cached" if present else "")
          + ("; bundle deleted" if removed else "") + ")")
    return {"id": jid, "handoff": manifest["id"], "keys": keys,
            "installed": installed, "already_cached": present, "status": st}
