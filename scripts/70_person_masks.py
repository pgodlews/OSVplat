#!/usr/bin/env python3
"""Per-frame person masks for equirectangular panoramas.

Why this exists: on a handheld 360 rig the operator is *rigidly attached* to the
camera. Their head and arm occupy the same solid angle in every frame, so they
are never observed from a second viewpoint, and the trainer bakes them in as a
smear anchored to the camera path. Anyone walking alongside behaves almost the
same way. Neither is scene geometry, and no trainer setting removes them.

Why not temporal variance (the obvious cheap trick, and what the project notes
originally planned): it assumes the rig is the only thing that holds still in
the camera frame. Walk down a footbridge and the parapets sit in the same
equirect pixels the whole way too -- measured on osmo360, the nadir band's mean
temporal std was 24 while the *sky* was 8.6. A low-variance mask deletes the
subject. See docs/troubleshooting.md #20.

So: segment people semantically. Two backends, both kept because they trade off
differently:

  maskrcnn  torchvision Mask R-CNN, COCO `person` class only. BSD-licensed, no
            weights to fetch, no licence to accept. A COCO detector has never
            seen an equirect -- at the nadir a person is smeared across the full
            image width and nothing fires -- so each panorama is resampled into
            13 overlapping 90-degree pinhole views, where the detector is
            in-distribution, and the masks are carried back to equirect.

  sam3      SAM 3, open-vocabulary text prompts. Reads the equirect DIRECTLY, so
            the 13-view scaffolding collapses to one inference plus a nadir view
            -- and measured on osmo360 it is both faster (0.53 vs 0.70 s/frame on
            a 3090) and tighter, because Mask R-CNN was blanketing real deck
            around the people. Needs its gated weights on disk
            (scripts/get_mask_weights.sh sam3), and comes under Meta's SAM
            licence rather than BSD.

Masks are written white = keep, black = ignore, at full source resolution --
the polarity and size both LichtFeld and COLMAP want. They disagree about the
FILENAME, though: LichtFeld matches by stem (`pano_0000.png`) while COLMAP wants
the image name with .png appended (`pano_0000.jpg.png`). Putting both in one
directory risks LichtFeld's ambiguous-match path, so --colmap-out writes the
second layout as hardlinks into a directory of its own.

usage:
  70_person_masks.py --panos DIR --out DIR [--colmap-out DIR] [--overlay DIR]
                     [--score 0.5] [--dilate 9] [--work-width 1920] [--force]

Run with venv_gs (torch + torchvision + opencv).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from io_pool import WriteBehind, read_ahead

PERSON_LABEL = 1  # COCO

# torch and the model libraries are imported inside the backends on purpose:
# both now run in venv_gs, but an install may still point a backend at its own
# interpreter (QUEUE_MASK_PY), and a module-scope import of one backend's
# library would make the script refuse to start where only the other exists.

# Tangent views. The two rings plus the nadir cover everything from just above
# the horizon down, which is where anyone on or beside the rig has to be: the
# operator holds the camera above their head or in front of their chest, and a
# companion walking alongside spans roughly +15 to -60 degrees at arm's length.
# 90-degree views at these pitches overlap enough that a person split across two
# views is caught whole by their union.
DEFAULT_VIEWS = ([(y, -30) for y in range(0, 360, 45)]
                 + [(y, -75) for y in range(0, 360, 90)]
                 + [(0, -90)])


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera->world rotation. Negative pitch looks DOWN.

    The sign matters and is easy to get backwards: with Rx as written, a
    positive angle tilts the forward axis up, so the pitch is negated here.
    Getting this wrong points every "downward" view at the sky, the detector
    finds nothing, and the masks come out empty but the run still succeeds.
    """
    y, p = np.deg2rad(yaw_deg), np.deg2rad(-pitch_deg)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return ry @ rx


def equirect_directions(width: int, height: int):
    """Unit direction per equirect pixel, and the latitude of each row."""
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    lon = (u * 2 - 1) * np.pi
    lat = (0.5 - v) * np.pi
    clat = np.cos(lat)[:, None]
    dirs = np.stack([clat * np.sin(lon)[None, :],
                     np.repeat(np.sin(lat)[:, None], width, axis=1),
                     clat * np.cos(lon)[None, :]], axis=-1).astype(np.float32)
    return dirs, lat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panos", required=True, help="directory of equirect frames")
    ap.add_argument("--out", required=True, help="directory to write masks into")
    ap.add_argument("--overlay", default="", help="optional dir for visual check images")
    ap.add_argument("--backend", choices=("maskrcnn", "sam3"), default="maskrcnn",
                    help="maskrcnn: torchvision Mask R-CNN, COCO person class "
                         "only, BSD-licensed, no setup. sam3: open-vocabulary "
                         "text prompts, needs its weights "
                         "(scripts/get_mask_weights.sh sam3). New backends: "
                         "queue/app/mask_backends.py.")
    ap.add_argument("--prompts", default="person",
                    help="sam3 only: comma-separated concepts, e.g. "
                         "'person,dog'. Each is a separate concept segmented "
                         "exhaustively; anything you do not name is simply not "
                         "masked, which is how 'people and cats but not dogs' "
                         "is expressed -- there is no textual negation.")
    ap.add_argument("--model", default=os.path.join(os.path.expanduser(
                        os.environ.get("SPLAT_ROOT", "~/splat")), "models/sam3"),
                    help="sam3 only: local weights directory")
    ap.add_argument("--nadir-view", dest="nadir_view", action="store_true",
                    default=True, help="sam3 only (default on)")
    ap.add_argument("--no-nadir-view", dest="nadir_view", action="store_false")
    ap.add_argument("--score", type=float, default=None,
                    help="detection confidence floor; defaults per backend "
                         "(0.5 maskrcnn, 0.3 sam3)")
    ap.add_argument("--dilate", type=int, default=9,
                    help="mask growth in working pixels; segmentation edges cut "
                         "through the colour halo around a person, and an "
                         "unmasked rim is still a camera-fixed object")
    ap.add_argument("--work-width", type=int, default=1920,
                    help="equirect working width; masks are written at full "
                         "source resolution regardless")
    ap.add_argument("--view-size", type=int, default=768)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="rewrite masks that exist")
    ap.add_argument("--colmap-out", default="",
                    help="also write COLMAP-named masks here: <image name>.png, "
                         "so pano_0000.jpg.png. COLMAP and LichtFeld disagree "
                         "about the filename -- LichtFeld matches by stem -- and "
                         "a mask either side cannot find is silently ignored, so "
                         "the two layouts get their own directories rather than "
                         "sharing one and risking an ambiguous match.")
    ap.add_argument("--sheet", default="",
                    help="write a contact sheet of evenly-spaced overlays here. "
                         "This is what a human actually reviews before "
                         "committing a GPU-hour to training; nobody flips "
                         "through 110 frames.")
    ap.add_argument("--sheet-tiles", type=int, default=12)
    ap.add_argument("--summary", default="",
                    help="write the JSON summary here as well as to stdout, so "
                         "a caller does not have to scrape the log for it")
    args = ap.parse_args()

    panos, out = Path(args.panos).expanduser(), Path(args.out).expanduser()
    files = sorted(p for p in panos.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        print(f"no panoramas in {panos}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    overlay_dir = Path(args.overlay).expanduser() if args.overlay else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir = Path(args.colmap_out).expanduser() if args.colmap_out else None
    if colmap_dir:
        colmap_dir.mkdir(parents=True, exist_ok=True)


    def publish_colmap(src_name: str, mask_file: Path) -> None:
        """Hardlink the mask under COLMAP's name. Same inode, no second copy."""
        if not colmap_dir:
            return
        link = colmap_dir / f"{src_name}.png"
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            os.link(mask_file, link)
        except OSError:
            shutil.copyfile(mask_file, link)

    ew = args.work_width
    eh = ew // 2
    vs = args.view_size
    f = vs / 2.0  # 90-degree horizontal FOV

    dirs, lat = equirect_directions(ew, eh)
    solid_angle_w = np.cos(lat)[:, None]

    def build_views(specs):
        """Both directions of the mapping for each tangent view.

        Depends only on geometry, not on the image, so it is computed once:
        equirect->view for sampling the tile, and view->equirect for pushing the
        resulting mask back onto the sphere.
        """
        xs, ys = np.meshgrid(np.arange(vs), np.arange(vs))
        ray = np.stack([(xs - vs / 2 + 0.5) / f, -(ys - vs / 2 + 0.5) / f,
                        np.ones_like(xs)], -1)
        ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
        built = []
        for yaw, pitch in specs:
            rot = rotation(yaw, pitch)
            w = ray @ rot.T
            la = np.arcsin(np.clip(w[..., 1], -1, 1))
            lo = np.arctan2(w[..., 0], w[..., 2])
            cam = dirs @ rot
            z = cam[..., 2]
            safe_z = np.where(z > 1e-6, z, 1e-6)
            px = f * cam[..., 0] / safe_z + vs / 2 - 0.5
            py = -f * cam[..., 1] / safe_z + vs / 2 - 0.5
            built.append({
                "label": f"y{yaw}p{pitch}",
                "map_x": ((lo / np.pi + 1) / 2 * ew).astype(np.float32),
                "map_y": ((0.5 - la / np.pi) * eh).astype(np.float32),
                "px": np.clip(px, 0, vs - 1).astype(np.int32),
                "py": np.clip(py, 0, vs - 1).astype(np.int32),
                "inside": (z > 1e-6) & (px >= 0) & (px < vs - 1)
                          & (py >= 0) & (py < vs - 1),
            })
        return built

    prompts = [t.strip() for t in args.prompts.split(",") if t.strip()]
    score = args.score if args.score is not None else (
        0.5 if args.backend == "maskrcnn" else 0.3)

    if args.backend == "maskrcnn":
        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2)
        if prompts != ["person"]:
            # Silently ignoring the prompts would produce person masks under a
            # config that claims to mask something else, and the cache key would
            # describe the wrong run.
            raise SystemExit(
                f"--backend maskrcnn segments the COCO person class only; "
                f"--prompts {args.prompts!r} needs --backend sam3")
        device = (args.device if torch.cuda.is_available() or args.device == "cpu"
                  else "cpu")
        net = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval().to(device)
        views = build_views(DEFAULT_VIEWS)
        # Where each equirect pixel lands in each view, kept on the GPU so the
        # view masks are carried back there and only the union comes home.
        for v in views:
            v["flat_t"] = torch.from_numpy(v["py"] * vs + v["px"]).long().to(device)
            v["inside_t"] = torch.from_numpy(v["inside"]).to(device)
        # byte -> [0, 1] exactly as the host computes it. CUDA divides by a
        # scalar as a multiply by its reciprocal, which can differ in the last bit.
        to_unit = torch.arange(256, dtype=torch.float32).div(255).to(device)

        def prepare(work):
            # Runs on a reader thread (cv2.remap releases the GIL), so the tiles
            # for the next frame are cut while the GPU works on this one.
            return np.stack([cv2.cvtColor(cv2.remap(work, v["map_x"], v["map_y"], cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_WRAP), cv2.COLOR_BGR2RGB)
                             for v in views])

        def segment(work, tiles):
            # Upload bytes and convert on the GPU: a quarter of the transfer. Each
            # tile is a CHW view of HWC memory, the layout the model always got.
            dev_tiles = to_unit[torch.from_numpy(tiles).to(device).long()].permute(0, 3, 1, 2)
            with torch.no_grad():
                outputs = net(list(dev_tiles))
            found = torch.zeros(eh * ew, dtype=torch.bool, device=device)
            n = 0
            for v, res in zip(views, outputs):
                keep = (res["labels"] == PERSON_LABEL) & (res["scores"] > score)
                if not bool(keep.any()):
                    continue
                n += int(keep.sum())
                vm = (res["masks"][keep, 0] > 0.5).any(0).reshape(-1)
                found |= v["inside_t"].reshape(-1) & vm[v["flat_t"].reshape(-1)]
            return found.reshape(eh, ew).cpu().numpy().astype(np.uint8), n
    else:
        import torch
        from PIL import Image
        from transformers import Sam3Model, Sam3Processor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        proc = Sam3Processor.from_pretrained(args.model)
        net = Sam3Model.from_pretrained(args.model, dtype=dtype).to(device).eval()
        # SAM 3 reads the equirect directly, so the 13-view scaffolding Mask
        # R-CNN needed collapses to one inference. The nadir view is kept as a
        # second: it is where the projection is most distorted AND where the
        # rigidly attached operator sits, which is the one place under-masking
        # cannot be recovered from any other frame.
        views = build_views([(0, -90)]) if args.nadir_view else []

        def sam(img_bgr, h, w_):
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            inp = proc(images=[pil] * len(prompts), text=prompts,
                       return_tensors="pt").to(device)
            with torch.no_grad():
                out_ = net(**inp)
            res = proc.post_process_instance_segmentation(
                out_, threshold=score, mask_threshold=0.5,
                target_sizes=[(h, w_)] * len(prompts))
            u = np.zeros((h, w_), bool)
            n = 0
            for r in res:
                m, sc = r["masks"], r["scores"]
                for i in range(len(sc)):
                    n += 1
                    u |= np.asarray(m[i].cpu() if hasattr(m[i], "cpu")
                                    else m[i]).astype(bool)
            return u, n

        def prepare(work):
            return None

        def segment(work, _):
            found, n = sam(work, eh, ew)
            for v in views:
                tile = cv2.remap(work, v["map_x"], v["map_y"], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_WRAP)
                vm, k = sam(tile, vs, vs)
                n += k
                found |= (v["inside"] & vm[v["py"], v["px"]])
            return found.astype(np.uint8), n

    kernel = (np.ones((args.dilate, args.dilate), np.uint8)
              if args.dilate > 0 else None)

    def load(src):
        """Everything about one frame that needs no GPU, on a reader thread."""
        dst = out / f"{src.stem}.png"
        if dst.exists() and not args.force:
            prev = cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE)
            if prev is not None:
                prev = cv2.resize(prev, (ew, eh), interpolation=cv2.INTER_NEAREST)
            return "cached", prev
        full = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if full is None:
            return "unreadable", None
        work = cv2.resize(full, (ew, eh), interpolation=cv2.INTER_AREA)
        return "new", (full.shape[1], full.shape[0], work, prepare(work))

    def write_mask(people, size, dst, src_name):
        # White = keep, black = ignore. Written at source resolution so the same
        # file can be handed to COLMAP, which requires an exact size match.
        mask_full = 255 - cv2.resize(people * 255, size, interpolation=cv2.INTER_NEAREST)
        if not cv2.imwrite(str(dst), mask_full):
            return False
        publish_colmap(src_name, dst)
        return True

    def write_overlay(work, people, dst):
        tinted = work.copy()
        sel = people > 0
        tinted[sel] = (0.35 * tinted[sel] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        return cv2.imwrite(str(dst), tinted, [cv2.IMWRITE_JPEG_QUALITY, 88])

    stats, empty, t0 = [], [], time.time()
    writer = WriteBehind()
    for i, (src, (kind, data)) in enumerate(read_ahead(load, files)):
        dst = out / f"{src.stem}.png"
        if kind == "cached":
            # Measure it anyway. The summary is what the caller gates on, and a
            # resumed run that reported "0% covered" because it did no work
            # would look exactly like a run where the detector found nobody.
            publish_colmap(src.name, dst)
            if data is not None:
                inv = (data < 128).astype(np.uint8)
                stats.append(float((inv * solid_angle_w).sum()
                                   / (np.ones_like(inv) * solid_angle_w).sum()))
            continue
        if kind == "unreadable":
            print(f"  skipping unreadable {src.name}", file=sys.stderr)
            continue
        full_w, full_h, work, prepared = data

        people, n_det = segment(work, prepared)

        if kernel is not None and people.any():
            people = cv2.dilate(people, kernel, iterations=1)

        cov_sr = float((people * solid_angle_w).sum()
                       / (np.ones_like(people) * solid_angle_w).sum())
        stats.append(cov_sr)
        if n_det == 0:
            empty.append(src.name)

        writer.submit(dst, write_mask, people, (full_w, full_h), dst, src.name)
        if overlay_dir:
            writer.submit(f"overlay for {src.name}", write_overlay, work, people,
                          overlay_dir / f"{src.stem}.jpg")

        # Every five, not every twenty: the queue scrapes this line for the
        # mask progress bar, and at ~0.7 s per panorama twenty is fourteen
        # seconds of a bar that does not move.
        if (i + 1) % 5 == 0 or i + 1 == len(files):
            print(f"  {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    writer.close()   # every mask, link and overlay on disk before counting them

    summary = {
        "frames": len(files),
        "masked": len(stats),
        "coverage_solid_angle_mean": float(np.mean(stats)) if stats else 0.0,
        "coverage_solid_angle_max": float(np.max(stats)) if stats else 0.0,
        "frames_without_detection": empty,
        "backend": args.backend,
        "prompts": prompts,
        "score": score,
        "dilate": args.dilate,
        "views": len(views) + (1 if args.backend == "sam3" else 0),
        "colmap_masks": len(list(colmap_dir.glob("*.png"))) if colmap_dir else 0,
        "seconds": round(time.time() - t0, 1),
    }
    # A frame with no detections is worth surfacing rather than averaging away.
    # The operator is in shot in every single frame by construction, so a zero
    # here is a detector miss, not an empty scene -- and that frame will train
    # the person straight back in.
    if empty:
        print(f"WARNING: {len(empty)} frame(s) had no person detected: "
              f"{', '.join(empty[:6])}{' ...' if len(empty) > 6 else ''}",
              file=sys.stderr)
    if args.sheet:
        # Sample across the clip rather than taking the first N: detection
        # failures cluster where the capture changes (someone turns, the sun
        # moves), and a sheet of the opening seconds would miss exactly that.
        # Any frame that found nobody is forced into the sheet, because that is
        # the frame a reviewer most needs to see.
        pool = sorted(overlay_dir.glob("*.jpg")) if overlay_dir else []
        if pool:
            empty_stems = {Path(n).stem for n in empty}
            forced = [p for p in pool if p.stem in empty_stems]
            rest = [p for p in pool if p.stem not in empty_stems]
            take = max(1, args.sheet_tiles - len(forced))
            step = max(1, len(rest) // take)
            chosen = (forced + rest[::step])[:args.sheet_tiles]
            chosen.sort()
            # Keep each tile paired with the file it came from. Filtering only
            # the images and then zipping them back against `chosen` shifted
            # every later label by one as soon as a single overlay failed to
            # read -- including the red NO DETECTION tag, on the one image a
            # human looks at before committing a GPU-hour.
            tiles = [(t, p) for p in chosen
                     if (t := cv2.imread(str(p))) is not None]
            if tiles:
                tw = 640
                first = tiles[0][0]
                th = int(first.shape[0] * tw / first.shape[1])
                cols = 2 if len(tiles) <= 4 else 3
                rows_n = (len(tiles) + cols - 1) // cols
                sheet = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
                for idx, (t, src_p) in enumerate(tiles):
                    t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
                    tag = src_p.stem + ("  NO DETECTION" if src_p.stem in empty_stems else "")
                    cv2.rectangle(t, (0, 0), (tw, 20), (0, 0, 0), -1)
                    cv2.putText(t, tag, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 255) if src_p.stem in empty_stems else (255, 255, 255),
                                1, cv2.LINE_AA)
                    r, c = divmod(idx, cols)
                    sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
                sp = Path(args.sheet).expanduser()
                sp.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
                summary["sheet"] = str(sp)
                # What is actually ON the sheet, not what was picked for it.
                summary["sheet_frames"] = [p.stem for _, p in tiles]

    if args.summary:
        sp = Path(args.summary).expanduser()
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
