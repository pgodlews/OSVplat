#!/usr/bin/env python3
"""Keep the sharpest frame (Laplacian variance on a 1/4-scale grayscale decode) out of every W consecutive candidates.
usage: 20_select_sharp.py W SRC DST   -> hardlinks DST/pano_NNNN.jpg"""
import cv2, os, sys, glob, time, numpy as np
from concurrent.futures import ProcessPoolExecutor
from io_pool import io_threads
if len(sys.argv) != 4: sys.exit(__doc__.strip().splitlines()[-1])
W, src, dst = int(sys.argv[1]), sys.argv[2], sys.argv[3]
files = sorted(glob.glob(src + '/*.jpg'))
if not files: sys.exit(f'no candidate JPEGs in {src}')
def score(f):
    img = cv2.imread(f, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    # A truncated JPEG -- the last frame of an interrupted ffmpeg dump is the
    # usual one -- decodes to None, and cv2.Laplacian(None) raises inside the
    # process pool, which failed the whole select stage over one bad frame.
    # -1 sorts below every real score, so a readable neighbour always wins.
    if img is None: return -1.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())
t = time.time()
# SPLAT_THREADS: the container's CPU allowance (queue/app/resources.py).
with ProcessPoolExecutor(io_threads(16)) as ex: scores = list(ex.map(score, files, chunksize=8))
os.makedirs(dst, exist_ok=True)
sel = []
for i in range(0, len(files), W):
    j = max(range(i, min(i + W, len(files))), key=lambda k: scores[k])
    # Every frame in this window was unreadable; skip it rather than hardlink a
    # file COLMAP will fail on later.
    if scores[j] >= 0: sel.append(j)
bad = sum(1 for s in scores if s < 0)
if not sel: sys.exit(f'every one of {len(files)} candidates was unreadable')
for k, j in enumerate(sel):
    out = f'{dst}/pano_{k:04d}.jpg'
    # Replace, never skip: an existing pano_0004.jpg from an earlier run with a
    # different window is a DIFFERENT frame, and keeping it silently mixes two
    # selections into one dataset.
    if os.path.lexists(out): os.unlink(out)
    os.link(files[j], out)
s = np.array(scores); ok = s[s >= 0]; ss = s[sel]
print(f'{len(files)} candidates -> {len(sel)} selected (window {W}) in {time.time()-t:.1f}s; '
      f'score all: median {np.median(ok):.1f} min {ok.min():.1f}; selected: median {np.median(ss):.1f} min {ss.min():.1f}'
      + (f'; SKIPPED {bad} unreadable candidate(s)' if bad else ''))
