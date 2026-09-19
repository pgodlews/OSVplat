#!/usr/bin/env python3
"""Print a ranked comparison table for a sweep.

usage:  summarize_sweep.py <sweep_id|job_id ...>
        summarize_sweep.py --latest

Run on the box (stdlib only). Emits a markdown table ready to paste into
notes, plus the config differences that explain it.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

B = os.environ.get("QUEUE_URL", "http://127.0.0.1:8090")
# Where deploy.sh keeps the shared secret, as a 0600 systemd EnvironmentFile.
ENV_FILE = Path(os.environ.get(
    "QUEUE_ENV_FILE", Path.home() / "splat" / "queue_app" / ".queue_env"))


def token() -> str:
    """The queue's shared secret, from the environment or deploy.sh's env file.

    Every request needs one. The loopback exemption in main.py applies only when
    the service has NO token configured, and deploy.sh always generates one --
    so sending nothing, as this script used to, is a 401 even from the box
    itself.
    """
    tok = os.environ.get("QUEUE_TOKEN", "").strip()
    if tok:
        return tok
    try:
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("QUEUE_TOKEN="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def get(path):
    req = urllib.request.Request(B + path)
    tok = token()
    if tok:
        req.add_header("X-Queue-Token", tok)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            sys.exit(f"{B} rejected the token. Set QUEUE_TOKEN, or run this on "
                     f"the box where {ENV_FILE} is readable.")
        raise


# limit=1000, not the 200 default: a sweep of 64 jobs is easily pushed off the
# first page by later work. Cleared ("hidden") jobs are still excluded -- clear
# finished only hides, so a purge is the only thing that really loses a sweep.
jobs = get("/api/jobs?limit=1000")
args = sys.argv[1:]
if not args or args[0] == "--latest":
    sweeps = [j for j in jobs if j.get("sweep_id")]
    if not sweeps:
        sys.exit("no sweeps found")
    # /api/jobs is ordered for the QUEUE view -- running first, then queued in
    # the order the dispatcher will take them, then finished newest-first -- so
    # its first row is whatever is running or the OLDEST queued job, not the
    # latest sweep. Go by the highest job id instead.
    target = max(sweeps, key=lambda j: j["id"])["sweep_id"]
    sel = [j for j in jobs if j.get("sweep_id") == target]
    print(f"sweep {target}\n")
elif len(args) == 1 and not args[0].isdigit():
    sel = [j for j in jobs if j.get("sweep_id") == args[0]]
    print(f"sweep {args[0]}\n")
else:
    ids = {int(a) for a in args}
    sel = [j for j in jobs if j["id"] in ids]

if not sel:
    sys.exit("no matching jobs")
sel.sort(key=lambda j: j["id"])


def label(j):
    n = j["name"]
    return n[n.index("[") + 1:n.rindex("]")] if "[" in n and "]" in n else n


def fmt(v, kind="", unit=""):
    if v is None:
        return "—"
    if kind == "bytes":
        return f"{v/1e6:.0f} MB"
    if kind == "time":
        if v < 120:
            return f"{v:.0f} s"
        return f"{v/60:.1f} min" if v < 3600 else f"{v/3600:.2f} h"
    if kind == "m":
        return f"{v/1e6:.2f}M"
    if kind == "int":
        return f"{int(round(v)):,}"
    if isinstance(v, float):
        # %g turns 15000.0 into 1.5e+04, which is unreadable in a results table.
        out = f"{int(round(v)):,}" if v == int(v) and abs(v) >= 1000 else f"{v:.4g}"
    else:
        out = str(v)
    return out + unit


rows = []
for j in sel:
    m = j["metrics"]
    t = j["config"]["train"]
    rows.append({
        "id": j["id"], "label": label(j), "state": j["state"],
        "psnr": m.get("psnr"), "ssim": m.get("ssim"),
        "splats": m.get("splats"), "iters": m.get("final_step"),
        "secs": m.get("train_seconds"), "vram": m.get("peak_vram_mib"),
        "ply": m.get("ply_bytes"), "sog": m.get("sog_bytes"),
        "width": t.get("max_width"), "sh": t.get("sh_degree"),
        "scaler": t.get("steps_scaler"),
    })

done = [r for r in rows if r["psnr"] is not None]
done.sort(key=lambda r: -r["psnr"])
base = next((r for r in rows if r["label"] == "baseline"), None)

print("| # | variant | PSNR | ΔPSNR | SSIM | splats | iters | train | peak VRAM | PLY | state |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in (done + [x for x in rows if x["psnr"] is None]):
    d = ("—" if not base or base["psnr"] is None or r["psnr"] is None
         else f"{r['psnr'] - base['psnr']:+.2f}")
    print(f"| {r['id']} | {r['label']} | {fmt(r['psnr'])} | {d} | "
          f"{fmt(r['ssim'])} | {fmt(r['splats'],'m')} | {fmt(r['iters'],'int')} | "
          f"{fmt(r['secs'],'time')} | "
          f"{fmt(r['vram'],unit=' MiB')} | {fmt(r['ply'],'bytes')} | "
          f"{r['state']} |")

if base and base["psnr"] is not None:
    print(f"\nbaseline = job {base['id']} "
          f"(PSNR {base['psnr']:.3f}, {fmt(base['secs'],'time')})")
    winners = [r for r in done if r["psnr"] > base["psnr"] and r["id"] != base["id"]]
    if winners:
        print("beats baseline: " + ", ".join(
            f"{r['label']} ({r['psnr']-base['psnr']:+.2f} dB, "
            f"{(r['secs'] or 0)/(base['secs'] or 1):.1f}x time)"
            for r in winners))
    else:
        print("nothing beat the baseline on PSNR")
