#!/usr/bin/env python3
"""Smoke tests for the queue API.

Run on the GPU workstation (stdlib only, no venv needed). deploy.sh installs it alongside
the service, and the token it needs is in the service's own env file:

    set -a; . ~/splat/queue_app/.queue_env; set +a
    python3 ~/splat/queue_app/test_api.py

Creates jobs while the queue is paused and cancels every one it created, so it
never schedules GPU work. It asserts the property the whole design rests on:
a sweep over training flags shares one frames/select/sfm key while each variant
gets a distinct train key.
"""
import json
import urllib.error
import urllib.request

import os

B = os.environ.get("QUEUE_URL", "http://127.0.0.1:8090")
TOKEN = os.environ.get("QUEUE_TOKEN", "")
fails = []

# Every job this test creates, so cleanup can cancel exactly those. Cancelling
# "every queued job" -- which is what this used to do -- destroys the queue of
# whoever happens to be running a sweep at the time.
MINE = []


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"}
    if TOKEN:
        headers["x-queue-token"] = TOKEN
    r = urllib.request.Request(B + path, data=data, method=method,
                               headers=headers)
    try:
        with urllib.request.urlopen(r) as resp:
            raw = resp.read()
            ctype = resp.headers.get("content-type", "")
            if "json" not in ctype:
                return resp.status, raw.decode("utf-8", "replace")
            return resp.status, json.loads(raw or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw.decode()[:300]


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def mine(body):
    """Record every job id in a create/sweep response, then pass it through."""
    if isinstance(body, dict):
        if isinstance(body.get("id"), int):
            MINE.append(body["id"])
        for j in body.get("jobs") or []:
            if isinstance(j, dict) and isinstance(j.get("id"), int):
                MINE.append(j["id"])
    return body


CLIP = os.environ.get("QUEUE_TEST_CLIP", "samples/lub.mp4")

_, _inputs = req("GET", "/api/inputs")
if isinstance(_inputs, list) and not any(
        i.get("file") == CLIP for i in _inputs if isinstance(i, dict)):
    # Fail here rather than 40 lines in. Without the clip every create returns
    # 400, which the "rejected with 400" checks read as a PASS while every
    # sweep check fails -- a confusing report of a problem that is not the
    # service's.
    raise SystemExit(
        f"{CLIP} is not one of this queue's inputs "
        f"({', '.join(str(i.get('file')) for i in _inputs) or 'none'}). "
        f"Set QUEUE_TEST_CLIP to one of them.")

# Never let this test schedule real work: remember the pause state, force
# paused, and restore it at the end.
_, _st0 = req("GET", "/api/status")
WAS_PAUSED = _st0.get("paused", True)
req("POST", "/api/pause", {"paused": True})

# 1. budget guard
code, body = req("POST", "/api/jobs", {"config": {
    "name": "bad", "input": {"file": CLIP},
    "train": {"iter": 15000, "steps_scaler": 0.5}}})
check("iter+steps_scaler rejected with 400", code == 400,
      f"HTTP {code}: {str(body)[:120]}")

# 2. bad trim rejected
code, body = req("POST", "/api/jobs", {"config": {
    "name": "bad2", "input": {"file": CLIP, "trim_start": 50, "trim_end": 10}}})
check("inverted trim rejected with 400", code == 400, f"HTTP {code}")

# 3. missing input rejected
code, body = req("POST", "/api/jobs", {"config": {
    "name": "bad3", "input": {"file": "samples/nope.mp4"}}})
check("missing input rejected with 400", code == 400, f"HTTP {code}")

# 4. sweep shares upstream keys
code, sweep = req("POST", "/api/jobs/sweep", {
    "base": {"name": "keytest", "input": {"file": CLIP},
             "train": {"steps_scaler": 0.5}},
    "axes": {"train.sh_degree": [1, 3], "train.enable_mip": [False, True]}})
mine(sweep)
check("sweep created 4 jobs", code == 200 and len(sweep.get("jobs", [])) == 4,
      f"HTTP {code}")

keys = []
for j in sweep.get("jobs", []):
    _, d = req("GET", f"/api/jobs/{j['id']}")
    keys.append((j["id"], j["name"],
                 {s["stage"]: s["cache_key"] for s in d["stages"]}))
for i, n, k in keys:
    print(f"      #{i} {n[:52]:52s} sfm={k['sfm']} train={k['train']}")

check("all variants share one frames key",
      len({k['frames'] for _, _, k in keys}) == 1)
check("all variants share one select key",
      len({k['select'] for _, _, k in keys}) == 1)
check("all variants share one sfm key",
      len({k['sfm'] for _, _, k in keys}) == 1,
      "<- this is what makes a sweep cheap")
check("every variant has a distinct train key",
      len({k['train'] for _, _, k in keys}) == 4)

# 4b. variants mode: one job per variant, all sharing the upstream cache
code, vs = req("POST", "/api/jobs/sweep", {
    "base": {"name": "vartest", "input": {"file": CLIP},
             "train": {"steps_scaler": 0.5}},
    "variants": [
        {"label": "baseline", "set": {}},
        {"label": "sh3", "set": {"train.sh_degree": 3}},
        {"label": "mip", "set": {"train.enable_mip": True}},
    ]})
mine(vs)
check("variants mode created 3 jobs",
      code == 200 and len(vs.get("jobs", [])) == 3, f"HTTP {code}")
vkeys = [j["keys"] for j in vs.get("jobs", [])]
check("variants share one sfm key",
      len({k["sfm"] for k in vkeys}) == 1)
check("each variant has a distinct train key",
      len({k["train"] for k in vkeys}) == 3)
check("the empty variant is the control",
      any(j["label"] == "baseline" for j in vs.get("jobs", [])))

code, _ = req("POST", "/api/jobs/sweep", {
    "base": {"name": "both", "input": {"file": CLIP}},
    "axes": {"train.sh_degree": [1, 3]},
    "variants": [{"label": "x", "set": {}}]})
check("axes and variants together rejected", code == 400, f"HTTP {code}")

code, _ = req("POST", "/api/jobs/sweep", {
    "base": {"name": "none", "input": {"file": CLIP}}})
check("empty sweep rejected", code == 400, f"HTTP {code}")

# 4c. A sweep is all-or-nothing. A bad LAST variant used to 400 with the earlier
# variants already queued.
_, before = req("GET", "/api/jobs")
n_before = len([j for j in before if j["state"] == "queued"])
code, body = req("POST", "/api/jobs/sweep", {
    "base": {"name": "atomic", "input": {"file": CLIP}},
    "variants": [
        {"label": "ok1", "set": {}},
        {"label": "ok2", "set": {"train.sh_degree": 3}},
        {"label": "bad", "set": {"train.sh_degree": 99}},   # ge=0 le=3
    ]})
check("invalid variant rejects the whole sweep", code == 400, f"HTTP {code}")
_, after = req("GET", "/api/jobs")
n_after = len([j for j in after if j["state"] == "queued"])
check("a rejected sweep queues nothing", n_after == n_before,
      f"{n_before} -> {n_after} queued")

# 4d. The cross product is bounded before it is materialised.
code, body = req("POST", "/api/jobs/sweep", {
    "base": {"name": "huge", "input": {"file": CLIP}},
    "axes": {"train.sh_degree": [0, 1, 2, 3],
             "train.max_width": [1920, 3840, 7680, 1280],
             "train.max_cap": [1000000, 2000000, 3000000, 4000000],
             "train.strategy": ["mrnf", "mcmc", "igs+", "mrnf"]}})
check("oversized sweep rejected", code == 400, f"HTTP {code}: {str(body)[:90]}")
_, after2 = req("GET", "/api/jobs")
check("a rejected oversized sweep queues nothing",
      len([j for j in after2 if j["state"] == "queued"]) == n_before)

code, body = req("POST", "/api/jobs/sweep", {
    "base": {"name": "emptyaxis", "input": {"file": CLIP}},
    "axes": {"train.sh_degree": []}})
check("axis with no values rejected", code == 400, f"HTTP {code}")

# 4e. Input identity is the server's to decide, not the caller's.
code, body = req("POST", "/api/jobs", {"config": {
    "name": "abs", "input": {"file": "/etc/passwd"}}})
check("absolute input path rejected", code == 400, f"HTTP {code}")
code, body = req("POST", "/api/jobs", {"config": {
    "name": "trav", "input": {"file": "../../../etc/hosts"}}})
check("traversal outside SPLAT_ROOT rejected", code == 400, f"HTTP {code}")
_, forged = req("POST", "/api/jobs", {"config": {
    "name": "forged", "input": {"file": CLIP, "quick_hash": "0" * 16}}})
mine(forged)
_, honest = req("POST", "/api/jobs", {"config": {
    "name": "honest", "input": {"file": CLIP}}})
mine(honest)
check("a supplied quick_hash cannot claim another input's cache",
      forged.get("keys") == honest.get("keys"),
      "server recomputes the hash")

# 4f. Values that cannot produce a run are refused at submit time.
code, _ = req("POST", "/api/jobs", {"config": {
    "name": "z", "input": {"file": CLIP}, "frames": {"fps": 0}}})
check("fps 0 rejected", code == 400, f"HTTP {code}")
code, _ = req("POST", "/api/jobs", {"config": {
    "name": "t", "input": {"file": CLIP, "trim_end": -5}}})
check("negative trim_end rejected", code == 400, f"HTTP {code}")
code, _ = req("POST", "/api/jobs", {"config": {
    "name": "typo", "input": {"file": CLIP}, "train": {"sh_degrees": 3}}})
check("misspelt field rejected instead of silently ignored", code == 400,
      f"HTTP {code}")
code, _ = req("POST", "/api/jobs", {"config": {
    "name": "dup", "input": {"file": CLIP},
    "train": {"extra_args": "--iter 500"}}})
check("extra_args repeating a managed flag rejected", code == 400, f"HTTP {code}")
code, _ = req("POST", "/api/jobs", {"config": {
    "name": "q", "input": {"file": CLIP},
    "train": {"extra_args": '--foo "bar'}}})
check("unbalanced quote in extra_args rejected", code == 400, f"HTTP {code}")
code, ok = req("POST", "/api/jobs", {"config": {
    "name": "okargs", "input": {"file": CLIP},
    "train": {"extra_args": "--far-seed-dose 4000"}}})
mine(ok)
check("legitimate extra_args still accepted", code == 200, f"HTTP {code}")

# 5. changing an ingest param must invalidate everything downstream
_, a = req("POST", "/api/jobs", {"config": {
    "name": "fpsA", "input": {"file": CLIP}, "frames": {"fps": 10}}})
mine(a)
_, b = req("POST", "/api/jobs", {"config": {
    "name": "fpsB", "input": {"file": CLIP}, "frames": {"fps": 5}}})
mine(b)
check("fps change invalidates sfm key", a["keys"]["sfm"] != b["keys"]["sfm"])
check("fps change invalidates train key", a["keys"]["train"] != b["keys"]["train"])

# 6. identical config -> identical keys (cache would hit)
_, c = req("POST", "/api/jobs", {"config": {
    "name": "fpsA-again", "input": {"file": CLIP}, "frames": {"fps": 10}}})
mine(c)
check("identical config yields identical keys",
      a["keys"] == c["keys"], "cache reuse works")

# 7. estimate reacts to budget
_, e1 = req("POST", "/api/estimate", {"config": {
    "name": "e1", "input": {"file": CLIP}}})
_, e2 = req("POST", "/api/estimate", {"config": {
    "name": "e2", "input": {"file": CLIP}, "train": {"steps_scaler": 0.5}}})
check("halving the budget roughly halves the training estimate",
      abs(e2["train"] / e1["train"] - 0.5) < 0.02,
      f"{e1['train']:.0f}s -> {e2['train']:.0f}s")

# 8. queue is paused and GPU 0 is seen as foreign-busy
_, st = req("GET", "/api/status")
check("queue reports paused while testing", st["paused"] is True)
g0 = next((g for g in st["gpus"] if g["index"] == 0), {})
print(f"      gpu0 foreign={g0.get('busy_foreign')} procs={len(g0.get('procs', []))}")

# 9. compare endpoint
sj = sweep.get("jobs", [])
ids = ",".join(str(sj[i]["id"]) for i in (0, 2))   # differ in sh_degree
code, cmp = req("GET", f"/api/compare?ids={ids}")
check("compare surfaces the differing field", code == 200 and
      "train.sh_degree" in cmp.get("diff", {}),
      f"diff keys={list(cmp.get('diff', {}))}")
ids2 = ",".join(str(sj[i]["id"]) for i in (0, 1))  # differ in enable_mip only
_, cmp2 = req("GET", f"/api/compare?ids={ids2}")
check("compare omits fields that are identical",
      "train.sh_degree" not in cmp2.get("diff", {}) and
      "train.enable_mip" in cmp2.get("diff", {}),
      f"diff keys={list(cmp2.get('diff', {}))}")

# 10. UI serves
code, html = req("GET", "/")
check("UI is served at /", code == 200 and "OSVplat" in str(html))

# 11. clean up every job THIS TEST created, and nothing else, so an unpause
# cannot run them and a sweep somebody else queued survives the test run.
killed = 0
for jid in MINE:
    code, _ = req("DELETE", f"/api/jobs/{jid}")
    if code == 200:
        killed += 1
print(f"      cancelled {killed} of {len(MINE)} jobs this test created")
_, jobs = req("GET", "/api/jobs")
still = [j["id"] for j in jobs if j["id"] in MINE and j["state"] == "queued"]
check("no jobs from this test left queued", not still, f"left: {still}")

# Restore the queue to the pause state we found it in.
req("POST", "/api/pause", {"paused": WAS_PAUSED})
_, st_end = req("GET", "/api/status")
check("pause state restored", st_end["paused"] == WAS_PAUSED,
      f"paused={st_end['paused']}")

# ------------------------------------------------------------- /metrics

def raw_get(path, headers):
    r = urllib.request.Request(B + path, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.headers.get("content-type", ""), \
                resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type", ""), \
            e.read().decode("utf-8", "replace")


# The endpoint is opt-in, so its absence is a configuration, not a failure --
# but a suite that skipped silently would let a broken exposition sit unnoticed
# on the one box where it IS enabled.
_st_cfg, _ = req("GET", "/api/status")
_metrics_on = bool(isinstance(_, dict) and _.get("metrics"))
_st, _ct, _body = raw_get("/metrics", {"x-queue-token": TOKEN} if TOKEN else {})

if not _metrics_on:
    check("/metrics is absent while QUEUE_METRICS is off", _st == 404,
          f"HTTP {_st} — the route should not exist at all when disabled")
    print("      metrics endpoint disabled on this service; "
          "redeploy with METRICS=1 to exercise it")
else:
  check("/metrics is served", _st == 200, f"HTTP {_st}")
  check("with the exposition content type",
        _ct.startswith("text/plain") and "version=0.0.4" in _ct, _ct)
  check("and carries the queue's own metrics",
        "splatqueue_up 1" in _body and "# TYPE splatqueue_jobs gauge" in _body)
  check("every family declares HELP and TYPE before its samples",
        all(f"# TYPE {n} " in _body
            for n in {l.split("{")[0].split(" ")[0]
                      for l in _body.splitlines() if l and not l.startswith("#")}),
        "a sample without a TYPE line is dropped by strict parsers")

if _metrics_on and TOKEN:
    # Scrapers send Authorization: Bearer; the middleware has to take it or
    # every Prometheus config needs a bespoke header block.
    _st_b, _, _ = raw_get("/metrics", {"authorization": f"Bearer {TOKEN}"})
    check("Authorization: Bearer is accepted", _st_b == 200, f"HTTP {_st_b}")
    _st_n, _, _ = raw_get("/metrics", {})
    check("and metrics are not open to the LAN unauthenticated",
          _st_n == 401, f"HTTP {_st_n}")
    _st_w, _, _ = raw_get("/metrics", {"authorization": "Bearer wrong"})
    check("nor to a wrong bearer token", _st_w == 401, f"HTTP {_st_w}")

# Bearer is now a general way in, not a metrics-only one.
if TOKEN:
    _sb, _ = req("GET", "/api/status")
    check("the status route still answers with the header form", _sb == 200,
          f"HTTP {_sb}")

print()
print("FAILURES:", fails if fails else "none")
raise SystemExit(1 if fails else 0)
