"""Does a self-hosted lab-fork build render the graphene flywire_public layer
with a CAVE token seeded as storage_state?  Serves dist/client from disk via
Playwright page.route on a synthetic origin (the mechanism proposed for
ChromeRenderer), then reports auth statuses, layer readiness and whether 2D
segmentation chunks and the old root's mesh actually arrive.

    NGL_DIST=/scratch/kp0374/ngl_fork_dist uv run --no-sync python scripts/probe_fork_graphene.py
"""
import json
import mimetypes
import os
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path

from playwright.sync_api import sync_playwright

DIST = Path(os.environ.get("NGL_DIST", "/scratch/kp0374/ngl_fork_dist"))
ORIGIN = "https://ngl.local"
APP = "https://prodv1.flywire-daf.com"
LOGIN_URL = "https://global.daf-apis.com/sticky_auth"
ROOT = os.environ.get("PROBE_ROOT", "720575940625112137")
OUT = os.environ.get("PROBE_OUT", "/scratch/kp0374/native_spike")
SECS = int(os.environ.get("PROBE_SECS", "90"))

STATE = {
    "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
    "position": [215675.25, 61215.0, 3828.825],
    "crossSectionScale": 8.0,
    "projectionOrientation": [-0.47, 0.80, -0.30, 0.199],
    "projectionScale": 16485.27,
    "layers": [
        {"type": "image", "source": "precomputed://gs://flywire_em/aligned/v1", "name": "EM"},
        {"type": "segmentation",
         "source": f"graphene://middleauth+{APP}/segmentation/1.0/flywire_public",
         "segments": [ROOT], "name": "pre-edit"},
    ],
    "layout": "xy-3d",
}


def storage_state():
    tok = json.load(open(os.path.expanduser("~/.cloudvolume/secrets/cave-secret.json")))["token"]
    entry = {"tokenType": "Bearer", "accessToken": tok, "url": LOGIN_URL, "appUrls": [APP]}
    return {"cookies": [], "origins": [{"origin": ORIGIN, "localStorage": [
        {"name": f"auth_token_v2_{LOGIN_URL}", "value": json.dumps(entry)}]}]}


def serve(route, request):
    path = urllib.parse.urlparse(request.url).path.lstrip("/") or "index.html"
    f = DIST / path
    if not f.is_file():
        route.fulfill(status=404, body="not found")
        return
    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
    route.fulfill(status=200, body=f.read_bytes(), headers={"content-type": ctype})


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=[
        "--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
        "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    ctx = browser.new_context(viewport={"width": 1000, "height": 700}, storage_state=storage_state())
    ctx.route(f"{ORIGIN}/**", serve)
    page = ctx.new_page()
    statuses, notable = Counter(), []
    def on_response(r):
        host = urllib.parse.urlparse(r.url).netloc
        statuses[(host, r.status)] += 1
        if r.status >= 400 and "ngl.local" not in host:
            notable.append(f"{r.status} {r.url[:130]}")
    page.on("response", on_response)
    errs = []
    page.on("console", lambda m: errs.append(f"console.{m.type}: {m.text[:150]}") if m.type == "error" else None)
    page.on("pageerror", lambda e: errs.append(f"pageerror: {str(e)[:150]}"))

    t0 = time.time()
    page.goto(ORIGIN + "/#!" + urllib.parse.quote(json.dumps(STATE), safe=""), timeout=90_000)
    ready = False
    while time.time() - t0 < SECS:
        try:
            ready = page.evaluate("() => !!(window.viewer && window.viewer.isReady && window.viewer.isReady())")
        except Exception:
            ready = False
        if ready and time.time() - t0 > 20:
            break
        time.sleep(0.5)

    def ev(js, label):
        try:
            return page.evaluate(js)
        except Exception as e:
            return f"err[{label}] {str(e)[:100]}"

    print(f"READY={ready} t={time.time()-t0:.0f}s")
    print("state keys:", ev("() => Object.keys(window.viewer.state.toJSON()).join(',')", "keys"))
    print("segments:  ", ev("() => JSON.stringify(window.viewer.state.toJSON().layers[1].segments)", "segs"))
    print("layer msgs:", ev("""() => { const l = window.viewer.layerManager.managedLayers[1].layer;
        if (!l) return 'layer not constructed';
        return JSON.stringify(l.dataSources.map(ds => ds.loadState && ds.loadState.error ? String(ds.loadState.error).slice(0,150) : 'loaded')); }""", "msgs"))
    # Did segmentation voxels and the mesh actually arrive?
    print("chunk hosts:")
    for (h, s), n in sorted(statuses.items()):
        if "ngl.local" not in h:
            print(f"   {h}: {s} x{n}")
    print("mesh requests:", sum(n for (h, s), n in statuses.items() if "flywire-daf" in h and s < 400))
    for e in errs[:5]:
        print("  ", e)
    for e in notable[:5]:
        print("   HTTP", e)
    page.screenshot(path=f"{OUT}/probe_fork_graphene.png")
    print("PROBE-DONE")
