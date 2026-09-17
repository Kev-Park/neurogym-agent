"""Can the upstream Neuroglancer build (neuroglancer-demo.appspot.com) load the
pre-edit flywire_public graphene layer with the cluster's CAVE token seeded as
a Playwright storage_state?  Runs headless on CPU (SwiftShader); prints per
case the response statuses per host, viewer readiness and the visible
segments.  Never prints the token.

    uv run --no-sync python scripts/probe_chrome_middleauth.py [out-dir]
"""
import json
import os
import sys
import time
import urllib.parse
from collections import Counter

from playwright.sync_api import sync_playwright

ORIGIN = "https://neuroglancer-demo.appspot.com"
APP = "https://prodv1.flywire-daf.com"
LOGIN_URL = "https://global.daf-apis.com/sticky_auth"
ROOT = os.environ.get("PROBE_ROOT", "720575940625112137")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/scratch/kp0374/native_spike"


def state(seg_source):
    return {
        "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
        "position": [215675.25, 61215.0, 3828.825],
        "crossSectionScale": 8.0,
        "projectionScale": 16485.27,
        "layers": [
            {"type": "image", "source": "precomputed://gs://flywire_em/aligned/v1", "name": "EM"},
            {"type": "segmentation", "source": seg_source, "segments": [ROOT], "name": "pre-edit"},
        ],
        "layout": "xy-3d",
    }


def storage_state():
    tok = json.load(open(os.path.expanduser("~/.cloudvolume/secrets/cave-secret.json")))["token"]
    entry = {"tokenType": "Bearer", "accessToken": tok, "url": LOGIN_URL, "appUrls": [APP]}
    return {"cookies": [], "origins": [{"origin": ORIGIN,
             "localStorage": [{"name": f"auth_token_v2_{LOGIN_URL}", "value": json.dumps(entry)}]}]}


CASES = [c for c in [
    ("middleauth+token", f"graphene://middleauth+{APP}/segmentation/1.0/flywire_public", True),
    ("middleauth-notoken", f"graphene://middleauth+{APP}/segmentation/1.0/flywire_public", False),
    ("plain+token", f"graphene://{APP}/segmentation/1.0/flywire_public", True),
] if c[0] in os.environ.get("PROBE_CASES", c[0])]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=[
        "--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
        "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    for name, src, with_token in CASES:
        ctx = browser.new_context(viewport={"width": 800, "height": 600},
                                  storage_state=storage_state() if with_token else None)
        page = ctx.new_page()
        statuses = Counter()
        urls = []
        def on_response(r):
            statuses[(urllib.parse.urlparse(r.url).netloc, r.status)] += 1
            if r.status != 206 and "appspot" not in r.url:
                urls.append(f"{r.status} {r.url[:150]}")
        page.on("response", on_response)
        console = []
        page.on("console", lambda m: console.append(f"console.{m.type}: {m.text[:160]}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console.append(f"pageerror: {str(e)[:160]}"))
        url = ORIGIN + "/#!" + urllib.parse.quote(json.dumps(state(src)), safe="")
        t0 = time.time()
        page.goto(url, timeout=60_000)
        ready = False
        for _ in range(int(os.environ.get('PROBE_POLLS', 240))):
            try:
                ready = page.evaluate("() => !!(window.viewer && window.viewer.isReady && window.viewer.isReady())")
            except Exception:
                ready = False
            if ready and time.time() - t0 > 15:
                break
            time.sleep(0.25)
        try:
            segs = page.evaluate("() => JSON.stringify(window.viewer.state.toJSON().layers[1].segments)")
        except Exception as e:
            segs = f"err {type(e).__name__}"
        try:
            layer = page.evaluate("""() => { const ml = window.viewer.layerManager.managedLayers[1]; const l = ml.layer;
                if (!l) return 'layer not constructed';
                return JSON.stringify({ready: ml.isReady ? ml.isReady() : null, ds: l.dataSources.map(ds => ({
                    load: ds.loadState ? (ds.loadState.error ? String(ds.loadState.error).slice(0, 200) : 'loaded') : 'loading',
                    msgs: ((ds.messages && ds.messages.messages) || []).map(m => m.severity + ': ' + m.message)}))}); }""")
        except Exception as e:
            layer = f"err {str(e)[:120]}"
        text = page.evaluate("() => document.body.innerText")
        msgs = [ln for ln in text.splitlines() if any(k in ln.lower() for k in ("login", "middleauth", "error", "unverified"))]
        page.screenshot(path=f"{OUT}/probe_middleauth_{name}.png")
        hosts = {}
        for (h, s), n in sorted(statuses.items()):
            hosts.setdefault(h, []).append(f"{s}x{n}")
        print(f"CASE {name}: ready={ready} t={time.time()-t0:.0f}s segments={segs}")
        print(f"   layer: {layer[:300]}")
        for c in console[:6]:
            print(f"   {c}")
        for h, v in hosts.items():
            print(f"   {h}: {' '.join(v)}")
        for m in msgs[:4]:
            print(f"   msg: {m[:120]}")
        for u in urls[:12]:
            print(f"   {u}")
        ctx.close()
    browser.close()
print("PROBE-DONE")
