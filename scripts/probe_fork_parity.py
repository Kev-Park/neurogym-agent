"""Does the self-hosted lab-fork build need recalibration vs the appspot build
ngllib.state was fitted against?

Loads the SAME states (public m783 link, no credentials) in both builds and
compares (a) the state Neuroglancer reports back -- position, crossSectionScale,
projectionScale, projectionOrientation, segments -- against what was requested,
(b) the rendered right pane pixel-for-pixel between builds, (c) the zoom
boundary rule ngllib.state encodes (non-positive projectionScale leaves
viewer.state unreadable).

    NGL_DIST=/scratch/kp0374/ngl_fork_dist uv run --no-sync python scripts/probe_fork_parity.py
"""
import json
import mimetypes
import os
import time
import urllib.parse
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

DIST = Path(os.environ.get("NGL_DIST", "/scratch/kp0374/ngl_fork_dist"))
FORK = "https://ngl.local"
APPSPOT = "https://neuroglancer-demo.appspot.com"
OUT = os.environ.get("PROBE_OUT", "/scratch/kp0374/native_spike")
W, H = 1800, 900          # ngllib PaneLayout.window_size
SEG = "precomputed://gs://flywire_v141_m783"
EM = "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14"

BASE = {
    "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
    "position": [143944.703125, 61076.59375, 192.58076477050781],
    "crossSectionScale": 2.0339912586467497,
    "projectionOrientation": [-0.4705163836479187, 0.8044001460075378,
                              -0.30343097448349, 0.1987067461013794],
    "projectionScale": 13976.00585680798,
    "layers": [
        {"type": "image", "source": EM, "tab": "source", "name": "em"},
        {"type": "segmentation", "source": SEG, "tab": "source",
         "segments": ["!720575940623044103", "720575940603464672"], "name": "seg"},
    ],
    "showDefaultAnnotations": False,
    "layout": "xy-3d",
}

def variant(**kw):
    s = json.loads(json.dumps(BASE))
    s.update(kw)
    return s

CASES = {
    "base": BASE,
    "zoom_in": variant(projectionScale=13976.00585680798 / 4),
    "zoom_out": variant(projectionScale=13976.00585680798 * 4),
    "rotated": variant(projectionOrientation=[0.0, 0.7071067811865476, 0.0, 0.7071067811865476]),
    "moved": variant(position=[144200.0, 61300.0, 200.0]),
    "xsection": variant(crossSectionScale=8.0),
    "hidden_swap": variant(layers=[BASE["layers"][0],
        {**BASE["layers"][1], "segments": ["720575940623044103", "!720575940603464672"]}]),
    "ps_zero": variant(projectionScale=0.0),
    "ps_negative": variant(projectionScale=-100.0),
}

FIELDS = ["position", "crossSectionScale", "projectionScale", "projectionOrientation"]


def serve(route, request):
    p = urllib.parse.urlparse(request.url).path.lstrip("/") or "index.html"
    f = DIST / p
    if not f.is_file():
        route.fulfill(status=404, body="not found")
        return
    route.fulfill(status=200, body=f.read_bytes(),
                  headers={"content-type": mimetypes.guess_type(f.name)[0] or "application/octet-stream"})


def right_pane(png_path):
    from PIL import Image
    a = np.asarray(Image.open(png_path).convert("RGB"))
    return a[:, a.shape[1] // 2:]


def run(browser, origin, tag):
    ctx = browser.new_context(viewport={"width": W, "height": H})
    if origin == FORK:
        ctx.route(f"{FORK}/**", serve)
    page = ctx.new_page()
    results = {}
    for name, state in CASES.items():
        url = origin + "/#!" + urllib.parse.quote(json.dumps(state), safe="")
        page.goto(url, timeout=90_000)
        raw, t0 = None, time.time()
        while time.time() - t0 < 25:
            raw = page.evaluate("() => (window.viewer && window.viewer.state) ? JSON.stringify(window.viewer.state) : null")
            if raw:
                break
            time.sleep(0.2)
        ready = False
        while time.time() - t0 < 70:
            try:
                ready = page.evaluate("() => !!(window.viewer && window.viewer.isReady && window.viewer.isReady())")
            except Exception:
                ready = False
            if ready:
                break
            time.sleep(0.25)
        time.sleep(1.0)
        raw = page.evaluate("() => (window.viewer && window.viewer.state) ? JSON.stringify(window.viewer.state) : null")
        st = json.loads(raw) if raw else None
        shot = f"{OUT}/parity_{tag}_{name}.png"
        for attempt in range(3):
            try:
                page.screenshot(path=shot, timeout=60_000)
                break
            except Exception as e:
                print(f"     screenshot retry {attempt}: {type(e).__name__}")
                time.sleep(2)
        results[name] = {"ready": ready, "state": st, "shot": shot}
        print(f"  [{tag}] {name}: ready={ready} readable={st is not None} "
              f"fields={sorted(set(FIELDS) & set(st or {}))}")
    ctx.close()
    return results


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=[
        "--no-sandbox", "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
        # same as ngllib ChromeRenderer._build_launch_args on Linux: the
        # compositor otherwise waits for a frame-rate-limited frame and
        # page.screenshot times out (seen on SwiftShader at 1800x900).
        "--disable-gpu-vsync", "--disable-frame-rate-limit"]
        + (["--use-gl=angle", "--use-angle=vulkan"] if os.environ.get("PROBE_GPU", "1") == "1"
           else ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]))
    print("PROBE_GPU=", os.environ.get("PROBE_GPU", "1"), "| APPSPOT:")
    a = run(browser, APPSPOT, "appspot")
    print("FORK:")
    f = run(browser, FORK, "fork")
    browser.close()

print("\n=== state readback: requested vs each build ===")
for name, state in CASES.items():
    sa, sf = a[name]["state"], f[name]["state"]
    print(f"{name}:")
    if sa is None or sf is None:
        print(f"   appspot readable={sa is not None} fork readable={sf is not None}"
              + ("   <- zoom-boundary rule AGREES" if (sa is None) == (sf is None) else "   <- DISAGREES"))
        continue
    for k in FIELDS:
        want, va, vf = state.get(k), sa.get(k), sf.get(k)
        def fmt(v):
            if isinstance(v, list):
                return "[" + ", ".join(f"{x:.6g}" for x in v) + "]"
            return f"{v:.6g}" if isinstance(v, (int, float)) else str(v)
        same = json.dumps(va) == json.dumps(vf)
        keptw = json.dumps(va) == json.dumps(want)
        print(f"   {k:22s} want={fmt(want):46s} appspot={fmt(va):46s} fork={fmt(vf):46s}"
              f" {'MATCH' if same else 'DIFFER'}{'' if keptw else ' (appspot != requested)'}")
    la = sa["layers"][1].get("segments"); lf = sf["layers"][1].get("segments")
    print(f"   segments               appspot={la} fork={lf} {'MATCH' if la == lf else 'DIFFER'}")

print("\n=== right-pane pixels, appspot vs fork ===")
for name in CASES:
    if a[name]["state"] is None or f[name]["state"] is None:
        print(f"{name}: skipped (unreadable state)")
        continue
    pa, pf = right_pane(a[name]["shot"]), right_pane(f[name]["shot"])
    if pa.shape != pf.shape:
        print(f"{name}: shape differs {pa.shape} vs {pf.shape}")
        continue
    ident = float((pa == pf).all(axis=2).mean())
    mad = float(np.abs(pa.astype(np.int16) - pf.astype(np.int16)).mean())
    # segmentation coverage: non-grey pixels (the coloured segment)
    def cover(x):
        mx, mn = x.max(axis=2).astype(np.int16), x.min(axis=2).astype(np.int16)
        return (mx - mn) > 25
    ca, cf = cover(pa), cover(pf)
    inter, union = float((ca & cf).sum()), float((ca | cf).sum())
    iou = inter / union if union else 1.0
    print(f"{name}: identical={ident:.4f} mean|diff|={mad:.2f} segment-IoU={iou:.4f} "
          f"cover appspot={ca.mean():.4f} fork={cf.mean():.4f}")
print("PARITY-DONE")
