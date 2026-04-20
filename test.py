from playwright.sync_api import sync_playwright
import json, io, time
from PIL import Image

with open("config.json") as f:
    cfg = json.load(f)

w, h = cfg["window_width"], cfg["window_height"]
url = cfg["default_ngl_start_url"]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=[
        "--no-sandbox", "--disable-dev-shm-usage",
        "--use-gl=angle", "--use-angle=vulkan",
        "--enable-features=Vulkan", "--enable-unsafe-swiftshader",
    ])
    page = browser.new_page(viewport={"width": w, "height": h})
    page.goto(url, wait_until="load", timeout=60000)

    print("Waiting for viewer.isReady()...")
    ready, deadline = False, time.time() + 60
    while time.time() < deadline:
        try:
            ready = page.evaluate("() => !!(window.viewer && window.viewer.isReady && window.viewer.isReady())")
            if ready: break
        except Exception: pass
        time.sleep(0.25)
    print("Viewer ready:", ready)

    iw = page.evaluate("() => window.innerWidth")
    ih = page.evaluate("() => window.innerHeight")
    print(f"Config:     {w} x {h}")
    print(f"Viewport:   {iw} x {ih}")
    print("Viewport:   " + ("MATCH" if iw == w and ih == h else "MISMATCH"))

    sc = page.screenshot(type="jpeg", quality=85)
    img = Image.open(io.BytesIO(sc))
    print(f"Screenshot: {img.width} x {img.height}")
    img.save("viewport_test.jpg")
    print("Saved: viewport_test.jpg")
    browser.close()