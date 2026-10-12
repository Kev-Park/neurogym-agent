"""Fresh-process CUDA, simulator EGL and Chromium Vulkan smoke checks."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys


def cuda():
    import torch

    a = torch.ones((512, 512), device="cuda")
    for _ in range(20):
        b = a @ a
    torch.cuda.synchronize()
    assert torch.all(b == 512).item()
    return {"device": torch.cuda.get_device_name(), "version": torch.version.cuda}


def egl():
    import numpy as np
    from ngllib.simulator.render3d import MeshRenderer

    r = MeshRenderer(128, 128)
    try:
        name = r.ctx.info["GL_RENDERER"]
        assert "nvidia" in (name + r.ctx.info["GL_VENDOR"]).lower(), name
        r.load_mesh("probe", np.array([[-250, -250, 0], [250, -250, 0], [0, 250, 0]],
                                    dtype=np.float32), np.array([[0, 1, 2]], dtype=np.uint32))
        for _ in range(20):
            image = r.render("probe", [0, 0, 0], [0, 0, 0, 1], 1000, [1, 0, 0])
            depth, _, _ = r.pick_depth("probe", [0, 0, 0], [0, 0, 0, 1], 1000)
            assert image.shape == (128, 128, 3) and (depth < 0.9999).any()
        return {"renderer": name, "mesh_pixels": int((depth < 0.9999).sum())}
    finally:
        r.close()


def chromium():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            "--ignore-gpu-blocklist", "--enable-gpu", "--use-gl=angle",
            "--use-angle=vulkan", "--enable-unsafe-swiftshader"])
        try:
            page = browser.new_page()
            page.goto("data:text/html,<canvas id='c' width='64' height='64'></canvas>")
            result = page.evaluate("""() => {
                const gl = document.getElementById('c').getContext('webgl2');
                if (!gl) throw Error('No WebGL2');
                const ext = gl.getExtension('WEBGL_debug_renderer_info');
                gl.clearColor(1, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
                const pixel = new Uint8Array(4);
                gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
                return {renderer: gl.getParameter(ext.UNMASKED_RENDERER_WEBGL),
                        pixel: Array.from(pixel)};
            }""")
            assert "nvidia" in result["renderer"].lower(), result
            assert result["pixel"] == [255, 0, 0, 255], result
            return result
        finally:
            browser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["cuda", "egl", "chromium"])
    args = parser.parse_args()
    if args.stage:
        print(json.dumps(globals()[args.stage]()), flush=True)
        return
    results = []
    for stage in ("cuda", "egl", "chromium"):
        for attempt in range(3):
            try:
                p = subprocess.run([sys.executable, __file__, "--stage", stage],
                                   capture_output=True, text=True, timeout=60)
                result = {"stage": stage, "attempt": attempt + 1, "ok": p.returncode == 0,
                          "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
            except subprocess.TimeoutExpired:
                result = {"stage": stage, "attempt": attempt + 1, "ok": False,
                          "error": "60-second timeout"}
            results.append(result)
            print(json.dumps(result), flush=True)
    print(json.dumps({"host": socket.gethostname(), "results": results,
                      "all_pass": all(r["ok"] for r in results)}), flush=True)
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
