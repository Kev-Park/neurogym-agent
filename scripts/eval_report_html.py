"""Build a self-contained HTML eval report from an eval_video manifest.

Embeds each episode's rollout MP4 (re-encoded to fit a 16MB page budget via
imageio-ffmpeg's bundled binary) as a data: URI next to an inline-SVG
"approach curve": normalized progress toward z_max (y=1.0 = target, green
band = +/- z_tolerance) over steps. Output is meant to be published as a
claude.ai Artifact (strict CSP: no external assets, so everything inlines).

    uv run --no-sync python scripts/eval_report_html.py \
        --manifest-dir eval_videos/v7_ckpt370_stoch_hud2 \
        --title "v7 Z-Nav Evals" \
        --summary "stochastic=49.6,argmax=2.5,random=2.5" \
        --quartile-rates "63.2,62.1,51.4,22.9" \
        --output eval_videos/v7_report.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import subprocess
import sys
import tempfile


def reencode(src: str, crf: int = 31) -> bytes:
    """Shrink an MP4 with the imageio-ffmpeg bundled binary; returns bytes."""
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out = tmp.name
    try:
        subprocess.run(
            [exe, "-y", "-i", src, "-c:v", "libx264", "-crf", str(crf),
             "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", out],
            check=True, capture_output=True)
        with open(out, "rb") as f:
            return f.read()
    finally:
        os.unlink(out)


def approach_svg(ep: dict) -> str:
    """Inline SVG: z-progress vs steps on the NEURON's own scale — y=0 is the
    segment's lowest skeleton z (z_min), y=1.0 its highest (z_max, dashed).
    The band is +/- z_tolerance; a hollow marker is the spawn height; the
    curve ends with an emphasized dot. Falls back to spawn-based
    normalization for manifests without z_min."""
    zs = ep["z_series"]
    zmax, tol = ep["z_max"], ep["z_tolerance"]
    zlo = ep.get("z_min", zs[0])
    span = (zmax - zlo) or 1e-6
    W, H, ml, mr, mt, mb = 560, 230, 44, 14, 14, 30
    iw, ih = W - ml - mr, H - mt - mb
    ymin, ymax = -0.08, 1.12
    n = max(len(zs) - 1, 1)

    def X(i):
        return ml + i / n * iw

    def Y(v):
        v = min(max(v, ymin), ymax)
        return mt + (ymax - v) / (ymax - ymin) * ih

    def N(z):
        return (z - zlo) / span

    pts = " ".join(f"{X(i):.1f},{Y(N(z)):.1f}" for i, z in enumerate(zs))
    band_top, band_bot = Y(1 + tol / abs(span)), Y(1 - tol / abs(span))
    gy = [f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{W - mr}" y2="{Y(v):.1f}" class="grid"/>'
          f'<text x="{ml - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v:g}</text>'
          for v in (0, 0.5)]
    step_ticks = [f'<text x="{X(i):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{i}</text>'
                  for i in range(0, len(zs), 100)]
    end_x, end_y = X(len(zs) - 1), Y(N(zs[-1]))
    ok = ep["outcome"] == "success"
    return f'''<svg viewBox="0 0 {W} {H}" role="img" aria-label="approach curve">
<rect x="{ml}" y="{band_top:.1f}" width="{iw}" height="{band_bot - band_top:.1f}" class="band"/>
<line x1="{ml}" y1="{Y(1):.1f}" x2="{W - mr}" y2="{Y(1):.1f}" class="target"/>
<text x="{ml - 6}" y="{Y(1) + 4:.1f}" class="tick target-t" text-anchor="end">1.0</text>
{''.join(gy)}{''.join(step_ticks)}
<polyline points="{pts}" class="curve {'curve-ok' if ok else 'curve-no'}"/>
<circle cx="{X(0):.1f}" cy="{Y(N(zs[0])):.1f}" r="4" class="spawn"/>
<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="4" class="{'dot-ok' if ok else 'dot-no'}"/>
<text x="{X(0) + 8:.1f}" y="{Y(N(zs[0])) + 4:.1f}" class="tick">spawn</text>
<text x="{ml}" y="{H - 8}" class="tick">step</text>
</svg>'''


CSS = '''
:root { --bg:#f4f6f5; --surface:#ffffff; --ink:#1c2327; --muted:#5d6b72;
  --line:#d7dde0; --accent:#0e7c86; --ok:#1e8f4e; --no:#c24437;
  --band:rgba(30,143,78,.16); --chip:#e8eeee; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#10151a; --surface:#171e25; --ink:#e6ebee; --muted:#93a1ab;
  --line:#2a343d; --accent:#4fc6d0; --ok:#46c77a; --no:#e06552;
  --band:rgba(70,199,122,.18); --chip:#1f2933; } }
:root[data-theme="dark"] {
  --bg:#10151a; --surface:#171e25; --ink:#e6ebee; --muted:#93a1ab;
  --line:#2a343d; --accent:#4fc6d0; --ok:#46c77a; --no:#e06552;
  --band:rgba(70,199,122,.18); --chip:#1f2933; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--ink); margin:0;
  font-family:"Archivo",system-ui,sans-serif; line-height:1.55; }
main { max-width:1020px; margin:0 auto; padding:40px 20px 80px; }
h1 { font-size:2rem; font-weight:650; letter-spacing:-.01em; margin:0 0 4px; text-wrap:balance; }
.sub { color:var(--muted); margin:0 0 28px; max-width:65ch; }
h2 { font-size:1.15rem; font-weight:650; margin:40px 0 4px; }
.mono, .tick, td, th { font-family:"IBM Plex Mono",ui-monospace,monospace; }
.stats { display:flex; gap:12px; flex-wrap:wrap; margin:20px 0 8px; }
.stat { background:var(--surface); border:1px solid var(--line); border-radius:8px;
  padding:12px 18px; min-width:150px; }
.stat b { display:block; font-size:1.5rem; font-variant-numeric:tabular-nums; }
.stat span { color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; }
.qrates { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 8px; }
.qr { background:var(--chip); border-radius:6px; padding:6px 12px; font-size:.85rem;
  font-variant-numeric:tabular-nums; }
.criterion { background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:8px; padding:12px 16px; margin:16px 0 4px; font-size:.92rem; max-width:70ch; }
.ep { background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:18px; margin:16px 0; display:grid; grid-template-columns:1fr 1fr; gap:18px; }
@media (max-width:820px) { .ep { grid-template-columns:1fr; } }
.ep video { width:100%; border-radius:6px; background:#000; display:block; }
.ep h3 { margin:0 0 8px; font-size:1rem; display:flex; align-items:center; gap:10px; }
.pill { font-size:.72rem; padding:2px 9px; border-radius:99px; font-weight:600;
  text-transform:uppercase; letter-spacing:.05em; }
.pill-ok { background:var(--ok); color:#fff; } .pill-no { background:var(--no); color:#fff; }
table { border-collapse:collapse; font-size:.82rem; margin-top:10px; width:100%; }
td, th { padding:3px 10px 3px 0; text-align:left; font-variant-numeric:tabular-nums; }
th { color:var(--muted); font-weight:400; }
svg { width:100%; height:auto; display:block; }
.grid { stroke:var(--line); stroke-width:1; }
.tick { fill:var(--muted); font-size:11px; }
.target { stroke:var(--accent); stroke-width:1.4; stroke-dasharray:5 4; }
.target-t { fill:var(--accent); }
.band { fill:var(--band); }
.curve { fill:none; stroke-width:2; } .curve-ok { stroke:var(--ok); } .curve-no { stroke:var(--no); }
.dot-ok { fill:var(--ok); } .dot-no { fill:var(--no); }
.spawn { fill:none; stroke:var(--muted); stroke-width:1.6; }
footer { color:var(--muted); font-size:.8rem; margin-top:48px; }
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--title", default="Eval report")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--summary", default="",
                    help='e.g. "stochastic=49.6,argmax=2.5,random=2.5"')
    ap.add_argument("--quartile-rates", default="",
                    help='pool success %% per length quartile, e.g. "63.2,62.1,51.4,22.9"')
    ap.add_argument("--thresholds-json", default="",
                    help="eval_thresholds.py --json output; rendered as a table.")
    ap.add_argument("--crf", type=int, default=31)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.manifest_dir, "manifest.json")) as f:
        manifest = json.load(f)

    stats = ""
    if args.summary:
        cells = []
        for part in args.summary.split(","):
            k, v = part.split("=")
            cells.append(f'<div class="stat"><b>{v}%</b><span>{html.escape(k)}</span></div>')
        stats = f'<div class="stats">{"".join(cells)}</div>'
    qrates = ""
    if args.quartile_rates:
        chips = [f'<div class="qr">q{i + 1} {r.strip()}%</div>'
                 for i, r in enumerate(args.quartile_rates.split(","))]
        qrates = f'<div class="qrates">{"".join(chips)} <div class="qr">pool success by neuron-length quartile</div></div>'

    thresholds = ""
    if args.thresholds_json:
        with open(args.thresholds_json) as f:
            tj = json.load(f)
        trs = "".join(
            f'<tr><td>{html.escape(r["criterion"])}</td><td><b>{r["overall"]}%</b></td>'
            + "".join(f'<td>{r[q]}%</td>' for q in ("q1", "q2", "q3", "q4"))
            + "</tr>"
            for r in tj["rows"])
        thresholds = f'''
<h2>Success under alternative tolerances</h2>
<p class="sub">Post-hoc from the {tj["n"]} recorded trajectories: an episode counts as a success
under a band if its trajectory ever entered it (exact for &ldquo;would have terminated&rdquo;).
Pool median z-extent: {tj["extent_median_vox"]} voxels.</p>
<div class="criterion" style="max-width:none;padding:8px 16px;">
<table><tr><th>criterion</th><th>overall</th><th>q1</th><th>q2</th><th>q3</th><th>q4</th></tr>{trs}</table>
</div>'''

    sections, total = [], 0
    for ep in manifest:
        vid = reencode(os.path.join(args.manifest_dir, ep["file"]), args.crf)
        total += len(vid)
        b64 = base64.b64encode(vid).decode()
        ok = ep["outcome"] == "success"
        pill = f'<span class="pill {"pill-ok" if ok else "pill-no"}">{ep["outcome"]}</span>'
        sections.append(f'''
<h2>{ep["quartile"]} &middot; {ep["length_nm"] / 1e6:.2f}M nm neuron</h2>
<div class="ep">
<div><video controls muted playsinline src="data:video/mp4;base64,{b64}"></video></div>
<div>
<h3>pair {ep["pair_idx"]} {pill}</h3>
{approach_svg(ep)}
<table><tr><th>steps</th><th>return</th><th>z start&rarr;final</th><th>&Delta;z at end</th></tr>
<tr><td>{ep["steps"]}</td><td>{ep["episode_return"]:+.3f}</td>
<td>{ep["z_start"]:g} &rarr; {ep["z_final"]:g} (target {ep["z_max"]:g})</td>
<td>{ep["dz_final"]:+.1f} vox</td></tr></table>
</div></div>''')
        print(f"[report] {ep['file']}: {len(vid) / 1e6:.2f}MB re-encoded", flush=True)

    doc = f'''<title>{html.escape(args.title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;650&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<main>
<h1>{html.escape(args.title)}</h1>
<p class="sub">Stochastic rollouts of {html.escape(args.checkpoint or "the checkpoint")} on the frozen
eval pool (eval_d0_v1, 200 pairs, seed 42). Each episode: rollout video beside its approach curve
&mdash; normalized progress toward the target, where 1.0 = the segment&rsquo;s max-z point.</p>
{stats}{qrates}
<div class="criterion mono">success &hArr; |viewer_z &minus; z_max| &le; {manifest[0]["z_tolerance"]:g} voxels
(&plusmn;{manifest[0]["z_tolerance"] * 40:g} nm) &mdash; the green band on each curve</div>
{thresholds}
{"".join(sections)}
<footer>Curves: y = (z &minus; z<sub>min</sub>) / (z<sub>max</sub> &minus; z<sub>min</sub>) — the neuron&rsquo;s own
z-extent, 0 = lowest skeleton node, 1.0 = target; hollow marker = spawn height; clamped for display.
Videos re-encoded (crf {args.crf}) from the archival MP4s in eval_videos/.</footer>
</main>'''

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(doc)
    size = os.path.getsize(args.output)
    print(f"[report] wrote {args.output}: {size / 1e6:.2f}MB "
          f"({'OK' if size < 15e6 else 'OVER 15MB — raise --crf'})", flush=True)
    return 0 if size < 16e6 else 1


if __name__ == "__main__":
    sys.exit(main())
