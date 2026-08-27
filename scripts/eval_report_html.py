"""Build a self-contained HTML eval report from an eval_video manifest.

Embeds each episode's rollout MP4 (re-encoded to fit a 16MB page budget via
imageio-ffmpeg's bundled binary) as a data: URI next to an inline-SVG
"approach curve". A criterion selector (abs / percent-of-extent bands)
re-evaluates the page live: each episode's success band, outcome label, and
curve color, plus the full pool's overall/per-quartile rates (from
--results-json's recorded trajectories). Output is meant to be published as
a claude.ai Artifact (strict CSP: no external assets, so everything inlines;
inline script is fine).

    uv run --no-sync python scripts/eval_report_html.py \
        --manifest-dir eval_videos/v8_ckpt740_vids \
        --results-json eval_results/v8_ckpt740.json \
        --title "Z-Nav Evals" --checkpoint "coord-v8 ckpt_000740" \
        --summary "v8@740=87.0,v7-baseline=64.5" \
        --thresholds-json eval_results/v8_ckpt740.json.thresholds.json \
        --output eval_videos/v8_report.html

Note: the videos' baked-in HUD always shows the RUN criterion; only the
charts/labels/stats re-evaluate. Criteria tighter than the run's are lower
bounds (episodes terminate at the run band).
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

import numpy as np

# SVG chart geometry — mirrored in the page's JS (CHART constant below).
W, H, ML, MR, MT, MB = 560, 230, 44, 14, 14, 30
IW, IH = W - ML - MR, H - MT - MB
YMIN, YMAX = -0.08, 1.12


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


def _y(v: float) -> float:
    v = min(max(v, YMIN), YMAX)
    return MT + (YMAX - v) / (YMAX - YMIN) * IH


def approach_svg(ep: dict) -> str:
    """Inline SVG on the NEURON's own scale — y=0 at z_min, y=1.0 at z_max.
    Band/curve/dot carry stable classes so the criterion selector can
    re-style them; the card's data-span/data-closest feed the re-evaluation."""
    zs = ep["z_series"]
    zmax, tol = ep["z_max"], ep["z_tolerance"]
    zlo = ep.get("z_min", zs[0])
    span = (zmax - zlo) or 1e-6

    def X(i):
        return ML + i / max(len(zs) - 1, 1) * IW

    def N(z):
        return (z - zlo) / span

    pts = " ".join(f"{X(i):.1f},{_y(N(z)):.1f}" for i, z in enumerate(zs))
    half = tol / abs(span)
    band_top, band_bot = _y(1 + half), _y(1 - half)
    if band_bot - band_top < 3.0:
        mid = (band_bot + band_top) / 2
        band_top, band_bot = mid - 1.5, mid + 1.5
    gy = [f'<line x1="{ML}" y1="{_y(v):.1f}" x2="{W - MR}" y2="{_y(v):.1f}" class="grid"/>'
          f'<text x="{ML - 6}" y="{_y(v) + 4:.1f}" class="tick" text-anchor="end">{v:g}</text>'
          for v in (0, 0.5)]
    step_ticks = [f'<text x="{X(i):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{i}</text>'
                  for i in range(0, len(zs), 100)]
    ok = ep["outcome"] == "success"
    return f'''<svg viewBox="0 0 {W} {H}" role="img" aria-label="approach curve">
<rect x="{ML}" y="{band_top:.1f}" width="{IW}" height="{band_bot - band_top:.1f}" class="band"/>
<line x1="{ML}" y1="{_y(1):.1f}" x2="{W - MR}" y2="{_y(1):.1f}" class="target"/>
<text x="{ML - 6}" y="{_y(1) + 4:.1f}" class="tick target-t" text-anchor="end">1.0</text>
{''.join(gy)}{''.join(step_ticks)}
<polyline points="{pts}" class="curve {'curve-ok' if ok else 'curve-no'}"/>
<circle cx="{X(0):.1f}" cy="{_y(N(zs[0])):.1f}" r="4" class="spawn"/>
<circle cx="{X(len(zs) - 1):.1f}" cy="{_y(N(zs[-1])):.1f}" r="4" class="enddot {'dot-ok' if ok else 'dot-no'}"/>
<text x="{X(0) + 8:.1f}" y="{_y(N(zs[0])) + 4:.1f}" class="tick">spawn</text>
<text x="{ML}" y="{H - 8}" class="tick">step</text>
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
.crit-picker { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:22px 0 6px; }
.crit-chip { background:var(--chip); color:var(--ink); border:1px solid var(--line);
  border-radius:99px; padding:7px 16px; font-size:.88rem; cursor:pointer;
  font-family:"IBM Plex Mono",ui-monospace,monospace; }
.crit-chip.active { background:var(--accent); color:#fff; border-color:var(--accent); }
.crit-chip:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.crit-note { color:var(--muted); font-size:.82rem; margin:2px 0 10px; max-width:75ch; }
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
                    help='e.g. "v8@740=87.0,v7-baseline=64.5"')
    ap.add_argument("--results-json", default="",
                    help="Full eval JSON (z_series per pair): powers the live "
                         "pool re-evaluation under the criterion selector. "
                         "Equivalent to a first --pool with the checkpoint name.")
    ap.add_argument("--pool", action="append", default=[],
                    help="label=path of an eval JSON; each becomes a top stat "
                         "card that RECOMPUTES under the criterion/budget "
                         "selectors. First pool also drives the quartile card. "
                         "Pools rolled at shorter budgets are lower bounds "
                         "beyond their own budget.")
    ap.add_argument("--thresholds-json", default="",
                    help="eval_thresholds.py --json output; rendered as a table.")
    ap.add_argument("--run-frac", type=float, default=0.05,
                    help="The run's own termination fraction (labels the default chip).")
    ap.add_argument("--abs-tol", type=float, default=10.0)
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

    # Pool data for live re-evaluation: closest approach (@300 and full) +
    # extent + length quartile per pair, PER RUN — every run's top stat card
    # recomputes under the selectors. Wedged/glitch pairs keep their recorded
    # trajectories (closest is still honest — they never got there).
    def load_pool(path):
        with open(path) as f:
            pairs = [r for r in json.load(f)["per_pair"] if "z_series" in r]
        lengths = np.asarray([p["length_nm"] for p in pairs])
        q1, q2, q3 = np.quantile(lengths, [0.25, 0.5, 0.75])
        pool = []
        for p in pairs:
            zs = np.asarray(p["z_series"], float)
            d = np.abs(zs - p["z_max"])
            c300 = float(d[:301].min())      # standard-budget closest approach
            cfull = float(d.min())           # full-rollout closest approach
            span = p["z_max"] - p["z_min"]
            qi = 0 if p["length_nm"] < q1 else 1 if p["length_nm"] < q2 \
                else 2 if p["length_nm"] < q3 else 3
            pool.append([round(c300, 1), round(cfull, 1), round(span, 1), qi])
        return pool

    pool_specs = []
    if args.results_json:
        pool_specs.append((args.checkpoint or "this run", args.results_json))
    for spec in args.pool:
        label, path = spec.split("=", 1)
        pool_specs.append((label, path))
    pools = {label: load_pool(path) for label, path in pool_specs}
    pool_labels = [label for label, _ in pool_specs]
    pools_js = json.dumps(pools, separators=(",", ":"))
    order_js = json.dumps(pool_labels)

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
<h2>Success under alternative tolerances (static table)</h2>
<div class="criterion" style="max-width:none;padding:8px 16px;">
<table><tr><th>criterion</th><th>overall</th><th>q1</th><th>q2</th><th>q3</th><th>q4</th></tr>{trs}</table>
</div>'''

    sections, total = [], 0
    for ep in manifest:
        vid = reencode(os.path.join(args.manifest_dir, ep["file"]), args.crf)
        total += len(vid)
        b64 = base64.b64encode(vid).decode()
        ok = ep["outcome"] == "success"
        span = ep["z_max"] - ep.get("z_min", ep["z_series"][0])
        dists = [abs(z - ep["z_max"]) for z in ep["z_series"]]
        closest, c300 = min(dists), min(dists[:301])
        pill = f'<span class="pill {"pill-ok" if ok else "pill-no"}">{ep["outcome"]}</span>'
        sections.append(f'''
<h2>{ep["quartile"]} &middot; {ep["length_nm"] / 1e6:.2f}M nm neuron</h2>
<div class="ep ep-card" data-span="{span:.1f}" data-closest="{closest:.1f}" data-closest300="{c300:.1f}">
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

    picker = f'''
<div class="crit-picker" role="group" aria-label="success criterion">
<span class="tick" style="font-size:.85rem">criterion:</span>
<button class="crit-chip crit-c" data-k="f5">&plusmn;5% extent (run)</button>
<button class="crit-chip crit-c" data-k="f10">&plusmn;10% extent</button>
<button class="crit-chip crit-c" data-k="f15">&plusmn;15% extent</button>
<button class="crit-chip crit-c" data-k="abs">abs &plusmn;{args.abs_tol:g} vox</button>
</div>
<div class="crit-picker" role="group" aria-label="step budget">
<span class="tick" style="font-size:.85rem">step budget:</span>
<button class="crit-chip crit-b" data-b="b300">@300 (standard)</button>
<button class="crit-chip crit-b" data-b="bfull">@600 (extended)</button>
</div>
<div class="stats">
{"".join(f'<div class="stat"><b id="sum-{i}">&ndash;</b><span>{html.escape(lab)}</span></div>'
         for i, lab in enumerate(pool_labels))}
<div class="stat"><b><span id="sel-q1">&ndash;</span> / <span id="sel-q2">&ndash;</span> /
<span id="sel-q3">&ndash;</span> / <span id="sel-q4">&ndash;</span></b><span>{html.escape(pool_labels[0] if pool_labels else "")} q1 / q2 / q3 / q4</span></div>
</div>
<p class="crit-note">Criterion and budget selections re-evaluate the pool stats above and every
episode below (band, label, curve color) from the recorded trajectories. Success@budget =
the trajectory entered the band within that many steps (exact: an extended episode&rsquo;s
first 300 steps ARE the standard episode). Criteria TIGHTER than the run&rsquo;s
(&plusmn;{args.run_frac:.0%}) are lower bounds — episodes terminated at the run band.
Video overlays are baked at the run criterion and full budget.</p>'''

    js = f'''<script>
const POOLS = {pools_js};
const ORDER = {order_js};
const CRIT = {{f5:{{t:"f",v:0.05}}, f10:{{t:"f",v:0.10}}, f15:{{t:"f",v:0.15}},
              abs:{{t:"a",v:{args.abs_tol}}}}};
const CH = {{mt:{MT}, ih:{IH}, ymin:{YMIN}, ymax:{YMAX}}};
let selC = "f5", selB = "bfull";
function tol(c, span) {{ return c.t === "a" ? c.v : c.v * span; }}
function Y(v) {{ v = Math.min(Math.max(v, CH.ymin), CH.ymax);
  return CH.mt + (CH.ymax - v) / (CH.ymax - CH.ymin) * CH.ih; }}
function apply() {{
  const c = CRIT[selC], b300 = selB === "b300";
  document.querySelectorAll(".ep-card").forEach(el => {{
    const span = +el.dataset.span;
    const closest = b300 ? +el.dataset.closest300 : +el.dataset.closest;
    const ok = closest <= tol(c, span);
    const pill = el.querySelector(".pill");
    pill.textContent = ok ? "success" : "failure";
    pill.setAttribute("class", "pill " + (ok ? "pill-ok" : "pill-no"));
    el.querySelector(".curve").setAttribute("class",
      "curve " + (ok ? "curve-ok" : "curve-no"));
    el.querySelector(".enddot").setAttribute("class",
      "enddot " + (ok ? "dot-ok" : "dot-no"));
    const half = tol(c, span) / span;
    let top = Y(1 + half), bot = Y(1 - half);
    if (bot - top < 3) {{ const m = (top + bot) / 2; top = m - 1.5; bot = m + 1.5; }}
    const band = el.querySelector(".band");
    band.setAttribute("y", top.toFixed(1));
    band.setAttribute("height", (bot - top).toFixed(1));
  }});
  ORDER.forEach((label, idx) => {{
    const pool = POOLS[label];
    const w = [0,0,0,0], n = [0,0,0,0]; let tot = 0;
    pool.forEach(p => {{
      const closest = b300 ? p[0] : p[1];
      n[p[3]]++; if (closest <= tol(c, p[2])) {{ w[p[3]]++; tot++; }}
    }});
    const el = document.getElementById("sum-" + idx);
    if (el) el.textContent = (100 * tot / pool.length).toFixed(1) + "%";
    if (idx === 0) {{
      ["sel-q1","sel-q2","sel-q3","sel-q4"].forEach((id, i) =>
        document.getElementById(id).textContent =
          n[i] ? (100 * w[i] / n[i]).toFixed(0) + "%" : "--");
    }}
  }});
  document.querySelectorAll(".crit-c").forEach(x =>
    x.classList.toggle("active", x.dataset.k === selC));
  document.querySelectorAll(".crit-b").forEach(x =>
    x.classList.toggle("active", x.dataset.b === selB));
}}
document.querySelectorAll(".crit-c").forEach(x =>
  x.addEventListener("click", () => {{ selC = x.dataset.k; apply(); }}));
document.querySelectorAll(".crit-b").forEach(x =>
  x.addEventListener("click", () => {{ selB = x.dataset.b; apply(); }}));
apply();
</script>'''

    doc = f'''<title>{html.escape(args.title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;650&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<main>
<h1>{html.escape(args.title)}</h1>
<p class="sub">Stochastic rollouts of {html.escape(args.checkpoint or "the checkpoint")} on the frozen
eval pool (eval_d0_v1, 200 pairs, seed 42). Each episode: rollout video beside its approach curve
&mdash; y=0 at the neuron&rsquo;s lowest skeleton z, 1.0 at the target z_max.</p>
{stats}
{picker}
{thresholds}
{"".join(sections)}
<footer>Curves: y = (z &minus; z<sub>min</sub>) / (z<sub>max</sub> &minus; z<sub>min</sub>); hollow marker =
spawn height; clamped for display. The success band tracks the selected criterion (drawn with a
minimum height — to scale it can be sub-pixel). Videos re-encoded (crf {args.crf}) from the
archival MP4s in eval_videos/.</footer>
</main>
{js}'''

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(doc)
    size = os.path.getsize(args.output)
    print(f"[report] wrote {args.output}: {size / 1e6:.2f}MB "
          f"({'OK' if size < 15e6 else 'OVER 15MB — raise --crf'})", flush=True)
    return 0 if size < 16e6 else 1


if __name__ == "__main__":
    sys.exit(main())
