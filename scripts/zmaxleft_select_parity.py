"""Chrome vs simulator LIGHT parity probe for zmax-left's left-pane mechanics.

The static 2D-pane parity (SSIM, single-click input parity) was settled in the
2026-08/09 campaigns; what zmax-left adds is interaction SEQUENCES: double-click
select of a NEIGHBOR segment, toggle-off, the SHOW_ALL flip, and a 3D move after
a hop. This probe replays identical CSS-pixel click sequences on both backends
from identical states and compares STATE-level outcomes (selection sets,
positions); images are saved for eyeballing, not gated.

Gates (state-level, where the two backends must agree):
  - select agreement: double-click at pixel p adds the SAME segment id on both
    (or both add none). Boundary pixels may disagree; gate on the rate.
  - toggle semantics: a second double-click at p removes it on both.
  - SHOW_ALL flip: deselecting everything empties `visible` on both.
Report-only (known pick-calibration offsets make a hard gate unfair):
  - post-hop 3D right-click position deltas between backends.

Run on a vulkan-good GPU node:
  EM_OUT=<dir> uv run --no-sync python scripts/zmaxleft_select_parity.py
Env knobs: PARQUET, CONFIG (chrome config.json), N_STATES (default 6), SEED.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from PIL import Image

from ngllib import state as S
from ngllib.chrome import ChromeRenderer
from ngllib.simulator.renderer import SimulatorRenderer

from ngllib_agent.providers.flywire_skeleton import FlywireSkeletonProvider

PARQUET = os.environ.get(
    "PARQUET", "/scratch/kp0374/neurogym-agent/segment_positions.parquet")
CONFIG = os.environ.get("CONFIG", "/scratch/kp0374/neurogym-agent/config.json")
OUT = os.environ.get("EM_OUT", "zmaxleft_parity_out")
N_STATES = int(os.environ.get("N_STATES", "6"))
SEED = int(os.environ.get("SEED", "7"))

# CSS-pixel scan grid over the 2D pane (panels span y in [23, 876], 2D pane
# x in [0, 900)). Margins keep clear of the pane edges the UI mask covers.
SCAN_X = range(150, 800, 130)
SCAN_Y = range(150, 800, 130)
CENTER_3D = (1350.0, 450.0)  # 3D pane centre in CSS click coords


def visible(state) -> set[str]:
    return set(S.visible_segments(state["segments"]))


def save(img: np.ndarray, name: str) -> None:
    try:
        Image.fromarray(img).save(os.path.join(OUT, name))
    except Exception as e:  # noqa: BLE001 — image saving must never fail the probe
        print(f"[warn] save {name}: {e}", flush=True)


class Backend:
    """One renderer + its last-OBSERVED state (ChromeRenderer has no .state
    property, so both backends track state via observe() returns)."""

    def __init__(self, name: str, renderer):
        self.name = name
        self.r = renderer
        self.state: dict | None = None

    def reset_to(self, state: dict):
        self.r.reset_to(dict(state))
        st, img = self.r.observe()
        self.state = st
        return st, img

    def dblclick(self, x: float, y: float):
        """Double-click at (x, y) -> (added_ids, removed_ids, img)."""
        before = visible(self.state)
        self.r.click("double_click", x, y, "")
        st, img = self.r.observe()
        self.state = st
        after = visible(st)
        return after - before, before - after, img

    def rclick(self, x: float, y: float):
        self.r.click("right_click", x, y, "")
        st, img = self.r.observe()
        self.state = st
        return st, img


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    provider = FlywireSkeletonProvider(
        PARQUET, projection_scale=8000.0, cross_section_scale=2.0)
    rng = np.random.default_rng(SEED)

    sim = Backend("sim", SimulatorRenderer(
        window_size=(1800, 900), capture_scale=0.5,
        left_pane=True, right_pane=True, config_path=CONFIG))
    chrome = Backend("chrome", ChromeRenderer(
        window_size=(1800, 900), capture_scale=0.5,
        left_pane=True, right_pane=True, config_path=CONFIG))
    sim.r.open()
    chrome.r.open()

    agree = mismatch = both_none = 0
    toggle_ok = toggle_bad = 0
    showall_ok = showall_bad = 0
    hop_dz = []

    for i in range(N_STATES):
        state, info = provider(rng, None)
        root = info["segment_id"]
        print(f"[state {i}] root={root} pos={state['position']}", flush=True)
        try:
            _, img_s = sim.reset_to(state)
            _, img_c = chrome.reset_to(state)
            save(img_s, f"s{i}_pre_sim.png")
            save(img_c, f"s{i}_pre_chrome.png")

            # -- SHOW_ALL flip FIRST (states still identical: only root visible;
            #    running it after the hop compared DIFFERENT states — the 3D
            #    pick offsets diverge positions, which is expected) -------------
            if i < 2:
                pix = None
                for y in SCAN_Y:
                    for x in SCAN_X:
                        rid = sim.r._segment_under_2d(float(x), float(y))
                        if rid is not None and str(rid) == root:
                            pix = (float(x), float(y))
                            break
                    if pix:
                        break
                if pix:
                    _, rem_s, img_s = sim.dblclick(*pix)
                    _, rem_c, img_c = chrome.dblclick(*pix)
                    empty_s = not visible(sim.state)
                    empty_c = not visible(chrome.state)
                    ok = empty_s and empty_c
                    showall_ok += int(ok)
                    showall_bad += int(not ok)
                    print(f"[state {i}] SHOW_ALL flip: sim_empty={empty_s} "
                          f"chrome_empty={empty_c}", flush=True)
                    save(img_s, f"s{i}_showall_sim.png")
                    save(img_c, f"s{i}_showall_chrome.png")
                    # re-select the root from the SHOW_ALL state on both; verify
                    # the round trip restored the identical starting selection
                    add_s, _, _ = sim.dblclick(*pix)
                    add_c, _, _ = chrome.dblclick(*pix)
                    if visible(sim.state) != {root} or visible(chrome.state) != {root}:
                        print(f"[state {i}] SHOW_ALL round-trip diverged "
                              f"(sim={visible(sim.state)} chrome={visible(chrome.state)}); "
                              f"re-resetting", flush=True)
                        sim.reset_to(state)
                        chrome.reset_to(state)

            # -- candidate neighbor pixels, chosen via the sim's id map --------
            cands, seen = [], {root}
            for y in SCAN_Y:
                for x in SCAN_X:
                    rid = sim.r._segment_under_2d(float(x), float(y))
                    rid = None if rid is None else str(rid)
                    if rid and rid not in seen:
                        seen.add(rid)
                        cands.append((float(x), float(y), rid))
                if len(cands) >= 3:
                    break
            print(f"[state {i}] {len(cands)} neighbor candidates", flush=True)

            first_agreed = None
            for j, (x, y, _sim_expect) in enumerate(cands[:3]):
                add_s, _, img_s = sim.dblclick(x, y)
                add_c, _, img_c = chrome.dblclick(x, y)
                a_s = next(iter(add_s), None)
                a_c = next(iter(add_c), None)
                if a_s is None and a_c is None:
                    both_none += 1
                    verdict = "both-none"
                elif a_s == a_c:
                    agree += 1
                    verdict = "AGREE"
                    if first_agreed is None:
                        first_agreed = (x, y, a_s)
                else:
                    mismatch += 1
                    verdict = f"MISMATCH sim={a_s} chrome={a_c}"
                    save(img_s, f"s{i}c{j}_mismatch_sim.png")
                    save(img_c, f"s{i}c{j}_mismatch_chrome.png")
                print(f"[state {i}] click({x:.0f},{y:.0f}) -> {verdict}", flush=True)

                # toggle back off (isolate trials); both must remove what they added
                if a_s or a_c:
                    _, rem_s, _ = sim.dblclick(x, y)
                    _, rem_c, _ = chrome.dblclick(x, y)
                    ok = (not a_s or a_s in rem_s) and (not a_c or a_c in rem_c)
                    toggle_ok += int(ok)
                    toggle_bad += int(not ok)
                    if not ok:
                        print(f"[state {i}] TOGGLE-OFF failed "
                              f"(sim removed {rem_s}, chrome removed {rem_c})",
                              flush=True)

            # -- hop: re-select the agreed neighbor, then a 3D move ------------
            if first_agreed is not None:
                x, y, rid = first_agreed
                sim.dblclick(x, y)
                chrome.dblclick(x, y)
                z0_s = sim.state["position"][2]
                z0_c = chrome.state["position"][2]
                st_s, img_s = sim.rclick(*CENTER_3D)
                st_c, img_c = chrome.rclick(*CENTER_3D)
                dz_s = st_s["position"][2] - z0_s
                dz_c = st_c["position"][2] - z0_c
                hop_dz.append((dz_s, dz_c))
                print(f"[state {i}] hop 3D move dz sim={dz_s:.1f} chrome={dz_c:.1f}",
                      flush=True)
                save(img_s, f"s{i}_hop_sim.png")
                save(img_c, f"s{i}_hop_chrome.png")
        except Exception:
            print(f"[state {i}] EXCEPTION:\n{traceback.format_exc()}", flush=True)

    sim.r.close()
    chrome.r.close()

    total = agree + mismatch
    rate = agree / total if total else 0.0
    dz_gap = [abs(a - b) for a, b in hop_dz]
    ok = total >= 6 and rate >= 0.8 and toggle_bad == 0 and showall_bad == 0
    print(f"select: agree={agree} mismatch={mismatch} both_none={both_none} "
          f"rate={rate:.2f}", flush=True)
    print(f"toggle: ok={toggle_ok} bad={toggle_bad} | "
          f"showall: ok={showall_ok} bad={showall_bad}", flush=True)
    print(f"hop dz pairs (sim, chrome): "
          f"{[(round(a, 1), round(b, 1)) for a, b in hop_dz]} | gap median="
          f"{np.median(dz_gap) if dz_gap else float('nan'):.1f} (report-only)",
          flush=True)
    print(f"ZMAXLEFT-PARITY {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
