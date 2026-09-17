"""Dump the datastack's NG viewer config (host, sources, resolution).

Authoritative source for WHICH Neuroglancer host + source strings to build
links against for flywire_fafb_public (vanilla neuroglancer-demo does not do
FlyWire middleauth; the datastack declares its own viewer).

    uv run --no-sync python scripts/probe_viewer_info.py
"""

from __future__ import annotations

import sys


def main() -> int:
    from caveclient import CAVEclient

    c = CAVEclient("flywire_fafb_public")
    info = c.info
    for name in ("viewer_site", "segmentation_source", "image_source",
                 "viewer_resolution", "get_datastack_info"):
        fn = getattr(info, name, None)
        try:
            print(f"{name}: {fn() if callable(fn) else fn!r}")
        except Exception as e:  # noqa: BLE001
            print(f"{name}: <err {type(e).__name__}: {e}>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
