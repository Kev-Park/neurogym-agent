"""Dummy learner/renderer for coordinator lifecycle tests.

Prints periodic heartbeat to stdout with hostname, PID, and elapsed time.
Optional `--die-after N` makes it exit non-zero after N seconds (for testing
coordinator's death detection and later respawn logic).

Not part of any release surface — pure test fixture.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="worker")
    ap.add_argument("--heartbeat-s", type=int, default=5)
    ap.add_argument("--die-after-s", type=int, default=0,
                    help="If >0, exit(1) after N seconds; else sleep forever.")
    args = ap.parse_args()

    host = socket.gethostname()
    pid = os.getpid()
    tag = f"[{host}] {args.role} pid={pid}"
    print(f"{tag} START (die_after={args.die_after_s})", flush=True)

    start = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - start
            print(f"{tag} elapsed={elapsed:.1f}s", flush=True)
            if args.die_after_s and elapsed >= args.die_after_s:
                print(f"{tag} DIE_AFTER reached; exiting 1", flush=True)
                return 1
            time.sleep(args.heartbeat_s)
    except KeyboardInterrupt:
        print(f"{tag} SIGINT/SIGTERM; exiting 0", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
