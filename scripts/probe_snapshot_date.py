"""Probe candidate frozen-snapshot dates T for the pre-edit training world.

Design (2026-09-14): pick one global timestamp T; each training object is
its neuron's root AS OF T (one coherent erroneous world-state); success
flags = coordinates of all edits with timestamp > T on that lineage. T
dials difficulty: early = error-rich but bigger objects, late = cleaner.
This probe produces the T-selection table from the pilot harvest:

  A. edits-after-T per neuron (from pilot timestamps — no API calls);
  B. object size at T (L2-node count, the merge-monster metric) + root@T
     resolvability, for --k sample neurons per candidate T (API).

Anchor for "the neuron at T": the root containing one of the CURRENT
root's supervoxels at T (first leaf — arbitrary but deterministic).

    uv run --no-sync python scripts/probe_snapshot_date.py \
        --edits edit_sites_pilot.parquet --k 6
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq

CANDIDATE_TS = ["2020-10-01", "2021-06-01", "2022-01-01",
                "2022-09-01", "2023-06-01"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits", default="edit_sites_pilot.parquet")
    ap.add_argument("--datastack", default="flywire_fafb_public")
    ap.add_argument("--k", type=int, default=6,
                    help="Sample neurons for the size-at-T API sweep.")
    args = ap.parse_args()

    tbl = pq.read_table(args.edits,
                        columns=["root_id", "operation_id", "timestamp"])
    rows = tbl.to_pylist()
    # Pilot timestamps are epoch ms; dedupe to one row per operation.
    ops = {}
    for r in rows:
        ops[(r["root_id"], r["operation_id"])] = r["timestamp"] / 1000.0
    ts = np.asarray(list(ops.values()))
    dates = [datetime.fromtimestamp(t, tz=timezone.utc) for t in
             np.quantile(ts, [0.05, 0.25, 0.5, 0.75, 0.95])]
    print(f"[snapshot] {len(ops)} ops across "
          f"{len({k[0] for k in ops})} neurons; edit-time quantiles "
          f"(5/25/50/75/95%): "
          + " ".join(d.strftime("%Y-%m") for d in dates), flush=True)

    print("\n== A. edits-after-T per neuron (pilot, no API) ==", flush=True)
    by_root: dict[str, list[float]] = {}
    for (rid, _), t in ops.items():
        by_root.setdefault(rid, []).append(t)
    print(f'{"T":<12}{"median":>8}{"q25":>6}{"q75":>6}{">=3 flags":>11}',
          flush=True)
    for tstr in CANDIDATE_TS:
        cut = datetime.fromisoformat(tstr).replace(
            tzinfo=timezone.utc).timestamp()
        after = np.asarray([sum(1 for t in v if t > cut)
                            for v in by_root.values()])
        print(f"{tstr:<12}{np.median(after):>8.0f}"
              f"{np.quantile(after, .25):>6.0f}{np.quantile(after, .75):>6.0f}"
              f"{100 * np.mean(after >= 3):>10.0f}%", flush=True)

    print(f"\n== B. root@T size sweep ({args.k} sample neurons, API) ==",
          flush=True)
    from caveclient import CAVEclient

    client = CAVEclient(args.datastack)
    sample = sorted(by_root, key=lambda r: -len(by_root[r]))[: args.k]
    print(f'{"T":<12}' + "".join(f"{r[-6:]:>10}" for r in sample)
          + "   (L2 count @T; x = unresolvable)", flush=True)
    anchors = {}
    for rid in sample:
        leaves = client.chunkedgraph.get_leaves(int(rid))
        anchors[rid] = int(leaves[0])
    for tstr in CANDIDATE_TS + [None]:
        t = (datetime.fromisoformat(tstr).replace(tzinfo=timezone.utc)
             if tstr else None)
        cells = []
        for rid in sample:
            try:
                root_t = client.chunkedgraph.get_roots(
                    [anchors[rid]], timestamp=t)[0]
                n_l2 = len(client.chunkedgraph.get_leaves(
                    int(root_t), stop_layer=2))
                cells.append(f"{n_l2:>10}")
            except Exception as e:  # noqa: BLE001 — resolvability is the finding
                cells.append(f"{'x':>10}")
                if tstr == CANDIDATE_TS[0]:
                    print(f"   [{rid}@{tstr}: {type(e).__name__}]", flush=True)
        print(f"{(tstr or 'now'):<12}" + "".join(cells), flush=True)
    print("\n[snapshot] read the knee: earliest T where sizes stay "
          "renderable AND >=3-flags coverage is high.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
