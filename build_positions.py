"""Build segment_positions.parquet from FlyWire L2 skeletons.

For each root_id in segment_ids.txt, fetches the L2 skeleton from CAVE
(~3 sec/neuron), extracts all node positions, and writes them as a flat
Parquet file with schema:

    root_id  str
    x        float32   (Neuroglancer voxel, 4 nm/voxel)
    y        float32   (Neuroglancer voxel, 4 nm/voxel)
    z        float32   (Neuroglancer voxel, 40 nm/voxel)

One row per skeleton node. Loaded at runtime by grouping on root_id.

Neurons are skipped if their z-extent (z_max - z_min in voxels) is below
--min_z_extent (default: 50 vox = 2000 nm). This prevents trivially easy
episodes where the agent starts at or near z_max.

Resume / retry
--------------
Progress is checkpointed to build_positions_progress.json after every
processed neuron. The output Parquet file is re-written every --flush_every
neurons (default 500) from the in-memory accumulator so partial progress
survives a crash. On restart, already-completed root_ids are skipped and
any previously failed ids are retried up to --max_retries times with
exponential backoff.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RESOLUTION = [4.0, 4.0, 40.0]  # nm per voxel, x/y/z
_SCHEMA = pa.schema([
    pa.field("root_id", pa.string()),
    pa.field("x", pa.float32()),
    pa.field("y", pa.float32()),
    pa.field("z", pa.float32()),
])


def _nm_to_voxel(x: float, y: float, z: float) -> tuple[float, float, float]:
    return x / RESOLUTION[0], y / RESOLUTION[1], z / RESOLUTION[2]


def _fetch_nodes(root_id: str) -> list[tuple[float, float, float]]:
    """Return list of (x, y, z) voxel positions from the L2 skeleton."""
    from fafbseg import flywire
    sk = flywire.get_l2_skeleton(int(root_id))
    nodes = sk.nodes[["x", "y", "z"]].dropna()
    return [_nm_to_voxel(r.x, r.y, r.z) for _, r in nodes.iterrows()]


def _flush(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["root_id", "x", "y", "z"])
    df["root_id"] = df["root_id"].astype("string")
    df["x"] = df["x"].astype("float32")
    df["y"] = df["y"].astype("float32")
    df["z"] = df["z"].astype("float32")
    pq.write_table(pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False), output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_ids", default="segment_ids.txt")
    parser.add_argument("--output", default="segment_positions.parquet")
    parser.add_argument("--progress", default="build_positions_progress.json")
    parser.add_argument("--min_z_extent", type=float, default=50.0,
                        help="Min z-extent in voxels (default 50 = 2000 nm).")
    parser.add_argument("--flush_every", type=int, default=500,
                        help="Write parquet after every N successful neurons.")
    parser.add_argument("--retry_base_delay", type=float, default=5.0)
    parser.add_argument("--retry_max_delay", type=float, default=300.0,
                        help="Cap on exponential backoff delay in seconds (default 300).")
    args = parser.parse_args()

    output = Path(args.output)
    progress_path = Path(args.progress)

    with open(args.segment_ids) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(all_ids)} root IDs from {args.segment_ids}")

    progress: dict = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "done": [], "failed": [], "skipped": []
    }
    done_set: set[str] = set(progress["done"])
    failed_set: set[str] = set(progress["failed"])
    skipped_set: set[str] = set(progress["skipped"])

    # Seed accumulator from existing parquet so flush overwrites correctly.
    if output.exists() and done_set:
        existing_rows = pq.read_table(output).to_pydict()
        rows: list[dict] = [
            {"root_id": r, "x": x, "y": y, "z": z}
            for r, x, y, z in zip(
                existing_rows["root_id"], existing_rows["x"],
                existing_rows["y"], existing_rows["z"],
            )
        ]
    else:
        rows = []

    # Previously failed IDs go to the front of the queue for retry.
    retry_ids = [r for r in failed_set if r not in done_set and r not in skipped_set]
    pending = [r for r in all_ids if r not in done_set and r not in skipped_set and r not in failed_set]
    queue = retry_ids + pending

    print(f"Done: {len(done_set)}  skipped: {len(skipped_set)}  "
          f"retry: {len(retry_ids)}  pending: {len(pending)}")

    stats = {"processed": 0, "skipped": 0, "failed": 0}

    for i, root_id in enumerate(queue):
        nodes: list[tuple[float, float, float]] | None = None
        attempt = 0
        while True:
            try:
                nodes = _fetch_nodes(root_id)
                break
            except Exception as exc:
                delay = min(args.retry_base_delay * (2 ** attempt), args.retry_max_delay)
                print(f"  attempt {attempt+1} failed for {root_id}: {exc}  retrying in {delay:.0f}s ...")
                time.sleep(delay)
                attempt += 1

        if not nodes:
            print(f"[{i+1}/{len(queue)}] SKIP {root_id}: empty skeleton")
            skipped_set.add(root_id)
            failed_set.discard(root_id)
            progress["skipped"] = list(skipped_set)
            progress["failed"] = list(failed_set)
            progress_path.write_text(json.dumps(progress, indent=2))
            stats["skipped"] += 1
            continue

        z_vals = [n[2] for n in nodes]
        z_extent = max(z_vals) - min(z_vals)
        if z_extent < args.min_z_extent:
            print(f"[{i+1}/{len(queue)}] SKIP {root_id}: z_extent={z_extent:.1f} vox < {args.min_z_extent}")
            skipped_set.add(root_id)
            failed_set.discard(root_id)
            progress["skipped"] = list(skipped_set)
            progress["failed"] = list(failed_set)
            progress_path.write_text(json.dumps(progress, indent=2))
            stats["skipped"] += 1
            continue

        for x, y, z in nodes:
            rows.append({"root_id": root_id, "x": x, "y": y, "z": z})

        done_set.add(root_id)
        failed_set.discard(root_id)
        progress["done"] = list(done_set)
        progress["failed"] = list(failed_set)
        progress_path.write_text(json.dumps(progress, indent=2))
        stats["processed"] += 1

        if stats["processed"] % args.flush_every == 0:
            _flush(rows, output)
            print(f"[{i+1}/{len(queue)}] flushed {len(rows):,} rows → {output}  {stats}")

    _flush(rows, output)
    print("\n=== done ===")
    print(f"  processed:    {stats['processed']}")
    print(f"  skipped:      {stats['skipped']}")
    print(f"  failed:       {stats['failed']}")
    print(f"  total rows:   {len(rows):,}")
    print(f"  output:       {output}  ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
