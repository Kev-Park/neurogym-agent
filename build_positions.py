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
survives a crash. On restart, already-completed root_ids are skipped.
Retries are infinite with exponential backoff capped at --retry_max_delay.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

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


def _process_one(
    root_id: str,
    retry_base_delay: float,
    retry_max_delay: float,
) -> tuple[str, list[tuple[float, float, float]]]:
    """Fetch with infinite retry. Runs in a worker thread."""
    attempt = 0
    while True:
        try:
            return root_id, _fetch_nodes(root_id)
        except Exception as exc:
            delay = min(retry_base_delay * (2 ** attempt), retry_max_delay)
            tqdm.write(f"  attempt {attempt+1} failed for {root_id}: {exc}  retrying in {delay:.0f}s ...")
            time.sleep(delay)
            attempt += 1


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
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel threads for CAVE queries (default 8).")
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

    stats = {"processed": 0, "skipped": 0}
    lock = threading.Lock()

    # Single-worker ProcessPoolExecutor for parquet writes — runs in a
    # separate process so pandas/pyarrow serialization never contends the GIL
    # with the network worker threads. submit() is called outside the lock so
    # pickling the row snapshot doesn't block result collection either.
    write_executor = ProcessPoolExecutor(max_workers=1)

    def _collect(root_id: str, nodes: list[tuple[float, float, float]], pbar: tqdm) -> list[dict] | None:
        """Called under lock. Returns a row snapshot if a flush is due, else None."""
        if not nodes:
            tqdm.write(f"SKIP {root_id}: empty skeleton")
            skipped_set.add(root_id)
            failed_set.discard(root_id)
            progress["skipped"] = list(skipped_set)
            progress["failed"] = list(failed_set)
            progress_path.write_text(json.dumps(progress, indent=2))
            stats["skipped"] += 1
            pbar.update(1)
            return None

        z_vals = [n[2] for n in nodes]
        z_extent = max(z_vals) - min(z_vals)
        if z_extent < args.min_z_extent:
            tqdm.write(f"SKIP {root_id}: z_extent={z_extent:.1f} vox < {args.min_z_extent}")
            skipped_set.add(root_id)
            failed_set.discard(root_id)
            progress["skipped"] = list(skipped_set)
            progress["failed"] = list(failed_set)
            progress_path.write_text(json.dumps(progress, indent=2))
            stats["skipped"] += 1
            pbar.update(1)
            return None

        for x, y, z in nodes:
            rows.append({"root_id": root_id, "x": x, "y": y, "z": z})

        done_set.add(root_id)
        failed_set.discard(root_id)
        progress["done"] = list(done_set)
        progress["failed"] = list(failed_set)
        progress_path.write_text(json.dumps(progress, indent=2))
        stats["processed"] += 1
        pbar.update(1)

        if stats["processed"] % args.flush_every == 0:
            return list(rows)  # snapshot to be flushed outside the lock
        return None

    with tqdm(total=len(queue), desc="neurons", unit="neuron") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _process_one, root_id, args.retry_base_delay, args.retry_max_delay
                ): root_id
                for root_id in queue
            }
            for future in as_completed(futures):
                root_id, nodes = future.result()
                with lock:
                    snapshot = _collect(root_id, nodes, pbar)
                if snapshot is not None:
                    write_executor.submit(_flush, snapshot, output)
                    tqdm.write(f"queued flush of {len(snapshot):,} rows → {output}")

    # Final flush then shut down the write process (waits for pending writes).
    write_executor.submit(_flush, list(rows), output)
    write_executor.shutdown(wait=True)

    print("\n=== done ===")
    print(f"  processed:    {stats['processed']}")
    print(f"  skipped:      {stats['skipped']}")
    print(f"  total rows:   {len(rows):,}")
    print(f"  output:       {output}  ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
