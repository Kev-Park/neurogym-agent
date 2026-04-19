"""Fetch L2 chunk positions for neurons from FlyWire.

For each segment ID in segment_ids.txt, fetches representative coordinates
along the neuron from the L2 cache. Appends results to segment_positions.csv
incrementally so progress survives interruptions.

Each row: root_id,x1;y1;z1|x2;y2;z2|...

Usage:
    python fetch_positions.py [--limit N] [--workers W]
"""

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from caveclient import CAVEclient

RESOLUTION = [4, 4, 40]  # nm per voxel in Neuroglancer viewer
OUT_PATH = "segment_positions.csv"


def load_existing_ids() -> set[str]:
    """Load root_ids already fetched from the CSV."""
    if not os.path.exists(OUT_PATH):
        return set()
    existing = set()
    with open(OUT_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                existing.add(row[0])
    return existing


def fetch_one(root_id: int) -> tuple[int, list[list[float]]]:
    client = CAVEclient("flywire_fafb_public")
    leaves = client.chunkedgraph.get_leaves(root_id, stop_layer=2)
    data = client.l2cache.get_l2data(leaves.tolist(), attributes=["rep_coord_nm"])
    coords = []
    for v in data.values():
        c = v.get("rep_coord_nm")
        if c:
            coords.append([c[0] / RESOLUTION[0], c[1] / RESOLUTION[1], c[2] / RESOLUTION[2]])
    return root_id, coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max neurons to fetch (default: all)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    with open("segment_ids.txt") as f:
        all_ids = [line.strip() for line in f if line.strip()]

    if args.limit is not None:
        all_ids = all_ids[: args.limit]

    existing = load_existing_ids()
    ids = [int(rid) for rid in all_ids if rid not in existing]

    print(f"{len(existing)} already fetched, {len(ids)} remaining")
    if not ids:
        print("Nothing to do.")
        return

    # Create file with header if it doesn't exist
    if not os.path.exists(OUT_PATH):
        with open(OUT_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["root_id", "positions"])

    outfile = open(OUT_PATH, "a", newline="")
    writer = csv.writer(outfile)

    succeeded = 0
    failed = 0
    t0 = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_one, rid): rid for rid in ids}
            for i, fut in enumerate(as_completed(futures), 1):
                rid = futures[fut]
                try:
                    root_id, coords = fut.result()
                    if coords:
                        pos_str = "|".join(f"{c[0]:.1f};{c[1]:.1f};{c[2]:.1f}" for c in coords)
                        writer.writerow([root_id, pos_str])
                        outfile.flush()
                        succeeded += 1
                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        print(f"  Failed {rid}: {e}")
                if i % 100 == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    eta = (len(ids) - i) / rate
                    print(f"  {i}/{len(ids)} done ({rate:.1f}/s, ETA {eta/60:.0f}m, {succeeded} ok, {failed} failed)")
    finally:
        outfile.close()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Added {succeeded} neurons ({failed} failed)")
    print(f"Total in {OUT_PATH}: {len(existing) + succeeded} neurons")


if __name__ == "__main__":
    main()
