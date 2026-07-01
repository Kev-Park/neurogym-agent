"""Verify a multi-node Ray cluster is formed correctly.

Run from the head node inside a multi-node SLURM allocation, AFTER
`ray start --head ...` on the head and `ray start --address=...` on each worker.

Validates:
  1. At least the expected number of nodes are alive and joined the cluster.
  2. Tasks pinned to different nodes actually land there (verifies scheduling
     across nodes, not just multi-actor on one node).
  3. Cross-node object store transfer: ray.put on head, ray.get on a worker
     node — matches the M3 sample-pipeline shape from agent_plan.md §6.

No GPU rendering needed — this is the pure-Ray infrastructure test that runs
on Mesa-only nodes just fine.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _alive_nodes():
    return [n for n in ray.nodes() if n.get("Alive")]


def _wait_for_nodes(expected: int, timeout: float = 60.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        nodes = _alive_nodes()
        if len(nodes) >= expected:
            return nodes
        time.sleep(2.0)
    return _alive_nodes()


@ray.remote(num_cpus=1)
def report_host_and_payload(size: int) -> tuple[str, int]:
    return socket.gethostname(), size


@ray.remote(num_cpus=1)
def resolve_object(ref) -> tuple[str, int]:
    payload = ray.get(ref)
    return socket.gethostname(), len(payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-ip", required=True, help="e.g. 10.32.15.3")
    ap.add_argument("--expected-nodes", type=int, default=2)
    ap.add_argument("--payload-size", type=int, default=10_000)
    args = ap.parse_args()

    ray.init(address=f"{args.head_ip}:6379")

    nodes = _wait_for_nodes(args.expected_nodes)
    print(f"[cluster] {len(nodes)} alive nodes:", flush=True)
    for n in nodes:
        print(
            f"  {n['NodeManagerHostname']:20s} "
            f"cpus={n.get('Resources', {}).get('CPU')}",
            flush=True,
        )
    if len(nodes) < args.expected_nodes:
        print(
            f"FAIL: expected {args.expected_nodes} nodes, got {len(nodes)}",
            flush=True,
        )
        return 1

    # 2. Pin tasks to specific nodes, verify they actually land there.
    print("\n[per-node scheduling]", flush=True)
    hosts_by_task = {}
    for n in nodes:
        strat = NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
        host, size = ray.get(
            report_host_and_payload.options(scheduling_strategy=strat).remote(42)
        )
        expected_host = n["NodeManagerHostname"]
        ok = host == expected_host
        hosts_by_task[expected_host] = host
        print(f"  expected {expected_host:20s} got {host:20s} {'OK' if ok else 'MISMATCH'}", flush=True)
    if any(v != k for k, v in hosts_by_task.items()):
        print("FAIL: at least one task landed on the wrong node", flush=True)
        return 1

    # 3. Cross-node object store transfer.
    print("\n[cross-node object store]", flush=True)
    payload = list(range(args.payload_size))
    ref = ray.put(payload)
    # Push it to a non-head node to force cross-node fetch.
    head_hostname = nodes[0]["NodeManagerHostname"]
    non_head = next(n for n in nodes if n["NodeManagerHostname"] != head_hostname)
    strat = NodeAffinitySchedulingStrategy(node_id=non_head["NodeID"], soft=False)
    host, size = ray.get(resolve_object.options(scheduling_strategy=strat).remote(ref))
    if host != non_head["NodeManagerHostname"] or size != args.payload_size:
        print(
            f"FAIL: cross-node ray.get returned host={host} size={size}, "
            f"expected host={non_head['NodeManagerHostname']} size={args.payload_size}",
            flush=True,
        )
        return 1
    print(
        f"  put on {head_hostname}, resolved on {host} (size={size}) OK",
        flush=True,
    )

    print("\nPASS: multi-node Ray cluster verified", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
