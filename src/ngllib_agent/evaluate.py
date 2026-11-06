"""Evaluate paired checkpoints on JSON frozen starts; optionally record video."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal

from .distributed.checkpoint import atomic_json, public_config
from .evaluation import StatePklPolicy, checkpoint_config, compare_results, rollout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-pkl")
    ap.add_argument("--config")
    ap.add_argument("--starts", help="JSON list of {id, state, task_info}; complete frozen states")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--backend", choices=["chrome", "simulator"])
    ap.add_argument("--video-dir")
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    ap.add_argument("--allow-backend-difference", action="store_true")
    args = ap.parse_args(argv)
    if args.compare:
        left, right = [json.loads(Path(path).read_text()) for path in args.compare]
        atomic_json(compare_results(left, right, args.allow_backend_difference), args.output)
        return 0
    if not args.state_pkl or not args.starts or args.max_steps < 1 or args.timeout_s < 1:
        ap.error("Supply --state-pkl, --starts, and a positive --max-steps")
    from .env_build import build_env, load_config
    import torch

    cfg = checkpoint_config(args.state_pkl, load_config(args.config) if args.config else None)
    if args.backend:
        cfg["env"]["backend"] = args.backend
    cfg["env"]["max_episode_steps"] = args.max_steps
    cfg["env"]["reset_ahead"] = False
    cfg["env"]["stagger_first_episode"] = False
    data = Path(args.starts).read_bytes()
    starts = json.loads(data)
    if not starts or len({str(row["id"]) for row in starts}) != len(starts):
        raise ValueError("Frozen starts must have unique IDs and contain at least one state")
    writer = None

    def timeout(signum, frame):
        raise TimeoutError("Evaluation episode exceeded --timeout-s")

    previous_handler = signal.signal(signal.SIGALRM, timeout)

    def frame_callback(frame):
        if writer is not None:
            writer.append_data(frame)

    env = build_env(cfg, frame_callback=frame_callback if args.video_dir else None)
    try:
        policy = StatePklPolicy(args.state_pkl, env, cfg.get("model", {}), args.stochastic)
        canonical_starts = json.dumps(sorted(starts, key=lambda row: str(row["id"])),
                                     sort_keys=True).encode()
        protocol = dict(starts_sha256=hashlib.sha256(canonical_starts).hexdigest(), seed=args.seed,
                        max_steps=args.max_steps, stochastic=args.stochastic,
                        env=public_config(cfg["env"]), reward=cfg["reward"],
                        dataset=asdict(env.unwrapped.renderer.dataset))
        result = dict(schema_version=1, checkpoint=str(Path(args.state_pkl).resolve()),
                      config=public_config(cfg), protocol=protocol, complete=False, episodes=[])
        atomic_json(result, args.output)
        for index, start in enumerate(starts):
            # The seed follows the frozen ID, so sharding or reordering does
            # not change the sampled actions for an episode.
            seed = (args.seed + int(hashlib.sha256(str(start["id"]).encode()).hexdigest()[:8], 16)) % (2**31)
            torch.manual_seed(seed)
            env.action_space.seed(seed)
            if args.video_dir:
                import imageio.v2 as imageio
                Path(args.video_dir).mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(str(Path(args.video_dir) / f"episode_{index:06d}.mp4"), fps=10)
            try:
                signal.alarm(args.timeout_s)
                result["episodes"].append(rollout(env, policy, start, args.max_steps, seed))
            finally:
                signal.alarm(0)
                if writer is not None:
                    writer.close()
                    writer = None
            atomic_json(result, args.output)
        result["complete"] = True
        atomic_json(result, args.output)
    finally:
        env.close()
        signal.signal(signal.SIGALRM, previous_handler)
    print("EVALUATION-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
