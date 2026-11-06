"""GPU-job validation of offline materialization, custom training and inference."""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from ngllib import DatasetSpec
from ngllib_agent.data import main as data_command, resolve_task_config
from ngllib_agent.env_build import build_env, load_config
from ngllib_agent.evaluate import main as evaluate
from ngllib_agent.train import build_argparser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(build_argparser().parse_args([]).config)
    dataset = {"id": "calibrated-validation", **asdict(DatasetSpec.from_config())}
    cfg["task"] = {"factory": "examples.axis_task:build_task", "manifest": str(out / "manifest.json"),
                   "precompute": {"factory": "examples.axis_task:precompute", "params": {
                       "source_parquet": "/scratch/kp0374/neurogym-agent/segment_positions.parquet",
                       "seed": 73, "count": 2, "delta_x": 100, "dataset": dataset}}}
    cfg["reward"] = {"mode": "custom_x"}
    cfg["ppo"].update(sgd_minibatch_size=32, num_sgd_iter=1)
    cfg["env"]["reset_ahead"] = False
    cfg["env"]["max_episode_steps"] = 12
    path = out / "task.yaml"
    path.write_text(yaml.safe_dump(cfg))
    data_command(["precompute", "--config", str(path), "--output", str(out / "manifest.json")])
    resolved, manifest = resolve_task_config(load_config(path))
    assert not resolved["env"].get("parquet_path")
    for backend in ("simulator", "chrome"):
        raw_cfg = copy.deepcopy(cfg)
        raw_cfg["obs"]["mode"] = "raw"
        raw_cfg["env"].update(backend=backend, capture_scale=.5)
        env = build_env(raw_cfg)
        try:
            obs, info = env.reset(seed=17)
            assert set(obs) == set(env.observation_space.spaces)
            assert "segments" not in obs and "segment_zmax" not in obs
            assert info["task_metadata"] == {"task_axis": "x"}
            obs, reward, _, _, info = env.step(env.action_space.sample())
            assert np.isfinite(reward) and "segments" not in obs and "segment_zmax" not in obs
        finally:
            env.close()
    subprocess.run([sys.executable, "-m", "ngllib_agent.train", "--config", str(path),
                    "--run-name", "custom-axis-validation", "--iters", "1",
                    "--train-batch-size", "128", "--num-env-runners", "1",
                    "--num-envs-per-env-runner", "1", "--num-cpus-per-env-runner", "4",
                    "--num-gpus-per-env-runner", ".5", "--learner-gpu", "--object-store-gb", "2",
                    "--checkpoint-dir", str(out / "checkpoints"), "--checkpoint-every", "1"],
                   check=True, timeout=600)
    checkpoint = out / "checkpoints" / "ckpt_000001.pkl"
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    assert metadata["config"]["task"]["data_identity"]["manifest_sha256"]
    evaluate(["--state-pkl", str(checkpoint), "--starts", manifest["artifacts"]["starts"]["path"],
              "--max-steps", "4", "--output", str(out / "evaluation.json")])
    print("CUSTOM-TASK-PRECOMPUTATION-TRAINING-INFERENCE-PASS", flush=True)


if __name__ == "__main__":
    main()
