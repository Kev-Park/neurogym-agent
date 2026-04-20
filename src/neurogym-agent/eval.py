from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from stable_baselines3 import PPO

from envs.action_translator import ActionSpec
from envs.ngl_gym_env import NGLGymEnv
from envs.reward import RewardConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, segment_positions_path: str, host: str, port: int) -> NGLGymEnv:
    env_cfg = cfg["env"]
    obs_cfg = cfg["obs"]
    action_spec = ActionSpec(
        grid_rows=env_cfg["click_grid_rows"],
        grid_cols=env_cfg["click_grid_cols"],
        pane_x0=env_cfg["pane_3d_bounds"][0],
        pane_y0=env_cfg["pane_3d_bounds"][1],
        pane_x1=env_cfg["pane_3d_bounds"][2],
        pane_y1=env_cfg["pane_3d_bounds"][3],
        rotation_bins_per_axis=env_cfg["rotation_bins_per_axis"],
        rotation_step_rad=env_cfg["rotation_step_rad"],
    )
    reward_cfg = RewardConfig(
        z_tolerance=env_cfg["z_tolerance"],
        success=env_cfg["reward_success"],
        noop_penalty=env_cfg["reward_noop_penalty"],
        noop_position_eps=env_cfg["noop_position_eps"],
    )
    return NGLGymEnv(
        host=host,
        port=port,
        segment_positions_path=segment_positions_path,
        action_spec=action_spec,
        reward_cfg=reward_cfg,
        max_episode_steps=env_cfg["max_episode_steps"],
        reset_rotation_perturb_rad=env_cfg["reset_rotation_perturb_rad"],
        reset_zoom_perturb_frac=env_cfg["reset_zoom_perturb_frac"],
        z_max=env_cfg.get("z_max", float("inf")),
        dino_repo=obs_cfg["dino_repo"],
        dino_model=obs_cfg["dino_model"],
        dino_input_size=obs_cfg["dino_input_size"],
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy on neurogym.")
    parser.add_argument("--config", type=str, default=str(_THIS_DIR / "config" / "default.yaml"))
    parser.add_argument("--segment_positions", type=str, required=True, help="Path to segment_positions.csv.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host where NGLServer is listening.")
    parser.add_argument("--port", type=int, default=7860, help="Port where NGLServer is listening.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = build_env(cfg, args.segment_positions, args.host, args.port)
    model = PPO.load(args.checkpoint, device=cfg["train"]["device"])

    all_successes: list[bool] = []
    all_returns: list[float] = []
    all_steps: list[int] = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            seg_id = info["segment_id"]
            total_reward = 0.0
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    break
            success = bool(info["episode_success"])
            all_successes.append(success)
            all_returns.append(total_reward)
            all_steps.append(steps)
            print(
                f"ep {ep:03d} seg={seg_id} success={success} "
                f"return={total_reward:.3f} steps={steps} z_now={info['z_now']:.2f}"
            )
    finally:
        env.close()

    print("\n=== aggregate ===")
    print(f"episodes:       {len(all_successes)}")
    print(f"success rate:   {np.mean(all_successes):.3f}")
    print(f"avg return:     {np.mean(all_returns):.3f}")
    print(f"avg steps:      {np.mean(all_steps):.1f}")


if __name__ == "__main__":
    main()
