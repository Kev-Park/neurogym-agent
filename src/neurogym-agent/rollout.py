from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import imageio
import numpy as np
import yaml

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.action_translator import ActionSpec
from envs.dino_vec_wrapper import DinoVecWrapper
from envs.ngl_gym_env import NGLGymEnv
from envs.reward import RewardConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, segment_positions_path: str) -> DinoVecWrapper:
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
        z_shaping_coef=env_cfg.get("z_shaping_coef", 0.001),
    )

    def _make():
        return NGLGymEnv(
            neurogym_config_path=env_cfg["neurogym_config_path"],
            segment_positions_path=segment_positions_path,
            action_spec=action_spec,
            reward_cfg=reward_cfg,
            max_episode_steps=env_cfg["max_episode_steps"],
            reset_rotation_perturb_rad=env_cfg["reset_rotation_perturb_rad"],
            reset_zoom_perturb_frac=env_cfg["reset_zoom_perturb_frac"],
            headless=env_cfg.get("headless", True),
        )

    venv = DummyVecEnv([_make])
    return DinoVecWrapper(
        venv,
        repo=obs_cfg["dino_repo"],
        model_name=obs_cfg["dino_model"],
        input_size=obs_cfg["dino_input_size"],
    )


def find_latest_checkpoint(folder: Path) -> Path:
    final = folder / "final.zip"
    if final.exists():
        return final

    candidates = list(folder.glob("ppo_ngl_*_steps.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {folder}. "
            "Expected 'final.zip' or 'ppo_ngl_*_steps.zip' files."
        )

    def _step_count(p: Path) -> int:
        m = re.search(r"ppo_ngl_(\d+)_steps", p.stem)
        return int(m.group(1)) if m else -1

    return max(candidates, key=_step_count)


def main():
    parser = argparse.ArgumentParser(description="Run inference rollouts from a checkpoint folder and save videos.")
    parser.add_argument("checkpoint_folder", type=str, help="Path to a run's checkpoint directory.")
    parser.add_argument("--rollouts", type=int, default=10, help="Number of rollouts to run.")
    parser.add_argument("--max_rollout_length", type=int, default=300, help="Max steps per rollout.")
    parser.add_argument("--config", type=str, default=str(_THIS_DIR / "config" / "default.yaml"))
    parser.add_argument(
        "--segment_positions",
        type=str,
        default=str((_THIS_DIR / "../../segment_positions.parquet").resolve()),
        help="Path to segment_positions.csv.",
    )
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    checkpoint_folder = Path(args.checkpoint_folder).resolve()
    checkpoint = find_latest_checkpoint(checkpoint_folder)
    print(f"Using checkpoint: {checkpoint}")

    cfg = load_config(args.config)
    cfg["env"]["max_episode_steps"] = args.max_rollout_length

    env = build_env(cfg, args.segment_positions)
    model = PPO.load(str(checkpoint), device=cfg["train"]["device"])

    video_dir = checkpoint_folder / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    all_successes: list[bool] = []
    all_returns: list[float] = []
    all_steps: list[int] = []

    try:
        for ep in range(args.rollouts):
            obs = env.reset()
            seg_id = env.get_attr("_last_seg_id")[0]
            frames: list[np.ndarray] = [env.get_attr("_last_image")[0]]
            total_reward = 0.0
            steps = 0
            done = False
            info: dict = {}

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, rewards, dones, infos = env.step(action)
                frames.append(env.get_attr("_last_image")[0])
                total_reward += float(rewards[0])
                info = infos[0]
                steps += 1
                done = bool(dones[0])

            success = bool(info.get("episode_success", False))
            all_successes.append(success)
            all_returns.append(total_reward)
            all_steps.append(steps)

            outcome = "success" if success else "fail"
            video_path = video_dir / f"rollout_{ep:03d}_{seg_id}_{outcome}.mp4"
            with imageio.get_writer(str(video_path), fps=10, macro_block_size=1) as writer:
                for frame in frames:
                    writer.append_data(frame)

            print(
                f"ep {ep:03d} seg={seg_id} success={success} "
                f"return={total_reward:.3f} steps={steps} "
                f"z_now={info.get('z_now', float('nan')):.2f} -> {video_path.name}"
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
