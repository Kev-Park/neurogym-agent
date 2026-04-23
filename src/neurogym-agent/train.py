from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

import yaml

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.action_translator import ActionSpec
from envs.ngl_gym_env import NGLGymEnv
from envs.reward import RewardConfig
from obs.features_extractor import DinoFeaturesExtractor


class SB3WandbCallback(BaseCallback):
    """Log SB3's internal metrics directly to wandb at each rollout end."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self._total = total_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        metrics = {k: float(v) for k, v in self.logger.name_to_value.items()}
        if metrics:
            wandb.log(metrics, step=self.num_timesteps)
        print(f"[{self.num_timesteps}/{self._total} steps]")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_factory(cfg: dict, segment_positions_path: str):
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
            dino_repo=obs_cfg["dino_repo"],
            dino_model=obs_cfg["dino_model"],
            dino_input_size=obs_cfg["dino_input_size"],
        )

    return _make


def make_vec_env(cfg: dict, segment_positions_path: str, n_envs: int):
    make_fn = build_env_factory(cfg, segment_positions_path)
    if n_envs <= 1:
        return DummyVecEnv([make_fn])
    return SubprocVecEnv([make_fn for _ in range(n_envs)])


def main():
    parser = argparse.ArgumentParser(description="Train click-based PPO agent on neurogym.")
    parser.add_argument("--config", type=str, default=str(_THIS_DIR / "config" / "default.yaml"))
    parser.add_argument("--segment_positions", type=str, required=True, help="Path to segment_positions.csv.")
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--resume", type=str, default=None, help="Path to an SB3 .zip checkpoint.")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    log_cfg = cfg["logging"]

    n_envs = args.n_envs if args.n_envs is not None else train_cfg["n_envs"]
    total_timesteps = args.total_timesteps if args.total_timesteps is not None else train_cfg["total_timesteps"]
    wandb_project = args.wandb_project or log_cfg["wandb_project"]
    wandb_mode = args.wandb_mode or log_cfg["wandb_mode"]

    from wandb.integration.sb3 import WandbCallback

    wandb_run = wandb.init(
        project=wandb_project,
        mode=wandb_mode,
        name=args.run_name,
        config=cfg,
        monitor_gym=False,
    )

    vec_env = make_vec_env(cfg, args.segment_positions, n_envs)

    policy_kwargs = dict(
        features_extractor_class=DinoFeaturesExtractor,
        features_extractor_kwargs=dict(pos_hidden_dim=64),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    if args.resume:
        model = PPO.load(
            args.resume,
            env=vec_env,
            device=train_cfg["device"],
        )
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            n_steps=train_cfg["n_steps"],
            batch_size=train_cfg["batch_size"],
            n_epochs=train_cfg["n_epochs"],
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
            clip_range=train_cfg["clip_range"],
            learning_rate=train_cfg["learning_rate"],
            ent_coef=train_cfg["ent_coef"],
            vf_coef=train_cfg["vf_coef"],
            max_grad_norm=train_cfg["max_grad_norm"],
            seed=train_cfg["seed"],
            device=train_cfg["device"],
            verbose=1,
        )

    checkpoint_dir = Path(log_cfg["checkpoint_dir"]) / wandb_run.id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = CallbackList(
        [
            SB3WandbCallback(total_timesteps=total_timesteps),
            WandbCallback(
                model_save_path=str(checkpoint_dir),
                model_save_freq=log_cfg["checkpoint_freq"],
                gradient_save_freq=0,
                verbose=1,
            ),
            CheckpointCallback(
                save_freq=log_cfg["checkpoint_freq"],
                save_path=str(checkpoint_dir),
                name_prefix="ppo_ngl",
            ),
        ]
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    finally:
        model.save(str(checkpoint_dir / "final.zip"))
        vec_env.close()
        wandb_run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
