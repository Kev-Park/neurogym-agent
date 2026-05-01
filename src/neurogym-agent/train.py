from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.browser_manager import BrowserManager
from envs.dino_vec_wrapper import DinoVecWrapper
from envs.env_factory import build_env_factory
from envs.ngl_gym_env import _load_segment_positions
from envs.threaded_vec_env import ThreadedVecEnv
from obs.features_extractor import DinoFeaturesExtractor


class SB3WandbCallback(BaseCallback):
    """Log SB3's internal metrics directly to wandb at each rollout end."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self._total = total_timesteps

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("browser_crash"):
                wandb.log({"env/browser_crash": 1}, step=self.num_timesteps)
            ep = info.get("episode")
            if ep is not None:
                wandb.log({
                    "env/ep_reward": ep["r"],
                    "env/ep_len": ep["l"],
                    "env/episode_success": float(info.get("episode_success", False)),
                }, step=self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        metrics = {k: float(v) for k, v in self.logger.name_to_value.items()}
        if metrics:
            wandb.log(metrics, step=self.num_timesteps)
        print(f"[{self.num_timesteps}/{self._total} steps]")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_vec_env(
    cfg: dict,
    segment_data: dict,
    segment_ids: list,
    browser_manager: BrowserManager,
    n_envs: int,
):
    obs_cfg = cfg["obs"]
    make_fn = build_env_factory(cfg, segment_data, segment_ids, browser_manager)
    if n_envs <= 1:
        venv = DummyVecEnv([make_fn])
    else:
        venv = ThreadedVecEnv([make_fn] * n_envs)
    return DinoVecWrapper(
        venv,
        repo=obs_cfg["dino_repo"],
        model_name=obs_cfg["dino_model"],
        input_size=obs_cfg["dino_input_size"],
    )


def main():
    parser = argparse.ArgumentParser(description="Train click-based PPO agent on neurogym.")
    parser.add_argument("--config", type=str, default=str(_THIS_DIR / "config" / "default.yaml"))
    parser.add_argument("--segment_positions", type=str, required=True, help="Path to segment_positions.parquet.")
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
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
    batch_size = args.batch_size if args.batch_size is not None else train_cfg["batch_size"]
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

    # Load segment data once in the main process — all worker threads share this dict.
    print("Loading segment positions...", flush=True)
    segment_data, segment_ids = _load_segment_positions(args.segment_positions)
    print(f"Loaded {len(segment_ids)} segments.", flush=True)

    browser_manager = BrowserManager(
        headless=cfg["env"].get("headless", True),
        extra_args=cfg["env"].get("chrome_args", []),
    )
    try:
        vec_env = make_vec_env(cfg, segment_data, segment_ids, browser_manager, n_envs)

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
                batch_size=batch_size,
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
    finally:
        browser_manager.close()


if __name__ == "__main__":
    main()
