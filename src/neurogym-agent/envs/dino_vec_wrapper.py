from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

import sys
from pathlib import Path
_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from gymnasium import spaces
from obs.dino_encoder import DinoEncoder


class DinoVecWrapper(VecEnvWrapper):
    """Encodes raw image observations from each env with a shared DINO model.

    Runs in the main process so CUDA is never initialized inside subprocesses,
    avoiding Vulkan/CUDA GPU conflicts with Chrome.  Also enables true batch
    inference across all envs simultaneously.
    """

    def __init__(
        self,
        venv,
        repo: str = "facebookresearch/dinov2",
        model_name: str = "dinov2_vits14",
        input_size: int = 224,
        device: str | None = None,
    ):
        self._dino = DinoEncoder(
            repo=repo,
            model_name=model_name,
            input_size=input_size,
            device=device,
        )

        obs_space = venv.observation_space
        new_obs_space = spaces.Dict({
            "image_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._dino.feature_dim,),
                dtype=np.float32,
            ),
            "pos_state": obs_space["pos_state"],
        })
        super().__init__(venv, observation_space=new_obs_space)

    def _encode(self, obs: dict) -> dict:
        images = list(obs["image"])  # list of (H, W, 3) uint8 arrays
        feats = self._dino.encode(images)  # (n_envs, feature_dim)
        return {
            "image_features": feats.astype(np.float32),
            "pos_state": obs["pos_state"],
        }

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return self._encode(obs)

    def step_wait(self) -> tuple:
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._encode(obs), rewards, dones, infos
