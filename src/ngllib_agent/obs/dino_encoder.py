"""Frozen DINOv2 encoder — ported verbatim from the legacy `obs/dino_encoder.py`.

Runs env-side (one instance per env-runner process, shared across that process's
envs via `get_dino_encoder`) per agent_plan.md Round 8. Swappable via config hook.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DinoEncoder:
    def __init__(
        self,
        repo: str = "facebookresearch/dinov2",
        model_name: str = "dinov2_vits14",
        input_size: int = 224,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load(repo, model_name, trust_repo=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

        self.input_size = input_size
        self._mean = _IMAGENET_MEAN.to(self.device)
        self._std = _IMAGENET_STD.to(self.device)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size, device=self.device)
            self.feature_dim = int(self.model(dummy).shape[-1])

    @torch.no_grad()
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode a list of RGB numpy arrays (H, W, 3) uint8 into DINO feature vectors."""
        batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255.0)
        batch = batch.to(self.device, non_blocking=True)
        batch = F.interpolate(
            batch, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
        )
        batch = (batch - self._mean) / self._std
        feats = self.model(batch)
        return feats.detach().cpu().numpy().astype(np.float32)


_ENCODER_CACHE: dict[tuple, DinoEncoder] = {}


def get_dino_encoder(
    repo: str = "facebookresearch/dinov2",
    model_name: str = "dinov2_vits14",
    input_size: int = 224,
    device: str | None = None,
) -> DinoEncoder:
    """Per-process singleton so all envs in an env-runner share one frozen model."""
    key = (repo, model_name, input_size, device)
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = DinoEncoder(
            repo=repo, model_name=model_name, input_size=input_size, device=device
        )
    return _ENCODER_CACHE[key]
