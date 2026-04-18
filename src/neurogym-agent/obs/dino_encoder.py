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
        # Stack into (B, H, W, 3) then convert to (B, 3, H, W) float32 [0, 1]
        batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255.0)
        batch = batch.to(self.device, non_blocking=True)
        batch = F.interpolate(batch, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        batch = (batch - self._mean) / self._std
        feats = self.model(batch)
        return feats.detach().cpu().numpy().astype(np.float32)

