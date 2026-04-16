from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((input_size, input_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size, device=self.device)
            self.feature_dim = int(self.model(dummy).shape[-1])

    @torch.no_grad()
    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(img.convert("RGB")) for img in images])
        batch = batch.to(self.device, non_blocking=True)
        feats = self.model(batch)
        return feats.detach().cpu().numpy().astype(np.float32)

    def split_panes(self, image: Image.Image, pane_bounds_3d: tuple[int, int, int, int]) -> list[Image.Image]:
        w, h = image.size
        x0, y0, x1, y1 = pane_bounds_3d
        left_pane = image.crop((0, 0, x0, h)) if x0 > 0 else image.crop((0, 0, w // 2, h))
        right_pane = image.crop((x0, y0, x1, y1))
        return [left_pane, right_pane]
