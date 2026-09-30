"""Frozen DINOv2 encoder — ported verbatim from the legacy `obs/dino_encoder.py`.

Runs env-side (one instance per env-runner process, shared across that process's
envs via `get_dino_encoder`) per agent_plan.md Round 8. Swappable via config hook.

CUDA-graph support (2026-09-30, ported from throughput-scaling): the frozen ViT +
fixed preprocess (permute/float/resize/normalize) is a static graph for a given
input signature, so we capture once and replay from a static input buffer to kill
the per-layer launch overhead. On the zmax-left CPU-readback `encode()` path the
H2D copy of the pane pixels stays OUTSIDE the graph (a static device buffer is fed
each step); only the device-side preprocess+ViT is captured. Isolated ceiling for
this workload = 1.42x at the co-pane batch of 2 (job 997385, bitwise-identical).
Gated by `obs.dino.cuda_graph`; eager is always correct, so a capture failure
falls back permanently for that signature and never crashes a run.
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
        use_cuda_graph: bool = False,
        use_compile: bool = False,
        use_noop: bool = False,
        use_fp16: bool = False,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # R_cap probe: skip the ViT forward, return zero features. Renders + env
        # still run identically, so training sps then measures the render+env
        # pipeline capacity with DINO removed.
        self._noop = bool(use_noop)
        self.model = torch.hub.load(repo, model_name, trust_repo=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

        self.input_size = input_size
        self._mean = _IMAGENET_MEAN.to(self.device)
        self._std = _IMAGENET_STD.to(self.device)

        # fp16: run the ViT in half precision (tensor cores). Numerics drift from
        # eager fp32 -> DYNAMICS-GATED (not a drop-in; needs a transfer/parity
        # check before adoption). Weights + running buffers cast once here.
        self._fp16 = bool(use_fp16) and self.device.type == "cuda"
        if self._fp16:
            self.model.half()

        # CUDA graphs and torch.compile both collapse the ViT's per-layer Python
        # dispatch; they are alternatives, not stacked (compile has its own graph).
        self._use_graph = bool(use_cuda_graph) and (not use_compile) and self.device.type == "cuda"
        self._graphs: dict = {}          # sig -> (graph, static_in_u8, static_out)
        self._graph_failed: set = set()  # sigs that failed capture -> stay eager

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size, device=self.device)
            if self._fp16:
                dummy = dummy.half()
            self.feature_dim = int(self.model(dummy).shape[-1])

        # torch.compile (Inductor): fused kernels + traced-away dispatch, no capture
        # race, but numerics drift -> dynamics-gated. NOT reduce-overhead (which
        # re-adds cudagraphs + the MPS capture race).
        if bool(use_compile) and self.device.type == "cuda":
            self.model = torch.compile(self.model)

    @torch.no_grad()
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode a list of RGB numpy arrays (H, W, 3) uint8 into DINO feature vectors."""
        if self._noop:
            return np.zeros((len(images), self.feature_dim), dtype=np.float32)
        if self._use_graph:
            feats = self._encode_graph(images)
            if feats is not None:
                return feats.detach().cpu().numpy().astype(np.float32)
        batch = self._to_model_batch(torch.from_numpy(np.stack(images)).to(
            self.device, non_blocking=True))
        feats = self.model(batch)
        return feats.detach().cpu().numpy().astype(np.float32)

    def _to_model_batch(self, dev_u8: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 3) uint8 device tensor -> normalized (B, 3, S, S) model input.
        Shared by the eager and CUDA-graph paths so they are numerically identical."""
        batch = dev_u8.permute(0, 3, 1, 2).float().div(255.0)
        batch = F.interpolate(
            batch, size=(self.input_size, self.input_size), mode="bilinear",
            align_corners=False)
        batch = (batch - self._mean) / self._std
        return batch.half() if self._fp16 else batch

    def _encode_graph(self, images: list[np.ndarray]):
        """CUDA-graph replay of the preprocess+ViT. The H2D pixel copy stays outside
        the graph (fed into a static device buffer each step); only the device-side
        compute is captured. Returns the static output tensor, or None to fall back
        to eager (unsupported signature or a capture that failed)."""
        arr = np.stack(images)  # (B, H, W, 3) uint8
        sig = (arr.shape, str(arr.dtype))
        if sig in self._graph_failed:
            return None
        entry = self._graphs.get(sig)
        if entry is None:
            entry = self._build_graph(sig, arr)
            if entry is None:
                self._graph_failed.add(sig)
                return None
        graph, static_in, static_out = entry
        static_in.copy_(torch.from_numpy(arr), non_blocking=True)  # feed this step's pixels
        graph.replay()
        return static_out

    def _build_graph(self, sig, arr: np.ndarray):
        """Warm up on a side stream (trigger all lazy allocs), then capture the
        preprocess+ViT into a CUDA graph reading from a static input buffer. Any
        failure returns None so the signature falls back to eager permanently -- a
        graph must never crash a training run."""
        import logging
        import os
        lf = None
        try:
            static_in = torch.from_numpy(arr).to(self.device)  # static (B,H,W,3) uint8
            # Serialize capture across co-resident MPS processes: concurrent capture
            # on a shared GPU races. An exclusive node-local file lock makes captures
            # happen one at a time. Best-effort (POSIX fcntl only).
            try:
                import fcntl
                lockpath = os.environ.get("NGL_DINO_CAPTURE_LOCK", "/tmp/ngl_dino_cudagraph.lock")
                lf = open(lockpath, "w")
                fcntl.flock(lf, fcntl.LOCK_EX)
            except Exception:
                if lf is not None:
                    lf.close()
                lf = None
            # Warm up to trigger ALL lazy allocations (cuDNN/cuBLAS workspaces,
            # pos-encoding interp) BEFORE capture, so capture never calls cudaMalloc.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.no_grad(), torch.cuda.stream(s):
                for _ in range(8):
                    _ = self.model(self._to_model_batch(static_in))
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                static_out = self.model(self._to_model_batch(static_in))
            torch.cuda.synchronize()
            self._graphs[sig] = (graph, static_in, static_out)
            logging.getLogger(__name__).info(
                "DinoEncoder: captured CUDA graph for sig=%s", sig)
            return self._graphs[sig]
        except Exception as e:  # capture is best-effort; eager is always correct
            logging.getLogger(__name__).warning(
                "DinoEncoder: CUDA-graph capture failed (%s) -> eager for sig=%s", e, sig)
            return None
        finally:
            if lf is not None:
                try:
                    import fcntl
                    fcntl.flock(lf, fcntl.LOCK_UN)
                    lf.close()
                except Exception:
                    pass


_ENCODER_CACHE: dict[tuple, DinoEncoder] = {}


def get_dino_encoder(
    repo: str = "facebookresearch/dinov2",
    model_name: str = "dinov2_vits14",
    input_size: int = 224,
    device: str | None = None,
    use_cuda_graph: bool = False,
    use_compile: bool = False,
    use_noop: bool = False,
    use_fp16: bool = False,
) -> DinoEncoder:
    """Per-process singleton so all envs in an env-runner share one frozen model."""
    key = (repo, model_name, input_size, device, bool(use_cuda_graph),
           bool(use_compile), bool(use_noop), bool(use_fp16))
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = DinoEncoder(
            repo=repo, model_name=model_name, input_size=input_size, device=device,
            use_cuda_graph=use_cuda_graph, use_compile=use_compile,
            use_noop=use_noop, use_fp16=use_fp16,
        )
    return _ENCODER_CACHE[key]
