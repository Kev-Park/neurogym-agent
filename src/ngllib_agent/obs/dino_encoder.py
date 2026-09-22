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
        use_cuda_graph: bool = False,
        use_noop: bool = False,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # R_cap probe: skip the ViT forward, return zero features. Renders + interop
        # + env still run identically (upstream in env.step), so training sps then
        # measures the render+env pipeline capacity with DINO removed.
        self._noop = bool(use_noop)
        self.model = torch.hub.load(repo, model_name, trust_repo=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

        self.input_size = input_size
        self._mean = _IMAGENET_MEAN.to(self.device)
        self._std = _IMAGENET_STD.to(self.device)

        # CUDA-graph capture of the encode_gpu path: the frozen ViT + fixed
        # preprocess (flip/pad/resize/norm) is a static graph for a given input
        # signature, so we capture once and replay from static input buffers to
        # kill per-launch overhead on the batch=M interop path. Only on CUDA.
        self._use_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self._graphs: dict = {}   # sig -> (graph, static_in list, static_out)
        self._graph_failed: set = set()  # sigs that failed capture -> stay eager

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size, device=self.device)
            self.feature_dim = int(self.model(dummy).shape[-1])

    @torch.no_grad()
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode a list of RGB numpy arrays (H, W, 3) uint8 into DINO feature vectors."""
        if self._noop:
            return np.zeros((len(images), self.feature_dim), dtype=np.float32)
        batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255.0)
        batch = batch.to(self.device, non_blocking=True)
        batch = F.interpolate(
            batch, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
        )
        batch = (batch - self._mean) / self._std
        feats = self.model(batch)
        return feats.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_gpu(self, imgs: list, *, gl_flip: bool | list = False,
                   top_pad: int | list = 0) -> np.ndarray:
        """Encode a list of (H, W, C) uint8 CUDA tensors already on the GPU (the
        CUDA-IPC path — pixels never touch the CPU). C in {3,4}; alpha is dropped.
        `gl_flip` reverses rows for GL's bottom-up framebuffer orientation — a
        bool applied to all, or a per-image list (the left EM pane is read
        unflipped, the right 3D pane flipped, so a mixed batch needs both).
        `top_pad` prepends N black rows in IMAGE order (after the flip), matching
        the numpy path's `canvas[TOOLBAR:] = pane` toolbar strip so the interop
        frame is framed identically to readback (else DINO sees a different
        aspect + no toolbar -> divergent obs). An int for all or a per-image list."""
        if self._noop:
            return np.zeros((len(imgs), self.feature_dim), dtype=np.float32)
        flips = [gl_flip] * len(imgs) if isinstance(gl_flip, bool) else list(gl_flip)
        pads = [int(top_pad)] * len(imgs) if isinstance(top_pad, int) else [int(p) for p in top_pad]
        if self._use_graph:
            feats = self._encode_gpu_graph(imgs, flips, pads)
            if feats is not None:
                return feats.detach().cpu().numpy().astype(np.float32)
        feats = self._forward_from(imgs, flips, pads)
        return feats.detach().cpu().numpy().astype(np.float32)

    def _forward_from(self, imgs: list, flips: list, pads: list):
        """Preprocess (flip/pad/RGB/normalize/resize) + ViT forward on GPU tensors.
        Shared by the eager path and the CUDA-graph capture so they are identical."""
        procd = []
        for t, flip, pad in zip(imgs, flips, pads):
            if flip:
                t = torch.flip(t, dims=[0])
            if pad:
                z = torch.zeros((int(pad), t.shape[1], t.shape[2]),
                                dtype=t.dtype, device=t.device)
                t = torch.cat([z, t], dim=0)          # black toolbar strip on top
            t = t[:, :, :3].permute(2, 0, 1).float().div(255.0)  # (3, H, W)
            procd.append(t)
        batch = torch.stack(procd, 0).to(self.device, non_blocking=True)
        batch = F.interpolate(
            batch, size=(self.input_size, self.input_size), mode="bilinear",
            align_corners=False)
        batch = (batch - self._mean) / self._std
        return self.model(batch)

    def _encode_gpu_graph(self, imgs: list, flips: list, pads: list):
        """CUDA-graph replay of _forward_from. Returns the (static) output tensor,
        or None to signal the caller to fall back to the eager path (unsupported
        signature or a capture that failed). Keyed by a signature that fixes every
        static aspect of the graph (per-image shape/dtype + flips + pads)."""
        sig = (tuple((tuple(t.shape), str(t.dtype)) for t in imgs),
               tuple(bool(f) for f in flips), tuple(int(p) for p in pads))
        if sig in self._graph_failed:
            return None
        entry = self._graphs.get(sig)
        if entry is None:
            entry = self._build_graph(sig, imgs, flips, pads)
            if entry is None:
                self._graph_failed.add(sig)
                return None
        graph, static_in, static_out = entry
        for buf, t in zip(static_in, imgs):
            buf.copy_(t, non_blocking=True)   # feed this step's pixels into the graph
        graph.replay()
        return static_out

    def _build_graph(self, sig, imgs: list, flips: list, pads: list):
        """Warm up on a side stream, then capture _forward_from into a CUDA graph
        reading from freshly-allocated static input buffers. Any capture failure
        (a data-dependent op in the model, an unsupported kernel) returns None so
        the signature falls back to eager permanently -- a graph must never crash
        a training run."""
        import logging, os
        lf = None
        try:
            static_in = [torch.empty_like(t) for t in imgs]
            for buf, t in zip(static_in, imgs):
                buf.copy_(t)
            # Serialize capture across co-resident MPS processes: concurrent capture
            # on a shared GPU races (one client's cudaMalloc/sync breaks another's
            # in-flight capture -> only ~half succeed). An exclusive NODE-LOCAL file
            # lock makes captures happen one at a time. Best-effort (fnctl only on
            # POSIX; if unavailable we just capture without the lock).
            try:
                import fcntl
                lockpath = os.environ.get("NGL_DINO_CAPTURE_LOCK", "/tmp/ngl_dino_cudagraph.lock")
                lf = open(lockpath, "w")
                fcntl.flock(lf, fcntl.LOCK_EX)
            except Exception:
                if lf is not None:
                    lf.close()
                lf = None
            # Warm up enough to trigger ALL lazy allocations (cuDNN/cuBLAS workspaces,
            # DINOv2 pos-encoding interpolation) BEFORE capture, so capture itself
            # never calls cudaMalloc (the other capture-failure cause).
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(8):
                    self._forward_from(static_in, flips, pads)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._forward_from(static_in, flips, pads)
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
    use_noop: bool = False,
) -> DinoEncoder:
    """Per-process singleton so all envs in an env-runner share one frozen model."""
    key = (repo, model_name, input_size, device, bool(use_cuda_graph), bool(use_noop))
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = DinoEncoder(
            repo=repo, model_name=model_name, input_size=input_size, device=device,
            use_cuda_graph=use_cuda_graph, use_noop=use_noop,
        )
    return _ENCODER_CACHE[key]
