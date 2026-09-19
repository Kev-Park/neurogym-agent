"""Shared DINO inference server — dedupe the per-process DINO copy + CUDA context.

In `threads` mode each env-runner PROCESS loads its own frozen DINO (N copies +
N CUDA contexts share one GPU). This module moves the encode into M `DinoServer`
Ray actors instead: runner ``r`` routes to server ``r % M``. Concurrent
``encode()`` calls from the runners' env threads are dynamically micro-batched
inside each server, so no caller waits on a global full-batch barrier (the
failure mode of the old M=1 service that had to assemble every env's image
before a forward pass).

The win: with DINO off the runner AND policy inference on CPU
(``num_gpus_per_env_runner=0``), a runner process holds no CUDA context at all —
only its GL/EGL render context — freeing ~0.4-0.6 GB/runner (context + weights +
activations) to respend on more runners. See the dino-server experiment.

`DinoServerClient` implements the same ``.encode(list[img]) -> (B, D)`` /
``.feature_dim`` interface as `DinoEncoder`, so it drops into
`DinoObservationWrapper` unchanged.
"""

from __future__ import annotations

import asyncio

import numpy as np
import ray

# Named-actor prefix so runners can look a server up without a handle passed
# through RLlib's env_config plumbing.
SERVER_NAME_FMT = "dino_server_{}"


@ray.remote
class DinoServer:
    """Async Ray actor: one frozen DINO + a dynamic micro-batching loop.

    Concurrency model: `encode` is an async method, so Ray runs each call as a
    coroutine on the actor's single event loop. Each call enqueues its images +
    a future and awaits; a background `_batch_loop` coalesces whatever is queued
    (up to `max_batch`, waiting at most `max_delay_ms` for stragglers) into one
    forward pass. The (blocking) torch forward runs in a thread executor so the
    event loop keeps accepting requests while the GPU computes — pipelining the
    next batch against the current one.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        input_size: int = 224,
        max_batch: int = 64,
        max_delay_ms: float = 3.0,
    ):
        from .dino_encoder import DinoEncoder

        self._enc = DinoEncoder(model_name=model_name, input_size=input_size)
        self._feature_dim = int(self._enc.feature_dim)
        self._max_batch = int(max_batch)
        self._max_delay = float(max_delay_ms) / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._loop_task: asyncio.Future | None = None
        # Lightweight counters for the experiment readout (batch efficiency).
        self._n_batches = 0
        self._n_images = 0
        # Rebuilt CUDA-IPC tensors, keyed by handle: each runner ships the SAME
        # payload every step (stable VRAM), so open/rebuild once and re-read.
        self._ipc_cache: dict = {}

    def feature_dim(self) -> int:
        return self._feature_dim

    def stats(self) -> dict:
        """(batches, images, mean_batch) since construction — batch efficiency."""
        mean = (self._n_images / self._n_batches) if self._n_batches else 0.0
        return {"batches": self._n_batches, "images": self._n_images, "mean_batch": mean}

    async def _ensure_loop(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.ensure_future(self._batch_loop())

    async def encode(self, images: np.ndarray) -> np.ndarray:
        """images: (n, H, W, 3) uint8 numpy -> (n, D) float32."""
        await self._ensure_loop()
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put(("np", images, fut, int(images.shape[0])))
        return await fut

    async def encode_ipc(self, payloads) -> np.ndarray:
        """payloads: a list of (reduce_tensor (rebuild, args), gl_flip) for the
        env's GPU panes -- [ (left,False), (right,True) ] both-panes, or
        [ (right,True) ] right-only -> (len, D) float32, one row per pane in
        order. Pixels stay in VRAM."""
        await self._ensure_loop()
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put(("ipc", payloads, fut, len(payloads)))
        return await fut

    def _run_batch(self, batch: list, total: int) -> np.ndarray:
        """Blocking forward for one coalesced batch (runs in a thread executor).
        A run is homogeneous in practice (all np OR all ipc); mixed is handled."""
        import torch

        kinds = {k for k, _ in batch}
        if kinds == {"np"}:
            stacked = np.concatenate([d for _, d in batch], axis=0)
            return self._enc.encode(list(stacked))
        # ipc (or mixed): rebuild each GPU frame; keep pixels in VRAM. Each ipc
        # item is a LIST of (payload, gl_flip) panes; keep per-pane flips so the
        # left EM pane (unflipped) and right 3D pane (flipped) coexist in a batch.
        gpu_imgs = []
        flips = []
        for kind, d in batch:
            if kind == "ipc":
                for payload, flip in d:
                    rebuild, args = payload
                    key = tuple(a for a in args if isinstance(a, bytes))  # IPC handles
                    t = self._ipc_cache.get(key)
                    if t is None:
                        t = rebuild(*args)      # open/map once per runner handle
                        self._ipc_cache[key] = t
                    gpu_imgs.append(t)          # (H, W, 4) uint8 cuda, re-read
                    flips.append(bool(flip))
            else:
                for im in d:
                    gpu_imgs.append(torch.from_numpy(im).cuda())
                    flips.append(False)         # np frames are already image-order
        return self._enc.encode_gpu(gpu_imgs, gl_flip=flips)

    async def _batch_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            kind, data, fut, count = await self._queue.get()
            batch = [(kind, data)]
            futs = [fut]
            counts = [count]
            total = count
            deadline = loop.time() + self._max_delay
            while total < self._max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    k2, d2, f2, c2 = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                batch.append((k2, d2))
                futs.append(f2)
                counts.append(c2)
                total += c2
            try:
                feats = await loop.run_in_executor(None, self._run_batch, batch, total)
                self._n_batches += 1
                self._n_images += total
                off = 0
                for f, c in zip(futs, counts):
                    if not f.done():
                        f.set_result(feats[off : off + c])
                    off += c
            except Exception as e:  # propagate to every waiter in the batch
                for f in futs:
                    if not f.done():
                        f.set_exception(e)


def ensure_dino_servers(
    num_servers: int,
    *,
    model_name: str = "dinov2_vits14",
    input_size: int = 224,
    max_batch: int = 64,
    max_delay_ms: float = 3.0,
    num_gpus: float = 0.1,
) -> list:
    """Create (or fetch, if already present) M named detached `DinoServer`s.

    Called once on the driver before `build_algo()`. Detached + named so the
    env-runner processes can `ray.get_actor(name)` them. Idempotent: an existing
    actor of the same name is reused (so a resumed run doesn't double-spawn).

    `num_gpus` is a logical Ray token that just pins the actor to the physical
    GPU (sets CUDA_VISIBLE_DEVICES); all M share device 0 via fractional GPUs.
    Keep sum(M*num_gpus) + runners' fractions <= RAY_NUM_GPUS.
    """
    servers = []
    for i in range(num_servers):
        name = SERVER_NAME_FMT.format(i)
        try:
            servers.append(ray.get_actor(name))
        except ValueError:
            servers.append(
                DinoServer.options(
                    name=name,
                    lifetime="detached",
                    num_gpus=num_gpus,
                    max_concurrency=1000,  # many concurrent encode() coroutines
                ).remote(
                    model_name=model_name,
                    input_size=input_size,
                    max_batch=max_batch,
                    max_delay_ms=max_delay_ms,
                )
            )
    return servers


class DinoServerClient:
    """Env-side encoder that ships panes to a `DinoServer` actor.

    Same interface as `DinoEncoder` (`.encode(list[img]) -> (B, D)`,
    `.feature_dim`) so `DinoObservationWrapper` is unchanged. Constructed inside
    an env-runner; looks its server up by name so no handle needs to thread
    through RLlib's env_config.
    """

    def __init__(self, server_index: int, feature_dim: int | None = None):
        self._actor = ray.get_actor(SERVER_NAME_FMT.format(server_index))
        self.feature_dim = (
            int(feature_dim)
            if feature_dim is not None
            else int(ray.get(self._actor.feature_dim.remote()))
        )

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        arr = np.ascontiguousarray(np.stack(images)).astype(np.uint8, copy=False)
        return ray.get(self._actor.encode.remote(arr))

    def encode_ipc(self, payloads) -> np.ndarray:
        """Ship a list of (reduce_tensor (rebuild, args), gl_flip) pane payloads;
        the server rebuilds them in-VRAM and runs DINO. Returns (len, D) float32,
        one row per pane in order (left EM then right 3D for both-panes)."""
        return ray.get(self._actor.encode_ipc.remote(payloads))
