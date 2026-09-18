# DINO-server throughput experiment (simulator backend)

Goal: cut the per-process DINO + CUDA-context VRAM cost so freed VRAM can host
MORE env-runners on one GPU slice, raising per-GPU SPS — and check the gain
beats the added image-shipping cost. Single-node first (this is about per-GPU
density). Baseline = current per-process DINO at its optimal packing.

## Why there is headroom

In `threads` mode each env-runner PROCESS holds one shared DINO + one CUDA
context; N runners on a 24 GB 3090 => N copies. At ~32 procs the GPU sits at
~15.3 G, of which the dominant term is the per-process CUDA context (~0.3-0.6 G
each), then DINO weights (~84 MB fp32 x N ~ 2.7 G), then GL render buffers.
Moving DINO to a server AND policy inference to CPU (`num_gpus_per_env_runner=0`)
leaves a runner with NO CUDA context — only its GL/EGL render context — freeing
~0.4-0.6 G/runner to respend on more runners.

## Design (Phase 1 — building)

- `obs/dino_server.py`: `DinoServer` async Ray actor (one frozen DINO + dynamic
  micro-batching: coalesce concurrent `encode()` calls up to `max_batch`,
  waiting <= `max_delay_ms`; forward runs in a thread executor so the loop keeps
  queuing during GPU compute). `ensure_dino_servers(M, ...)` spawns M named
  detached actors on the driver. `DinoServerClient` (same `.encode`/`.feature_dim`
  interface as `DinoEncoder`) ships stacked uint8 panes; runner `worker_index % M`
  picks the server.
- `env_build.py`: when `obs.dino.server.enabled`, inject `DinoServerClient`
  instead of `get_dino_encoder`; `worker_index` threaded through `build_env`.
- `train.py`: after `ray.init`, spawn the servers when enabled.
- Launch with `--num-gpus-per-env-runner 0` (runner has no CUDA; render via EGL,
  inference on CPU). Servers hold fractional GPU tokens (all share device 0).

### Integration risk to validate EARLY
EGL device selection is independent of `CUDA_VISIBLE_DEVICES`, so a runner with
`num_gpus=0` (empty CUDA_VISIBLE_DEVICES) SHOULD still render on device 0. Must
confirm the simulator renders under `num_gpus_per_env_runner=0` before the sweep.

### Sweep
Start aggressive: **M=1** (max dedup). Push runner count up until VRAM/CPU
saturates; record peak SPS + VRAM occupancy + server `stats()` (mean batch).
Scale M up only if the single server bottlenecks (its GPU can't keep up ->
callers queue -> SPS caps). For each M, re-find the runner count that maxes SPS.
Readout: does freed-VRAM->more-runners beat the ship cost — net SPS vs baseline?

## Phase 2 — on-GPU direct feed (stretch; after Phase 1 finds optimal M/runners)

VRAM is per-process, so "never leave VRAM" splits two ways:
- **2a (cheaper, latency-only):** per-process DINO + GL<->CUDA interop
  (`cudaGraphicsGLRegisterImage`) so the rendered texture feeds the ViT with no
  `glReadPixels`/numpy/re-upload. Does NOT dedup VRAM. Sizes the readback cost.
- **2b (VRAM + latency):** server + **CUDA IPC** — runner interops the texture
  into a CUDA buffer, sends the `cudaIpcGetMemHandle` (a ~64-byte handle, not the
  pixels) over Ray; server `cudaIpcOpenMemHandle` reads it from VRAM directly.
  Same-GPU only; needs double-buffering + IPC events for read-before-overwrite.
  Requires ngllib's SimulatorRenderer to expose the GL texture (currently returns
  numpy) -> a paired ngllib worktree.

Do 2a first (bounds the win), then 2b if Phase 1 shows VRAM is the binding lever.

## MPS experiment (separate worktree `mps`, no code)

Run the existing per-process sim under an `nvidia-cuda-mps-control` daemon;
measure whether shared context frees enough VRAM to fit more runners and whether
kernel overlap raises SPS. Report independently; combine with the server only
after both are measured (per user).
