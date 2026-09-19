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

## Phase 1 validation (2026-09-18, job 923652, 16x2, M=1)

WORKS end-to-end. Critical risk CLEARED: runners launched with
`--num-gpus-per-env-runner 0` still render — log shows `simulator GL: NVIDIA
GeForce RTX 3090` on the EnvRunners despite an empty CUDA_VISIBLE_DEVICES, i.e.
EGL device selection is independent of CUDA_VISIBLE_DEVICES. DinoServer spawned
and encoded; `iter 1: sps=41.5 H=16/16` (warmup). Next: density sweep (32x2 to
match the per-process ~172 sps baseline, then 48x2 / 64x2 to spend freed VRAM).

## Phase 1 RESULTS (2026-09-18, real PPO load, >=10-iter steady state, fragment-125)

VRAM (the win): per-process 32x2 ~15 GB -> DINO-server 32x2 ~4.2 GB (server ~1.5 G
+ learner ~0.5 G + 32 runners' GL ~2 G). Runners hold ZERO CUDA context
(num_gpus=0; render via EGL). ~11 GB freed on a 24 GB card; GPU util 43% at 32x2.

SPS (M=1 unless noted):
  16x2 @48cpu ......... 167
  32x2 @48cpu ......... 176   <- PEAK (= per-process baseline ~172)
  48x2 @48cpu(0.9/rnr)  155
  48x2 @64cpu(1.2/rnr)  162
  60x2 @64cpu(1.0/rnr)  156
  48x2 M=2 @48cpu ..... 155   (== M=1: server NOT the limiter)
  32x2 NO-BATCH ....... ~92   (batching ~2x FASTER: coalescing helps, not a straggler)

Conclusions:
1. Server works; num_gpus=0 runners render via EGL (CUDA_VISIBLE_DEVICES-independent).
2. VRAM slashed ~15 -> ~4 GB (per-process CUDA context + N DINO copies eliminated).
3. SPS does NOT rise. Peak ~176 at 32x2 = per-process. Adding runners does not beat
   it even at MAX CPU (64 cores): 48x2 and 60x2 stay ~155-163 < 176.
4. Bottleneck is NOT VRAM (freed), NOT DINO count (M2==M1), NOT CPU (64cpu barely
   helped, +7 sps). It is the SYNCHRONOUS-PPO barrier + per-step pipeline latency:
   adding envs adds straggler/sync overhead that cancels the extra parallelism
   (matches the prior "3x doesn't survive RLlib" finding).
5. Batching is beneficial (no-batch ~2x slower); NOT a straggler source here,
   because the server is far from compute-saturated so bigger forwards just amortize
   RPC + GPU-launch overhead.

Implications:
- On ONE node under synchronous PPO the DINO server is a VRAM optimization, not a
  throughput one. Freed VRAM buys headroom (bigger model / more panes / higher-res
  obs / bigger batch), NOT more-runners-for-SPS.
- To turn VRAM into SPS you must attack the real ceiling: (a) async PPO (APPO/IMPALA)
  to kill the barrier, or (b) cut per-step LATENCY -> which reframes Phase 2b
  (on-GPU direct DINO feed, no readback) as the higher-value SPS lever, above packing
  runners. Or (c) scale across nodes (each GPU hosts more runners, but multi-node
  re-adds broadcast + straggler tax).

## Phase 2b implementation notes (CUDA IPC, fully-in-VRAM; ngllib pairing)

Paired worktrees: agent `wt/neurogym-agent-dino-server` <-> ngllib
`wt/neurogym-dino-server` (branch dino-server, both). Repoint the agent
pyproject ngllib source to `../neurogym-dino-server` + resync BEFORE Phase 2b
(kept at ../../neurogym shared main through Phase 1 so the sweep is unperturbed).
Rule: >=1 DinoServer per GPU (self-contained) so IPC stays same-device.

Pure Python, no CUDA kernels. Pipeline per step:
  moderngl render (GL texture, in runner)
  -> GL->CUDA map: PyCUDA `pycuda.gl` (cudaGraphicsGLRegisterImage +
     MapResources), or a thin cffi binding  [FRAGILE piece: EGL-context interop]
  -> cudaMemcpy (device->device) into a persistent, IPC-capable torch CUDA tensor
  -> torch native CUDA IPC: storage._share_cuda_() -> small handle
  -> ship handle (not pixels) over Ray to the server
  -> server rebuilds tensor (torch.multiprocessing reductions) -> DINO forward
  -> return features (small) over Ray
Sync: double-buffer the render target + torch.cuda.Event so the server reads
before the runner's next render overwrites. ngllib change: SimulatorRenderer must
expose the GL texture / a CUDA handle instead of only the numpy image.
Fallback if GL-interop is too fragile: 2a (per-process DINO + GL-CUDA map, no
IPC) — proves the latency win but not the VRAM dedup.

## Phase 2b RESULTS (2026-09-18) — WORKS end-to-end

The fully-in-VRAM CUDA-IPC on-GPU DINO feed runs: v7 (8x1, right-pane-only)
reached H=8/8 healthy at ~199 sps, COMPLETED. render -> GL->CUDA export (in VRAM)
-> reduce_tensor IPC handle over Ray -> server rebuilds+caches in VRAM -> DINO,
no pixel to CPU.

Bugs fixed to get there (all subtle, none fundamental):
1. 208 cudaErrorInvalidGraphicsContext: self._color was the ACTIVE FBO color
   attachment. Bind a scratch FBO (+ glFinish) before BOTH register and map.
2. Per-step worker crash ("int() ... not 'tuple'"): cuda-python returns a 1-tuple
   (err,); `int(rt.cudaMemcpy2DFromArray(...))` raised every step. Unpack `(e,)=`.
   This was the real crasher; the resource_tracker KeyError / "Producer terminated
   before shared CUDA tensors released" were DOWNSTREAM cleanup noise, NOT a
   torch-IPC-over-Ray incompatibility -> the raw-cudaIpc redesign was unnecessary.
3. IPC churn: build the reduce_tensor payload ONCE (stable VRAM); server caches
   the rebuilt tensor per handle (rebuild once, re-read each step).
Runners need --num-gpus-per-env-runner > 0 (a small CUDA context for the interop);
cuda-python's cudaIpcMemHandle_t serializes via getPtr()+ctypes (no .reserved).

Note: 199 sps @ 8x1 is low-density + right-pane-only, NOT comparable to the
both-panes ~176 peak (64 envs).

### Matched-density result (32x2 = 64 envs, all H=32/32, steady-state)
  both-panes numpy (Phase-1 baseline) ... ~176 sps
  right-pane  numpy control ............. ~263 sps   (+49% from dropping the left pane)
  right-pane  IPC (on-GPU feed) ......... ~333 sps   (+27% over numpy readback; +89% vs baseline)
Threaded multi-context CUDA-IPC scales (32 procs, each own GL ctx + IPC handle).
Attribution: the on-GPU-IPC feed itself is a clean, accuracy-neutral +27% at this
density (bigger than the Phase-1 "small gain" call, because with the lighter
right-pane-only step the readback/ship latency is a larger share). Right-pane-only
is a separate +49% throughput win but an ACCURACY tradeoff (3D-pane-only lost
~19pp on Chrome historically). Left (2D EM) pane WAS CPU-composed; Phase 3 below
moves it onto the GPU too (render_em), so both-panes now runs fully on-GPU.

## Phase 3 — left pane on GPU (both-panes CUDA-IPC) + overnight plan (2026-09-19)

Left pane IS GPU-implementable: compose_left_parts is per-pixel (EM->RGB, 0.5 tint
for visible ids, SHOW_ALL when empty, + crosshair). Implemented as MeshRenderer.
render_em: EM->R8 tex, ids->compact-index tex (per fetch, np.unique), per-step
K-entry color/vis LUT, fragment shader blend + in-shader crosshair. Fixed a
vertical-flip bug offline (EM quad needs NO [::-1], unlike the projection-oriented
3D scene). scripts/em_gl_probe.py validates render_em vs CPU compose_left_parts
(numeric diff + GPU/CPU/diff PNGs to em_out/).

### DONE (2026-09-19, autonomous)
1. **Parity gate PASSED** — em_gl_probe (job 930156): subset mean 0.067, showall
   0.206, noids 0.071 (worst 0.21/255; only ~0.1-0.3% crosshair-AA px differ).
   PNGs copied to Seung Lab/out/em_*_{gpu,cpu,diff8x}.png.
2. **Both-panes CUDA-IPC implemented.** Refactored the proven single-pane path
   into slot-based helpers `_ipc_register`/`_ipc_copy(gl_tex, slot)` shared by both
   panes (render3d.py). render_em(to_cuda=True) ships the _em_color payload
   (unflipped); render(to_cuda=True) ships the _color payload (flipped).
   SimulatorRenderer returns a [ (left,False), (right,True) ] list of
   (payload, gl_flip) tuples; the left_pane=False guard is gone. Server
   encode_ipc takes the pane list, rebuilds each in VRAM with per-pane flip, and
   encode_gpu returns (n_panes, D) -> 768 feats. Config:
   configs/native_dinoserver_ipc_bp.yaml (both panes, use_left_pane: true).
3. **SPS test (job 930359, sarekl15-4, 32x2 BATCH=8000):** RUNNING clean, all 32
   envs healthy (H=32/32). Early iters: iter2 226.8, iter3 228.8 sps — ~+29% over
   the ~176 both-panes numpy baseline (consistent with the +27% right-pane on-GPU
   gain). Full 20-iter mean pending. NO CUDA/interop errors — both-panes on-GPU
   feed works end-to-end.
4. **MPS bench queued (job 930443)** — mps worktree, native.yaml 32x2 under a
   per-job nvidia-cuda-mps daemon vs the ~176 non-MPS baseline.

### RESULT MATRIX (both-panes, 32x2, single 3090, good node, 20 iters)
All measured fresh with bench_single_node_sps.slurm so they are matched (the
old ~176 was NOT trusted blind — control A below reproduces it, validating it).

  A. per-process DINO, numpy, NO MPS ... ~173 sps  (job 930608 COMPLETED; = the ~176 baseline)
  B. per-process DINO, numpy, MPS ...... ~256 sps  (job 930443 COMPLETED) = +48% vs A  <-- max SPS
  C. DINO server, numpy ................ ~162 sps  (job 930609 COMPLETED) = -6% vs A (RPC cost)
  D. DINO server, CUDA-IPC (Phase 3) ... ~227 sps  (job 930359 COMPLETED) = +31% vs A, +40% vs C
  E. DINO server, CUDA-IPC, + MPS ...... ~243 sps  (job 938454 COMPLETED) = +41% vs A, +7% vs D
  (all: 32x2=64 envs, H=32/32 every iter, 20 iters, good node, steady-state)

E = the combined-lever cell (both panes on-GPU via CUDA-IPC, run UNDER MPS).
MPS lifts the server/IPC path +7% (D 227 -> E 243) -- the 32 GL->CUDA interop
copies now overlap instead of time-slicing -- but E stays BELOW B (256): the DINO
forwards in E all funnel through the ONE server context (batched, serial), which
MPS cannot parallelize, whereas B's 32 per-process forwards are exactly what MPS
runs concurrently. So the two levers do NOT fully stack.
Recommendation: max SPS + VRAM is free -> B (per-process + MPS, zero code). VRAM
constrained (many runners / bigger encoder) -> E (server + CUDA-IPC + MPS): ~95%
of B's throughput while keeping the single-DINO ~4GB footprint; turn MPS on there
too, it recovers most of the gap to B for free.

Two independent throughput levers, ranked:
- MPS (B): +50% vs A, ZERO code — the biggest single-GPU lever. Just a per-job
  nvidia-cuda-mps daemon; the 32 per-process CUDA contexts' kernels run
  concurrently instead of driver-time-sliced.
- CUDA-IPC in the server (D): the server ALONE (C) COSTS ~7% (one Ray RPC/step),
  but keeping the panes in VRAM (D) more than pays it back: +42% over C, +31%
  over baseline. So the server is worth it for SPS *only* with CUDA-IPC; its
  standalone value is the VRAM cut (15GB->4GB).

KEY FINDING: **MPS (a runtime daemon, ZERO code) is the bigger throughput lever
than the CUDA-IPC engineering** — +48% (B) vs +29% (D), and MPS BEATS CUDA-IPC
outright (~260 vs ~227). Mechanism: MPS lets the 32 per-process CUDA contexts'
kernels run concurrently on the GPU instead of serializing through the driver's
time-slicer; the DINO server instead funnels every encode through ONE actor
(batched, single stream) + a Ray RPC per step, which caps it below MPS.
Implication for the "combine server+MPS?" question: the server's value is VRAM
reduction (15GB->4GB), not SPS; for SPS alone, MPS on the per-process path wins.
Combining (MPS + server) would test whether MPS's concurrency also lifts the
server path — worth a follow-up, but MPS-only is the throughput recommendation.
Both-panes CUDA-IPC (D) is still a clean +29% and remains the right choice when
per-process DINO VRAM is the binding constraint (many runners / bigger models).

## MPS experiment (separate worktree `mps`, no code)

Run the existing per-process sim under an `nvidia-cuda-mps-control` daemon;
measure whether shared context frees enough VRAM to fit more runners and whether
kernel overlap raises SPS. Report independently; combine with the server only
after both are measured (per user).
