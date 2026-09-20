# Render-batching throughput experiment (throughput-scaling)

Question: does batching the per-env 3D renders through ONE shared GL context per
runner (instead of one GL context per env) raise single-GPU SPS? The prior
finding was that GPU **context contention** is the per-step ceiling (MPS on the
CUDA side gave +48%), and MPS does NOT cover GL — so N per-env GL contexts still
time-slice. This tests collapsing them.

## Design

- `ngllib/simulator/render_service.py` — `RenderService`, a per-process singleton
  owning ONE `MeshRenderer` (one GL context, one shared mesh slot pool) on ONE
  dedicated thread. Env threads marshal render/load_mesh/pick via a queue+future
  and block; the service thread **coalesces** the per-step render calls into a
  single ATLAS render (grid of cells) + a single readback (or a single GL->CUDA
  interop copy). So `#GL contexts = #DINO = #runner processes`; batch size =
  `num_envs_per_env_runner`.
- DINO stays **in-process** (per-process singleton) — the controlled variable, so
  the experiment isolates render batching. (A server would add IPC/RPC that
  confounds it; its VRAM-dedup value is orthogonal, measured separately.)
- Transfer A/B: `readback` (one glReadPixels of the atlas) vs `interop`
  (one GL->CUDA copy of the atlas, split into per-cell CUDA views, fed to
  `encode_gpu`). Interop's device-wide sync is amortized over the whole batch.
- Configs `native_rb_{base,readback,interop}.yaml` (right-pane-only, in-process
  DINO; only the render path differs). `env.render_batch` + `NGL_RENDER_BATCH_SIZE`.

## Gates (both PASS)

- **Parity** (`render_batch_probe`, job 942326): batched-atlas cells are
  pixel-identical to per-env `MeshRenderer.render` — readback worst max diff = 1,
  interop worst max diff = 1 (rounding on one rotated scene).
- **Integration**: the batched service runs healthy under real PPO + concurrent
  env threads + MPS, both transfer modes, up to 256 envs (H all-healthy).

## Results — envs/runner axis (16 runners, 1x 3090, right-pane, MPS, real PPO)

Mean SPS (iters 2+, batched INTEROP vs per-env-context BASELINE):

| batch (envs/runner) | total envs | baseline | interop | Δ |
|---|---|---|---|---|
| 2  | 32  | ~315 | ~310 | -1.6% (wash) |
| 4  | 64  | 349  | 355  | +1.6% (wash) |
| 8  | 128 | 353  | 373  | +5.7% |
| 12 | 192 | ~350 -> CRATERS (H=15/16, ~255) | **378** | batched wins; baseline unstable |
| 16 | 256 | ~350 -> CRATERS (H=15/16, ~255) | 360  | batched wins; baseline unstable |

readback tracked interop but a bit lower (batch 8: 362 vs 373) — interop >=
readback across the batched regime.

### Findings
1. **Batching's advantage grows monotonically with batch size** (-1.6% -> +5.7%
   -> decisive). At low density it is a wash: per-env GL contexts aren't contended
   enough to beat batching's coalescing/marshaling overhead.
2. **At high density the win is STABILITY, not just throughput.** At >=192 per-env
   GL contexts the BASELINE collapses — a runner dies (H=15/16) and iters crater
   to ~255 sps — because 192+ GL contexts + per-env mesh pools exhaust the GPU.
   The batched arm (16 shared contexts + one shared mesh pool) stays steady at
   ~360-378. Collapsing N contexts into one removes the pressure that breaks the
   per-env design at scale.
3. **Interop >= readback** in the batched regime: the atlas amortizes the
   device-wide `cudaDeviceSynchronize` over the whole batch, reversing the
   in-process per-env loss seen earlier (dino-server config F).
4. Peak healthy single-GPU: **~378 sps at 16x12 batched-interop** — and it stays
   healthy where the per-env baseline cannot.

## Results — runner axis ("how many runners/GPU"), in-process DINO

Batched interop, envs/runner=8, 1x 3090, MPS, VAO_LRU=128 / CHUNK_LRU=192:

| runners | envs | result |
|---|---|---|
| 16 | 128 | 373, healthy |
| **24** | 192 | **~384, H=24/24 — PEAK** |
| 32 | 256 | DEGRADES: H=25/32 (7 runners die), sps craters to ~150 |
| 40 | 320 | unstable (mass actor restarts) |
| 48 | 384 | CRASH: `CUDA error: device busy/unavailable` at init |

**Ceiling ~24 runners; the wall is per-process DINO context count.** Batching
already collapsed the RENDER contexts (1 GL ctx/runner), so what binds is each
runner carrying its OWN DINO (weights + CUDA context) + a GL interop context —
past ~24 the GPU can't init that many contexts (H drops at 32, CUDA-device-busy
at 48). Note a naive r40 also failed on GPU-token oversubscription
(40x0.02 + learner > 1.0); dropping to 0.015 fixed the tokens but 40/48 still hit
the real context wall.

Peak healthy single-GPU (in-process DINO) = **~384 sps at 24x8 = 192 envs.**

## Results — batched render + DINO SERVER (removes per-runner DINO)

Pairing batched render with the SHARED DINO server (M instances): runners hold NO
DINO / DINO-CUDA context. Readback runners can even run num_gpus=0 (EGL render
only); interop runners keep a tiny CUDA context for the GL->CUDA handoff.

Server + batched, 32 runners, 1x 3090, MPS, real PPO (mean sps):

| arm      | batch8 M1 | batch8 M2 | batch12 M1 | batch12 M2 |
|----------|-----------|-----------|------------|------------|
| interop  | 355       | 370       | **~385** (pk 397) | ~380 |
| readback | 239       | ~237      | —          | ~220 |

Runner-ceiling probes (readback, num_gpus=0 runners):

| runners | envs | health | sps |
|---|---|---|---|
| 32 | 256 | H=32/32 | ~239 |
| 48 | 384 | **H=48/48 healthy** | ~186 |
| 64 | 512 | **H=64/64 healthy** | ~160 |

### Findings (server + batched)
1. **The server ~doubles+ the runner ceiling.** In-process DINO capped at ~24
   runners (CUDA-context exhaustion). With the shared server + num_gpus=0 runners
   (no per-runner CUDA context), **64 runners run H=64/64 healthy** (48 too) — the
   crash point is above 64, not found. This is the answer to "how many runners/GPU".
2. **SPS is PPO-barrier-capped ~370-384; the server does NOT raise it.** Stable
   server-interop peak is 32x8 ~370, which is BELOW the in-process peak (24x8 ~384)
   because the server adds a Ray RPC per step. Adding runners does NOT raise total
   sps; for readback it LOWERS it (32->239, 48->186, 64->160) as the per-iter batch
   splits thinner + Ray-ship grows. The server's value is CAPACITY / VRAM, not sps.
3. **batch12 at 32 runners is UNSTABLE (OOM collapse).** Both e12 jobs (M1 and M2)
   ran ~13 healthy iters at ~385 then ALL 32 workers died (SYSTEM_ERROR exit 1,
   H=0/32) — a memory cascade (384 envs: bigger atlas + per-cell dst + fetch caches
   over the edge). The earlier "~385" was pre-collapse. e8 is rock-stable (20/20
   H=32/32). **So batch8 is the stable sweet spot; batch12 needs lower runners or
   trimmed caches to be usable.**
4. **With a server, INTEROP is essential: +50-65% over readback** (e8: ~365 vs
   ~239). Readback ships full numpy panes over Ray's object store to the server
   (expensive); interop ships ~64-byte CUDA-IPC handles. This is the opposite of
   the IN-PROCESS batched result (where readback ~= interop) — because there the
   pixels never crossed a process. So interop only earns its keep across a
   process boundary, which is exactly the server case.
5. **M=1 ~= M=2:** a second DINO server never helps SPS — not the bottleneck. But
   M=1 is a single point of failure (if the one server OOMs, ALL runners lose their
   encoder -> H=0/32), so M>=2 is worth it for robustness at scale.

## Utilization — where the fastest-stable config actually spends (job 952786)

Instrumented 24x8 in-process interop (~390 sps, H=24/24) with nvidia-smi dmon +
mpstat during training:
- **GPU SM ~98-100% through the rollout** (mem-controller ~25-40%) => GPU
  COMPUTE-bound (not memory-bandwidth). GL rasterization + DINO forwards fill the
  SMs; MPS is what lets them pack to 100%.
- **CPU ~76-80% busy (~20% idle)** => NOT CPU-bound; headroom to spare.
- Brief dips (GPU->~15%, CPU idle->~80%) at each iteration boundary = the
  synchronous-PPO barrier + learner update + weight sync (~9%, per the timers).

Bottleneck (ignoring the barrier): **GPU compute is saturated.** Hence more
runners/envs don't help (GPU already full), and the encoder-path swaps barely move
peak SPS (rearranging work on a full GPU). Levers to go faster: cheaper per-step
GPU work (smaller/quantized DINO, lower render res, fewer overlay draws) or async
PPO (reclaims the ~9% inter-iter idle + removes straggler gating). CPU/runner count
are not the constraint.

## OVERALL CONCLUSION (all axes)

- **Single-GPU SPS ceiling ~370-384 (stable) is set by synchronous PPO**, not by
  the encoder path, GL contexts, or runner/DINO count. Every stable config tops out
  there. To go higher needs async/off-policy PPO, not more packing.
- **FASTEST STABLE single-GPU config:** in-process DINO + batched render + interop
  + ~24 runners x 8 envs + MPS = **~384 sps** (job 946146, 20/20 H=24/24). The DINO
  server does NOT beat this (its per-step Ray RPC costs ~15 sps: 32x8 server-interop
  = ~370). So the server is for CAPACITY, not speed.
- **To pack the most runners on one GPU:** shared DINO server + num_gpus=0 readback
  runners -> 64+ healthy (crash point >64, not found), but SPS DROPS with runners
  (barrier + Ray-ship): 32->239, 48->186, 64->160. Use this only when you need the
  env count / freed VRAM (bigger model/obs), not for throughput.
- **Levers:** render batching = throughput+stability at high envs/runner (neutral
  low); DINO server = VRAM/capacity win, NOT an SPS win; interop = matters only
  across a process boundary (server), +50-65% vs readback; batch8 stable, batch12
  OOM-collapses at 32 runners.

## Takeaway

Render-batching (in-process DINO, interop transfer) is a **"turn on when packing
envs densely" lever**: neutral at low envs/runner, but a throughput AND stability
win as envs/runner grows — it lets one runner host many envs without the per-env
GL-context explosion that collapses the default design at scale. Not a universal
speedup; the per-env-context default is fine at low density.

## Reproduce

- Parity: `sbatch scripts/rb_parity.slurm`
- Sweep: `bash scripts/rb_sweep.sh` (GRID/ARMS env vars), or `scripts/rb_bench.slurm`
  with CONFIG/RUNNERS/ENVS. All under a per-job CUDA MPS daemon.
