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

## Results — batched render + DINO SERVER (removes per-runner DINO): IN PROGRESS

To lift the per-process-DINO ceiling: pair batched render with the SHARED DINO
server (M instances) so runners hold NO DINO/CUDA-DINO context. Composition
built (per-cell CUDA-IPC handles for the interop arm; numpy cells for readback).
Sweep: RUNNERS (push past 24) x batch(envs/runner) x M(server instances) x
{readback, interop}. Smoke: server+interop 32x8 M=1 (32 crashed in-process).
Will populate the runners-vs-SPS curve under the server + the batch/M/interop
tradeoffs.

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
