# Dedicated GL / CUDA GPU disaggregation (next experiment round)

Motivation: on one GPU the fastest-stable config is GPU-compute-SATURATED (SM
~100%) by GL rasterization + DINO forwards COMBINED, with CPU ~80% (headroom) and
SPS ~385 (barrier-adjacent). Two open levers this round targets:
1. **Efficiency:** SM 100% is availability, not occupancy. 24 small per-process
   DINO forwards may under-fill the SMs; ONE big batched forward on a DEDICATED
   DINO GPU could do more work per SM-cycle. Dedicating GPUs by role (render vs
   encode) may raise per-GPU efficiency vs mixed GPUs.
2. **CPU packing (per-node GPU count):** render GPUs are CPU-hungry (fetch/decode);
   a DINO-only GPU needs ~no CPU. So a node can host MORE total GPUs within its
   ~48-core budget by MIXING roles (few render GPUs + several DINO GPUs) than by
   uniform mixed-role GPUs (which each need ~30 cores). This attacks the CPU wall
   the multi-GPU-per-node test measures.

## Architecture

Split the pipeline across physical GPUs by ROLE:
- **Render GPU(s)** — env-runner processes render the 3D pane via EGL (no DINO,
  no learner). CPU-heavy (fetch/decode).
- **DINO GPU(s)** — the shared DINO server(s), one big batched forward. CPU-light.
- **Learner** — its own (fractional) GPU or shares a DINO GPU.

Most of this is a PLACEMENT problem on the existing DINO-server + batched-render
code, not new rendering code:
- Server actor pinned to the DINO GPU(s) via Ray num_gpus / CUDA_VISIBLE_DEVICES.
- Runners pinned to render on the render GPU(s): EGL device selection is
  independent of CUDA_VISIBLE_DEVICES (confirmed), so runners need an explicit
  EGL device_index (or CVD scoping) to land on the render GPU, not the DINO GPU.
  This is the one bit of new plumbing to add.

## Transfer, staged (per user)
- **Stage 1 — readback (easy, do first):** runners readback the atlas to numpy,
  ship over Ray to the server (server on a DIFFERENT GPU). Ray moves the numpy via
  host RAM; server uploads to its GPU. NO cross-GPU CUDA needed — works with the
  existing readback+server path once placement pins roles to GPUs. Cost: the numpy
  Ray-ship (the ~240-sps cap seen single-GPU) — but here it buys role dedication.
- **Stage 2 — cross-GPU interop (NVLink / GPUDirect P2P):** keep the pane in VRAM
  and move it GPU_render VRAM -> GPU_dino VRAM directly (cudaMemcpyPeer / P2P
  handle), skipping host RAM. 3090 P2P works over PCIe; NVLink only if bridged
  (likely not on these nodes -> PCIe P2P). Removes the readback+Ray-ship cost.

## Sizing (gated on the profiling run rb_profile / gl_dino_profile)
The render:DINO GPU RATIO comes from the GL-vs-DINO GPU-time split:
- If DINO dominates (e.g., 60% DINO / 40% GL): ~2 render : 3 DINO, or big-batch
  DINO on fewer-but-saturated GPUs.
- If GL dominates: more render GPUs, consolidate DINO.
DO NOT size the layout until that split is measured (rb_profile job).

## Experiment (once sized)
Compare, on a multi-GPU node, at equal total envs:
- **Replicate** (baseline): N independent mixed single-GPU jobs (= rb_multigpu).
- **Disaggregate**: 1 job, roles split across the N GPUs by the profiled ratio.
Metric: aggregate SPS and sps/GPU, plus how many GPUs/node fit within CPU/RAM
(disaggregation should fit MORE because DINO GPUs are CPU-light). Stage-1 readback
first; Stage-2 P2P if the host-RAM ship is shown to bottleneck.

## Status
Worktree paired (neurogym-disaggregation <-> neurogym-agent-disaggregation, branch
disaggregation off throughput-scaling). Architecture only — implementation waits
on (a) the GL/DINO split from rb_profile and (b) the CPU/RAM saturation curve from
rb_multigpu, so the render:DINO ratio and GPU-count target are data-driven.
