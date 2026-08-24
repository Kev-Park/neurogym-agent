# REFINEMENT — infra hardening to "fully established"

> Working tracker for the remaining gaps between "production-ready supervised"
> and "unattended at full scale". Work through top to bottom; check items off
> as they land. Conventions: **[DECIDED]** locked, **[decide]** needs a call.
> Context: `../neurogym/agent_plan.md` (design), dist-rl-v1/v2 findings
> (2026-07-03 / 07-06).

---

## R1. Completion semantics — iteration-based **[DECIDED]**

Coordinator can't distinguish "workload finished" from "learner died" (v2:
respawn-trained past target until the salloc wall).

Design: iteration count as the completion measure (RL-native).
- `train.py`: write `meta.json` **every iteration** (currently only on
  checkpoint cadence) — `{iteration, wandb_id, run_name}`.
- `coordinator.py`: new `--target-iterations N` (0 = disabled) +
  `--progress-file <path to meta.json>`. Monitor loop reads it each cycle;
  on `iteration >= N` → log, teardown (scancel), exit 0.
- Keep `--max-cycles` as the outer safety bound.

Tests:
- [ ] unit: coordinator teardown fires when a stub meta.json crosses target;
      not before; tolerates missing/partial file (atomic write already used).
- [ ] cluster: dummy-worker run with fake meta progression → clean self-stop.
- [ ] real: next training run launched with `--target-iterations`.

## R2. Zombie-browser watchdog **[DECIDED]**

Hang → nothing raises → permanent capacity loss. Port legacy PID-kill.

Design (env-internal, in ngllib `Environment`):
- Capture Chrome PID at launch (psutil child-walk, legacy-verbatim).
- `threading.Timer` around step/reset browser calls; on fire → SIGKILL PID →
  blocked Playwright call raises → funnel into the existing sick-browser
  escalation (`_needs_browser_restart` → full restart next reset).
- Timeouts **[DECIDED]**: step **30s**, reset **240s** (cold init is slow);
  constructor kwargs. Adds `psutil` dep to ngllib.

Tests:
- [x] **Implemented 2026-07-08** (ngllib `ed292bf`): `_BrowserWatchdog` timer +
      psutil PID capture at launch; step fires → BrowserError → sick-browser
      escalation; navigate fires → retryable error → browser-restart retry
      path. step 30s / reset 240s kwargs. 58/58 tests incl. timer semantics.
      **PROVISIONAL** — code + unit level only.
- [x] **GPU-CONFIRMED 2026-07-08** (`probe_watchdog.py`): SIGSTOP'd a live
      Chrome, watchdog fired at the 8s test-timeout ("watchdog killed hung
      Chrome pid ..."), step raised BrowserError → resilient truncation →
      recovery reset relaunched a fresh browser + stepped. **R2 DONE.**

> **GPU-first principle (2026-07-08):** CPU-only / dummy-worker validations are
> **provisional** — the gold-standard test is always the full env (browsers,
> DINO, GPU) at scale. Each provisional item below carries a GPU-confirmation
> step that runs when the pool frees.

## R3. Real-preemption chaos test

v2 validated re-salloc on *expiry* (COMPLETING). Real preemption unobserved;
`salloc --no-shell` may be cancelled outright rather than requeued
(REQUEUE only applies to batch) — both coordinator branches exist, neither
ground-truthed.

- [x] Step 0: privilege check — account CAN submit to `highpri` (2026-07-06).
- [x] **DONE 2026-07-08 — with a major finding.** Squeeze test (coordinator
      salloc on sarekl15-2/4, then highpri job pinned + sized to require
      eviction): **SLURM never preempted the salloc** — the highpri job queued
      with a ~6-day ETA instead (5-min observation window; PreemptExemptTime
      NONE). Meanwhile the r4-soak — a BATCH job — WAS preempted+REQUEUEd by
      the same highpri campaign a day earlier. Conclusion (observational):
      on this config, REQUEUE-mode preemption applies to batch jobs only;
      interactive/salloc allocations are skipped, not cancelled.
      **Design implication:** the coordinator's allocation is far more
      contention-stable than designed for — its dominant loss mode is TIME
      expiry, which v2 already validated end-to-end (re-salloc + resume).
      Provisional caveat: config-dependent; re-verify if SLURM is upgraded.
- [ ] GPU confirmation (when pool frees): repeat squeeze during a small REAL
      training run — confirm the salloc holds and only batch-side artifacts
      (if any) are affected.
- [ ] Fold findings into `scripts/cluster_probe_findings.md` + CLAUDE.md.

## R4. Per-node density parity

Throughput parity reached at M=8/node; legacy density (32/node) untested.
Process-per-env costs ~0.3–0.5GB VRAM CUDA context + DINO copy per env →
caps M well below 32 on a 24GB 3090.

- [x] M-sweep benchmark: threads knee ≈ M=16 @ 24c (31 sps, legacy parity);
      M=24 @ 24c regressed — CPU-starvation vs GIL disambiguation below.
- [ ] **Degradation-curve sweep (job 837858)**: threads M = 20/24/28/32/**36**
      at 32 cpus / 200G — where sps peaks/degrades; M=32 is the VRAM-safe
      ceiling (~0.6GB Vulkan per Chrome ⇒ ~21GB; cgroup isolation pins all
      Chromes to the one allocated GPU). **M=36 is a deliberate over-the-edge
      probe**: does VRAM exhaustion fail loudly (Vulkan errors → sick-browser
      escalation) or silently (SwiftShader fallback)? Phase grep watches for
      SwiftShader markers. Beyond-one-GPU note: nodes have 8x3090; 2 runners
      in separate srun steps (own device cgroups) could steer 2xM Chromes onto
      2 GPUs — the ceiling-raiser if the curve is still rising at 32.
- [x] **ThreadedVectorEnv: BUILT + benchmarked** (Phase 1; `6d7914c`). Curve
      (jobs 837608/837858): threads M=8=18 sps (thread tax at low M, loses to
      spawn's 26), **M=16≈31 (legacy parity)**, M=20 peak 37 (=legacy peak
      band), then **flat 24-30 through M=36** — renderer-bound plateau, NOT
      thread/CPU/GIL/VRAM. No SwiftShader fallback even at M=36 (VRAM edge
      higher than feared). M=32 had a transient env-runner crash (nan), not a
      systematic ceiling. **[DECIDED] production M=16, threads topology.**
      Phase-2 vector-level DINO batching NOT needed (renderer-bound, not
      encoder-bound). Remaining: flip `--vector` default to threads after the
      clean production soak + threaded-coordinator learner-kill confirm a long
      run (the bench/curve were only 3-iter phases).

## R5. Failover with real workload

v2 accidentally validated learner-death → full respawn → head migration →
resume. Make it deliberate + cover the promotion branch.

- [x] **DONE 2026-07-08 — real workload, threaded** (salloc 838109): trained to
      iter 4 (ckpt written), scancel'd the learner step (SIGKILL exit 137) →
      coordinator detected next cycle, correctly classed as death (not denial,
      via workload_ran()), respawned → new learner `resuming from ckpt_000004`
      → iter advanced 4→5→6 with REAL samples (1024 steps) → renderer
      RECONNECTED to the migrated ray head (ray-joins=2). Head migration +
      resume + renderer rejoin all validated. Also validated `--vector threads`
      under the coordinator.
- [x] **DONE 2026-07-08** — `--force-promotion-once` exercised the promotion
      branch (sacrifice → learner into slot → re-heal); exposed + fixed the
      crash-vs-denial false positive (see R5 finding above).
- [x] wandb continuity: v2 already showed run-id continuity across respawn
      (meta.json wandb_id + resume="allow"); this run used offline mode.

## R4-ext. Long-run soak (isolated) + multi-GPU/node

- [x] **CORRECTION 2026-07-09: sustained per-GPU throughput is ~23 sps, NOT 31.**
      The 31 sps M=16 "parity" was a 3-iter artifact (measured the clean period
      before glitches accumulated). Root cause: viewer-state "missing fields"
      glitches scale with GPU contention (0.36/iter @M=8 → ~10/iter @M=16 = 28x),
      and step() escalated to a FULL browser restart on EVERY glitch → restart-
      storm → sustained sps ~20. **Fixed** (ngllib `37e15bc`): escalate only
      after N consecutive failures (single miss = truncate); poll up to 2s for
      state to settle; nav_timeout 90s. Post-fix M=16 12-iter: mean ~23 sps,
      FLAT (no drift). So the isolated-soak "-23% drift" was co-tenancy; the
      real story is a lower-but-flat plateau.
      **Reframe:** per-GPU throughput is render-bound at ~23 sps for BOTH
      topologies (threads M=16 ≈ spawn M=8 ≈ 23/GPU); threads' win is VRAM (one
      CUDA ctx + one DINO), not throughput. We're ~25% under legacy's ~30/GPU —
      likely render-config (resolution / NGL state / DINO co-tenancy / state-
      poll overhead); worth a look, not urgent.
- [x] **Multi-GPU/node: PASS (job 840289).** Vulkan rendering DOES follow Ray's
      per-runner CUDA_VISIBLE_DEVICES — 2 runners x 1 GPU x M=16: GPU#0 util 70%,
      GPU#1 util 62% (both busy), aggregate **~35 sps/node, 0 restarts**
      (~1.5x single-GPU; sub-2x = shared CPU/learner). Multi-GPU/node works via
      native RLlib config and **beats legacy's ~30/node**.
- [x] **SPS-gap investigation (probes 0c9a7d8/7f242c8): not an efficiency gap.**
      M=1 step breakdown: screenshot=101ms (84%), DINO=14ms (11%), logic=7ms.
      The screenshot has a **fixed ~67ms floor** (CDP capture + GPU-readback
      sync) that is resolution- AND jpeg-quality-INDEPENDENT (1800x900 q85 = 67ms,
      900x450 q30 = 67ms; only PNG scales with res). So downscale/quality won't
      help. Conclusion: **per-env we're ~1.5x FASTER than legacy** (23 sps/16
      envs = 1.44/env vs legacy 30/32 = 0.94/env); the lower aggregate was
      simply fewer envs/GPU (16 vs legacy's 32). Levers to match/beat legacy
      aggregate, ranked: (1) more envs/GPU M=24-32 — legacy's approach, now
      viable post restart-storm-fix; (2) multi-GPU/node — validated above;
      (3) CDP screencast to cut the 67ms floor — real ngllib work, deferred.

## R4-frontier. (GPU count x threads/GPU) scaling — RESULTS 2026-07-10

Single-GPU post-fix (job 840369): M16=28, M24=30, **M32=34 sps** (monotonic;
beats legacy ~30). SPS-gap CLOSED — it was the restart-storm on M=16, not a
real deficit.

Frontier (job 840371, 4-GPU node, R runners x M browsers):
  R1M16=33  R1M24=24  R1M32=28  R2M24=22  **R2M32=34**  R4M16=17  R4M24=17
Clean pairing: R2M32 (2 GPU, 64 browsers) = 34 ≈ R1M32 (1 GPU, 32 browsers) = 34.
**Multi-GPU-per-node does NOT scale throughput** — rendering splits (both GPUs
~65% util) but aggregate hits a **per-node ceiling ~34 sps**; R4 COLLAPSES
(17 sps, wall 5-10x) into a CPU/GIL/glitch-storm at 64-96 browsers/node.
Numbers noisy — the stochastic glitch storm (state-race under load, 2..126 per
run) dominates and is the true ceiling.

**[DECIDED] scaling model: ~34 sps/node (1 GPU + M=32) x N nodes** (multi-node
linear, v2-validated). Do NOT pack GPUs onto fewer nodes. Production per-runner
M=32, one runner/node.

**Root-cause confirmed 2026-07-10 (pure-env throughput probes, no ray/ppo):**
- Q1 M-ceiling: M32=36.5, M40=37.4, M48=35.4 sps — flat ~37 plateau, per-env
  drops 1.14→0.74. M=32 optimal; M>32 gives nothing.
- Q2 why-no-multi-GPU: concurrent 2-GPU processes each slow 37→28 sps while
  **CPU 53-76% IDLE and GPU util only 22-47%**. Neither compute resource is the
  wall — the bottleneck is the shared per-node **screenshot readback** (GPU→CPU
  framebuffer copy over CDP/PCIe/mem-bw = the 67ms floor). Adding GPUs adds
  render compute (not the limit) but not readback bandwidth. Explains the
  per-node ceiling, M-plateau, AND multi-GPU non-scaling with one cause.

**Compositor-flag lever — VALIDATED 2026-07-10 (job 844494 / flags2).** The
non-hacky readback speedup that *does* work safely: two Chrome flags in ngllib's
headless launch args (`449b3d2`), `--disable-gpu-vsync` + `--disable-frame-rate-limit`.
Removes the vsync/frame-throttle wait between action-apply and frame-commit.
- Single-env step: **133ms → 88ms median** (~33% faster), p90 96ms. **Zero
  hangs, zero 30s watchdog timeouts** across the full run — the safe 2-flag set
  is stable. (The 3rd flag `--run-all-compositor-stages-before-draw` was dropped:
  it intermittently deadlocks page.screenshot → 30s timeout. Do NOT re-add.)
- **BUT aggregate is unchanged: M16=31.5, M32=33.1 sps** — same ~33 sps node
  ceiling as pre-flag baseline. per_env collapses 1.97 (M16) → 1.04 (M32).
- **Conclusion:** the flags cut *per-capture latency* but NOT the *shared-node
  readback bandwidth*, which is what saturates at M=32. Confirms the root cause
  above: at training M the many concurrent browsers already contend for the one
  GPU→CPU readback path; making each faster doesn't widen the pipe. Flags are
  **kept** (free, stable, help low-contention / warmup / low-M) but are **not**
  the path to >34 sps/node. Raw-CDP capture (50 vs 67ms) is the same story —
  single-env win, no aggregate lift — so not worth the page.screenshot swap yet.

**Process-packing vs GPU-scaling — pure-env sweeps 2026-07-11 (jobs 844573 /
844878, `probe_procscale.py` + `run_procscale2.sh`).** Two questions, cleanly
separated. Both ran on sarekl15-2 *while the highpri array co-tenanted the node*
(40 cpu + 5 GPUs busy), so absolute sps are depressed ~25% vs the uncontended
~36 baseline — trust the SHAPES, not the absolute numbers.
- **(B) Process-packing on ONE GPU, fixed 32 browsers, 24 cpu:** 1×M32 = 27.7,
  2×M16 = **39.4 (+42%)**, 4×M8 = 40.0 (plateau). **Multi-PROCESS beats
  single-process threading by ~44% on one GPU**, saturating at 2 processes. Cause
  = the single Python process serializes JPEG-decode + DINO under the GIL (one
  CUDA context); a 2nd process recovers most of it. ⇒ our single-process
  `ThreadedVectorEnv` leaves ~44%/GPU on the table; **~2 EnvRunner processes per
  GPU** is a real lever (RLlib-native), no extra GPUs needed.
- **(A) GPU-scaling, cgroup-isolated GPUs (`srun --gres=gpu:1`, verified
  `visible_gpus=1`/proc), 8 cpu/proc:** 1 GPU = 20.3, 2 GPU = **27.6**, 3 GPU =
  **27.6**. **The 3rd GPU added exactly zero**; per-proc collapsed 20→14→9 so the
  aggregate stayed pinned. A hard node ceiling *divided* among GPUs, not lifted.
- **ngllib GPU-selection finding:** ngllib sets **no per-GPU Vulkan device
  select** (environment.py only passes `--use-angle=vulkan`); Chrome renders on
  GPU0 regardless of `CUDA_VISIBLE_DEVICES` (which steers only CUDA/DINO). The v1
  A-series thus piled all browsers on GPU0 and OOM'd past ~32 (A2=64, A3=96 →
  FAIL). Multi-GPU-per-node requires cgroup isolation (Ray/SLURM 1 GPU/worker).
- **Caveat / not-fully-clean:** the contended fleet forced 8 cpu/proc (vs 24 for
  the B baseline) + co-tenancy, so the 1→2 GPU bump is partly cpu and only the
  **2→3 flat** is a clean within-run signal. A definitive multi-GPU-at-M32 run
  needs a low-co-tenancy node with ≥~16 cpu/GPU free (blocked while the array
  saturates the fleet). Even so, 2→3-flat + the readback root-cause agree:
  **adding GPUs per node does NOT scale throughput.**
- **[DECIDED] reaffirmed:** scale by NODES, not GPUs/node; per node use 1 GPU
  with ~2 EnvRunner processes (the +44% packing win), M≈16/proc.

**DEFINITIVE clean re-test + capture-flood — 2026-07-11 (jobs 845147 / 845149,
`run_procscale3.sh` + `probe_flood.py`), fleet IDLE, exclusive node.** These
resolve the two open questions and OVERTURN the "readback is the ~40 wall" claim.
- **(A′) GPU-scaling, EXCLUSIVE node, 16 cpu/proc, no co-tenant, verified 1
  GPU/proc:** 1 GPU = 30.0, 2 GPU = 35.7, 3 GPU = 35.4, 4 GPU = 31.1 sps.
  **FLAT even with abundant CPU** (per-proc collapses 30→18→12→8). So the earlier
  flat was NOT starvation — there is a genuine **node-level full-step ceiling
  ~35 sps** that adding GPUs cannot lift. Q1 answered: real node limit.
- **(FLOOD) pure-capture ceiling, ONE GPU** (`captureScreenshot` in a tight loop,
  no DINO/decode/action; each browser built+flooded in its own thread — Playwright
  sync is thread-bound): M1=24, M4=83, M8=148, **M16=170**, M32=154 caps/s.
  **The raw readback ceiling is ~170/GPU — ~4× the ~35 full-step rate.**
- **⇒ CORRECTION:** capture readback is **NOT** the training bottleneck (it has
  ~4× headroom). The ~35 full-step ceiling is the **post-capture host path**:
  JPEG-decode (PIL, GIL) + DINO input-prep + memory copies. It's **node-shared**
  (adding independent processes on separate GPUs still drags each down 30→18→…),
  most likely **host memory-bandwidth / LLC** contention (NOT capture, NOT GPU
  compute — util low & per-GPU, NOT VRAM, NOT CPU cores — 16 dedicated each, idle
  in the flood). Exact node-shared sub-component not micro-profiled (mem-bw is the
  leading hypothesis; a `nsys`/membw probe would confirm).

**Revised lever priority (was: screencast is #1 — now DEMOTED):**
1. **Attack the host-side per-frame cost** — the real wall. Faster JPEG decode
   (PyTurboJPEG / GPU-side decode), **batched DINO** across the vector envs (one
   ViT call not M), fewer numpy copies. This is what can push a node past ~40.
2. **~2 EnvRunner processes/GPU** — the proven +44% (escapes the GIL on that same
   host path). Fold into the training config.
3. **Scale by NODES** — unchanged; each node independent, linear.
4. **CDP screencast — DEPRIORITIZED.** It attacks *capture*, which is NOT the wall
   (~170 headroom). Only relevant if you ever need to exceed the ~170/GPU capture
   ceiling itself — i.e. after the host-side path is already optimized. Not the
   milestone I earlier billed it as.

**Bottleneck hunt cont. — 2026-07-12.** Pinned the training topology to **2
EnvRunner procs x 16 threaded envs/GPU** (`train.py` defaults; commit a080637),
then probed the host path stage by stage.
- **JPEG-decode RULED OUT (job 845150, `probe_decode.py`).** Real frame = 237KB.
  CPU PIL decode+resize224 = 18.7ms → 54/s *single worker* (already > the ~40
  full-step rate). CPU-decode flood **scales ~linearly** across processes
  (P=1/2/4/8 → 51/90/170/326 dec/s, per-proc ~41–51) ⇒ **compute-bound per core,
  NOT a shared memory-bandwidth wall.** So the earlier host-mem-bw hypothesis is
  **refuted** — decode has 8x headroom and parallelizes. GPU nvJPEG decode =
  3.7ms (267/s, ~5x CPU) but won't help since decode isn't binding.
- **Next suspect = per-env DINO.** `dino_obs.py` calls `encoder.encode([left,
  right])` **per env (batch 2)**; a 32-env vector-step = 32 tiny forwards + 32
  host↔GPU round-trips instead of one batched forward. Probing throughput
  (per-env batch2 vs batched 32/64, + threaded ceiling) in job 845151
  (`probe_dino.py`). If threaded-per-env ≈40 and batched ≫, **batch DINO at the
  vector level** is the fix.
- **DINO RULED OUT too (job 845151).** Per-env batch2 = 87 env-steps/s, batched
  (32/64) = 117, threaded-per-env plateaus ~100. All **≫ the ~40 full-step rate**
  → DINO has 2–3x headroom; batching gives only ~+34% (87→117), not the 2.5x
  needed. **Every stage in isolation (capture ~170, decode 54–326, DINO ~100) is
  faster than the ~40 full step.** ⇒ **No single-stage villain.** The wall is the
  *serialized per-step critical path* (stages don't overlap — at M=32 per-env
  slows 90ms→~800ms) + GIL-bound orchestration in one process, not any one op.
  Consistent with: 2 procs = +44% (2 GILs), multi-GPU flat (node-level, not
  per-stage). **Decisive next diagnostic = real-loop per-stage wall-clock at
  M=16/32** (where does the 800ms/env-step actually go under contention).
- **Optimized-obs architecture — Phase 1 GO (job 845152, `probe_obs_arch.py`).**
  Reframe: GPU-decode + batched-DINO aren't about stage speed — they REMOVE the
  GIL-held per-env work into one batched GPU call, attacking the actual wall.
  Head-to-head (no browsers, same frames): current per-env (CPU decode + DINO
  batch2, M threads) vs batched (GPU nvJPEG decode + one DINO forward over 2M
  panes, 1 thread): **M16 = 82→185 obs/s (2.3x), M32 = 85→166 obs/s (2.0x).**
  ⇒ the vector-level batched-obs rewrite is worth building.
  **Phase 2 (the real refactor):** (a) ngllib: add a raw-JPEG obs mode
  (`_get_screenshot` returns bytes, defer decode) so decode can move downstream;
  (b) move DINO OUT of the per-env `DinoObservationWrapper` into a VECTOR-level
  batched obs step in `ThreadedVectorEnv` (collect M raw JPEGs -> batched GPU
  decode+DINO -> distribute features); (c) re-run `probe_throughput` end-to-end
  at 2x16 + sweep process count (opt 2). Touches shipped ngllib — needs care.
- **Phase 2 end-to-end — REFUTED (job 845153, `probe_e2e_batched.py`).** Batched
  DINO REGRESSED the real loop: 1x32 28.1→24.4, **2x16 39.9→35.1**; proto process
  sweep 1x32/2x16/4x8 = 24/35/40 (needs 4 procs to match baseline's 2-proc 40).
  **Why (the lesson):** the per-env threaded design ALREADY hides DINO behind
  browser-I/O waits — each sticky thread does capture (GIL released on CDP I/O)
  then its per-env DINO, so 32 threads overlap browser-wait with DINO-compute.
  Batching DINO as a POST-step barrier serializes it onto the critical path,
  losing that overlap; the isolated 2x compute win doesn't translate. (Also a
  prototype artifact: raw mode concatenates+re-splits M full 900x1800 images
  through the vector env — extra memory traffic baseline avoids; ~part of the
  12%.) Net: **batching a stage that was never on the critical path can't help**
  — consistent with "no single-stage villain; DINO has headroom & is hidden."
- **[DECIDED] per-node optimization CLOSED.** Tested capture/decode/DINO/batching/
  GPU-scaling/process-count — **none beats the current 2x16 (~40 sps/GPU-node).**
  The threaded per-env design is near-optimal (hides DINO behind browser I/O);
  the wall is the browser-step critical path + GIL, no single fixable stage. Only
  untried idea = a fully PIPELINED async obs (overlap batched-DINO(t) with
  browser-step(t+1)) — complex, uncertain, only if per-node >40 is truly needed.
  **Ship 2x16/node; scale by NODES.**

## R6. Housekeeping

- [x] **DONE 2026-07-08** — `uv.lock` committed (`368b7e7`; scp'd back
      checksum-verified, synced to cluster identically).
- [x] **DONE 2026-07-08** — sarekl15-1 probed = **GOOD** (nvidia ICD+EGL, HW
      angle-vulkan). Good pool now 12 nodes; CLAUDE.md + node-list memory
      updated. Minor follow-up: add sarekl15-1 to `--nodelist` allowlists in
      the r4/coord scripts (currently omit it).
- [ ] Coordinator `--force-promotion-once` and any test hooks: mark clearly
      as test-only in --help.
- [ ] Overnight full-scale run (8+ nodes, best-M from R4, 24h): the closing
      validation for all of the above at once.

---

## Status log

- 2026-07-06: file created. R1/R2 designs locked (iteration-based completion;
  watchdog 30s step / 240s reset). d0 eval of dist-rl-v2 ckpt_000360 pending
  in queue (job 837471).
- 2026-07-06: **R1 DONE** (`6d5d2b8`) — meta.json heartbeat + coordinator
  `--target-iterations`; unit-tested. R3 step-0 done: account CAN submit highpri.
- 2026-07-06: **R4 phase 1 DONE** (`6d7914c`) — ThreadedVectorEnv (sticky
  thread per env, SyncVectorEnv-equivalent semantics, 51/51 tests) behind
  `--vector threads`; default stays spawn pending hardware A/B.
  Benchmark queued (job 837477: threads-vs-spawn M=8 + threaded M=16/24 sweep
  w/ VRAM maxima). [decide]-threads-default resolves on its numbers.
- 2026-07-07: first bench attempt failed — stale `/tmp/ray/ray_current_cluster`
  markers made plain ray.init() join a dead head and hang. Fix `54c7807`:
  `RAY_ADDRESS=local` setdefault in ppo_smoke/train + full-log tee + per-GPU
  VRAM sampling. Verified on the poisoned node.
- 2026-07-08: GPU pool saturated by a 55-job highpri campaign (SLURM ETA for
  our jobs: 07-13/14). Pivoted to CPU-side validations:
  **R1 CLUSTER-VALIDATED** (dummy run: meta progressed → "target iterations
  reached (5 >= 5)" → clean scancel). **R5 promotion branch VALIDATED** via
  --force-promotion-once (sacrifice → learner into slot → pool re-healed) —
  and exposed a false-positive loop: died_quickly alone treats a fast-CRASHING
  learner as srun denial, serially sacrificing renderers. Fixed (`5243078`):
  promotion additionally requires workload_ran() is not True (srun denial
  leaves no workload output in the launch log). 54/54 tests.
  Still GPU-blocked: soak, curve sweep, real-workload learner-kill.
  Side observation: the soak was preempted+REQUEUEd once by the highpri
  campaign — first in-the-wild PreemptMode=REQUEUE datapoint for R3.
- 2026-07-07: **R4 bench DONE (job 837608)**: threads M=8 = 17.9-18.6 sps /
  5.9GB (loses to spawn at low M — thread tax); spawn M=8 = 25.7-26.3 / 9.3GB;
  **threads M=16 = 29.7-31.0 sps / 10.7GB ≈ legacy sustained parity**;
  threads M=24 = 26.9-27.5 / 16.0GB (past the knee at 24 cpus). VRAM saving
  vs spawn confirmed (~0.43GB/env). Soak (837609, M=16) + ceiling probe
  (837856, M=20/24 @ 32c) queued.

## R7. Straggler-mitigation campaign — FINAL VERDICT 2026-08-16

Goal (user): consistent SPS near the peak — pull unlucky runs up, no constant
taxes (jitter/backoff ruled out). Data-first: storm trials (851928/929) found
the real straggler causes = synchronized episode-boundary reset WAVES (83% of
resets are 300-step TimeLimit truncations, phase-locked from run start; 52-96
resets/10s), cold-start resets (40-73s vs 3s steady), actor deaths, and an
unattributed slow-step class (F4). Glitch-retry storms did NOT reproduce.

4-arm x 3-seed sweep (150 iters, jobs 856613-21 + 861935-42, `r_mitig.slurm`;
preemption-resilient resume after the fleet-holder's rolling batch requeued
arms 6x; final arms ran on highpri):

  arm      mean±sd      worst   straggler  waves(max/10s)
  base     96.5 ± 2.7   92.9    8.5%       63-85
  m1      101.1 ± 2.1   98.2    3.1%       8-13      <- WINNER, default ON
  m1m2     86.7 ± 1.7   84.3    1.0%       ~9
  m1m2m5   94.9 ± 6.6   86.5    5.3%       ~10

- **M1a stagger SHIPPED (train.py default ON)**: +5% mean, worst-seed +5.3,
  stragglers /2.7, waves /7, slow-steps /3. Zero steady-state cost. Mechanism
  verified (wave metric excludes first=True cold bursts — requeued attempts
  APPEND to event files).
- M2 small-frag (frag8/5376/timeout60): lowest stragglers (1%) but -14% mean —
  dropped rounds discard work + more learner passes. NOT shipped; keep for
  runs where tail-latency matters more than mean.
- M5 reset-ahead: smoke = 200x per-reset win (17s->81ms) but at scale the prep
  ticks (goto-commit + polls) run inside the vector join barrier -> slow-step
  events 50-101/run (vs m1's 14-36), net m1m2m5 < m1. With waves already dead,
  the remaining reset cost is too small for M5's overhead to beat. Flag stays
  (`--reset-ahead`, default off); untested combo m1+m5 (no m2) might differ.
- M4 (drop sarekl15-8, add 15-1) folded into all arms' node pool.

Ops lessons: own big pending jobs block backfill of own small jobs (hold/release
to slot probes); rolling highpri fleets starve preempt arms — preemption-
resilient resume (--open-mode=append + ckpt-resume + remaining-iters) is now in
r_mitig.slurm; account can politely queue on highpri (no preemption of running
jobs) when the preempt partition is starved.

R7 significance addendum (2026-08-16, seed-level Welch tests, n=3/arm):
straggler reduction 8.7->3.1% p~0.002 and wave collapse 75->12 p~0.01 are
SIGNIFICANT; the mean-SPS gain (+4.6, p~0.13) is NOT — treat as point estimate
("mean-neutral-at-worst"). ~7 seeds/arm would be needed to power it. Ship
rationale for M1a default rests on the consistency wins + zero cost only.

**[DECIDED] 2026-08-16 (user): M1a stagger LOCKED as the production default**
(train.py `--stagger-first-episode` default ON since `9ed5e68`). M2/M5 remain
opt-in flags; M1b and the m1+m5 combo stay unexplored unless a need arises.

## R8. Cycle-time levers (Playwright/Chrome layer) — 2026-08-16

Probed 3 untried browser-level levers (job 862345) + poll tightening:
- **capture_scale (browser-side GPU downscale, 2:1 aspect kept): WINNER.**
  Single-env step 98.6 -> 49.3ms @0.5 (2.0x); 0.25 adds nothing (latency
  floor) and is visibly softer. Aggregate confirm (862346, M=16 same node):
  26.5 -> **34.8 sps (+31%)**. Visual QA: 0.5 pristine (frames in
  out/2026-08-16_capture-*.png); DINO input unchanged (encoder resizes to 224
  anyway). **SHIPPED: dino mode defaults capture_scale=0.5**
  (env_build; env.capture_scale overrides).
- Skip per-episode HTTP-cache clear: NULL (reset ~2.4s both; RSS flat both
  869->994 vs 904->944MB — no blow-up either way). Default unchanged
  (clear=True); `clear_cache_on_recycle` param exists. Note: single-env
  uncontended — a warm cache could still matter under 96-env reset herds.
- Footprint flag bundle (8 flags incl. site-isolation off): REGRESSION
  (+21% step time), renderer stayed ANGLE/NVIDIA. Dropped, not bisected.
- Poll tightening SHIPPED: nav 1s->100ms, isReady 50->25ms, settle 100->50ms.
Remaining unexplored: same-page JS state reset (vs navigation) — biggest
reset-side lever; hybrid (swap N, recycle Nth) if P2 accumulation appears.

## R9. Coordinator end-to-end test run — coord-test-v7 COMPLETE (2026-08-18)

**Goal**: a coordinator-driven 2x16-per-node run to --target-iterations under
monitoring, surviving whatever the browser stack throws, with zero manual
intervention. Attempts v1-v6 each exposed (and fixed) one gap; **v7 ran 370/370
iterations to autonomous completion** (`PARTITION=highpri SALLOC_TIME=08:00:00
launch_coord_train.sh coord-test-v7 2 370`, nodes sarekl16-[6-8]).

Result census (370 unique iters, 1.60M env steps, wandb ka2ovlhs):
- cruise: median 39.9s/iter, median 102 sps; return_mean 0.13 -> 1.28
  (agent genuinely learning — reward signal healthy end-to-end).
- mean 100.4s/iter dragged by two tails: 27 Chrome-hang incidents (see below)
  each pricing one ~360-430s iter, and one degraded stretch (iters 257-273,
  ~360s/iter for ~100 min at segment-1 end) while an env-runner cycled through
  RLlib restart; the salloc-wall re-salloc cleared it.
- **Recovery ladder: 27/27 hangs self-healed, zero run-ending failures.**
  Uniform signature every incident: watchdog tree-kill (6 procs) + driver kill
  -> wedged caller thread hits the 300s ThreadedVectorEnv backstop
  ("abandoning thread and rebuilding") -> occasionally the runner still dies
  on a dead-driver touch -> RLlib restart_failed_env_runners brings the actor
  back ~2-4 min later. Every rung production-fired.
- **Coordinator salloc-wall handoff PROVEN**: 863308 hit its 8h TIME LIMIT at
  iter 273; coordinator re-salloc'd (863366 granted 6 min later), workload
  resumed from ckpt-270, segment 2 cruised (100 iters, 10 self-healed hangs)
  and hit target -> clean teardown + scancel. Zero human input across 10.5h.

Fix ledger that made v7 possible (v1-v6, all committed):
  1. tree-kill ascends to the topmost Chrome parent (child-kill didn't raise);
  2. driver-pid tracking + ALWAYS kill the Playwright driver (wedged driver
     never propagates browser death — py-spy-verified greenlet wedge);
  3. driver-SCOPED chrome-pid attribution (process-wide first-match could kill
     a sibling env's Chrome);
  4. 120s launch guard (child sweep since pre_launch);
  5. ThreadedVectorEnv abandon-and-rebuild backstop (result_timeout_s=300) —
     the only rung a playwright wedge cannot veto;
  6. sample_timeout_s 600 (tight 90s creates terminal phase-lock — v2/v3);
  7. coord_learner crash-exit (nonzero workload exit = learner death, no more
     zombie-sleep);
  8. node pool excludes sarekl16-2 (hang factory, 6/9 incidents one night) and
     sarekl15-5 (persistent CUDA-init failure) on top of the Mesa trio + 15-8.

Open notes: 16-2/15-5 may be transient node states — re-probe before treating
as permanent; steps/iter is ~4080 at train-batch 4000 (launcher comment said
~5450 — 370 iters = 1.6M steps, not 2M; size targets accordingly); the
segment-1-tail degradation (runner restart churn under sustained load) is the
remaining known soft spot — a future lever is restart_failed_env_runners
diagnostics or per-runner health metrics.

**Verdict: the coordinator path is the production default for real RL runs.**

## R10. Degraded-throughput catch + handle (2026-08-21)

Closes v7's one blemish (R9): an EnvRunner restart churned ~100 min at ~360s/
iter (15% throughput) and nothing noticed — RLlib only distinguishes dead from
alive. The cure was already proven (workload restart from ckpt, ~5 min); R10
automates invoking it. Two rungs:

1. **train.py degraded-exit** (`--degraded-exit`, default ON;
   `degradation.DegradationDetector`): trip when median of the last 5 iter
   times > max(3x run-median baseline, 180s floor). Median-of-5 makes v7's 27
   benign single-iter hang recoveries invisible; v7's churn block would have
   tripped at its 3rd degraded iter (~18 min in, vs 100 min). On trip: force
   sync checkpoint -> MARK degraded_exit -> exit 43 -> coord_learner set -e
   propagates (v6-production-proven path) -> coordinator respawns + --resume.
   Cross-restart cap via meta.json `degraded_exits`: >=3 exits in 2h disables
   the detector (degradation surviving restarts = environmental; ride it out).
2. **coordinator progress-stall net** (`--progress-stall-timeout-s`, launcher
   sets 1800): SIGTERM a nominally-ALIVE learner whose meta.json iteration is
   frozen past the timeout (catches wedges rung 1 can't see — e.g. the loop
   itself stuck outside algo.train()). Clock resets on every learner launch so
   startup never counts. Kill counts toward the respawn circuit breaker.

Forensics: iter lines now carry `H=<healthy>/<total>` env-runners (v7's
100-min outage had to be reconstructed from scattered actor-manager warnings).
70/70 tests (12 new: detector properties + coordinator stall behavior).
First live validation rides the next production run; the detector is pure
logic and the exit->respawn->resume plumbing is already production-fired.

## R11. First frozen-d0 eval of coord-test-v7 — eval protocol decided (2026-08-22)

Frozen pool `eval_d0_v1.parquet` (committed): 200 pairs, seed 42, uniform over
the full 124k-id skeleton x cell_stats intersection (= training distribution).
Jobs 868581/868582/868584, ckpt_000370:

  arm                          success   mean_return  notes
  v7 STOCHASTIC (sampled)      49.6%     1.37         n=139 (job wedged @142)
  v7 ARGMAX (per-head argmax)   2.5%     -            = chance; mean 15 steps
  RANDOM baseline               2.5%     -            chance floor

- **The v7 policy is real: ~50% vs 2.5% chance (20x)**, mean_return matches
  training (1.3-1.9), successful episodes median 148 steps. Success grades
  cleanly by neuron length: q1 63% / q2 62% / q3 51% / q4 23% — natural
  curriculum axis for IndexedStateProvider.
- **[DECIDED] Eval protocol = STOCHASTIC sampling** (`eval_d0.py --stochastic
  --torch-seed N`). Per-head argmax is degenerate for this policy (the
  1024-way click head is only ever optimized under sampling); argmax's 5
  "successes" were near-spawn z_max pairs (mean 15 steps). Deterministic
  eval understates a working policy by 20x here.
- Eval-side robustness gap: the single-env eval loop has no vector-level
  hang backstop — job 868584 wedged at pair 142 on the known
  playwright-greenlet failure (watchdog killed Chrome+driver, relaunch
  wedged; in-process kills can't unblock the sync caller). Per-pair results
  are log-recoverable. Fix candidate if evals get longer: run the env inside
  ThreadedVectorEnv(1) to inherit the 300s abandon-and-rebuild, or a
  hard per-pair alarm that os._exits (partial JSON flush + resume).

## R11 addendum — eval v2 (instrumented) + threshold study + action census (2026-08-24)

Second full stochastic eval of ckpt_000370 (job 869583, all 200 pairs, no
wedge; z_series recorded, incremental flush live): **55.5%** overall
(q1 72 / q2 50 / q3 60 / q4 40) — consistent with R11's 49.6%/139.

Post-hoc tolerance study (eval_thresholds.py, first-crossing on recorded
trajectories — exact for "would have terminated"):

  criterion       overall   q1    q2    q3    q4
  abs +/-10 vox    55.5%   72    50    60    40
  +/-5%  extent    70.5%   80    68    68    66
  +/-10% extent    82.0%   84    72    82    90
  +/-15% extent    84.5%   88    74    84    92

Pool median z-extent 1866 vox => abs +/-10 is a median +/-0.54%-of-extent
band. Key reading: under proportional bands the q4 deficit EVAPORATES
(40 -> 90% at +/-10%): long neurons reach the target's neighborhood as
reliably as short ones — the abs-band gap is a FINE-APPROACH problem, not
gross navigation. Returns saturate past +/-10% (82 -> 84.5). If a
percentage criterion is adopted, +/-5% keeps discrimination (70.5%).

Action census (video manifests, 2.2k steps): click 63% / rotate 23% /
**zoom 0.14%** — the zoom verb is effectively unused. Architecture is NOT
the cause (HierarchicalMultiCategorical verb head is a flat 3-way softmax;
click's 1024 options are gated behind it): suspects are zoom's zero
z-shaping payoff + the entropy bonus favoring the high-H click branch
(H_max log1024 vs log9). Fine-approach levers: per-head-normalized entropy,
zoom-in shaping near the band, or the percentage criterion itself.

Eval artifact pipeline (all committed): r_eval_video.slurm (HUD: z/target/dz
+ action label) -> eval_report_html.py (embedded MP4s + z_min-based approach
curves + threshold table) -> claude.ai artifact. z_min now rides task_info.
