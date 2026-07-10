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

- [ ] **Isolated clean soak (job 839615)**: M=16 threads, restart_every=90,
      `--exclusive`. First attempt (838108) FAILED (mean 22.6, -23% drift, 259
      glitches) but was CONTAMINATED — shared a node the whole run with the
      pathological restart-storm soak (837609, restart_every=10). Rerun on a
      dedicated node for the true endurance verdict. If it still drifts/glitches,
      that's a real long-run degradation to chase (memory? cumulative browser
      sickness?) before calling threads production-ready.
- [ ] **Multi-GPU/node (job 839614)**: 2 env-runners x 1 GPU x M=16 = 32
      browsers, both GPUs sampled. Tests if Chrome VULKAN rendering follows
      Ray's per-runner CUDA_VISIBLE_DEVICES (the single-GPU render ceiling was
      the M-curve plateau cause; 2 GPUs is the real ceiling-raiser). PASS = both
      GPUs busy + ~2x sps; FAIL = renderer pins GPU0, need explicit Vulkan
      device steering.

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
