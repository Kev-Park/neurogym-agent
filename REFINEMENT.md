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
- [ ] unit (browser-free): timer fires → kill callable invoked; cancelled on
      normal completion.
- [ ] cluster: SIGSTOP a Chrome process mid-run (simulated zombie) → watchdog
      kills it → episode truncates → next reset relaunches → run continues.

## R3. Real-preemption chaos test

v2 validated re-salloc on *expiry* (COMPLETING). Real preemption unobserved;
`salloc --no-shell` may be cancelled outright rather than requeued
(REQUEUE only applies to batch) — both coordinator branches exist, neither
ground-truthed.

- [ ] Step 0: privilege check — can this account submit to `highpri`?
      (`sbatch -p highpri -A pni -t 00:02:00 --wrap hostname`)
- [ ] Coordinator + dummy workers pinned to 2 nodes (`-w`), then highpri job
      onto the same nodes. Record: signal/grace observed, requeue vs cancel,
      coordinator detection branch, recovery time, resume correctness.
- [ ] Fallback if no highpri access: `scancel` / `scontrol requeue` simulation.
- [ ] Fold findings into `scripts/cluster_probe_findings.md` + CLAUDE.md.

## R4. Per-node density parity

Throughput parity reached at M=8/node; legacy density (32/node) untested.
Process-per-env costs ~0.3–0.5GB VRAM CUDA context + DINO copy per env →
caps M well below 32 on a 24GB 3090.

- [x] M-sweep benchmark: threads knee ≈ M=16 @ 24c (31 sps, legacy parity);
      M=24 @ 24c regressed — CPU-starvation vs GIL disambiguation below.
- [ ] **Degradation-curve sweep (job 837857)**: threads M = 20/24/28/32 at
      32 cpus / 200G — find where sps peaks and where it degrades; M=32 is the
      VRAM-safe ceiling (~0.6GB Vulkan per Chrome ⇒ ~21GB at 32). Answers
      whether the M=24@24c dip was CPU-bound and whether we can beat legacy
      peak (32-36 sps).
- [ ] **[decide] ThreadedVectorEnv port** (the density endgame): custom
      `gym.vector.VectorEnv` via our existing `vector_entry_point` hook —
      one process/node, M browser threads (legacy-proven I/O-bound pattern),
      ONE CUDA context, ONE DINO batching across M (port `DinoVecWrapper`
      semantics). Tradeoff vs process-per-env: loses per-env crash isolation
      (mitigated by R2 watchdog per browser); wins ~10×+ VRAM headroom and
      vector-level DINO batching. Decide after the M-sweep shows where
      process-per-env tops out.

## R5. Failover with real workload

v2 accidentally validated learner-death → full respawn → head migration →
resume. Make it deliberate + cover the promotion branch.

- [ ] Manual learner-kill mid-training (real workload, small run): kill the
      learner srun step; verify renderers exit, coordinator respawns all,
      new ray head + endpoint, `--resume` continues from latest ckpt with
      correct iteration.
- [ ] `--force-promotion-once` test hook in coordinator (genuine srun denial
      is hard to stage under `--overlap`): learner death → skip respawn →
      renderer sacrifice → learner into freed slot → pool re-heals.
- [ ] Verify wandb run continuity across both.

## R6. Housekeeping

- [ ] Commit `uv.lock` (reproducibility; regenerate cluster-side w/ /tmp cache,
      port back via git).
- [ ] Probe sarekl15-1 when it frees (icd check + `gl_probe.py`) → update
      CLAUDE.md node lists.
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
- 2026-07-07: **R4 bench DONE (job 837608)**: threads M=8 = 17.9-18.6 sps /
  5.9GB (loses to spawn at low M — thread tax); spawn M=8 = 25.7-26.3 / 9.3GB;
  **threads M=16 = 29.7-31.0 sps / 10.7GB ≈ legacy sustained parity**;
  threads M=24 = 26.9-27.5 / 16.0GB (past the knee at 24 cpus). VRAM saving
  vs spawn confirmed (~0.43GB/env). Soak (837609, M=16) + ceiling probe
  (837856, M=20/24 @ 32c) queued.
