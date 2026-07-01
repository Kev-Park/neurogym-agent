"""Distributed-RL orchestration primitives for `ngllib-agent`.

- `coordinator` — login-node process manager (Milestone 4). Holds a SLURM
  allocation via `salloc --no-shell`, launches learner + renderer processes
  via `srun --overlap`, monitors liveness, and (in later milestones) handles
  respawn, renderer→learner promotion, and salloc re-request on preemption.
"""
