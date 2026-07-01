# ngllib-agent (branch `dist-rl-rewrite`)

Distributed RL for FlyWire neuron navigation in Neuroglancer, built on
[`ngllib`](../neurogym) 0.2's Gymnasium-compliant `Environment`.

- **Algorithm:** RLlib PPO (synchronous large-batch, `sample_async`).
- **Task:** Z-navigate — reach a segment's max-z point within tolerance.
- **Topology:** login-node coordinator + `salloc --no-shell` renderer/learner
  pool restricted to vulkan-capable nodes (see repo `../.claude/CLAUDE.md`
  "Node quirks"). Design: `../neurogym/agent_plan.md`.

Package layout under `src/ngllib_agent/` (Milestone 1 scaffold):

```
providers/flywire_skeleton.py   StateProvider over the (root_id,x,y,z) parquet
rewards/z_navigate.py           reward + termination factories (Z-tolerance)
wrappers/action.py              MultiDiscrete -> ngllib Dict action wrapper
```

Scripts (`scripts/`) and configs (`configs/`) drive the single-process smoke
(Milestone 1). Multi-node coordinator lands in later milestones.
