# ngllib-agent

New training defaults to a single 3D-pane max-Z navigation task with spatial
DINO patch tokens. Supply the reset dataset explicitly:

```bash
python -m ngllib_agent.train --set env.parquet_path=/path/to/positions.parquet
```

`configs/native_zmaxleft_v7.yaml` explicitly selects the two-pane multi-neuron
climbing experiment. New DINO training requires `obs.dino.patch_tokens: true`.
The CLI respects the observation mode in YAML unless `--obs` overrides it.
W&B logging is disabled by default. RLlib is pinned to 2.56.0.

Each checkpoint has a JSON sidecar containing its task/model configuration,
launch arguments, and progress. Matching configurations in the same checkpoint
directory continue automatically, including the curriculum and entropy schedule.
Use a new directory for an independent run; `--resume` explicitly permits an
intentional configuration change or a legacy checkpoint. Restoring weights does
not preserve an in-flight browser episode. Credentials must be supplied at
runtime, and should never be embedded in a task config or launch override.

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
