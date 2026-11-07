# ngllib-agent

## Installation

Clone both repositories as siblings (Python 3.12 and uv required):

```bash
git clone https://github.com/Kev-Park/ngllib.git
git clone https://github.com/Kev-Park/ngllib-agent.git
cd ngllib-agent
uv sync --locked
uv run --no-sync playwright install chromium
```

The editable dependency points to `../ngllib`. Keep that sibling directory
name, or explicitly change the uv source when using a different layout.
Run the commands below with `uv run --no-sync` to use this environment.
Dataset files and runtime credentials are supplied separately.

Existing W&B runs retain their original project; pass `--wandb-project`
explicitly when continuing one in a project created before the repository rename.

New training defaults to a single 3D-pane max-Z navigation task with spatial
DINO patch tokens. Supply the reset dataset explicitly:

```bash
python -m ngllib_agent.train --set env.parquet_path=/path/to/positions.parquet
```

Alternatively, set `task.manifest` to a content-verified data manifest. Create
one for existing materialized data with:

```bash
python -m ngllib_agent.data manifest --reset-data /data/positions.parquet \
  --dataset-id my-dataset --release my-release --output /data/task.json
python -m ngllib_agent.train --set task.manifest=/data/task.json
```

Supply `--start-url` for another volume and `--parameters` with a JSON record
of the precomputation settings. Artifact paths are relative to the manifest;
its dataset, coordinates, parameters, and content hashes bind the run to its
data. Changed artifacts are rejected. Existing table-only configs remain
supported and now record table hashes. Intentional data transfer on resume
requires `--resume --allow-data-change`; older checkpoints without data identity
require explicit `--resume` when first upgraded.

`task.factory: package.module:build_task` selects an installed callable with
signature `(cfg, *, manifest)` returning `ngllib_agent.tasks.TaskComponents`:
reset provider, reward factory, termination factory, and optional metadata
provider. To replace only reset sampling, set `task.provider_factory` with the
same signature, returning the provider. Custom factories do not require a
FlyWire table. A Python caller can also pass `task_factory=` to `build_env`.

Offline precomputation is explicit:

```yaml
task:
  factory: examples.axis_task:build_task
  manifest: /data/axis-task.json
  precompute:
    factory: examples.axis_task:precompute
    params: {seed: 7, count: 20, delta_x: 100, source_parquet: /data/positions.parquet,
             dataset: {id: my-dataset}}
```

`python -m ngllib_agent.data precompute --config task.yaml --output /data/axis-task.json`
calls the materializer with `(params, output_dir)`. It returns dataset metadata,
coordinate declarations, and artifact paths; the command writes their hashes.
Initialization never runs precomputation. The X-navigation example reads JSON
starts at runtime, demonstrating a different task and provider format. Factory
modules must be available on every runner and inference host.

Selection/geometry for climbing rewards now travels through a separate
`TransitionContext`, not observations. Diagnostics read `info.task_metadata`.
The single-neuron task collects no selection/geometry metadata by default.

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

Entropy controls are explicit in the YAML's `model` section:
`normalize_entropy`, `cell_entropy_stopgrad`, `cell_entropy_scale`,
`dbl_entropy_scale`, and `verb_entropy_scale`. Each scale supports its
corresponding `*_entropy_schedule`; `ppo.entropy_coeff` is the global multiplier.
These configure learner exploration, not renderer dynamics.

In DINO mode, use `obs.use_left_pane` and `obs.dino.input_size`. Contradictory
renderer pane settings or a non-null `env.image_size` now fail rather than
being silently ignored. Existing DINO recipes omit those redundant settings.
For multi-layer viewer URLs, set `env.image_layer` and/or
`env.segmentation_layer`; the selected pair is used by both backends.

Evaluate new paired checkpoints on complete frozen starts:

```bash
python -m ngllib_agent.evaluate --state-pkl checkpoints/run/ckpt_000100.pkl \
  --starts frozen_starts.json --max-steps 500 --output evaluation.json
```

The JSON input is a list of `{id, state, task_info}` records; `state` contains
the full camera/selection state, and `task_info` supplies the chosen task's
targets. The checkpoint sidecar supplies the config; legacy weights require
`--config`. `--backend`, `--stochastic`, `--seed`, and `--video-dir` are optional.
`--progress-steps` fixes step-annealed reward progress (default zero); frozen
episodes seed randomized termination independently of rollout order.
Evaluation preserves task termination; a different evaluation reward or
termination recipe must be supplied explicitly with `--config`, with matching
observation/action/model settings. Traces record positions in viewer voxel
coordinates, reward, actions, and separate termination/truncation flags.

Use `--compare LEFT.json RIGHT.json --output differences.json` for paired
return/Z-gain differences. Incomplete runs or different starts/protocols are
rejected. `--allow-backend-difference` explicitly permits a simulator/Chrome
comparison. Historical task-specific metrics remain in the existing scripts,
which now share policy loading with the supported evaluator.

Distributed RL for FlyWire neuron navigation in Neuroglancer, built on
[`ngllib`](../ngllib) 0.2's Gymnasium-compliant `Environment`.

- **Algorithm:** RLlib PPO (synchronous large-batch, `sample_async`).
- **Task:** Z-navigate — reach a segment's max-z point within tolerance.
- **Topology:** login-node coordinator + `salloc --no-shell` renderer/learner
  pool restricted to vulkan-capable nodes (see repo `../.claude/CLAUDE.md`
  "Node quirks"). Design: `../ngllib/agent_plan.md`.

Package layout under `src/ngllib_agent/` (Milestone 1 scaffold):

```
providers/flywire_skeleton.py   StateProvider over the (root_id,x,y,z) parquet
rewards/z_navigate.py           reward + termination factories (Z-tolerance)
wrappers/action.py              MultiDiscrete -> ngllib Dict action wrapper
```

Scripts (`scripts/`) and configs (`configs/`) drive the single-process smoke
(Milestone 1). Multi-node coordinator lands in later milestones.
