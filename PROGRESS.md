# neurogym-agent — training code progress

Goal: train a PPO agent that right-clicks on a neuron mesh in Neuroglancer to navigate to the highest Z-position. Talks to `ngllib.Environment` in-process (one Chrome per `SubprocVecEnv` worker). Plan file: `C:\Users\kevy0\.claude\plans\giggly-popping-eagle.md`.

## What's built

### Dependencies ([pyproject.toml](pyproject.toml))

Added via `uv add`:

- `stable-baselines3`, `gymnasium`, `wandb`, `pyyaml`
- `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124` — pinned to the `pytorch-cu124` uv index via `[tool.uv.sources]`. Verified `torch.cuda.is_available() == True` on the RTX 3050 dev box; same wheels work on RTX 3090 for real training.

### Package layout ([src/neurogym-agent/](src/neurogym-agent/))

```
src/neurogym-agent/
├── __init__.py
├── train.py                          PPO entry point
├── eval.py                           rollout + per-link success rate
├── config/
│   └── default.yaml                  hyperparams, action/reward/DINO config
├── envs/
│   ├── __init__.py
│   ├── ngl_gym_env.py                gymnasium.Env wrapping ngllib.Environment
│   ├── action_translator.py          MultiDiscrete -> 17-d neurogym action
│   └── reward.py                     sparse reward + no-op penalty
├── obs/
│   ├── __init__.py
│   ├── dino_encoder.py               frozen DINOv2 ViT-S/14 wrapper
│   └── features_extractor.py         SB3 BaseFeaturesExtractor (DINO + pos MLP)
└── policies/
    └── __init__.py
```

### Preserved (not touched)

- [relay.py](relay.py), [test.py](test.py) per user instruction. Also left alone: [host.py](host.py), [client.py](client.py), [config.json](config.json), [job.slurm](job.slurm).

### Design decisions fixed in code

- **Action space.** `MultiDiscrete([1024, 2, 9, 9, 9])` = `[click_cell, click_type, dEulerX_bin, dEulerY_bin, dEulerZ_bin]`. 32×32 click grid restricted to the 3D pane `[x=900..1800, y=0..900]`. Rotation bins: 9 per axis (4 neg / noop / 4 pos) at 0.08 rad step.
- **Frozen neurogym action indices** (always 0): `left_click`, `double_click`, modifier keys, `delta_position_xyz`, `delta_crossSectionScale`, `delta_projectionScale`.
- **`json_change` auto-set** by the translator when `right_click == 0` and any rotation bin is non-zero. (`apply_actions` uses an `elif` chain, so clicks and rotations can't fire in the same step — policy will learn to alternate.)
- **Reward.** Sparse `+1` when `|Z_now - Z_max| <= z_tolerance` (default 10); `noop_penalty = -0.01` when a right-click doesn't move the crosshair (within `noop_position_eps = 0.5`).
- **Termination.** Success tolerance reached, or `max_episode_steps = 300` truncation.
- **Observation.** `Dict({"image_features": Box(2*384,), "pos_state": Box(8,)})`. Each env worker holds a frozen DINOv2 ViT-S/14 on GPU; image is split into left EM / right 3D panes, each resized to 224×224, encoded, and concatenated. `pos_state = [pos_xyz, csScale, euler_xyz, projScale]`.
- **Episode reset.** Sample a URL from `start_links.txt`, `env.reset(url=...)` (which sets `prev_state`/`prev_json` internally), then apply one random-rotation + random-zoom perturbation action before handing the first obs to the agent.
- **Euler mode.** `env.options["euler_angles"] = True` set at wrapper construction.

### Start-links file format

Two plain text files, paired by line, blank lines and `#` comments ignored, counts must match:

- `--start_links path/to/urls.txt` — one Neuroglancer URL per line
- `--z_max path/to/zmax.txt` — one float per line, same order

### Reward portability

`envs/reward.py::make_env_reward_fn(z_max, cfg)` returns a closure with the `(state, action, prev_state) -> (reward, done)` signature that `ngllib.Environment(reward_function=...)` expects. Lets the same reward live inside `Environment` directly for eval / manual rollouts without the Gym wrapper.

## Verification status

| Check | Status |
|---|---|
| Dependency install on Windows w/ uv + cu124 torch | done — `torch.cuda.is_available()` = True |
| `py_compile` of all new modules | done |
| Action translator round-trip (cell 0 and 1023 map to expected pixels on 3D pane; rotation bins produce correct signed deltas) | done |
| Reward factory (`make_env_reward_fn`) returns `(reward, done)` matching `Environment` signature | done — smoke-tested with fake states |
| DINO model download + forward pass | **not yet** — first `.reset()` will trigger `torch.hub.load` |
| End-to-end `env.reset() -> env.step()` against a real Neuroglancer URL | **not yet** — needs a `start_links.txt` + `z_max.txt` |
| `PPO.learn()` smoke run (n_envs=1, ~1k steps) | **not yet** |
| `SubprocVecEnv` w/ n_envs=4 | **not yet** |

## How to run

Set up a `start_links.txt` and a `z_max.txt` somewhere, e.g.:

```
# start_links.txt
https://neuroglancer-demo.appspot.com/#!<state1>
https://neuroglancer-demo.appspot.com/#!<state2>
```

```
# z_max.txt
4200.0
3876.5
```

### Smoke train (1 env, short)

```
cd neurogym-agent
.venv/Scripts/python src/neurogym-agent/train.py \
  --start_links path/to/start_links.txt \
  --z_max       path/to/z_max.txt \
  --n_envs 1 \
  --total_timesteps 2000 \
  --wandb_mode offline
```

### Full train (4 envs, W&B online)

```
.venv/Scripts/python src/neurogym-agent/train.py \
  --start_links path/to/start_links.txt \
  --z_max       path/to/z_max.txt \
  --n_envs 4 \
  --total_timesteps 200000
```

### Eval

```
.venv/Scripts/python src/neurogym-agent/eval.py \
  --start_links path/to/held_out_urls.txt \
  --z_max       path/to/held_out_z.txt \
  --checkpoint  checkpoints/<run_id>/final.zip \
  --episodes 20 --deterministic
```

## Outstanding items

1. Provide `start_links.txt` + `z_max.txt`. Needed for every end-to-end test below.
2. Smoke test: one env, random policy for a few hundred steps — confirm obs shapes, no exceptions, reward fires under manual neuron-top click.
3. DINO forward latency on GPU — check it's not the step-time bottleneck vs. Selenium / Chrome.
4. N=1 overfit on a single start-link — expect `rollout/ep_rew_mean` and episode-success to trend up in W&B.
5. Scale to N=4 with `SubprocVecEnv`; confirm no IPC / Chrome collision.
6. Potential tuning: rotation step size (`rotation_step_rad`), click-grid resolution, action-masking for cells outside the pane (currently unneeded — the grid is already pane-scoped).

## Reference

- Plan file (full design, open questions, critical-files list): `C:\Users\kevy0\.claude\plans\giggly-popping-eagle.md`
- Env source of truth: [../neurogym/ngllib/environment.py](../neurogym/ngllib/environment.py)
- Click dispatch: [../neurogym/ngllib/utils/MouseActionHandler.py](../neurogym/ngllib/utils/MouseActionHandler.py)
