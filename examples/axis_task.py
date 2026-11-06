"""An X-navigation task over frozen JSON starts, independent of skeleton tables at runtime."""
import copy
import json

import numpy as np

from ngllib_agent.tasks import TaskComponents


def precompute(params, output_dir):
    """Explicit offline sampling; no extraction occurs in build_task."""
    from ngllib_agent.providers import FlywireSkeletonProvider

    rng = np.random.default_rng(params["seed"])
    source = FlywireSkeletonProvider(params["source_parquet"])
    rows = []
    for index in range(params["count"]):
        state, _ = source(rng, {})
        rows.append({"id": str(index), "state": state,
                     "task_info": {"target_x": state["position"][0] + params["delta_x"]}})
    path = output_dir / "axis_starts.json"
    path.write_text(json.dumps(rows))
    return {"dataset": params["dataset"],
            "coordinates": {"frame": "viewer_voxel", "units": "voxel"},
            "artifacts": {"starts": path}}


def provider(cfg, *, manifest):
    rows = json.loads(open(manifest["artifacts"]["starts"]["path"]).read())

    def sample(rng, options):
        row = rows[int(rng.integers(len(rows)))]
        return copy.deepcopy(row["state"]), dict(row["task_info"])

    return sample


def build_task(cfg, *, manifest):
    tolerance = cfg["task"].get("params", {}).get("tolerance", 2.0)

    def terminal_factory(task_info):
        target = task_info["target_x"]
        return lambda obs, action, previous: abs(float(obs["position"][0]) - target) <= tolerance

    def reward_factory(task_info):
        target = task_info["target_x"]

        def reward(obs, action, previous, terminated, *, context):
            assert context.current_metadata["task_axis"] == "x"
            before = abs(float(previous["position"][0]) - target)
            after = abs(float(obs["position"][0]) - target)
            return .001 * (before - after) + float(terminated)

        return reward

    return TaskComponents(provider(cfg, manifest=manifest), reward_factory, terminal_factory,
                          lambda renderer, state: {"task_axis": "x"})
