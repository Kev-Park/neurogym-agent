"""Task factories compose reset providers and episode hooks."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable


@dataclass(frozen=True)
class TaskComponents:
    reset_state_provider: Any
    reward_factory: Callable
    termination_factory: Callable
    metadata_provider: Callable | None = None


def import_factory(reference):
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError("Factory must be module:callable")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise TypeError(f"Factory {reference!r} is not callable")
    return factory


def build_task(cfg, manifest=None, factory=None):
    reference = cfg.get("task", {}).get("factory")
    factory = factory or (import_factory(reference) if reference else flywire_task)
    task = factory(cfg, manifest=manifest)
    if not isinstance(task, TaskComponents):
        raise TypeError("Task factory must return TaskComponents")
    return task


def flywire_task(cfg, *, manifest=None):
    from .providers import FlywireSkeletonProvider
    from .rewards import ZRewardConfig, make_z_reward_factory, make_z_termination_factory
    ec, rc = cfg["env"], cfg["reward"]
    if not cfg.get("task", {}).get("provider_factory") and not ec.get("parquet_path"):
        raise ValueError("Supply env.parquet_path or task.manifest for the default reset provider")
    if rc.get("mode", "z_navigate") not in ("z_navigate", "z_free"):
        raise ValueError(f"unknown reward.mode: {rc['mode']!r}")
    reference = cfg.get("task", {}).get("provider_factory")
    if reference:
        provider = import_factory(reference)(cfg, manifest=manifest)
    else:
        # env.holdout_parquet: a frozen eval pool whose root_ids are EXCLUDED from
        # training resets, so eval measures unseen-neuron generalization. The eval
        # CLI resets with explicit states and is unaffected.
        exclude = None
        if ec.get("holdout_parquet"):
            import pyarrow.parquet as pq
    
            exclude = [
                str(r) for r in
                pq.read_table(ec["holdout_parquet"], columns=["root_id"])
                .column("root_id").to_pylist()
            ]
        psr = ec.get("projection_scale_range")
        provider = FlywireSkeletonProvider(
            ec["parquet_path"],
            projection_scale_range=tuple(psr) if psr else None,
            spawn_curriculum=ec.get("spawn_curriculum"),
            exclude_root_ids=exclude,
        )

    # reward.mode selects the task's reward/termination pair:
    #   (absent)  z-navigate — reach the start segment's known z-max (terminal).
    #   "z_free"  zmax-left free climb — best-so-far z + annealed selection-
    #             novelty scaffold, NO task terminal (TimeLimit-only).
    if rc.get("mode") == "z_free":
        from .rewards import ZFreeRewardConfig, make_no_termination_factory, \
            make_v3_termination_factory, make_zfree_reward_factory

        zfree = ZFreeRewardConfig(
            shaping_coef=rc.get("shaping_coef", 0.001),
            select_novelty_bonus=rc.get("select_novelty_bonus", 0.1),
            select_novelty_end_steps=rc.get("select_novelty_end_steps", 1_500_000),
            select_novelty_cap=rc.get("select_novelty_cap", 10),
            step_penalty=rc.get("step_penalty", 0.0),
            delta_coef=rc.get("delta_coef", 0.0),
            select_cost=rc.get("select_cost", 0.0),
            explore_penalty_coef=rc.get("explore_penalty_coef", 0.0),
            noop_click_cost=rc.get("noop_click_cost", 0.0),
        )
        reward_factory = make_zfree_reward_factory(zfree)
        # v3 earliness pressure (TRAINING configs only — eval uses the v2
        # config so episodes stay TimeLimit-only; see z_free.py):
        if rc.get("budget_max") or rc.get("stall_limit"):
            termination_factory = make_v3_termination_factory(
                budget_min=rc.get("budget_min", 100),
                budget_max=rc.get("budget_max", 500),
                stall_limit=rc.get("stall_limit", 125),
            )
        else:
            termination_factory = make_no_termination_factory()
    else:
        rcfg = ZRewardConfig(
            z_tolerance=rc["z_tolerance"],
            success=rc["success"],
            z_shaping_coef=rc["z_shaping_coef"],
            step_penalty=rc["step_penalty"],
            z_tolerance_frac=rc.get("z_tolerance_frac"),
        )
        reward_factory = make_z_reward_factory(rcfg)
        termination_factory = make_z_termination_factory(rcfg)

    from .rewards.z_free import selection_metadata
    metadata = selection_metadata if (rc.get("mode") == "z_free" or
                                      cfg.get("diagnostics", {}).get("click_credit")) else None
    return TaskComponents(provider, reward_factory, termination_factory, metadata)
