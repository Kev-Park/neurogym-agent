"""Ray plumbing for the per-node render+encode service.

`create_render_services(cfg)` (driver, after ray.init) starts one
RenderServiceActor per alive GPU node, named by node IP. Envs resolve their
own node's actor lazily via `service_factory()` (called inside the runner
process), keeping ngllib Ray-free — the env just sees a handle with
`.features(...)` / `.pick(...)`.

Measured basis (native/probe_render_service.py --pipeline): 255 sps/node at
K=64 clients vs ~150 for per-env encoding — the service owns the node's
single GL context and single DINO, batching across all client envs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SERVICE_NAME_PREFIX = "ngl_render_service"


def _service_name(node_ip: str) -> str:
    return f"{SERVICE_NAME_PREFIX}_{node_ip}"


def create_render_services(cfg: dict, num_gpus: float = 0.05,
                           fetch_workers: int = 8) -> list:
    """One service actor per alive GPU node. Idempotent by actor name."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(max_concurrency=256)
    class RenderServiceActor:
        def __init__(self, dino_cfg: dict, cache_dir, fetch_workers: int):
            from ngllib.native.service import RenderEncodeService

            from .obs import get_dino_encoder

            encoder = get_dino_encoder(
                model_name=dino_cfg.get("model_name", "dinov2_vits14"),
                input_size=dino_cfg.get("input_size", 224),
            )
            self._core = RenderEncodeService(
                encoder, cache_dir=cache_dir, fetch_workers=fetch_workers)

        def features(self, client_id, state, block_canvas=False):
            return self._core.features(client_id, state,
                                       block_canvas=block_canvas)

        def pick(self, state, px, py):
            return self._core.pick(state, px, py)

        def ping(self):
            return True

    dino_cfg = (cfg.get("obs", {}) or {}).get("dino", {}) or {}
    cache_dir = cfg.get("env", {}).get("cv_cache")
    actors = []
    for node in ray.nodes():
        if not node.get("Alive") or node.get("Resources", {}).get("GPU", 0) < 1:
            continue
        name = _service_name(node["NodeManagerAddress"])
        try:
            actors.append(ray.get_actor(name))
            continue
        except ValueError:
            pass
        actor = RenderServiceActor.options(
            name=name,
            num_gpus=num_gpus,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"], soft=False),
        ).remote(dino_cfg, cache_dir, fetch_workers)
        actors.append(actor)
        logger.info("render service started on %s", name)
    for a in actors:
        import ray as _ray

        _ray.get(a.ping.remote(), timeout=600)  # DINO + GL warm before runners
    return actors


class _ServiceProxy:
    """Blocking adapter: env-facing handle -> Ray actor calls."""

    def __init__(self, actor):
        self._a = actor

    def features(self, client_id, state, block_canvas=False):
        import ray

        return ray.get(self._a.features.remote(client_id, state, block_canvas))

    def pick(self, state, px, py):
        import ray

        return ray.get(self._a.pick.remote(state, px, py))


def service_factory():
    """Zero-arg factory for NativeEnvironment(render_service=...): resolves
    THIS node's service actor. Runs inside the env-runner process."""
    import ray
    from ray.util import get_node_ip_address

    return _ServiceProxy(ray.get_actor(_service_name(get_node_ip_address())))
