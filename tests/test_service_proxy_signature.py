"""The service crosses a Ray actor boundary, so a kwarg added to the core
service is a runtime TypeError in every env runner if the actor and the
blocking proxy are not updated with it (cost us a crash-looping multi-node
run, 2026-08-31). Keep the three signatures in lockstep."""

from __future__ import annotations

import inspect
import re


def _params(fn):
    return list(inspect.signature(fn).parameters)


def test_proxy_matches_core_service():
    from ngllib.native.service import RenderEncodeService

    from ngllib_agent.service_actor import _ServiceProxy

    for name in ("features", "pick", "warm"):
        core = getattr(RenderEncodeService, name, None)
        proxy = getattr(_ServiceProxy, name, None)
        if core is None or proxy is None:
            continue
        assert _params(proxy) == _params(core), (
            f"_ServiceProxy.{name} signature drifted from RenderEncodeService"
        )


def test_actor_forwards_every_proxy_arg():
    """The actor class is built inside `service_factory` under @ray.remote,
    so read its source rather than importing ray here."""
    from ngllib_agent import service_actor
    from ngllib.native.service import RenderEncodeService

    src = inspect.getsource(service_actor)
    body = re.search(
        r"def features\(self, client_id[^)]*\):\s*\n\s*return self\._core\.features\(([^)]*)\)",
        src, re.S)
    assert body, "actor's features() forwarder not found"
    forwarded = [a.strip() for a in body.group(1).split(",")]
    expected = _params(RenderEncodeService.features)[1:]  # drop self
    assert forwarded == expected, (
        f"actor forwards {forwarded}, core takes {expected}")
