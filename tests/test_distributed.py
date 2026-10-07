from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from sr_adapter.distributed import run_asyncio, run_ray, run_threadpool


@pytest.mark.parametrize("runner", [run_asyncio, run_threadpool])
def test_local_backends_preserve_input_order(runner):
    assert runner(lambda x: x * 2, [3, 1, 2], workers=2) == [6, 2, 4]


def test_asyncio_backend_rejects_nested_event_loop_cleanly():
    async def scenario():
        with pytest.raises(RuntimeError, match="active event loop"):
            run_asyncio(lambda x: x, [1], workers=1)
    asyncio.run(scenario())


def test_ray_cluster_connection_does_not_override_node_resources(monkeypatch):
    calls = []
    ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: calls.append(kwargs),
        remote=lambda func: SimpleNamespace(remote=func),
        get=lambda refs: refs,
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    assert run_ray(lambda x: x * 2, [1, 2], address="auto", workers=3) == [2, 4]
    assert calls == [{"ignore_reinit_error": True, "address": "auto"}]
