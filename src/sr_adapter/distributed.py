"""Helpers for running conversion workloads across concurrency backends."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Sequence, TypeVar

S = TypeVar("S")
R = TypeVar("R")


def run_threadpool(func: Callable[[S], R], items: Sequence[S], *, workers: int) -> List[R]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        return list(executor.map(func, items))


async def _async_gather(
    func: Callable[[S], R],
    items: Sequence[S],
    *,
    workers: int | None,
) -> List[R]:
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(workers) if workers and workers > 0 else None

    async def _invoke(item: S) -> R:
        if semaphore is None:
            return await loop.run_in_executor(None, func, item)
        async with semaphore:
            return await loop.run_in_executor(None, func, item)

    tasks = [_invoke(item) for item in items]
    results = await asyncio.gather(*tasks)
    return list(results)


def run_asyncio(func: Callable[[S], R], items: Sequence[S], *, workers: int | None) -> List[R]:
    if not items:
        return []
    return asyncio.run(_async_gather(func, items, workers=workers))


def run_dask(
    func: Callable[[S], R],
    items: Sequence[S],
    *,
    scheduler: str | None,
    workers: int | None,
) -> List[R]:
    try:
        import dask
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("dask is not installed") from exc

    delayed_tasks = [dask.delayed(func)(item) for item in items]
    compute_kwargs = {}
    if scheduler:
        compute_kwargs["scheduler"] = scheduler
    if workers:
        compute_kwargs["num_workers"] = workers
    results = dask.compute(*delayed_tasks, **compute_kwargs)
    return list(results)


def run_ray(
    func: Callable[[S], R],
    items: Sequence[S],
    *,
    address: str | None,
    workers: int | None,
) -> List[R]:
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ray is not installed") from exc

    if not ray.is_initialized():  # pragma: no cover - depends on runtime
        init_kwargs = {"ignore_reinit_error": True}
        if address:
            init_kwargs["address"] = address
        if workers:
            init_kwargs["num_cpus"] = workers
        ray.init(**init_kwargs)

    remote_func = ray.remote(func)
    refs = [remote_func.remote(item) for item in items]
    results = ray.get(refs)
    return list(results)


__all__ = [
    "run_asyncio",
    "run_dask",
    "run_ray",
    "run_threadpool",
]

