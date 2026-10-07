import asyncio

import pytest

from sr_adapter.drivers.base import LLMDriver


class _StubDriver(LLMDriver):
    def generate(self, prompt: str, *, metadata=None):  # type: ignore[override]
        return {"prompt": prompt, "metadata": metadata}


class _StreamingDriver(LLMDriver):
    def generate(self, prompt: str, *, metadata=None):  # type: ignore[override]
        raise NotImplementedError

    def stream_generate(self, prompt: str, *, metadata=None):  # type: ignore[override]
        yield {"step": 1, "prompt": prompt}
        raise RuntimeError("stream exploded")


def test_async_generate_falls_back_to_thread() -> None:
    driver = _StubDriver("stub", {})
    result = asyncio.run(driver.async_generate("hello", metadata={"foo": "bar"}))
    assert result["prompt"] == "hello"
    assert result["metadata"] == {"foo": "bar"}


def test_stream_generate_default_yields_once() -> None:
    driver = _StubDriver("stub", {})
    items = list(driver.stream_generate("hello"))
    assert len(items) == 1
    assert items[0]["prompt"] == "hello"


def test_async_stream_generate_wraps_sync_stream() -> None:
    driver = _StubDriver("stub", {})

    async def _collect() -> list[dict[str, object]]:
        chunks: list[dict[str, object]] = []
        async for chunk in driver.async_stream_generate("hello"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    assert len(chunks) == 1
    assert chunks[0]["prompt"] == "hello"


def test_async_stream_generate_propagates_chunks_and_errors() -> None:
    driver = _StreamingDriver("stream", {})

    async def _collect() -> list[dict[str, object]]:
        stream = driver.async_stream_generate("hello")
        seen: list[dict[str, object]] = []
        with pytest.raises(RuntimeError):
            async for chunk in stream:
                seen.append(chunk)
        return seen

    chunks = asyncio.run(_collect())

    assert chunks == [{"step": 1, "prompt": "hello"}]


def test_async_stream_fallback_applies_backpressure_and_closes() -> None:
    seen = []
    closed = []

    class InfiniteDriver(_StubDriver):
        def stream_generate(self, prompt, *, metadata=None):
            try:
                while True:
                    seen.append(len(seen))
                    yield {"step": seen[-1]}
            finally:
                closed.append(True)

    async def consume_one():
        stream = InfiniteDriver("infinite", {}).async_stream_generate("hello")
        assert await anext(stream) == {"step": 0}
        await asyncio.sleep(0.02)
        assert seen == [0]
        await stream.aclose()

    asyncio.run(consume_one())
    assert closed == [True]
