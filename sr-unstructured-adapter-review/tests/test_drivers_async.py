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
        async for chunk in await driver.async_stream_generate("hello"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    assert len(chunks) == 1
    assert chunks[0]["prompt"] == "hello"


def test_async_stream_generate_propagates_chunks_and_errors() -> None:
    driver = _StreamingDriver("stream", {})

    async def _collect() -> list[dict[str, object]]:
        stream = await driver.async_stream_generate("hello")
        seen: list[dict[str, object]] = []
        with pytest.raises(RuntimeError):
            async for chunk in stream:
                seen.append(chunk)
        return seen

    chunks = asyncio.run(_collect())

    assert chunks == [{"step": 1, "prompt": "hello"}]
