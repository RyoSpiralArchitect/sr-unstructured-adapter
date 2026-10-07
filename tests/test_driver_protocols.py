"""Exercise real HTTPX request/stream paths without network access."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from sr_adapter.drivers import (
    AnthropicDriver, AzureDriver, BedrockDriver, DockerDriver, DriverError,
    GoogleAIDriver, MistralDriver, OpenAIDriver, VLLMDriver, XaiDriver,
)

CASES = [
    (OpenAIDriver, {"api_key": "test", "model": "test"}),
    (AnthropicDriver, {"api_key": "test", "model": "test"}),
    (AzureDriver, {"api_key": "test", "endpoint": "https://example.test", "deployment": "test", "api_version": "2024-10-21"}),
    (VLLMDriver, {"model": "test"}),
    (DockerDriver, {"url": "https://example.test/v1/chat/completions"}),
    (MistralDriver, {"api_key": "test", "model": "test"}),
    (XaiDriver, {"api_key": "test", "model": "test"}),
    (BedrockDriver, {"api_key": "test", "model": "test", "endpoint": "https://example.test/v1/chat/completions"}),
]


@pytest.fixture(params=CASES, ids=lambda case: case[0].__name__)
def driver(request):
    cls, config = request.param
    return cls("test", {**config, "max_retries": 2, "retry_backoff_base": 0, "retry_jitter": 0})


@pytest.fixture
def install_transport(monkeypatch):
    client_cls = httpx.Client
    async_client_cls = httpx.AsyncClient

    def install(handler):
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_cls(transport=transport, **kwargs))
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: async_client_cls(transport=transport, **kwargs))
    return install


@pytest.mark.parametrize("asynchronous", [False, True])
def test_permanent_error_is_not_retried(driver, install_transport, asynchronous):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    install_transport(handler)
    with pytest.raises(DriverError, match="HTTP 401"):
        if asynchronous:
            asyncio.run(driver.async_generate("hello"))
        else:
            driver.generate("hello")
    assert len(requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_transient_error_retries_then_returns_object(driver, install_transport, asynchronous):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429 if len(requests) == 1 else 200, json={"choices": []})

    install_transport(handler)
    if asynchronous:
        result = asyncio.run(driver.async_generate("hello"))
    else:
        result = driver.generate("hello")
    assert result == {"choices": []}
    assert len(requests) == 2


@pytest.mark.parametrize("payload", [b"not JSON", b"[]", b'{"error": {"message": "bad"}}'])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_malformed_response_is_driver_error_without_retry(driver, install_transport, payload, asynchronous):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=payload)

    install_transport(handler)
    with pytest.raises(DriverError):
        if asynchronous:
            asyncio.run(driver.async_generate("hello"))
        else:
            driver.generate("hello")
    assert len(requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_stream_decodes_multiline_sse_and_query_parameters(driver, install_transport, asynchronous):
    requests = []

    def handler(request):
        requests.append(request)
        assert json.loads(request.content)["stream"] is True
        if isinstance(driver, AzureDriver):
            assert request.url.params.multi_items() == [("api-version", "2024-10-21")]
        return httpx.Response(200, content=': heartbeat\nevent: completion\ndata:{"choices": [],\ndata: "text": "猫"}\n\ndata:[DONE]\n\n'.encode())

    install_transport(handler)

    async def collect():
        return [chunk async for chunk in driver.async_stream_generate("hello")]

    chunks = asyncio.run(collect()) if asynchronous else list(driver.stream_generate("hello"))
    assert chunks == [{"choices": [], "text": "猫"}]
    assert len(requests) == 1


class _BrokenSyncStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b'data: {"choices": [{"delta": {"content": "first"}}]}\n\n'
        raise httpx.ReadError("connection lost")


class _BrokenAsyncStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices": [{"delta": {"content": "first"}}]}\n\n'
        raise httpx.ReadError("connection lost")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_partial_stream_is_never_replayed(driver, install_transport, asynchronous):
    requests = []
    seen = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, stream=_BrokenAsyncStream() if asynchronous else _BrokenSyncStream())

    install_transport(handler)

    async def collect():
        async for item in driver.async_stream_generate("hello"):
            seen.append(item)

    with pytest.raises(DriverError, match="ReadError"):
        if asynchronous:
            asyncio.run(collect())
        else:
            for item in driver.stream_generate("hello"):
                seen.append(item)
    assert len(requests) == 1
    assert len(seen) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("event", [b'data: {"choices": []}\n\n', b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error"}}\n\n'])
def test_incomplete_or_error_stream_is_failure(driver, install_transport, asynchronous, event):
    install_transport(lambda request: httpx.Response(200, content=event))

    async def collect():
        return [chunk async for chunk in driver.async_stream_generate("hello")]

    with pytest.raises(DriverError):
        if asynchronous:
            asyncio.run(collect())
        else:
            list(driver.stream_generate("hello"))


def test_provider_payloads_keep_adapter_metadata_separate():
    metadata = {"recipe": "test", "block_count": 2, "indices": [0, 1]}
    openai = OpenAIDriver("openai", {"model": "reasoning-model", "api_key": "test", "max_completion_tokens": 123, "reasoning_effort": "low"})
    payload = openai._build_payload("hi", metadata)
    assert "temperature" not in payload
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 123
    assert payload["reasoning_effort"] == "low"
    assert "metadata" not in payload
    assert "store" not in payload
    anthropic = AnthropicDriver("anthropic", {"model": "claude", "api_key": "test"})
    assert "metadata" not in anthropic._build_payload("hi", metadata)
    assert anthropic._build_payload("hi", {**metadata, "user_id": "opaque"})["metadata"] == {"user_id": "opaque"}
    google = GoogleAIDriver("googleai", {"model": "gemini", "api_key": "test", "system_prompt": "review", "max_tokens": 32, "temperature": 0})
    payload = google._build_payload("hi", metadata)
    assert "model" not in payload
    assert "safetySettings" not in payload
    assert payload["systemInstruction"] == {"parts": [{"text": "review"}]}
    assert payload["generationConfig"] == {"maxOutputTokens": 32, "temperature": 0}
    assert google.supports_streaming() is False


def test_mistral_serializes_typed_escalation_metadata(install_transport):
    metadata = {
        "recipe": "test", "tenant": "default", "block_count": 1,
        "indices": [0], "context_indices": [],
    }
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["metadata"] == {
            "recipe": "test", "tenant": "default", "block_count": "1",
            "indices": "[0]", "context_indices": "[]",
        }
        assert "store" not in payload
        return httpx.Response(200, json={"choices": []})

    install_transport(handler)
    driver = MistralDriver("mistral", {"model": "test", "api_key": "test"})
    driver.generate("Review this block", metadata=metadata)
    assert len(requests) == 1
    assert metadata["block_count"] == 1
    assert metadata["indices"] == [0]
    assert metadata["context_indices"] == []


@pytest.mark.parametrize("cls,config", [CASES[0], CASES[2]], ids=["openai", "azure"])
@pytest.mark.parametrize("store", [None, False, True])
@pytest.mark.parametrize("mode", ["sync", "async", "stream", "async_stream"])
def test_openai_metadata_requires_explicit_storage_opt_in(cls, config, store, mode, install_transport):
    metadata = {"recipe": "test", "block_count": 2, "indices": [0, 1]}
    if store is not None:
        config = {**config, "store": store}
    driver = cls("test", config)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if "stream" in mode:
            return httpx.Response(200, content=b'data: {"choices": []}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"choices": []})

    install_transport(handler)

    async def collect():
        return [chunk async for chunk in driver.async_stream_generate("hi", metadata=metadata)]

    if mode == "sync":
        driver.generate("hi", metadata=metadata)
    elif mode == "async":
        asyncio.run(driver.async_generate("hi", metadata=metadata))
    elif mode == "stream":
        list(driver.stream_generate("hi", metadata=metadata))
    else:
        asyncio.run(collect())

    assert len(requests) == 1
    payload = requests[0]
    if store is None:
        assert "store" not in payload
    else:
        assert payload["store"] is store
    if store is True:
        assert payload["metadata"] == {"recipe": "test", "block_count": "2", "indices": "[0, 1]"}
    else:
        assert "metadata" not in payload
    assert metadata == {"recipe": "test", "block_count": 2, "indices": [0, 1]}


@pytest.mark.parametrize("cls,config", [CASES[0], CASES[2]], ids=["openai", "azure"])
@pytest.mark.parametrize("store", ["true", "false", 1, 0])
def test_openai_storage_option_rejects_ambiguous_values(cls, config, store):
    driver = cls("test", {**config, "store": store})
    with pytest.raises(DriverError, match="must be a boolean"):
        driver._build_payload("hi", {"recipe": "test"})


@pytest.mark.parametrize("endpoint", ["http://localhost:8000", "http://localhost:8000/v1/", "http://localhost:8000/v1/chat/completions"])
def test_vllm_version_path_and_authentication(endpoint):
    driver = VLLMDriver("vllm", {"model": "test", "endpoint": endpoint, "api_key": "test"})
    assert driver._endpoint() == "http://localhost:8000/v1/chat/completions"
    assert driver._headers()["Authorization"] == "Bearer test"


def test_bedrock_and_vertex_require_explicit_endpoints():
    with pytest.raises(DriverError, match="endpoint"):
        BedrockDriver("bedrock", {"model": "test", "api_key": "test"})
    with pytest.raises(DriverError, match="endpoint"):
        GoogleAIDriver("vertex", {"model": "test", "api_key": "test"})


def test_anthropic_message_stop_completes_stream(install_transport):
    driver = AnthropicDriver("anthropic", {"model": "claude", "api_key": "test"})
    install_transport(lambda request: httpx.Response(200, content=b'event: message_stop\ndata: {"type": "message_stop"}\n\n'))
    assert list(driver.stream_generate("hi")) == [{"type": "message_stop"}]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_circuit_breaker_stops_remaining_retries(driver, install_transport, asynchronous):
    from sr_adapter.drivers.resilience import CircuitBreaker
    driver._circuit_breaker = CircuitBreaker(failure_threshold=1, recovery_time=30)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503, json={"error": "unavailable"})

    install_transport(handler)
    with pytest.raises(DriverError, match="HTTP 503"):
        if asynchronous:
            asyncio.run(driver.async_generate("hello"))
        else:
            driver.generate("hello")
    assert len(requests) == 1
    with pytest.raises(DriverError, match="circuit breaker is open"):
        driver.generate("hello")
    assert len(requests) == 1
