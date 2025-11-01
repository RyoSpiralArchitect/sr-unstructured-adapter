"""Azure OpenAI driver implementation."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Iterator, Mapping

from .base import DriverError, LLMDriver, register_driver
from .resilience import BackoffPolicy
from ..llm_metrics import get_llm_registry

try:  # pragma: no cover - exercised indirectly when dependency is present
    import httpx
except ImportError:  # pragma: no cover - handled gracefully at runtime
    httpx = None  # type: ignore[assignment]


class AzureDriver(LLMDriver):
    """Simple Azure OpenAI chat completion driver."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        super().__init__(name, config)
        for key in ("endpoint", "deployment", "api_version", "api_key"):
            if key not in self.config:
                raise DriverError(f"Azure driver requires '{key}' in configuration")

    def _coerce_timeout(self) -> httpx.Timeout:
        timeout = self.config.get("timeout", 30.0)
        if isinstance(timeout, Mapping):
            return httpx.Timeout(**timeout)
        return httpx.Timeout(float(timeout))

    def _endpoint(self) -> str:
        endpoint = self.config["endpoint"].rstrip("/")
        deployment = self.config["deployment"]
        return f"{endpoint}/openai/deployments/{deployment}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "api-key": self.config["api_key"],
            "content-type": "application/json",
        }
        user_agent = self.config.get("user_agent")
        if user_agent:
            headers["user-agent"] = str(user_agent)
        return headers

    def _build_payload(self, prompt: str, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        system_prompt = self.config.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": self.config.get("temperature", 0.2),
            "max_tokens": self.config.get("max_tokens", 512),
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return payload

    def _params(self) -> dict[str, str]:
        return {"api-version": str(self.config["api_version"])}

    def _should_retry(self, exc: httpx.HTTPError) -> bool:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(exc, httpx.TimeoutException):
            return True
        if status is None:
            return isinstance(exc, httpx.TransportError)
        if status in {408, 409, 429}:
            return True
        return status >= 500

    def _request_bytes(self, payload: Mapping[str, Any]) -> int:
        try:
            return len(json.dumps(payload).encode("utf-8"))
        except Exception:
            return 0

    def _record_success(
        self,
        *,
        start: float,
        request_bytes: int,
        response_bytes: int,
    ) -> None:
        duration_ms = (time.perf_counter() - start) * 1000.0
        get_llm_registry().record_success(
            self.name,
            latency_ms=duration_ms,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
        )
        self.circuit_breaker.record_success()

    def _record_failure(self, *, start: float, exc: Exception) -> None:
        duration_ms = (time.perf_counter() - start) * 1000.0
        get_llm_registry().record_failure(
            self.name,
            latency_ms=duration_ms,
            error=exc.__class__.__name__,
        )
        self.circuit_breaker.record_failure()

    def generate(self, prompt: str, *, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if httpx is None:  # pragma: no cover - dependency guard
            raise DriverError("httpx is required to use the Azure driver")
        self._ensure_circuit_closed()
        url = self._endpoint()
        headers = self._headers()
        payload = self._build_payload(prompt, metadata)
        params = self._params()
        timeout = self._coerce_timeout()
        retries = int(self.config.get("max_retries", 2))
        backoff = BackoffPolicy(
            base_delay=float(self.config.get("retry_backoff_base", 0.5)),
            max_delay=float(self.config.get("retry_backoff_max", 8.0)),
            jitter=float(self.config.get("retry_jitter", 0.5)),
        )
        request_bytes = self._request_bytes(payload)
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            start = time.perf_counter()
            try:
                with httpx.Client(timeout=timeout) as client:
                    request = client.build_request(
                        "POST",
                        url,
                        headers=headers,
                        params=params,
                        json=payload,
                    )
                    response = client.send(request)
                    response.raise_for_status()
                    data = response.json()
                    response_bytes = len(response.content or b"")
                    self._record_success(
                        start=start,
                        request_bytes=request_bytes,
                        response_bytes=response_bytes,
                    )
                    return data
            except httpx.HTTPError as exc:  # pragma: no cover - network failure path
                last_error = exc
                self._record_failure(start=start, exc=exc)
                if attempt >= retries or not self._should_retry(exc):
                    break
                delay = backoff.compute(attempt + 1)
                time.sleep(delay)
        if last_error:
            raise DriverError(f"Azure request failed: {last_error}") from last_error
        raise DriverError("Azure driver failed without exception context")

    def stream_generate(
        self,
        prompt: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        if httpx is None:  # pragma: no cover - dependency guard
            raise DriverError("httpx is required to use the Azure driver")
        self._ensure_circuit_closed()
        url = self._endpoint()
        headers = self._headers()
        payload = self._build_payload(prompt, metadata)
        payload["stream"] = True
        params = self._params()
        timeout = self._coerce_timeout()
        retries = int(self.config.get("max_retries", 2))
        backoff = BackoffPolicy(
            base_delay=float(self.config.get("retry_backoff_base", 0.5)),
            max_delay=float(self.config.get("retry_backoff_max", 8.0)),
            jitter=float(self.config.get("retry_jitter", 0.5)),
        )
        request_bytes = self._request_bytes(payload)
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            start = time.perf_counter()
            accumulated = 0
            try:
                with httpx.Client(timeout=timeout) as client:
                    request = client.build_request(
                        "POST",
                        url,
                        headers=headers,
                        params=params,
                        json=payload,
                    )
                    with client.stream(
                        request.method,
                        request.url,
                        headers=request.headers,
                        params=request.params,
                        content=request.content,
                    ) as response:
                        response.raise_for_status()
                        for chunk in response.iter_lines():
                            if not chunk:
                                continue
                            accumulated += len(chunk)
                            if chunk.startswith("data: "):
                                chunk = chunk[6:]
                            if chunk.strip() == "[DONE]":
                                break
                            try:
                                yield json.loads(chunk)
                            except json.JSONDecodeError:
                                continue
                    self._record_success(
                        start=start,
                        request_bytes=request_bytes,
                        response_bytes=accumulated,
                    )
                    return
            except httpx.HTTPError as exc:  # pragma: no cover - network failure path
                last_error = exc
                self._record_failure(start=start, exc=exc)
                if attempt >= retries or not self._should_retry(exc):
                    break
                delay = backoff.compute(attempt + 1)
                time.sleep(delay)
        if last_error:
            raise DriverError(f"Azure streaming failed: {last_error}") from last_error
        raise DriverError("Azure streaming failed without exception context")

    async def async_generate(
        self,
        prompt: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if httpx is None:  # pragma: no cover - dependency guard
            raise DriverError("httpx is required to use the Azure driver")
        self._ensure_circuit_closed()
        url = self._endpoint()
        headers = self._headers()
        payload = self._build_payload(prompt, metadata)
        params = self._params()
        timeout = self._coerce_timeout()
        retries = int(self.config.get("max_retries", 2))
        backoff = BackoffPolicy(
            base_delay=float(self.config.get("retry_backoff_base", 0.5)),
            max_delay=float(self.config.get("retry_backoff_max", 8.0)),
            jitter=float(self.config.get("retry_jitter", 0.5)),
        )
        request_bytes = self._request_bytes(payload)
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            start = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        params=params,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    response_bytes = len(response.content or b"")
                    self._record_success(
                        start=start,
                        request_bytes=request_bytes,
                        response_bytes=response_bytes,
                    )
                    return data
            except httpx.HTTPError as exc:  # pragma: no cover - network failure path
                last_error = exc
                self._record_failure(start=start, exc=exc)
                if attempt >= retries or not self._should_retry(exc):
                    break
                delay = backoff.compute(attempt + 1)
                await asyncio.sleep(delay)
        if last_error:
            raise DriverError(f"Azure async request failed: {last_error}") from last_error
        raise DriverError("Azure async request failed without exception context")

    async def async_stream_generate(
        self,
        prompt: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        if httpx is None:  # pragma: no cover - dependency guard
            raise DriverError("httpx is required to use the Azure driver")
        self._ensure_circuit_closed()
        url = self._endpoint()
        headers = self._headers()
        payload = self._build_payload(prompt, metadata)
        payload["stream"] = True
        params = self._params()
        timeout = self._coerce_timeout()
        retries = int(self.config.get("max_retries", 2))
        backoff = BackoffPolicy(
            base_delay=float(self.config.get("retry_backoff_base", 0.5)),
            max_delay=float(self.config.get("retry_backoff_max", 8.0)),
            jitter=float(self.config.get("retry_jitter", 0.5)),
        )
        request_bytes = self._request_bytes(payload)
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            start = time.perf_counter()
            accumulated = 0
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        url,
                        headers=headers,
                        params=params,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_lines():
                            if not chunk:
                                continue
                            accumulated += len(chunk)
                            if chunk.startswith("data: "):
                                chunk = chunk[6:]
                            if chunk.strip() == "[DONE]":
                                break
                            try:
                                yield json.loads(chunk)
                            except json.JSONDecodeError:
                                continue
                    self._record_success(
                        start=start,
                        request_bytes=request_bytes,
                        response_bytes=accumulated,
                    )
                    return
            except httpx.HTTPError as exc:  # pragma: no cover - network failure path
                last_error = exc
                self._record_failure(start=start, exc=exc)
                if attempt >= retries or not self._should_retry(exc):
                    break
                delay = backoff.compute(attempt + 1)
                await asyncio.sleep(delay)
        if last_error:
            raise DriverError(f"Azure async streaming failed: {last_error}") from last_error
        raise DriverError("Azure async streaming failed without exception context")


register_driver(
    "azure",
    "azure_openai",
    factory=AzureDriver,
    metadata={"provider": "azure", "transport": "rest"},
    overwrite=True,
)
