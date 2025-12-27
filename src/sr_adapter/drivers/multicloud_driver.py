"""Lightweight REST chat drivers for additional providers."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from .base import DriverError, LLMDriver, register_driver
from .resilience import BackoffPolicy
from ..llm_metrics import get_llm_registry

try:  # pragma: no cover - optional dependency guard
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


class JSONChatProxyDriver(LLMDriver):
    """Generic OpenAI-style JSON chat driver with configurable headers and endpoints."""

    def __init__(
        self,
        name: str,
        config: Mapping[str, Any],
        *,
        default_endpoint: str,
        provider: str,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer ",
    ):
        super().__init__(name, config)
        for key in ("api_key", "model"):
            if key not in self.config:
                raise DriverError(f"{provider} driver requires '{key}' in configuration")
        self.default_endpoint = default_endpoint
        self.provider = provider
        self.api_key_header = self.config.get("api_key_header", api_key_header)
        self.api_key_prefix = self.config.get("api_key_prefix", api_key_prefix)

    # ---------------------------------------------------------------- request plumbing
    def _endpoint(self) -> str:
        endpoint = str(self.config.get("endpoint", self.default_endpoint)).rstrip("/")
        if "{model}" in endpoint:
            return endpoint.format(model=self.config["model"]).rstrip("/")
        return endpoint

    def _headers(self) -> dict[str, str]:
        api_key = str(self.config["api_key"])
        header = self.api_key_header
        prefix = "" if self.api_key_prefix is None else self.api_key_prefix
        headers = {
            header: f"{prefix}{api_key}",
            "content-type": "application/json",
        }
        user_agent = self.config.get("user_agent")
        if user_agent:
            headers["user-agent"] = str(user_agent)
        headers.update({str(k): str(v) for k, v in self.config.get("headers", {}).items()})
        return headers

    def _build_payload(self, prompt: str, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        messages = []
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
        }
        temperature = self.config.get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        max_tokens = self.config.get("max_tokens")
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if metadata:
            payload["metadata"] = dict(metadata)
        return payload

    # ---------------------------------------------------------------- metrics
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

    # ---------------------------------------------------------------- execution
    def generate(self, prompt: str, *, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if httpx is None:  # pragma: no cover - dependency guard
            raise DriverError("httpx is required to use this driver")
        self._ensure_circuit_closed()
        url = self._endpoint()
        headers = self._headers()
        payload = self._build_payload(prompt, metadata)
        timeout = self.config.get("timeout", 30.0)
        retries = int(self.config.get("max_retries", 2))
        backoff = BackoffPolicy(
            base_delay=float(self.config.get("retry_backoff_base", 0.5)),
            max_delay=float(self.config.get("retry_backoff_max", 8.0)),
            jitter=float(self.config.get("retry_jitter", 0.5)),
        )
        last_error: Exception | None = None
        try:
            request_bytes = len(json.dumps(payload).encode("utf-8"))
        except Exception:
            request_bytes = 0

        for attempt in range(retries + 1):
            start = time.perf_counter()
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                response_bytes = len(response.content or b"")
                self._record_success(
                    start=start,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                )
                return data
            except httpx.HTTPError as exc:  # pragma: no cover - network error path
                last_error = exc
                self._record_failure(start=start, exc=exc)
                if attempt >= retries:
                    break
                delay = backoff.compute(attempt + 1)
                time.sleep(delay)
        if last_error:
            raise DriverError(f"{self.provider} request failed: {last_error}") from last_error
        raise DriverError(f"{self.provider} driver failed without exception context")


class MistralDriver(JSONChatProxyDriver):
    """Mistral chat driver using the OpenAI-compatible API surface."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        super().__init__(
            name,
            config,
            default_endpoint="https://api.mistral.ai/v1/chat/completions",
            provider="mistral",
        )


class GoogleAIDriver(JSONChatProxyDriver):
    """Google Gemini/Vertex chat driver with JSON payload."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        endpoint = config.get(
            "endpoint",
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        )
        super().__init__(
            name,
            config,
            default_endpoint=endpoint,
            provider="google-ai",
            api_key_header="x-goog-api-key",
            api_key_prefix="",
        )

    def _build_payload(self, prompt: str, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        if metadata:
            payload["safetySettings"] = metadata  # pass-through for caller-provided settings
        return payload


class XaiDriver(JSONChatProxyDriver):
    """xAI driver using the Grok chat endpoint."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        super().__init__(
            name,
            config,
            default_endpoint="https://api.x.ai/v1/chat/completions",
            provider="xai",
        )


class BedrockDriver(JSONChatProxyDriver):
    """AWS Bedrock proxy driver for OpenAI-compatible gateways."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        super().__init__(
            name,
            config,
            default_endpoint="https://bedrock-runtime.amazonaws.com/model/{model}/invoke",
            provider="aws-bedrock",
            api_key_header=str(config.get("api_key_header", "x-api-key")),
            api_key_prefix=str(config.get("api_key_prefix", "")),
        )


register_driver(
    "mistral",
    factory=MistralDriver,
    metadata={"provider": "mistral", "transport": "rest"},
    overwrite=True,
)
register_driver(
    "googleai",
    "google",
    "vertex",
    factory=GoogleAIDriver,
    metadata={"provider": "google-ai", "transport": "rest"},
    overwrite=True,
)
register_driver(
    "xai",
    "grok",
    factory=XaiDriver,
    metadata={"provider": "xai", "transport": "rest"},
    overwrite=True,
)
register_driver(
    "bedrock",
    "aws",
    factory=BedrockDriver,
    metadata={"provider": "aws-bedrock", "transport": "rest"},
    overwrite=True,
)
