"""Driver targeting a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Any, Mapping

from .base import DriverError, LLMDriver, register_driver

try:  # pragma: no cover - optional dependency guard
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


class VLLMDriver(LLMDriver):
    """Simple REST driver for local vLLM deployments."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        super().__init__(name, config)
        if "endpoint" not in self.config:
            self.config["endpoint"] = "http://localhost:8000"
        if "model" not in self.config:
            raise DriverError("vLLM driver requires 'model' in configuration")

    def generate(self, prompt: str, *, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if httpx is None:  # pragma: no cover - dependency guard
            raise DriverError("httpx is required to use the vLLM driver")
        endpoint = self.config.get("endpoint", "http://localhost:8000").rstrip("/")
        url = f"{endpoint}/v1/chat/completions"
        headers = {
            "content-type": "application/json",
        }
        user_agent = self.config.get("user_agent")
        if user_agent:
            headers["user-agent"] = str(user_agent)
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.get("temperature", 0.2),
            "max_tokens": self.config.get("max_tokens", 512),
        }
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            payload["messages"].insert(0, {"role": "system", "content": system_prompt})
        if metadata:
            payload["metadata"] = dict(metadata)
        timeout = self.config.get("timeout", 30.0)
        retries = int(self.config.get("max_retries", 0))
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError as exc:  # pragma: no cover - network error path
                last_error = exc
        if last_error:
            raise DriverError(f"vLLM request failed: {last_error}") from last_error
        raise DriverError("vLLM driver failed without exception context")


register_driver(
    "vllm",
    "local_vllm",
    factory=VLLMDriver,
    metadata={"provider": "vllm", "transport": "rest"},
    overwrite=True,
)

