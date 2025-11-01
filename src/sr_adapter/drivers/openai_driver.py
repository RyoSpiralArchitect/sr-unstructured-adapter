"""OpenAI Chat Completions driver."""

from __future__ import annotations

from typing import Any, Mapping

from .base import DriverError, LLMDriver, register_driver

try:  # pragma: no cover - optional dependency guard
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


class OpenAIDriver(LLMDriver):
    """Minimal OpenAI client using the chat completions REST API."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        super().__init__(name, config)
        for key in ("api_key", "model"):
            if key not in self.config:
                raise DriverError(f"OpenAI driver requires '{key}' in configuration")

    def generate(self, prompt: str, *, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if httpx is None:  # pragma: no cover - dependency guard
            raise DriverError("httpx is required to use the OpenAI driver")
        endpoint = self.config.get("endpoint", "https://api.openai.com/v1").rstrip("/")
        url = f"{endpoint}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "content-type": "application/json",
        }
        user_agent = self.config.get("user_agent")
        if user_agent:
            headers["user-agent"] = str(user_agent)
        messages = []
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config.get("temperature", 0.2),
            "max_tokens": self.config.get("max_tokens", 512),
        }
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
            raise DriverError(f"OpenAI request failed: {last_error}") from last_error
        raise DriverError("OpenAI driver failed without exception context")


register_driver(
    "openai",
    factory=OpenAIDriver,
    metadata={"provider": "openai", "transport": "rest"},
    overwrite=True,
)

