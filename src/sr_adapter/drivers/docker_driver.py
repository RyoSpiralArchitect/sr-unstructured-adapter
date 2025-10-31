"""Docker hosted model driver."""

from __future__ import annotations

from typing import Any, Mapping

from .base import DriverError, LLMDriver

try:  # pragma: no cover - exercised indirectly when dependency is present
    import httpx
except ImportError:  # pragma: no cover - handled gracefully at runtime
    httpx = None  # type: ignore[assignment]


class DockerDriver(LLMDriver):
    """HTTP driver for models exposed from local Docker containers."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        super().__init__(name, config)
        if "url" not in self.config:
            raise DriverError("Docker driver requires 'url' in configuration")

    def generate(self, prompt: str, *, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if httpx is None:  # pragma: no cover - dependency guard
            raise DriverError("httpx is required to use the Docker driver")
        url = self.config["url"]
        model = self.config.get("model")
        system_prompt = self.config.get("system_prompt")
        payload = {
            "temperature": self.config.get("temperature", 0.2),
            "max_tokens": self.config.get("max_tokens", 512),
        }
        if model:
            payload["model"] = model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload["messages"] = messages
        if metadata:
            payload["metadata"] = dict(metadata)
        try:
            with httpx.Client(timeout=self.config.get("timeout", 30.0)) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            raise DriverError(f"Docker driver request failed: {exc}") from exc
