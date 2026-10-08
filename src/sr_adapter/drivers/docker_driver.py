"""Docker hosted model driver."""

from __future__ import annotations

from typing import Any, Mapping

from .base import DriverError, LLMDriver, register_driver
from .openai_driver import OpenAIDriver


class DockerDriver(OpenAIDriver):
    """OpenAI-compatible HTTP models hosted locally, with optional authentication."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        LLMDriver.__init__(self, name, config)
        if not self.config.get("url"):
            raise DriverError("Docker driver requires 'url' in configuration")

    def _endpoint(self) -> str:
        return str(self.config["url"])

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.config.get("api_key"):
            headers["Authorization"] = f"Bearer {self.config['api_key']}"
        if self.config.get("user_agent"):
            headers["user-agent"] = str(self.config["user_agent"])
        return headers

    def _build_payload(self, prompt: str, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        messages = []
        if self.config.get("system_prompt"):
            messages.append({"role": "system", "content": self.config["system_prompt"]})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": self.config.get("temperature", 0.2),
            "max_tokens": self.config.get("max_tokens", 512),
        }
        if self.config.get("model"):
            payload["model"] = self.config["model"]
        return payload


register_driver(
    "docker",
    "http",
    factory=DockerDriver,
    metadata={"provider": "generic", "transport": "http"},
    overwrite=True,
)
