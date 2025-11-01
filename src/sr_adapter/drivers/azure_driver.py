"""Azure OpenAI driver implementation."""

from __future__ import annotations

from typing import Any, Mapping

from .base import DriverError, LLMDriver, register_driver

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

    def generate(self, prompt: str, *, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if httpx is None:  # pragma: no cover - dependency guard
            raise DriverError("httpx is required to use the Azure driver")
        endpoint = self.config["endpoint"].rstrip("/")
        deployment = self.config["deployment"]
        api_version = self.config["api_version"]
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        headers = {
            "api-key": self.config["api_key"],
            "content-type": "application/json",
        }
        user_agent = self.config.get("user_agent")
        if user_agent:
            headers["user-agent"] = str(user_agent)
        system_prompt = self.config.get("system_prompt")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: Mapping[str, Any] = {
            "messages": messages,
            "temperature": self.config.get("temperature", 0.2),
            "max_tokens": self.config.get("max_tokens", 512),
        }
        if metadata:
            payload = dict(payload)
            payload["metadata"] = dict(metadata)
        params = {"api-version": api_version}
        retries = int(self.config.get("max_retries", 0))
        timeout = self.config.get("timeout", 30.0)
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, headers=headers, params=params, json=payload)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError as exc:  # pragma: no cover - network failure path
                last_error = exc
        if last_error:
            raise DriverError(f"Azure request failed: {last_error}") from last_error
        raise DriverError("Azure driver failed without exception context")


register_driver(
    "azure",
    "azure_openai",
    factory=AzureDriver,
    metadata={"provider": "azure", "transport": "rest"},
    overwrite=True,
)
