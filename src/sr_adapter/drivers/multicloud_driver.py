"""Lightweight REST chat drivers for additional providers."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .base import DriverError, register_driver
from .openai_driver import OpenAIDriver
from .protocol import GeminiSSEDecoder, openai_metadata


class JSONChatProxyDriver(OpenAIDriver):
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
        token_field = "max_completion_tokens" if "max_completion_tokens" in self.config else "max_tokens"
        max_tokens = self.config.get(token_field, 512)
        if max_tokens is not None:
            payload[token_field] = max_tokens
        for key in ("top_p", "reasoning_effort", "response_format", "random_seed"):
            if self.config.get(key) is not None:
                payload[key] = self.config[key]
        if metadata:
            payload["metadata"] = openai_metadata(metadata)
        return payload


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
    """Google Gemini generateContent driver; Vertex requires an explicit endpoint."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        if name.lower() == "vertex" and not config.get("endpoint"):
            raise DriverError("Vertex requires an explicit project/location endpoint and authentication headers")
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
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        if self.config.get("system_prompt"):
            payload["systemInstruction"] = {"parts": [{"text": self.config["system_prompt"]}]}
        generation = dict(self.config.get("generation_config") or {})
        generation.setdefault("maxOutputTokens", self.config.get("max_tokens", 512))
        if self.config.get("temperature") is not None:
            generation["temperature"] = self.config["temperature"]
        if self.config.get("top_p") is not None:
            generation["topP"] = self.config["top_p"]
        payload["generationConfig"] = generation
        if self.config.get("safety_settings") is not None:
            payload["safetySettings"] = self.config["safety_settings"]
        return payload

    def _stream_endpoint(self) -> str:
        endpoint = str(self.config.get("stream_endpoint") or self._endpoint())
        if "{model}" in endpoint:
            endpoint = endpoint.format(model=self.config["model"])
        parts = urlsplit(endpoint)
        path = parts.path
        if path.endswith(":generateContent"):
            path = path.removesuffix(":generateContent") + ":streamGenerateContent"
        elif not path.endswith(":streamGenerateContent") and not self.config.get("stream_endpoint"):
            raise DriverError("Google streaming requires a :generateContent endpoint or explicit stream_endpoint")
        query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key != "alt"]
        query.append(("alt", "sse"))
        return urlunsplit((parts.scheme, parts.netloc, path, urlencode(query), parts.fragment))

    def _stream_payload(self, prompt: str, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        return self._build_payload(prompt, metadata)

    def _stream_decoder(self) -> GeminiSSEDecoder:
        count = (self.config.get("generation_config") or {}).get("candidateCount", 1)
        return GeminiSSEDecoder(candidate_count=count)


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
        if not config.get("endpoint"):
            raise DriverError("Bedrock requires an explicit OpenAI-compatible chat completions endpoint")
        super().__init__(
            name,
            config,
            default_endpoint=str(config["endpoint"]),
            provider="aws-bedrock",
            api_key_header=str(config.get("api_key_header", "Authorization")),
            api_key_prefix=str(config.get("api_key_prefix", "Bearer ")),
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
