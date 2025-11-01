"""Normalize LLM driver responses into a shared schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class NormalizedChoice:
    """Normalized representation of an LLM choice."""

    text: str
    finish_reason: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class NormalizedLLMResult:
    """Standard payload returned by the adapter after normalization."""

    provider: str
    model: Optional[str]
    prompt: str
    choices: List[NormalizedChoice]
    usage: Dict[str, Any]
    raw: Mapping[str, Any]


class LLMNormalizer:
    """Convert vendor-specific responses into :class:`NormalizedLLMResult`."""

    def normalize(self, provider: str, raw: Mapping[str, Any], *, prompt: str) -> NormalizedLLMResult:
        return self._normalize_openai_like(provider, raw, prompt)

    def _normalize_openai_like(
        self, provider: str, raw: Mapping[str, Any], prompt: str
    ) -> NormalizedLLMResult:
        choices: List[NormalizedChoice] = []
        for choice in raw.get("choices", []):
            message = choice.get("message") or {}
            text = message.get("content") or choice.get("text") or ""
            metadata: Dict[str, Any] = {}
            if "index" in choice:
                metadata["index"] = choice["index"]
            if message.get("role"):
                metadata["role"] = message.get("role")
            for key in ("logprobs", "delta", "content_filter_results"):
                if key in choice:
                    metadata[key] = choice[key]
            finish_reason = choice.get("finish_reason")
            choices.append(NormalizedChoice(text=text, finish_reason=finish_reason, metadata=metadata))
        model = raw.get("model") or raw.get("deployment_id") or raw.get("id")
        usage = dict(raw.get("usage", {}))
        return NormalizedLLMResult(
            provider=provider,
            model=model,
            prompt=prompt,
            choices=choices,
            usage=usage,
            raw=raw,
        )
