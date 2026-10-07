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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        part["text"]
        for part in content
        if isinstance(part, Mapping)
        and isinstance(part.get("text"), str)
        and not part.get("thought")
        and part.get("type", "text") in {"text", "output_text"}
    )


class LLMNormalizer:
    """Normalize by response shape so registered aliases behave identically."""

    def normalize(self, provider: str, raw: Mapping[str, Any], *, prompt: str) -> NormalizedLLMResult:
        if not isinstance(raw, Mapping):
            raise ValueError("LLM response must be an object")
        if raw.get("error") is not None or raw.get("type") == "error":
            raise ValueError("LLM response contains a provider error")

        choices: List[NormalizedChoice] = []
        usage = dict(_mapping(raw.get("usage")))
        model = raw.get("model") or raw.get("modelVersion") or raw.get("deployment_id")
        if "choices" in raw:
            for choice in raw.get("choices") or []:
                if not isinstance(choice, Mapping):
                    raise ValueError("LLM choices must contain objects")
                message = _mapping(choice.get("message") or choice.get("delta"))
                metadata: Dict[str, Any] = {}
                if "index" in choice:
                    metadata["index"] = choice["index"]
                for key in ("role", "refusal", "tool_calls", "function_call"):
                    if message.get(key) is not None:
                        metadata[key] = message[key]
                for key in ("logprobs", "delta", "content_filter_results"):
                    if key in choice:
                        metadata[key] = choice[key]
                choices.append(NormalizedChoice(
                    text=_text(message.get("content")) or _text(choice.get("text")),
                    finish_reason=choice.get("finish_reason"),
                    metadata=metadata,
                ))
        elif "content" in raw and (raw.get("type") == "message" or "stop_reason" in raw):
            choices.append(NormalizedChoice(
                text=_text(raw.get("content")),
                finish_reason=raw.get("stop_reason"),
                metadata={"role": raw.get("role", "assistant")},
            ))
            input_tokens = sum(
                usage.get(key, 0) or 0
                for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            )
            if "input_tokens" in usage:
                usage["prompt_tokens"] = input_tokens
            if "output_tokens" in usage:
                usage["completion_tokens"] = usage["output_tokens"]
            if "input_tokens" in usage and "output_tokens" in usage:
                usage["total_tokens"] = input_tokens + (usage["output_tokens"] or 0)
        elif "candidates" in raw or "promptFeedback" in raw:
            for candidate in raw.get("candidates") or []:
                if not isinstance(candidate, Mapping):
                    raise ValueError("LLM candidates must contain objects")
                content = _mapping(candidate.get("content"))
                metadata = {key: candidate[key] for key in ("index", "safetyRatings", "citationMetadata") if key in candidate}
                if content.get("role"):
                    metadata["role"] = content["role"]
                choices.append(NormalizedChoice(
                    text=_text(content.get("parts")),
                    finish_reason=candidate.get("finishReason"),
                    metadata=metadata,
                ))
            usage = dict(_mapping(raw.get("usageMetadata")))
            for source, target in (("promptTokenCount", "prompt_tokens"), ("candidatesTokenCount", "completion_tokens"), ("totalTokenCount", "total_tokens")):
                if source in usage:
                    usage[target] = usage[source]
        else:
            raise ValueError("Unsupported LLM response format")

        return NormalizedLLMResult(
            provider=provider,
            model=str(model) if model is not None else None,
            prompt=prompt,
            choices=choices,
            usage=usage,
            raw=raw,
        )
