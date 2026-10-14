"""Normalize LLM driver responses into a shared schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional


CompletionStatus = Literal["complete", "truncated", "blocked", "tool_call", "empty", "unknown", "mixed"]


@dataclass
class NormalizedChoice:
    """Normalized representation of an LLM choice."""

    text: str
    finish_reason: Optional[str]
    metadata: Dict[str, Any]
    status: CompletionStatus = "unknown"


@dataclass
class NormalizedLLMResult:
    """Standard payload returned by the adapter after normalization."""

    provider: str
    model: Optional[str]
    prompt: str
    choices: List[NormalizedChoice]
    usage: Dict[str, Any]
    raw: Mapping[str, Any]
    status: CompletionStatus = "unknown"

    @property
    def is_complete(self) -> bool:
        """Whether every returned choice is finished, nonempty text.

        This is a protocol check, not a claim about factual or extraction quality.
        """
        return self.status == "complete"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _records(value: Any, name: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"LLM {name} must be an array of objects")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"LLM {name} must contain objects")
    return value


def _status(choice: NormalizedChoice) -> CompletionStatus:
    reason = choice.finish_reason
    if reason is not None and not isinstance(reason, str):
        raise ValueError("LLM finish reason must be a string or null")
    if choice.metadata.get("refusal") or reason in {
        "content_filter", "refusal", "SAFETY", "RECITATION", "LANGUAGE",
        "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "ESCALATION",
        "PUP_LIMITED_DISABLED",
    }:
        return "blocked"
    if reason in {"length", "model_length", "max_tokens", "MAX_TOKENS", "model_context_window_exceeded"}:
        return "truncated"
    if reason in {"tool_calls", "function_call", "tool_use"} or any(
        choice.metadata.get(key) for key in ("tool_calls", "function_call")
    ):
        return "tool_call"
    if reason not in {"stop", "end_turn", "stop_sequence", "STOP"}:
        return "unknown"
    return "complete" if choice.text.strip() else "empty"


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
            for choice in _records(raw.get("choices"), "choices"):
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
            content = _records(raw.get("content"), "content")
            metadata = {"role": raw.get("role", "assistant")}
            tools = [part for part in content if part.get("type") == "tool_use"]
            if tools:
                metadata["tool_calls"] = tools
            choices.append(NormalizedChoice(
                text=_text(content),
                finish_reason=raw.get("stop_reason"),
                metadata=metadata,
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
        elif "candidates" in raw or "promptFeedback" in raw or "usageMetadata" in raw:
            for candidate in _records(raw.get("candidates"), "candidates"):
                content = _mapping(candidate.get("content"))
                metadata = {key: candidate[key] for key in ("index", "safetyRatings", "citationMetadata") if key in candidate}
                parts = _records(content.get("parts"), "parts")
                tools = [part["functionCall"] for part in parts if "functionCall" in part]
                if tools:
                    metadata["tool_calls"] = tools
                if content.get("role"):
                    metadata["role"] = content["role"]
                choices.append(NormalizedChoice(
                    text=_text(parts),
                    finish_reason=candidate.get("finishReason"),
                    metadata=metadata,
                ))
            usage = dict(_mapping(raw.get("usageMetadata")))
            for source, target in (("promptTokenCount", "prompt_tokens"), ("candidatesTokenCount", "completion_tokens"), ("totalTokenCount", "total_tokens")):
                if source in usage:
                    usage[target] = usage[source]
        else:
            raise ValueError("Unsupported LLM response format")

        for choice in choices:
            choice.status = _status(choice)
        statuses = {choice.status for choice in choices}
        status: CompletionStatus = next(iter(statuses)) if len(statuses) == 1 else "mixed" if statuses else "empty"
        feedback = raw.get("promptFeedback")
        if feedback is not None and not isinstance(feedback, Mapping):
            raise ValueError("LLM prompt feedback must be an object")
        if _mapping(feedback).get("blockReason") not in (None, "", "BLOCK_REASON_UNSPECIFIED"):
            status = "blocked"

        return NormalizedLLMResult(
            provider=provider,
            model=str(model) if model is not None else None,
            prompt=prompt,
            choices=choices,
            usage=usage,
            raw=raw,
            status=status,
        )
