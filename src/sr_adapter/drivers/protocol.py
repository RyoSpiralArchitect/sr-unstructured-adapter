"""Small protocol helpers shared by the HTTP drivers."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .base import DriverError


def response_object(data: Any) -> Mapping[str, Any]:
    """Reject malformed responses before recording a successful request."""

    if not isinstance(data, Mapping):
        raise DriverError("LLM response must be a JSON object")
    if data.get("error") is not None or data.get("type") == "error":
        # Provider error messages may echo request data. Keep exceptions safe to log.
        raise DriverError("LLM provider returned an error response")
    return data


def openai_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Encode structured adapter metadata within Chat Completions limits."""

    result: dict[str, str] = {}
    for key, value in metadata.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if len(result) >= 16 or len(str(key)) > 64 or len(text) > 512:
            # The full adapter metadata is retained locally on the escalation.
            continue
        result[str(key)] = text
    return result


def error_description(exc: Exception) -> str:
    """Describe transport failures without exposing credential-bearing URLs."""

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return f"HTTP {status}"
    if isinstance(exc, DriverError):
        return str(exc)
    return type(exc).__name__


class SSEDecoder:
    """Decode JSON SSE events, including multiline data and provider errors."""

    def __init__(self) -> None:
        self._data: list[str] = []
        self.finished = False

    def feed(self, line: str) -> Mapping[str, Any] | None:
        if line:
            field, _, value = line.partition(":")
            if field == "data":
                self._data.append(value.removeprefix(" "))
            return None
        if not self._data:
            return None
        data = "\n".join(self._data)
        self._data.clear()
        if data.strip() == "[DONE]":
            self.finished = True
            return None
        try:
            payload = response_object(json.loads(data))
        except json.JSONDecodeError as exc:
            raise DriverError("LLM stream contained malformed JSON") from exc
        if payload.get("type") == "message_stop":
            self.finished = True
        return payload

    def ensure_complete(self) -> None:
        if not self.finished:
            raise DriverError("LLM stream ended before its completion event")
