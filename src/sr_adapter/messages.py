"""Helpers for producing chat-friendly message payloads."""

from __future__ import annotations

from typing import Dict, Iterator, List

from .models import Payload


def _iter_chunks(text: str, chunk_size: int) -> Iterator[str]:
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def to_llm_messages(payload: Payload | Dict[str, object], *, chunk_size: int = 2000) -> List[Dict[str, str]]:
    """Convert a payload into chat messages that respect chunking."""

    if isinstance(payload, Payload):
        source = str(payload.source)
        mime = payload.mime
        text = payload.text
    else:
        source = str(payload.get("source", ""))
        mime = str(payload.get("mime", ""))
        text = str(payload.get("text", ""))

    if not text:
        return [
            {
                "role": "user",
                "content": f"[{mime}] {source}\n(no textual preview available)",
            }
        ]

    messages: List[Dict[str, str]] = []
    multi_part = len(text) > chunk_size
    for index, chunk in enumerate(_iter_chunks(text, chunk_size), start=1):
        header = f"[{mime}] {source}"
        if multi_part:
            header = f"{header} (part {index})"
        messages.append({"role": "user", "content": f"{header}\n{chunk}"})
    return messages
