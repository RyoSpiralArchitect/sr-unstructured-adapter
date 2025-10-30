"""Helpers for producing chat-friendly message payloads."""

from __future__ import annotations

from typing import Dict, Iterator, List

from .models import Payload


def _iter_chunks(text: str, chunk_size: int) -> Iterator[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    length = len(text)
    start = 0
    while start < length:
        original_start = start
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            chosen = None
            for sep in ("\n\n", "\n", " "):
                idx = window.rfind(sep)
                if idx != -1 and idx >= int(chunk_size * 0.5):
                    chosen = start + idx
                    break
            if chosen is not None and chosen > start:
                end = chosen
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        start = end
        if start <= original_start:
            start = original_start + chunk_size
        while start < length and text[start] in {"\n", " "}:
            start += 1


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

    chunks = list(_iter_chunks(text, chunk_size))
    if not chunks:
        return [
            {
                "role": "user",
                "content": f"[{mime}] {source}\n(text available but empty after trimming)",
            }
        ]

    total = len(chunks)
    messages: List[Dict[str, str]] = []
    for index, chunk in enumerate(chunks, start=1):
        header = f"[{mime}] {source}"
        if total > 1:
            header = f"{header} (part {index}/{total})"
        messages.append({"role": "user", "content": f"{header}\n{chunk}"})
    return messages
