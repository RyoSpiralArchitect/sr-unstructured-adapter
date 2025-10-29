"""Metadata helpers for the adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def _count_lines(text: str) -> int:
    """Return the number of logical lines in *text*."""

    if not text:
        return 0

    line_count = text.count("\n")
    if not text.endswith("\n"):
        line_count += 1
    return line_count


def _count_words(text: str) -> int:
    """Return the number of whitespace-delimited tokens in *text*."""

    count = 0
    in_word = False
    for char in text:
        if char.isspace():
            if in_word:
                count += 1
                in_word = False
        else:
            in_word = True
    if in_word:
        count += 1
    return count


def collect_metadata(
    path: Path,
    text: Optional[str],
    mime: str,
    extra: Dict[str, object],
) -> Dict[str, object]:
    """Gather metadata for the given file."""

    stat = path.stat()
    metadata: Dict[str, object] = {
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }

    treat_as_text = mime.startswith("text/") or bool(extra.get("extracted_as_text"))

    if text is not None:
        metadata.update(
            {
                "line_count": _count_lines(text),
                "char_count": len(text),
            }
        )

        if treat_as_text:
            metadata["word_count"] = _count_words(text)

    if treat_as_text:
        metadata.setdefault("encoding", "utf-8")

    metadata.update(extra)
    return metadata
