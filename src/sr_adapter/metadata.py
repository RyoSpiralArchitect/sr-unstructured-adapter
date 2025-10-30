"""Metadata helpers for the adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
import hashlib
import re

import re


def collect_metadata(path: Path, text: str, mime: str, extra: Dict[str, object]) -> Dict[str, object]:
    """Gather metadata for the given file."""

    stat = path.stat()
    metadata: Dict[str, object] = {
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "mime": mime,
    }

    if text:
        lines = text.splitlines() or [text]
        non_empty = sum(1 for line in lines if line.strip())
        avg_line_length = (
            sum(len(line) for line in lines) / max(len(lines), 1)
        )
        words = re.findall(r"\w+", text)
        metadata.update(
            {
                "line_count": len(lines),
                "non_empty_lines": non_empty,
                "char_count": len(text),
                "word_count": len(words),
                "avg_line_length": round(avg_line_length, 2),
                "text_preview": text[:160],
                "text_sha256": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
            }
        )

    if mime.startswith("text/"):
        metadata.setdefault("encoding", "utf-8")

    metadata.update(extra)

    if "length_bytes" in metadata and "size" in metadata:
        try:
            diff = abs(float(metadata["length_bytes"]) - float(metadata["size"]))
            metadata["length_bytes_mismatch"] = diff > 1
        except Exception:
            metadata.pop("length_bytes_mismatch", None)

    return metadata
