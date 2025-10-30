# SPDX-License-Identifier: AGPL-3.0-or-later
"""Metadata helpers for the adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import re


def collect_metadata(path: Path, text: str, mime: str, extra: Dict[str, object]) -> Dict[str, object]:
    """Gather metadata for the given file."""

    stat = path.stat()
    metadata: Dict[str, object] = {
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }

    if text:
        metadata.update(
            {
                "line_count": text.count("\n") + 1,
                "char_count": len(text),
                "word_count": len([w for w in re.split(r"\s+", text.strip()) if w]),
            }
        )

    if mime.startswith("text/"):
        metadata.setdefault("encoding", "utf-8")

    metadata.update(extra)
    return metadata
