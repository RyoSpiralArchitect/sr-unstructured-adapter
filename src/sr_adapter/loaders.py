"""Utilities for reading various unstructured file formats."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, Tuple

_TEXT_EXTENSIONS = {
    ".csv",
    ".log",
    ".md",
    ".rst",
    ".text",
    ".txt",
}


def read_file_contents(path: Path, mime: str) -> Tuple[str, Dict[str, object]]:
    """Return a textual representation and extra metadata for *path*.

    Parameters
    ----------
    path:
        File to read.
    mime:
        MIME type determined by the caller.
    """

    suffix = path.suffix.lower()
    extra: Dict[str, object] = {}

    if mime == "application/json" or suffix == ".json":
        raw = path.read_text(encoding="utf-8", errors="ignore")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            extra.update({"json_valid": False})
            return raw, extra

        extra.update(
            {
                "json_valid": True,
                "json_top_level_type": type(data).__name__,
            }
        )
        if isinstance(data, dict):
            extra["json_top_level_keys"] = sorted(map(str, data.keys()))
        return json.dumps(data, ensure_ascii=False, indent=2), extra

    if mime.startswith("text/") or suffix in _TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text, extra

    # Fallback for binary files – provide a short base64 preview.
    blob = path.read_bytes()
    extra.update(
        {
            "binary_preview_bytes": min(len(blob), 32),
            "binary_preview_base64": base64.b64encode(blob[:32]).decode("ascii"),
        }
    )
    return "", extra
