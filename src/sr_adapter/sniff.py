"""Simple file type detection used by the conversion pipeline."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Dict

mimetypes.init()

_MAGIC_SIGNATURES: Dict[bytes, str] = {
    b"%PDF-": "pdf",
    b"PK\x03\x04": "zip",  # differentiates via file suffix
}

_EXTENSION_MAP: Dict[str, str] = {
    ".txt": "text",
    ".log": "text",
    ".md": "md",
    ".markdown": "md",
    ".rst": "text",
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".pptx": "pptx",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".properties": "ini",
    ".xml": "text",
    ".rtf": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".eml": "eml",
    ".ics": "ics",
}


def _sniff_magic(path: Path) -> str | None:
    try:
        header = path.read_bytes()[:4]
    except OSError:
        return None
    for signature, kind in _MAGIC_SIGNATURES.items():
        if header.startswith(signature):
            return kind
    return None


def detect_type(path: str | Path) -> str:
    """Return a lightweight type identifier for *path*."""

    path = Path(path)
    magic = _sniff_magic(path)
    if magic == "pdf":
        return "pdf"
    if magic == "zip":
        suffix = path.suffix.lower()
        if suffix == ".docx":
            return "docx"
        if suffix == ".xlsx":
            return "xlsx"
        if suffix == ".pptx":
            return "pptx"

    suffix = path.suffix.lower()
    if suffix in _EXTENSION_MAP:
        return _EXTENSION_MAP[suffix]

    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if mime.startswith("text/"):
            return "text"
        if mime in {"application/json", "application/xml"}:
            return "text"

    return "text"

