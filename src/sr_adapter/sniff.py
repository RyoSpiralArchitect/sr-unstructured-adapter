"""Utilities for inferring the best parser for an input file."""

from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path
from typing import Optional

_TEXT_EXTENSIONS = {
    ".txt": "text",
    ".md": "md",
    ".markdown": "md",
    ".log": "text",
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".html": "html",
    ".htm": "html",
}

_MAGIC_SIGNATURES = [
    (b"%PDF", "pdf"),
    (b"PK\x03\x04", "zip"),
]


def _read_prefix(path: Path, size: int = 4096) -> bytes:
    with path.open("rb") as stream:
        return stream.read(size)


def _sniff_magic(prefix: bytes) -> Optional[str]:
    for signature, label in _MAGIC_SIGNATURES:
        if prefix.startswith(signature):
            return label
    return None


def _sniff_json(prefix: bytes) -> bool:
    snippet = prefix.decode("utf-8", errors="ignore").lstrip()
    if not snippet:
        return False
    if snippet[0] in "[{":
        try:
            json.loads(snippet[:200])
            return True
        except json.JSONDecodeError:
            return False
    return False


def _sniff_html(prefix: bytes) -> bool:
    text = prefix.decode("utf-8", errors="ignore").lower()
    return "<html" in text or re.search(r"<!doctype\\s+html", text) is not None


def detect_type(path: str | Path) -> str:
    """Return a symbolic type string for *path*.

    The detection strategy roughly follows magic-number inspection, extension
    heuristics, and finally lightweight content sniffing.
    """

    normalised = Path(path).expanduser().resolve()
    prefix = _read_prefix(normalised)
    magic = _sniff_magic(prefix)
    if magic == "pdf":
        return "pdf"
    if magic == "zip":
        return _detect_from_zip(normalised)

    ext = normalised.suffix.lower()
    if ext in _TEXT_EXTENSIONS:
        return _TEXT_EXTENSIONS[ext]

    if _sniff_json(prefix):
        return "json"
    if _sniff_html(prefix):
        return "html"

    mime, _ = mimetypes.guess_type(str(normalised))
    if mime:
        if mime.startswith("text/"):
            return "text"
        if mime == "application/pdf":
            return "pdf"
        if mime in {"application/json", "application/x-ndjson"}:
            return "json"

    return "text"


def _detect_from_zip(path: Path) -> str:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return "text"

    if any(name.startswith("word/") for name in names):
        return "docx"
    if any(name.startswith("xl/") for name in names):
        return "xlsx"

    return "text"
