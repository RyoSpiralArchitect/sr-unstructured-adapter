"""Output helpers for persisting pipeline results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, TextIO

import sys

from .schema import Document


def _serialise(document: object) -> str:
    if hasattr(document, "model_dump_json"):
        return document.model_dump_json(ensure_ascii=False)  # type: ignore[attr-defined]
    if hasattr(document, "json"):
        return document.json(ensure_ascii=False)  # type: ignore[call-arg]
    if hasattr(document, "model_dump"):
        return json.dumps(document.model_dump(), ensure_ascii=False)  # type: ignore[attr-defined]
    if isinstance(document, Mapping):
        return json.dumps(document, ensure_ascii=False)
    return json.dumps(getattr(document, "__dict__", {}), ensure_ascii=False)


def write_jsonl(documents: Iterable[Document | Mapping[str, object]], destination: str | Path | TextIO) -> None:
    """Write *documents* to *destination* as JSON Lines.

    ``destination`` can be a filesystem path, ``"-"`` to indicate ``stdout``, or
    any text IO handle. The writer will ensure parent directories exist when a
    path is provided and will avoid closing file-like objects it did not open.
    """

    handle: TextIO
    must_close = False

    if isinstance(destination, (str, Path)):
        if str(destination) == "-":
            handle = sys.stdout
        else:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w", encoding="utf-8")
            must_close = True
    elif hasattr(destination, "write"):
        handle = destination  # type: ignore[assignment]
    else:
        raise TypeError("destination must be a path, '-', or a text IO handle")

    try:
        for document in documents:
            handle.write(_serialise(document))
            handle.write("\n")
    finally:
        if must_close:
            handle.close()

