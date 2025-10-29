"""Export helpers for persisting processed documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import Document


def write_jsonl(documents: Iterable[Document], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            try:
                payload = document.model_dump()
            except AttributeError:  # pragma: no cover - pydantic v1 fallback
                payload = document.dict()
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
