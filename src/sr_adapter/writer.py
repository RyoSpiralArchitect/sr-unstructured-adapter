# SPDX-License-Identifier: AGPL-3.0-or-later
"""Output helpers for persisting pipeline results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import Document


def write_jsonl(documents: Iterable[Document], destination: str | Path) -> None:
    """Write *documents* to *destination* as JSON Lines."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            if hasattr(document, "model_dump_json"):
                payload = document.model_dump_json(ensure_ascii=False)  # type: ignore[attr-defined]
            elif hasattr(document, "json"):
                payload = document.json(ensure_ascii=False)  # type: ignore[call-arg]
            elif hasattr(document, "model_dump"):
                payload = json.dumps(document.model_dump(), ensure_ascii=False)  # type: ignore[attr-defined]
            else:
                payload = json.dumps(getattr(document, "__dict__", {}), ensure_ascii=False)
            handle.write(payload)
            handle.write("\n")

