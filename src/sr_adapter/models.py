"""Lightweight data structures used by the adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class Payload:
    """Normalized representation of an unstructured file."""

    source: Path
    mime: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return {
            "source": str(self.source),
            "mime": self.mime,
            "text": self.text,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Payload":
        """Instantiate a :class:`Payload` from a dictionary."""

        return cls(
            source=Path(data["source"]),
            mime=data["mime"],
            text=data.get("text", ""),
            meta=dict(data.get("meta", {})),
        )
