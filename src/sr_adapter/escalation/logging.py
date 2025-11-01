"""Persistent JSONL logging for escalation decisions."""

from __future__ import annotations

import json
import time
from threading import Lock
from typing import Iterable, Mapping, Optional

from ..schema import Block
from ..settings import EscalationSettings, get_settings

try:  # pragma: no cover - optional type checking import
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover - python <3.11 fallback
    TYPE_CHECKING = False

if TYPE_CHECKING:  # pragma: no cover
    from .policy import SelectionResult
    from ..normalizer.llm_normalizer import NormalizedLLMResult
class EscalationLogger:
    """Append structured events describing the escalation lifecycle."""

    def __init__(self, settings: Optional[EscalationSettings] = None) -> None:
        self._settings = settings or get_settings().escalation
        self._enabled = bool(self._settings.logging_enabled)
        self._path = self._settings.resolved_log_path
        self._feature_version = self._settings.feature_version
        self._lock = Lock()
        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _write(self, payload: Mapping[str, object]) -> None:
        if not self._enabled:
            return
        record = dict(payload)
        record.setdefault("timestamp", time.time())
        record.setdefault("feature_version", self._feature_version)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_selection(
        self,
        recipe: str,
        selection: "SelectionResult" | None,
        blocks: Iterable[Block],
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if not self._enabled or selection is None:
            return
        block_list = list(blocks)
        context = dict(metadata or {})
        for candidate in selection.candidates:
            if candidate.index >= len(block_list):
                continue
            block = block_list[candidate.index]
            entry = {
                "event": "selection",
                "recipe": recipe,
                "block_index": candidate.index,
                "block_id": block.id,
                "score": candidate.score,
                "selected": candidate.selected,
                "confidence": float(block.confidence),
                "type": block.type,
                "features": dict(candidate.features),
                "metadata": context,
            }
            layout_conf = None
            if isinstance(block.attrs, dict):
                layout_conf = block.attrs.get("layout_confidence")
            if layout_conf is not None:
                entry["layout_confidence"] = float(layout_conf)
            self._write(entry)

    def log_result(
        self,
        recipe: str,
        block: Block,
        *,
        index: int,
        candidate_score: float,
        llm_result: "NormalizedLLMResult",
        rank: Optional[int] = None,
    ) -> None:
        if not self._enabled:
            return
        choice_text = ""
        if llm_result.choices:
            choice_text = llm_result.choices[0].text
        entry = {
            "event": "llm_result",
            "recipe": recipe,
            "block_index": index,
            "block_id": block.id,
            "score": candidate_score,
            "rank": rank,
            "confidence": float(block.confidence),
            "type": block.type,
            "llm": {
                "provider": llm_result.provider,
                "model": llm_result.model,
                "usage": dict(llm_result.usage),
            },
            "response_preview": choice_text[:5000],
        }
        self._write(entry)

    def log_failure(
        self,
        recipe: str,
        *,
        reason: str,
        selection: "SelectionResult" | None,
    ) -> None:
        if not self._enabled:
            return
        entry = {
            "event": "llm_failure",
            "recipe": recipe,
            "reason": reason,
            "attempted_blocks": len(selection.indices) if selection else 0,
        }
        self._write(entry)


_LOGGER: EscalationLogger | None = None


def get_escalation_logger(settings: Optional[EscalationSettings] = None) -> EscalationLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = EscalationLogger(settings)
    return _LOGGER


__all__ = ["EscalationLogger", "get_escalation_logger"]
