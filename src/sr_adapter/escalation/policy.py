"""Meta-model driven escalation policy orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..schema import Block
from ..settings import EscalationSettings, get_settings
from .features import build_features
from .model import EscalationModel, load_escalation_model


@dataclass
class SelectionCandidate:
    index: int
    score: float
    features: Dict[str, float]
    selected: bool = False
    rank: Optional[int] = None


@dataclass
class SelectionResult:
    indices: List[int]
    candidates: List[SelectionCandidate] = field(default_factory=list)
    threshold: float = 0.0
    limit: Optional[int] = None
    _index_map: Dict[int, SelectionCandidate] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index_map = {candidate.index: candidate for candidate in self.candidates}

    def _refresh_index_map(self) -> None:
        if len(self._index_map) != len(self.candidates):
            self._index_map = {candidate.index: candidate for candidate in self.candidates}

    def find(self, index: int) -> Optional[SelectionCandidate]:
        if not self.candidates:
            return None
        candidate = self._index_map.get(index)
        if candidate is not None:
            return candidate
        self._refresh_index_map()
        return self._index_map.get(index)


class EscalationPolicyEngine:
    """Apply a learned escalation model to incoming blocks."""

    def __init__(
        self,
        model: Optional[EscalationModel] = None,
        *,
        settings: Optional[EscalationSettings] = None,
    ) -> None:
        self._settings = settings or get_settings().escalation
        self._model = model or load_escalation_model(self._settings)
        min_score = self._settings.min_score or self._model.threshold
        self._threshold = float(min_score)
        self._last: Optional[SelectionResult] = None

    @property
    def threshold(self) -> float:
        return self._threshold

    def evaluate(
        self,
        blocks: Sequence[Block],
        *,
        max_confidence: Optional[float] = None,
        allow_types: Sequence[str] | None = None,
        limit: Optional[int] = None,
    ) -> SelectionResult:
        if isinstance(limit, int) and limit <= 0:
            result = SelectionResult(indices=[], candidates=[], threshold=self._threshold, limit=limit)
            self._last = result
            return result

        allow_set = {t for t in allow_types or ()}
        enforce_types = bool(allow_set)

        candidates: List[SelectionCandidate] = []
        for idx, block in enumerate(blocks):
            if max_confidence is not None and block.confidence > max_confidence:
                continue
            if enforce_types and block.type not in allow_set:
                continue
            features = build_features(block)
            score = self._model.score(features, block=block)
            candidates.append(SelectionCandidate(index=idx, score=score, features=features))

        ordered = sorted(candidates, key=lambda cand: cand.score, reverse=True)
        selected: List[int] = []
        for rank, candidate in enumerate(ordered, start=1):
            if candidate.score < self._threshold:
                continue
            candidate.selected = True
            candidate.rank = rank
            selected.append(candidate.index)
            if isinstance(limit, int) and limit > 0 and len(selected) >= limit:
                break

        result = SelectionResult(indices=selected, candidates=candidates, threshold=self._threshold, limit=limit)
        self._last = result
        return result

    def select(
        self,
        blocks: Sequence[Block],
        *,
        max_confidence: Optional[float] = None,
        allow_types: Sequence[str] | None = None,
        limit: Optional[int] = None,
    ) -> List[int]:
        return self.evaluate(
            blocks,
            max_confidence=max_confidence,
            allow_types=allow_types,
            limit=limit,
        ).indices

    def last(self) -> Optional[SelectionResult]:
        return self._last


_POLICY: EscalationPolicyEngine | None = None


def get_escalation_policy() -> EscalationPolicyEngine:
    global _POLICY
    if _POLICY is None:
        _POLICY = EscalationPolicyEngine()
    return _POLICY


def reset_escalation_policy() -> None:
    global _POLICY
    _POLICY = None


__all__ = [
    "EscalationPolicyEngine",
    "SelectionCandidate",
    "SelectionResult",
    "get_escalation_policy",
    "reset_escalation_policy",
]
