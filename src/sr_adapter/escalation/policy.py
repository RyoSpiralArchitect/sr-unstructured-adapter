"""Meta-model driven escalation policy orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..confidence import semantic_confidence, structural_confidence
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
        max_semantic_confidence: Optional[float] = None,
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
        semantic_forced: List[int] = []
        for idx, block in enumerate(blocks):
            if enforce_types and block.type not in allow_set:
                continue

            struct_score = structural_confidence(block)
            struct_low = max_confidence is not None and struct_score <= max_confidence

            semantic_low = False
            semantic_score = None
            if max_semantic_confidence is not None:
                semantic_score = semantic_confidence(block)
                if semantic_score is not None and semantic_score <= max_semantic_confidence:
                    semantic_low = True

            if max_confidence is not None or max_semantic_confidence is not None:
                if not (struct_low or semantic_low):
                    continue
            features = build_features(block)
            features.setdefault("confidence_structural", float(struct_score))
            if semantic_score is not None:
                features.setdefault("semantic_confidence", float(semantic_score))
            score = self._model.score(features, block=block)
            candidates.append(SelectionCandidate(index=idx, score=score, features=features))
            if semantic_low:
                semantic_forced.append(idx)

        ordered = sorted(candidates, key=lambda cand: cand.score, reverse=True)
        selected: List[int] = []
        candidate_map = {candidate.index: candidate for candidate in candidates}
        rank = 1

        for index in semantic_forced:
            selected.append(index)
            candidate = candidate_map.get(index)
            if candidate is not None:
                candidate.selected = True
                candidate.rank = rank
            rank += 1
            if isinstance(limit, int) and limit > 0 and len(selected) >= limit:
                break

        if not (isinstance(limit, int) and limit > 0 and len(selected) >= limit):
            for candidate in ordered:
                if candidate.index in candidate_map and candidate.index in selected:
                    continue
                if candidate.score < self._threshold:
                    continue
                candidate.selected = True
                candidate.rank = rank
                selected.append(candidate.index)
                rank += 1
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
        max_semantic_confidence: Optional[float] = None,
        allow_types: Sequence[str] | None = None,
        limit: Optional[int] = None,
    ) -> List[int]:
        return self.evaluate(
            blocks,
            max_confidence=max_confidence,
            max_semantic_confidence=max_semantic_confidence,
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
