"""Hybrid native + learned block refiner for low confidence content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .escalation.features import build_features
from .escalation.model import EscalationModel, load_escalation_model
from .schema import Block, clone_model
from .settings import EscalationSettings, get_settings


@dataclass
class RefinerConfig:
    confidence_threshold: float
    layout_threshold: float


class HybridRefiner:
    """Apply lightweight ML-based clean-up to low-confidence blocks."""

    def __init__(
        self,
        *,
        config: RefinerConfig | None = None,
        model: EscalationModel | None = None,
        settings: EscalationSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings().escalation
        self._config = config or RefinerConfig(
            confidence_threshold=max(0.0, min(1.0, self._settings.min_score)),
            layout_threshold=0.45,
        )
        self._model = model or load_escalation_model(self._settings)

    def _should_refine(self, block: Block) -> bool:
        if block.confidence < self._config.confidence_threshold:
            return True
        layout_conf = 1.0
        if isinstance(block.attrs, dict) and "layout_confidence" in block.attrs:
            try:
                layout_conf = float(block.attrs.get("layout_confidence", 1.0))
            except (TypeError, ValueError):
                layout_conf = 1.0
        return layout_conf < self._config.layout_threshold

    def _refine_text(self, text: str) -> str:
        candidate = text.strip()
        if candidate.isupper() and len(candidate) < 160:
            candidate = candidate.title()
        candidate = candidate.replace("\u3000", " ")
        candidate = " ".join(candidate.split())
        return candidate

    def refine(self, blocks: Sequence[Block]) -> List[Block]:
        refined: List[Block] = []
        for block in blocks:
            if not self._should_refine(block):
                refined.append(block)
                continue
            features = build_features(block)
            score = self._model.score(features, block=block)
            attrs = dict(block.attrs)
            meta = dict(attrs.get("ml_refine", {}))
            meta.update(
                {
                    "score": float(score),
                    "original_confidence": float(block.confidence),
                }
            )
            attrs["ml_refine"] = meta
            updated_text = self._refine_text(block.text or "")
            new_confidence = max(block.confidence, min(1.0, score))
            refined.append(
                clone_model(
                    block,
                    text=updated_text,
                    confidence=new_confidence,
                    attrs=attrs,
                )
            )
        return refined


__all__ = ["HybridRefiner", "RefinerConfig"]
