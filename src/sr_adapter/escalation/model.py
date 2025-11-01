"""Escalation model abstractions and loaders."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

from ..schema import Block
from ..settings import EscalationSettings, get_settings


class EscalationModel:
    """Base interface for learned escalation policies."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        self._threshold = float(threshold)

    @property
    def threshold(self) -> float:
        return self._threshold

    def score(self, features: Mapping[str, float], *, block: Optional[Block] = None) -> float:
        raise NotImplementedError


class LinearEscalationModel(EscalationModel):
    """Simple linear model with optional logistic activation."""

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        bias: float = 0.0,
        logistic: bool = True,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(threshold=threshold)
        self._weights = dict(weights)
        self._bias = float(bias)
        self._logistic = bool(logistic)

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = max(min(value, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-clipped))

    def score(self, features: Mapping[str, float], *, block: Optional[Block] = None) -> float:
        activation = self._bias
        for name, weight in self._weights.items():
            activation += weight * float(features.get(name, 0.0))
        return self._sigmoid(activation) if self._logistic else activation


def _load_json_model(path: Path, *, fallback_threshold: float) -> LinearEscalationModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    weights = data.get("weights") or {}
    bias = data.get("bias")
    if isinstance(weights, Mapping) and "bias" in weights and bias is None:
        bias = weights.get("bias")
        weights = {key: value for key, value in weights.items() if key != "bias"}
    logistic = bool(data.get("logistic", True))
    threshold = float(data.get("threshold", fallback_threshold))
    numeric_weights = {key: float(value) for key, value in dict(weights).items()}
    return LinearEscalationModel(numeric_weights, bias=float(bias or 0.0), logistic=logistic, threshold=threshold)


def load_escalation_model(settings: Optional[EscalationSettings] = None) -> EscalationModel:
    """Load the configured escalation model or fall back to defaults."""

    if settings is None:
        settings = get_settings().escalation

    model_path = settings.resolved_model_path
    if model_path and model_path.exists():
        try:
            return _load_json_model(model_path, fallback_threshold=settings.min_score)
        except Exception:
            pass

    weights = dict(settings.default_weights)
    bias = float(weights.pop("bias", 0.0))
    return LinearEscalationModel(weights, bias=bias, logistic=True, threshold=settings.min_score)


__all__ = ["EscalationModel", "LinearEscalationModel", "load_escalation_model"]
