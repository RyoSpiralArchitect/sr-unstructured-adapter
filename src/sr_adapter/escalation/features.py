"""Feature extraction helpers for the escalation meta-model."""

from __future__ import annotations

import math
from typing import Dict

from ..schema import Block


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(number) or math.isinf(number):
        return float(default)
    return float(number)


def build_features(block: Block) -> Dict[str, float]:
    """Return a dense feature dictionary derived from *block*."""

    text = block.text or ""
    tokens = text.split()
    lines = text.splitlines() or [text]

    features: Dict[str, float] = {
        "confidence": _safe_float(block.confidence, 0.0),
        "text_length": float(len(text)),
        "word_count": float(len(tokens)),
        "line_count": float(len(lines)),
        "has_spans": 1.0 if block.spans else 0.0,
    }

    layout_conf = block.attrs.get("layout_confidence") if isinstance(block.attrs, dict) else None
    if layout_conf is not None:
        features["layout_confidence"] = _safe_float(layout_conf, features["confidence"])
    else:
        features["layout_confidence"] = 0.0

    if lines:
        avg_line = sum(len(line) for line in lines) / max(len(lines), 1)
        features["avg_line_length"] = float(avg_line)

    if tokens:
        avg_token = sum(len(token) for token in tokens) / max(len(tokens), 1)
        features["avg_token_length"] = float(avg_token)

    alpha_chars = [ch for ch in text if ch.isalpha()]
    upper_chars = [ch for ch in alpha_chars if ch.isupper()]
    digit_chars = [ch for ch in text if ch.isdigit()]
    if alpha_chars:
        features["uppercase_ratio"] = float(len(upper_chars) / len(alpha_chars))
    if text:
        features["digit_ratio"] = float(len(digit_chars) / len(text))

    provenance = block.prov
    if provenance and getattr(provenance, "page", None) is not None:
        features["page_index"] = float(provenance.page or 0)

    features[f"type={block.type}"] = 1.0

    language_scores = {}
    if isinstance(block.attrs, dict):
        language_scores = block.attrs.get("language_scores", {}) or {}
    if isinstance(language_scores, dict):
        for lang, score in list(language_scores.items())[:5]:
            features[f"lang={lang}"] = _safe_float(score, 0.0)

    return features


__all__ = ["build_features"]
