# SPDX-License-Identifier: AGPL-3.0-or-later
"""Language detection utilities for block-level enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence

from langdetect import DetectorFactory, LangDetectException, detect_langs


# Stabilise detection output for reproducibility across runs.
DetectorFactory.seed = 0


@dataclass(frozen=True)
class LanguageGuess:
    """Language prediction with probability."""

    lang: str
    prob: float


_MIN_SAMPLE_CHARS = 12
_DEFAULT_MAX_CANDIDATES = 3
_DEFAULT_MIN_PROB = 0.12


def _normalise_text(sample: str) -> str:
    text = sample.strip()
    if len(text) > 2000:
        # langdetect has quadratic behaviour on extremely long strings – trim.
        return text[:2000]
    return text


@lru_cache(maxsize=1024)
def _detect_language_inner(text: str) -> Sequence[LanguageGuess]:
    if len(text) < _MIN_SAMPLE_CHARS:
        return ()
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return ()
    guesses: List[LanguageGuess] = []
    for candidate in candidates:
        guesses.append(LanguageGuess(lang=candidate.lang, prob=float(candidate.prob)))
    return tuple(guesses)


def detect_language_guesses(
    text: str,
    *,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    min_probability: float = _DEFAULT_MIN_PROB,
) -> List[LanguageGuess]:
    """Return language guesses sorted by probability.

    Parameters
    ----------
    text:
        Sample text to analyse.
    max_candidates:
        Maximum number of guesses to return.
    min_probability:
        Ignore guesses whose probability is below this threshold.
    """

    sample = _normalise_text(text)
    guesses = _detect_language_inner(sample)
    filtered: List[LanguageGuess] = []
    for guess in guesses:
        if guess.prob < min_probability:
            continue
        filtered.append(guess)
        if len(filtered) >= max_candidates:
            break
    return filtered


def merge_language_hints(*groups: Iterable[str]) -> List[str]:
    """Merge multiple language sequences while preserving order."""

    seen: set[str] = set()
    merged: List[str] = []
    for group in groups:
        for lang in group:
            if not lang:
                continue
            normalised = lang.strip()
            if not normalised or normalised in seen:
                continue
            seen.add(normalised)
            merged.append(normalised)
    return merged

