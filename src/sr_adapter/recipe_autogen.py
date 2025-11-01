# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate regex based recipes from a handful of labelled examples."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


def _classify(char: str) -> str:
    if char.isdigit():
        return "digit"
    if char.isspace():
        return "space"
    if char.isalpha():
        if char.isupper():
            return "upper"
        if char.islower():
            return "lower"
        return "alpha"
    if char.isalnum():
        return "alnum"
    return "punct"


def _group_chars(text: str) -> List[tuple[str, str]]:
    groups: List[tuple[str, str]] = []
    if not text:
        return groups
    current_kind = _classify(text[0])
    buffer = [text[0]]
    for char in text[1:]:
        kind = _classify(char)
        if kind == current_kind:
            buffer.append(char)
            continue
        groups.append((current_kind, "".join(buffer)))
        current_kind = kind
        buffer = [char]
    groups.append((current_kind, "".join(buffer)))
    return groups


def _kind_to_regex(kind: str, sample: str) -> str:
    if not sample:
        return ""
    if kind == "digit":
        return r"\d{1,%d}" % max(1, len(sample))
    if kind == "space":
        return r"\s{1,%d}" % max(1, len(sample))
    if kind == "upper":
        return r"[A-Z]{1,%d}" % max(1, len(sample))
    if kind == "lower":
        return r"[a-z]{1,%d}" % max(1, len(sample))
    if kind == "alpha":
        return r"[A-Za-z]{1,%d}" % max(1, len(sample))
    if kind == "alnum":
        return r"[A-Za-z0-9]{1,%d}" % max(1, len(sample))
    escaped = re.escape(sample)
    if len(escaped) == 1:
        return escaped
    return f"(?:{escaped})"


def _merge_groups(examples: Sequence[str]) -> str:
    if not examples:
        return r".*"
    grouped = [_group_chars(text) for text in examples]
    max_len = max(len(groups) for groups in grouped)
    pattern_parts: List[str] = ["^"]
    for index in range(max_len):
        kinds = []
        samples = []
        for groups in grouped:
            if index < len(groups):
                kinds.append(groups[index][0])
                samples.append(groups[index][1])
        if not kinds:
            continue
        if len(set(kinds)) == 1:
            pattern_parts.append(_kind_to_regex(kinds[0], max(samples, key=len)))
        else:
            escaped = [re.escape(sample) for sample in samples if sample]
            if escaped:
                pattern_parts.append("(?:%s)" % "|".join(sorted(set(escaped))))
            else:
                pattern_parts.append(r".+?")
    pattern_parts.append("$")
    return "".join(pattern_parts)


@dataclass
class RecipeExample:
    """A labelled example describing the desired block classification."""

    text: str
    target_type: str
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class RecipeSuggestion:
    """Candidate regex pattern and its evaluation metrics."""

    pattern: str
    target_type: str
    attrs: dict[str, str]
    matched: int
    missed: int
    false_positives: int
    score: float


class RecipeSuggester:
    """Derive regex patterns from a few positive examples and evaluate them."""

    def __init__(self, examples: Sequence[RecipeExample]) -> None:
        if not examples:
            raise ValueError("At least one example is required")
        self.examples = list(examples)

    def build_pattern(self) -> str:
        texts = [example.text for example in self.examples]
        pattern = _merge_groups(texts)
        if pattern.count("|") > len(texts) * 2:
            # Guard against runaway alternations – fall back to a loose pattern.
            pattern = r"^.*$"
        return pattern

    def suggest(self, negatives: Sequence[str] | None = None) -> RecipeSuggestion:
        pattern = self.build_pattern()
        regex = re.compile(pattern, re.MULTILINE)
        matched = sum(1 for example in self.examples if regex.search(example.text))
        missed = len(self.examples) - matched
        false_positives = 0
        for text in negatives or []:
            if regex.search(text):
                false_positives += 1
        positive_rate = matched / len(self.examples) if self.examples else 0.0
        negative_rate = 0.0
        if negatives:
            negative_rate = 1.0 - (false_positives / max(len(negatives), 1))
        score = max(0.0, (positive_rate + negative_rate) / 2.0)
        attrs = {}
        for example in self.examples:
            attrs.update(example.attrs)
        return RecipeSuggestion(
            pattern=pattern,
            target_type=self.examples[0].target_type,
            attrs=attrs,
            matched=matched,
            missed=missed,
            false_positives=false_positives,
            score=round(score, 4),
        )

    @staticmethod
    def load_examples(path: Path | str) -> List[RecipeExample]:
        records: List[RecipeExample] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                text = str(payload.get("text", ""))
                target_type = str(payload.get("type", payload.get("target_type", "paragraph")))
                attrs = payload.get("attrs", {})
                if not isinstance(attrs, dict):
                    attrs = {}
                records.append(RecipeExample(text=text, target_type=target_type, attrs=attrs))
        return records

    @staticmethod
    def load_negative_samples(path: Path | str) -> List[str]:
        samples: List[str] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                text = payload.get("text") if isinstance(payload, dict) else None
                if isinstance(text, str):
                    samples.append(text)
                elif isinstance(payload, str):
                    samples.append(payload)
        return samples

def render_yaml(name: str, suggestion: RecipeSuggestion) -> str:
    """Render a YAML payload describing *suggestion* suitable for recipes."""

    import yaml

    payload = {
        "name": name,
        "patterns": [
            {
                "when": suggestion.pattern,
                "as": suggestion.target_type,
                "attrs": suggestion.attrs or {},
                "confidence": max(0.5, min(0.99, suggestion.score)),
            }
        ],
        "fallback": {
            "as": suggestion.target_type,
            "confidence": max(0.4, min(0.95, suggestion.score)),
        },
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


__all__ = [
    "RecipeExample",
    "RecipeSuggestion",
    "RecipeSuggester",
    "render_yaml",
]
