"""Shared normalization routines used by all parsers."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from .schema import Block, clone_model

_BULLET_NORMALISER = re.compile(r"^[\u2022\u30fb]\s*")
_HEADER_PREFIX = re.compile(r"^(?:\d+[.)]|[（(][^)]+[)）])\s+")


def _normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _BULLET_NORMALISER.sub("- ", text)
    return text.strip()


def _infer_type(block: Block) -> str:
    if block.type != "paragraph":
        return block.type
    text = block.text.strip()
    if not text:
        return "other"
    if text.count("\n") >= 1 and all(line.startswith("- ") for line in text.splitlines()):
        return "list"
    if _HEADER_PREFIX.match(text) and len(text) < 120:
        return "header"
    if len(text.split()) <= 6 and text.isupper():
        return "header"
    if ":" in text and text.count(":") == 1 and len(text) < 80:
        return "kv"
    return "paragraph"


def normalize_blocks(blocks: Iterable[Block]) -> List[Block]:
    """Apply text normalisation and lightweight type inference."""

    normalised: List[Block] = []
    for block in blocks:
        text = _normalise_text(block.text)
        attrs = dict(block.attrs)
        if "text" in attrs:
            attrs["text"] = _normalise_text(attrs["text"])
        candidate = clone_model(block, text=text)
        inferred_type = _infer_type(candidate)
        updated = clone_model(candidate, attrs=attrs, type=inferred_type)
        if not updated.text:
            updated = clone_model(updated, confidence=min(updated.confidence, 0.2))
        normalised.append(updated)
    return normalised

