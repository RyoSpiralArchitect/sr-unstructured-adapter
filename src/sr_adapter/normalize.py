"""Normalization utilities applied to parsed blocks."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from .schema import Block


_WHITESPACE_RE = re.compile(r"[ \t]+")
_FULLWIDTH_MAP = str.maketrans({
    "：": ":",
    "（": "(",
    "）": ")",
    "，": ",",
    "．": ".",
})


def normalize_blocks(blocks: Iterable[Block]) -> List[Block]:
    """Return a list of blocks with text normalized in a common fashion."""

    normalised: List[Block] = []
    for block in blocks:
        text = _normalize_text(block.text)
        attrs = dict(block.attrs)
        if block.type == "paragraph" and text.endswith(":"):
            attrs.setdefault("is_key", "true")
        updated = block.with_updates(attrs=attrs)
        try:
            normalised.append(updated.model_copy(update={"text": text}))
        except AttributeError:  # pragma: no cover - pydantic v1 fallback
            normalised.append(updated.copy(update={"text": text}))
    return normalised


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_FULLWIDTH_MAP)
    text = text.replace("\u3000", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = text.strip()
    return text
