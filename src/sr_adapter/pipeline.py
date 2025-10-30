"""High level conversion pipeline wiring parsers and writers together."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from .delegate import escalate_low_conf
from .normalize import normalize_blocks
from .parsers import (
    parse_csv,
    parse_docx,
    parse_email,
    parse_html,
    parse_image,
    parse_json,
    parse_log,
    parse_md,
    parse_pptx,
    parse_pdf,
    parse_txt,
    parse_xml,
    parse_yaml,
    parse_xlsx,
    parse_zip,
)
from .recipe import apply_recipe
from .schema import Block, Document
from .sniff import detect_type


PARSERS: Dict[str, callable] = {
    "text": parse_txt,
    "md": parse_md,
    "log": parse_log,
    "html": parse_html,
    "csv": parse_csv,
    "json": parse_json,
    "yaml": parse_yaml,
    "xml": parse_xml,
    "pdf": parse_pdf,
    "docx": parse_docx,
    "pptx": parse_pptx,
    "xlsx": parse_xlsx,
    "email": parse_email,
    "zip": parse_zip,
    "image": parse_image,
}


def _parse(path: Path, detected: str) -> List[Block]:
    parser = PARSERS.get(detected, parse_txt)
    try:
        return parser(path)
    except Exception:  # pragma: no cover - defensive: fall back to plain text
        return parse_txt(path)


def convert(path: str | Path, recipe: str, llm_ok: bool = True) -> Document:
    """Convert *path* to the unified :class:`Document` representation."""

    source = Path(path)
    detected = detect_type(source)
    blocks = _parse(source, detected)
    blocks = normalize_blocks(blocks)
    blocks = apply_recipe(blocks, recipe)
    if llm_ok:
        blocks = escalate_low_conf(blocks, recipe)
    document = Document(blocks=list(blocks), meta={"source": str(source), "type": detected})
    return document


def batch_convert(paths: Iterable[str | Path], recipe: str, llm_ok: bool = True) -> List[Document]:
    return [convert(path, recipe=recipe, llm_ok=llm_ok) for path in paths]

