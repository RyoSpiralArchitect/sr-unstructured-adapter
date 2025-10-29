"""High level pipeline orchestrating sniffing, parsing, normalization and export."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List

from .delegate import escalate_low_conf
from .normalize import normalize_blocks
from .parsers import (
    parse_csv,
    parse_docx,
    parse_html,
    parse_json,
    parse_md,
    parse_pdf,
    parse_txt,
    parse_unknown,
    parse_xlsx,
)
from .recipe import apply_recipe
from .schema import Block, Document
from .sniff import detect_type
from .writer import write_jsonl

PARSERS: Dict[str, Callable[[str | Path], List[Block]]] = {
    "text": parse_txt,
    "md": parse_md,
    "html": parse_html,
    "csv": parse_csv,
    "json": parse_json,
    "pdf": parse_pdf,
    "docx": parse_docx,
    "xlsx": parse_xlsx,
}


def convert(path: str | Path, recipe: str, out: str, llm_ok: bool = True) -> Document:
    """Convert *path* to the unified document schema and write to *out*."""

    path = Path(path)
    doc = _process(path, recipe, llm_ok=llm_ok)
    write_jsonl([doc], out)
    return doc


def _process(path: Path, recipe: str, *, llm_ok: bool) -> Document:
    file_type = detect_type(path)
    parser = PARSERS.get(file_type, parse_unknown)
    blocks = parser(path)
    normalised = normalize_blocks(blocks)
    resolved = apply_recipe(normalised, recipe)
    if llm_ok:
        resolved = escalate_low_conf(resolved, recipe)
    document = Document(blocks=resolved, meta={"source": str(path), "type": file_type})
    return document


def convert_many(paths: Iterable[str | Path], recipe: str, out: str, llm_ok: bool = True) -> List[Document]:
    documents = [_process(Path(path), recipe, llm_ok=llm_ok) for path in paths]
    write_jsonl(documents, out)
    return documents
