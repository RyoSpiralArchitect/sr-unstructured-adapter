"""Helpers for producing LLM-friendly renderings of unified payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _format_metadata(unified: Dict[str, Any]) -> List[str]:
    meta_lines = ["## Overview"]
    meta_lines.append(f"- Doc ID: `{unified.get('doc_id', '')}`")
    meta_lines.append(f"- Doc Type: `{unified.get('doc_type', '')}`")
    meta_lines.append(f"- Source: `{unified.get('source', '')}`")
    meta_lines.append(f"- MIME: `{unified.get('mime', '')}`")
    confidence = unified.get("confidence")
    if confidence is not None:
        meta_lines.append(f"- Confidence: {confidence}")

    highlights = unified.get("highlights") or {}
    key_points = highlights.get("key_points") or []
    if key_points:
        meta_lines.append("- Highlights:")
        for point in key_points[:5]:
            meta_lines.append(f"  - {point}")

    return meta_lines


def _format_amounts(amounts: Iterable[Dict[str, Any]]) -> List[str]:
    amount_lines = []
    entries = list(amounts)
    if not entries:
        return amount_lines

    amount_lines.append("## Amounts")
    for entry in entries[:10]:
        currency = entry.get("currency") or ""
        raw = entry.get("raw") or entry.get("value") or ""
        block_index = entry.get("block_index")
        block_ref = f" (block {block_index})" if block_index is not None else ""
        amount_lines.append(f"- {currency}{raw}{block_ref}")
    return amount_lines


def _format_dates(dates: Iterable[Dict[str, Any]]) -> List[str]:
    date_lines: List[str] = []
    entries = sorted({entry.get("value") for entry in dates if entry.get("value")})
    if not entries:
        return date_lines

    date_lines.append("## Dates")
    for value in entries[:10]:
        date_lines.append(f"- {value}")
    return date_lines


def _format_parties(parties: Iterable[Dict[str, Any]]) -> List[str]:
    party_lines: List[str] = []
    entries = list(parties)
    if not entries:
        return party_lines

    party_lines.append("## Parties")
    by_role: Dict[str, List[str]] = {}
    for party in entries:
        role = str(party.get("role") or "other")
        by_role.setdefault(role, []).append(str(party.get("name") or ""))

    for role, names in sorted(by_role.items()):
        unique = sorted({name for name in names if name})
        if not unique:
            continue
        party_lines.append(f"- {role}: {', '.join(unique)}")
    return party_lines


def _markdown_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    if not header:
        return ""
    header_line = " | ".join(cell or "" for cell in header)
    separator = " | ".join("---" for _ in header)
    body_lines = [" | ".join(cell or "" for cell in row) for row in body]
    return "\n".join([header_line, separator, *body_lines])


def _format_tables(tables: Iterable[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for index, table in enumerate(tables, start=1):
        rows = table.get("rows") or []
        markdown = _markdown_table(rows)
        if not markdown:
            continue
        lines.append(f"## Table {index}")
        lines.append(markdown)
    return lines


def _format_items(items: Iterable[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    entries = list(items)
    if not entries:
        return lines

    lines.append("## Items")
    for entry in entries[:20]:
        if "values" in entry and isinstance(entry["values"], dict):
            kv = ", ".join(f"{key}: {value}" for key, value in entry["values"].items())
            lines.append(f"- {kv}")
        elif entry.get("label") and entry.get("value"):
            lines.append(f"- {entry['label']}: {entry['value']}")
        elif entry.get("text"):
            lines.append(f"- {entry['text']}")
    return lines


def _chunk_text_blocks(blocks: List[Dict[str, Any]], max_chars: int = 1400) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    current_chars = 0
    chunk_index = 1

    for block in blocks:
        text = block.get("text") or ""
        if not text.strip():
            continue
        snippet = text.strip()
        if current_chars + len(snippet) > max_chars and current_lines:
            chunk_text = "\n".join(current_lines)
            chunks.append(
                {
                    "title": f"Chunk {chunk_index}",
                    "text": chunk_text,
                    "approx_tokens": max(1, len(chunk_text) // 4),
                }
            )
            chunk_index += 1
            current_lines = []
            current_chars = 0

        current_lines.append(snippet)
        current_chars += len(snippet)

    if current_lines:
        chunk_text = "\n".join(current_lines)
        chunks.append(
            {
                "title": f"Chunk {chunk_index}",
                "text": chunk_text,
                "approx_tokens": max(1, len(chunk_text) // 4),
            }
        )

    return chunks


def build_llm_bundle(unified: Dict[str, Any]) -> Dict[str, Any]:
    lines: List[str] = [f"# Document: {unified.get('doc_type', 'document').title()}"]
    lines.extend(_format_metadata(unified))

    lines.extend(_format_parties(unified.get("parties") or []))
    lines.extend(_format_amounts(unified.get("amounts") or []))
    lines.extend(_format_dates(unified.get("dates") or []))
    lines.extend(_format_items(unified.get("items") or []))
    lines.extend(_format_tables(unified.get("tables") or []))

    text_blocks = unified.get("text_blocks") or []
    if text_blocks:
        lines.append("## Text Blocks")
        for block in text_blocks[:20]:
            block_type = block.get("type") or "text"
            prefix = f"### {block_type.title()} (block {block.get('block_index', '')})"
            lines.append(prefix)
            lines.append(block.get("text") or "")

    markdown = "\n\n".join(line for line in lines if line)
    chunks = _chunk_text_blocks(text_blocks)

    sections: List[Dict[str, Any]] = []
    for entry in chunks:
        sections.append({"title": entry["title"], "text": entry["text"]})

    return {
        "markdown": markdown,
        "sections": sections,
        "chunks": chunks,
    }

