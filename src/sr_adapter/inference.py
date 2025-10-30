"""Light-weight heuristics for producing LLM-friendly views of documents."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

from .schema import Block

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")
_HEADER_TYPES = {"header", "title", "heading"}
_LIST_TYPES = {"list", "kv"}


def _clean_text(text: str) -> str:
    collapsed = _WHITESPACE.sub(" ", text.strip())
    return collapsed.strip()


def _sentence_preview(text: str, *, limit: int = 180) -> str:
    if not text:
        return ""
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    parts = _SENTENCE_SPLIT.split(cleaned)
    preview = parts[0] if parts else cleaned
    if len(preview) > limit:
        preview = preview[: limit - 1].rstrip() + "…"
    return preview


def _list_preview(lines: Iterable[str], *, limit: int = 4) -> str:
    collected: List[str] = []
    for line in lines:
        clean = _clean_text(line)
        if not clean:
            continue
        collected.append(clean)
        if len(collected) >= limit:
            break
    preview = "; ".join(collected)
    if preview and len(collected) == limit:
        preview += "…"
    return preview


def _block_preview(block: Block) -> str:
    if block.type in _LIST_TYPES:
        return _list_preview(block.text.splitlines())
    return _sentence_preview(block.text)


def build_outline(blocks: Sequence[Block]) -> List[Dict[str, Any]]:
    """Return a coarse outline grouped by detected headers."""

    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    def _finalise(section: Dict[str, Any] | None) -> None:
        if not section:
            return
        highlights = section.get("highlights", [])
        preview_parts: List[str] = []
        for highlight in highlights[:3]:
            label = highlight.get("kind", "block").capitalize()
            text = highlight.get("preview", "")
            if text:
                preview_parts.append(f"{label}: {text}")
        preview = section.get("title") or ""
        if preview_parts:
            preview = " | ".join(preview_parts)
        confidence_values = section.get("confidences", [])
        confidence = 0.0
        if confidence_values:
            confidence = sum(confidence_values) / len(confidence_values)
        sections.append(
            {
                "title": section.get("title"),
                "kind": section.get("kind", "section"),
                "preview": preview,
                "highlights": highlights[:5],
                "block_indices": section.get("block_indices", []),
                "confidence": round(confidence, 3) if confidence_values else 0.0,
            }
        )

    for index, block in enumerate(blocks):
        preview = _block_preview(block)
        if not preview:
            continue
        if block.type in _HEADER_TYPES:
            _finalise(current)
            current = {
                "title": preview,
                "kind": "section",
                "block_indices": [index],
                "confidences": [block.confidence],
                "highlights": [],
            }
            continue

        highlight = {
            "kind": block.type,
            "preview": preview,
            "block_index": index,
            "confidence": round(block.confidence, 3),
        }

        if current:
            current.setdefault("block_indices", []).append(index)
            current.setdefault("confidences", []).append(block.confidence)
            current.setdefault("highlights", []).append(highlight)
        else:
            sections.append(
                {
                    "title": None,
                    "kind": block.type,
                    "preview": preview,
                    "highlights": [highlight],
                    "block_indices": [index],
                    "confidence": round(block.confidence, 3),
                }
            )

    _finalise(current)

    return sections


def _truncate(text: str, *, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def build_focus(
    meta: Dict[str, Any],
    parties: List[Dict[str, Any]],
    amounts: List[Dict[str, Any]],
    dates: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    attachments: List[Dict[str, Any]],
    hints: List[str],
) -> List[Dict[str, Any]]:
    focus: List[Dict[str, Any]] = []

    def _add(label: str, summary: str, *, confidence: float = 0.65, source: str = "meta") -> None:
        summary = _truncate(summary)
        if not summary:
            return
        focus.append(
            {
                "label": label,
                "summary": summary,
                "confidence": round(confidence, 3),
                "source": source,
            }
        )

    title = meta.get("html_title") or meta.get("email_subject") or meta.get("pptx_titles", [None])[0]
    if title:
        _add("title", f"Primary title detected: {title}", confidence=0.75, source="meta")

    if parties:
        senders = [party["name"] for party in parties if party.get("role") == "from" and party.get("name")]
        recipients = [
            party["name"]
            for party in parties
            if party.get("role") in {"to", "cc"} and party.get("name")
        ]
        if senders:
            _add("parties.from", f"Senders include {', '.join(senders[:3])}", confidence=0.8, source="parties")
        if recipients:
            _add(
                "parties.recipients",
                f"Recipients include {', '.join(recipients[:3])}",
                confidence=0.7,
                source="parties",
            )

    if amounts:
        top = max(amounts, key=lambda entry: float(entry.get("value", 0) or 0), default=amounts[0])
        display = top.get("value")
        currency = top.get("currency") or ""
        origin = f"block[{top.get('block_index', '?')}]"
        if display:
            formatted = f"{currency}{display}".strip()
            _add("amounts", f"Contains monetary values such as {formatted}", confidence=top.get("confidence", 0.7), source=origin)

    if dates:
        sorted_dates = sorted(
            [entry for entry in dates if entry.get("value")],
            key=lambda entry: entry["value"],
        )
        if sorted_dates:
            first = sorted_dates[0]
            last = sorted_dates[-1]
            if first == last:
                _add(
                    "dates",
                    f"Key date observed: {first['value']} ({first.get('origin', 'unknown')})",
                    confidence=first.get("confidence", 0.6),
                    source=first.get("origin", "dates"),
                )
            else:
                _add(
                    "dates",
                    f"Dates range from {first['value']} to {last['value']}",
                    confidence=min(first.get("confidence", 0.6), last.get("confidence", 0.6)),
                    source="dates",
                )

    if tables:
        _add(
            "tables",
            f"Structured tables detected ({len(tables)} total)",
            confidence=max(table.get("confidence", 0.6) for table in tables),
            source="tables",
        )

    if items:
        sample = items[:3]
        texts: List[str] = []
        for item in sample:
            if "label" in item and item.get("label"):
                texts.append(f"{item['label']}: {item.get('value', item.get('values'))}")
            elif "text" in item and item.get("text"):
                texts.append(str(item["text"]))
            elif "values" in item and isinstance(item["values"], dict):
                pair_preview = ", ".join(f"{key}={val}" for key, val in list(item["values"].items())[:2])
                texts.append(pair_preview)
        if texts:
            _add("items", f"Important bullet points: {'; '.join(texts)}", confidence=0.68, source="items")

    if attachments:
        names = [
            att.get("name") or att.get("filename") or att.get("source")
            for att in attachments
            if any(att.get(key) for key in ("name", "filename", "source"))
        ]
        if names:
            _add(
                "attachments",
                f"References {len(attachments)} attachment(s): {', '.join(names[:3])}",
                confidence=0.6,
                source="attachments",
            )

    word_count = meta.get("word_count")
    if isinstance(word_count, int) and word_count > 0:
        _add("word_count", f"Approximate length: {word_count} words", confidence=0.55, source="meta.word_count")

    log_high = meta.get("log_high_severity_count")
    if isinstance(log_high, int) and log_high > 0:
        _add(
            "log.high_severity",
            f"Log severity alerts present ({log_high} entries)",
            confidence=0.75,
            source="meta.log_high_severity_count",
        )

    if hints and len(focus) < 3:
        for hint in hints[:3 - len(focus)]:
            _add("hint", hint, confidence=0.5, source="llm_hints")

    return focus


def build_brief(
    document_meta: Dict[str, Any],
    meta: Dict[str, Any],
    hints: List[str],
    outline: List[Dict[str, Any]],
    focus: List[Dict[str, Any]],
) -> str:
    doc_type = document_meta.get("type") or meta.get("detected_type") or "document"
    parts: List[str] = [f"{doc_type.capitalize()} prepared for downstream LLM consumption."]

    title = meta.get("html_title") or meta.get("email_subject")
    if title:
        parts.append(f"Title: {title}.")

    if focus:
        focus_bits = ", ".join(entry["summary"] for entry in focus[:2])
        if focus_bits:
            parts.append(f"Key facts: {focus_bits}.")

    if hints:
        parts.append(f"Signals: {', '.join(hints[:2])}.")

    if outline:
        section_titles = [
            section["title"] or section["preview"]
            for section in outline[:2]
            if section.get("title") or section.get("preview")
        ]
        if section_titles:
            parts.append(f"Sections include {', '.join(section_titles)}.")

    brief = " ".join(part.strip() for part in parts if part)
    return brief or "Document converted for LLM consumption."


def build_llm_facets(
    blocks: Sequence[Block],
    meta: Dict[str, Any],
    parties: List[Dict[str, Any]],
    amounts: List[Dict[str, Any]],
    dates: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    attachments: List[Dict[str, Any]],
    document_meta: Dict[str, Any],
    hints: List[str],
) -> Dict[str, Any]:
    outline = build_outline(blocks)
    focus = build_focus(meta, parties, amounts, dates, items, tables, attachments, hints)
    brief = build_brief(document_meta, meta, hints, outline, focus)

    return {
        "brief": brief,
        "outline": outline,
        "focus": focus,
        "hints": hints,
    }
