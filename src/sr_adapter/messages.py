"""Helpers for producing chat-friendly message payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

from .models import Payload


def _iter_chunks(text: str, chunk_size: int) -> Iterator[str]:
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


_META_PRIORITY = [
    "html_title",
    "email_subject",
    "pdf_page_count",
    "pptx_slide_count",
    "workbook_sheet_count",
    "yaml_top_level_keys",
    "xml_root_tag",
    "log_line_count",
    "log_high_severity_count",
    "word_count",
    "line_count",
]


def _format_meta_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        preview = [str(item) for item in value[:5]]
        if len(value) > 5:
            preview.append("…")
        return ", ".join(preview)
    if isinstance(value, dict):
        items = list(value.items())
        preview = [f"{key}={val}" for key, val in items[:5]]
        if len(items) > 5:
            preview.append("…")
        return ", ".join(preview)
    return str(value)


def _summarize_meta(meta: Dict[str, Any] | None, *, limit: int = 8) -> str:
    if not meta:
        return ""

    lines: List[str] = []
    seen: set[str] = set()
    ordered_keys = _META_PRIORITY + [key for key in sorted(meta.keys()) if key not in _META_PRIORITY]
    for key in ordered_keys:
        if key in seen or key not in meta:
            continue
        value = meta[key]
        if value in (None, "", [], {}):
            if not isinstance(value, (int, float)) or value == 0:
                continue
        formatted = _format_meta_value(value)
        if not formatted:
            continue
        lines.append(f"- {key}: {formatted}")
        seen.add(key)
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _normalise_llm_block(data: Dict[str, Any]) -> Dict[str, Any]:
    llm = data.get("llm")
    if isinstance(llm, dict):
        return llm
    meta = data.get("meta")
    if isinstance(meta, dict):
        llm = meta.get("llm")
        if isinstance(llm, dict):
            return llm
    return {}


def _format_focus(focus: List[Dict[str, Any]], limit: int = 5) -> str:
    lines: List[str] = []
    for entry in focus[:limit]:
        summary = entry.get("summary")
        if not summary:
            continue
        label = entry.get("label")
        if label:
            lines.append(f"- {label}: {summary}")
        else:
            lines.append(f"- {summary}")
    return "\n".join(lines)


def _format_outline(outline: List[Dict[str, Any]], limit: int = 4) -> str:
    lines: List[str] = []
    for section in outline[:limit]:
        title = section.get("title") or section.get("kind")
        preview = section.get("preview")
        if not preview:
            continue
        if title:
            lines.append(f"- {title}: {preview}")
        else:
            lines.append(f"- {preview}")
    return "\n".join(lines)


def to_llm_messages(payload: Payload | Dict[str, object], *, chunk_size: int = 2000) -> List[Dict[str, str]]:
    """Convert a payload into chat messages that respect chunking."""

    if isinstance(payload, Payload):
        source = str(payload.source)
        mime = payload.mime
        text = payload.text
        meta = payload.meta
        llm_info: Dict[str, Any] = payload.meta.get("llm") if isinstance(payload.meta.get("llm"), dict) else {}
    else:
        source = str(payload.get("source", ""))
        mime = str(payload.get("mime", ""))
        text = str(payload.get("text", ""))
        meta = payload.get("meta") if isinstance(payload, dict) else None
        if not isinstance(meta, dict):
            meta = {}
        llm_info = _normalise_llm_block(payload if isinstance(payload, dict) else {})

    meta_summary = _summarize_meta(meta)
    brief = ""
    focus_summary = ""
    outline_summary = ""
    if llm_info:
        brief = str(llm_info.get("brief") or "").strip()
        focus_entries = llm_info.get("focus")
        if isinstance(focus_entries, list):
            focus_summary = _format_focus(focus_entries)
        outline_entries = llm_info.get("outline")
        if isinstance(outline_entries, list):
            outline_summary = _format_outline(outline_entries)

    if not text:
        preface = f"[{mime}] {source}\n(no textual preview available)"
        system_sections: List[str] = []
        if meta_summary:
            system_sections.append(f"Key metadata:\n{meta_summary}")
        if brief:
            system_sections.append(f"Brief:\n{brief}")
        if focus_summary:
            system_sections.append(f"Focus:\n{focus_summary}")
        if outline_summary:
            system_sections.append(f"Outline:\n{outline_summary}")
        if system_sections:
            preface = f"Source: {source}\n" + "\n\n".join(system_sections) + f"\n\n{preface}"
        return [
            {
                "role": "user",
                "content": preface,
            }
        ]

    messages: List[Dict[str, str]] = []
    system_sections: List[str] = []
    if meta_summary:
        system_sections.append(f"Key metadata:\n{meta_summary}")
    if brief:
        system_sections.append(f"Brief:\n{brief}")
    if focus_summary:
        system_sections.append(f"Focus:\n{focus_summary}")
    if outline_summary:
        system_sections.append(f"Outline:\n{outline_summary}")
    if system_sections:
        messages.append({"role": "system", "content": f"Source: {source}\n" + "\n\n".join(system_sections)})

    multi_part = len(text) > chunk_size
    for index, chunk in enumerate(_iter_chunks(text, chunk_size), start=1):
        header = f"[{mime}] {source}"
        if multi_part:
            header = f"{header} (part {index})"
        messages.append({"role": "user", "content": f"{header}\n{chunk}"})
    return messages
