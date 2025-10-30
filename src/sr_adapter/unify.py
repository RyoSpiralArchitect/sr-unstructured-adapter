"""Helpers for mapping raw payloads to the unified downstream schema."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from .inference import build_llm_facets
from .models import Payload
from .schema import Block, Document

_AMOUNT_PATTERN = re.compile(
    r"(?P<currency>[\$€¥£])?\s*(?P<number>(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)(?:[.,]\d{2})?)"
)
_DATE_CANDIDATE_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b"
)
_LIST_BULLET = re.compile(r"^(?:[-*\u2022\u30fb]\s+)")


def _avg_confidence(blocks: Iterable[Block]) -> float:
    blocks = list(blocks)
    if not blocks:
        return 0.0
    total = sum(block.confidence for block in blocks)
    return round(total / len(blocks), 3)


def _normalize_amount(raw: str) -> str:
    value = raw.replace(" ", "")
    if "," in value and "." in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif "," in value:
        parts = value.split(",")
        if len(parts) == 2 and len(parts[-1]) == 2:
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")
    value = value.replace(" ", "")
    if value.count(".") > 1:
        value = value.replace(".", "")
    return value


def _normalize_date(raw: str) -> str:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%d %B %Y"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return raw


def _extract_amounts(blocks: List[Block]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    amounts: List[Dict[str, Any]] = []
    provenance: List[Dict[str, Any]] = []
    for index, block in enumerate(blocks):
        for match in _AMOUNT_PATTERN.finditer(block.text):
            raw = match.group("number")
            normalized = _normalize_amount(raw)
            entry = {
                "value": normalized,
                "raw": match.group(0).strip(),
                "currency": match.group("currency") or "",
                "block_index": index,
                "confidence": min(block.confidence + 0.2, 1.0),
            }
            amounts.append(entry)
            provenance.append({"value": normalized, "block_index": index})
    return amounts, provenance


def _extract_dates(blocks: List[Block], meta: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dates: List[Dict[str, Any]] = []
    provenance: List[Dict[str, Any]] = []

    def _append(value: str, origin: str, confidence: float, block_index: int | None) -> None:
        normalised = _normalize_date(value)
        entry = {
            "value": normalised,
            "raw": value,
            "origin": origin,
            "confidence": confidence,
        }
        if block_index is not None:
            entry["block_index"] = block_index
        dates.append(entry)
        provenance.append({"value": normalised, "origin": origin, "block_index": block_index})

    for index, block in enumerate(blocks):
        for match in _DATE_CANDIDATE_PATTERN.finditer(block.text):
            _append(match.group(0), "block", min(block.confidence + 0.1, 1.0), index)

    if meta.get("email_date"):
        _append(str(meta["email_date"]), "meta.email_date", 0.9, None)

    if meta.get("modified_at"):
        _append(str(meta["modified_at"]), "meta.modified_at", 0.6, None)

    return dates, provenance


def _extract_parties(meta: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    parties: List[Dict[str, Any]] = []
    provenance: List[Dict[str, Any]] = []
    for role in ("from", "to", "cc"):
        key = f"email_{role}"
        if key not in meta:
            continue
        addresses = meta.get(key) or []
        if not isinstance(addresses, list):
            continue
        for addr in addresses:
            entry = {
                "name": addr,
                "role": role,
                "origin": f"meta.{key}",
                "confidence": 0.85 if role == "from" else 0.7,
            }
            parties.append(entry)
            provenance.append({"name": addr, "role": role, "origin": entry["origin"]})
    return parties, provenance


def _extract_items_and_tables(blocks: List[Block]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    for index, block in enumerate(blocks):
        if block.type == "table":
            rows: List[List[str]] = []
            if "rows" in block.attrs:
                try:
                    decoded = json.loads(block.attrs["rows"])
                    if isinstance(decoded, list):
                        rows = [list(map(str, row)) for row in decoded]
                except json.JSONDecodeError:
                    rows = []
            tables.append(
                {
                    "block_index": index,
                    "rows": rows,
                    "source": block.source,
                    "confidence": block.confidence,
                }
            )
            if rows:
                header = rows[0]
                for row in rows[1:]:
                    if len(row) != len(header):
                        continue
                    record = {header[pos]: row[pos] for pos in range(len(header))}
                    items.append(
                        {
                            "values": record,
                            "block_index": index,
                            "confidence": block.confidence,
                        }
                    )
            continue

        if block.type in {"list", "kv"}:
            lines = [segment for segment in block.text.splitlines() if segment.strip()]
            for line in lines:
                clean = _LIST_BULLET.sub("", line).strip()
                if not clean:
                    continue
                key_value = clean.split(":", 1)
                payload: Dict[str, Any]
                if len(key_value) == 2:
                    payload = {"label": key_value[0].strip(), "value": key_value[1].strip()}
                else:
                    payload = {"text": clean}
                items.append(
                    {
                        **payload,
                        "block_index": index,
                        "confidence": min(block.confidence + 0.05, 1.0),
                    }
                )

    return items, tables


def _extract_text_blocks(blocks: List[Block]) -> List[Dict[str, Any]]:
    structured: List[Dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.type == "table":
            continue
        structured.append(
            {
                "block_index": index,
                "type": block.type,
                "text": block.text,
                "attrs": dict(block.attrs),
                "source": block.source,
                "confidence": block.confidence,
            }
        )
    return structured


def _item_provenance_label(item: Dict[str, Any]) -> str:
    if "label" in item and item["label"]:
        return str(item["label"])
    if "text" in item and item["text"]:
        return str(item["text"])
    if "values" in item and isinstance(item["values"], dict):
        return ", ".join(f"{key}={value}" for key, value in item["values"].items())
    return ""


def _list_preview(values: Iterable[Any], limit: int = 5) -> str:
    sequence = [str(value) for value in values if value not in (None, "")]
    if not sequence:
        return ""
    preview = sequence[:limit]
    if len(sequence) > limit:
        preview.append("…")
    return ", ".join(preview)


def _compile_validation(meta: Dict[str, Any], blocks: List[Block], payload: Payload) -> Dict[str, List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    if not blocks:
        warnings.append("No blocks were produced by the parser; downstream enrichment may be limited.")
    if not payload.text:
        warnings.append("No primary text extracted – some analytics may be unavailable.")
    if meta.get("json_valid") is False:
        errors.append("JSON payload could not be parsed cleanly.")
    if meta.get("pdf_has_text") is False:
        warnings.append("PDF did not expose text – consider OCR for scans.")
    if meta.get("zip_contains_nested"):
        warnings.append("ZIP file contains nested archives; deep extraction recommended.")
    high_severity = meta.get("log_high_severity_count")
    if isinstance(high_severity, int):
        if high_severity > 0:
            warnings.append("Log file contains high-severity entries.")
    else:
        log_levels = meta.get("log_levels")
        if isinstance(log_levels, list):
            if any(level in {"ERROR", "CRITICAL", "FATAL"} for level in log_levels):
                warnings.append("Log file contains high-severity entries.")

    return {"warnings": warnings, "errors": errors}


def _build_llm_hints(
    meta: Dict[str, Any],
    parties: List[Dict[str, Any]],
    amounts: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    attachments: List[Dict[str, Any]],
    document_meta: Dict[str, Any],
) -> List[str]:
    hints: List[str] = []

    def _add(message: str) -> None:
        if message and message not in hints:
            hints.append(message)

    doc_type = document_meta.get("type")
    if doc_type:
        _add(f"Detected type: {doc_type}")

    word_count = meta.get("word_count")
    if isinstance(word_count, int) and word_count > 0:
        _add(f"Word count: {word_count}")
    line_count = meta.get("line_count")
    if isinstance(line_count, int) and line_count > 0:
        _add(f"Line count: {line_count}")

    title = meta.get("html_title") or meta.get("email_subject")
    if title:
        _add(f"Title: {title}")

    pptx_count = meta.get("pptx_slide_count")
    if isinstance(pptx_count, int) and pptx_count > 0:
        _add(f"Presentation with {pptx_count} slides.")
        titles = meta.get("pptx_titles")
        if isinstance(titles, list) and titles:
            preview = _list_preview(titles)
            if preview:
                _add(f"Slide highlights: {preview}")

    yaml_keys = meta.get("yaml_top_level_keys")
    if isinstance(yaml_keys, list) and yaml_keys:
        preview = _list_preview(yaml_keys)
        if preview:
            _add(f"YAML keys: {preview}")

    xml_root = meta.get("xml_root_tag")
    if xml_root:
        element_count = meta.get("xml_element_count")
        if isinstance(element_count, int) and element_count > 0:
            _add(f"XML root '{xml_root}' with {element_count} elements.")
        else:
            _add(f"XML root '{xml_root}'.")

    sheet_count = meta.get("workbook_sheet_count")
    if isinstance(sheet_count, int) and sheet_count > 0:
        _add(f"Workbook with {sheet_count} sheets.")

    log_count = meta.get("log_line_count")
    if isinstance(log_count, int) and log_count > 0:
        severity = meta.get("log_high_severity_count")
        if isinstance(severity, int) and severity > 0:
            _add(f"Log summary: {log_count} entries ({severity} high severity).")
        else:
            _add(f"Log summary: {log_count} entries.")
        levels = meta.get("log_levels")
        if isinstance(levels, list) and levels:
            preview = _list_preview(levels)
            if preview:
                _add(f"Log levels: {preview}")

    if parties:
        senders = [party.get("name") for party in parties if party.get("role") == "from"]
        recipients = [
            party.get("name")
            for party in parties
            if party.get("role") in {"to", "cc"}
        ]
        preview = _list_preview(senders)
        if preview:
            _add(f"Senders: {preview}")
        preview = _list_preview(recipients)
        if preview:
            _add(f"Recipients: {preview}")

    if amounts:
        sample = amounts[0]
        value = sample.get("value")
        currency = sample.get("currency") or ""
        if value:
            display = f"{currency}{value}".strip()
            _add(f"Monetary amounts detected (e.g. {display}).")

    if tables:
        _add(f"Structured tables detected ({len(tables)} total).")

    if attachments:
        names = [
            att.get("name") or att.get("filename") or att.get("source")
            for att in attachments
        ]
        preview = _list_preview(name for name in names if name)
        if preview:
            _add(f"Attachments: {preview}")
        else:
            _add(f"{len(attachments)} attachments referenced.")

    if len(hints) > 10:
        return hints[:10]
    return hints


def build_unified_payload(payload: Payload, document: Document) -> Dict[str, Any]:
    blocks = list(document.blocks)
    meta = dict(payload.meta)

    text_blocks = _extract_text_blocks(blocks)
    items, tables = _extract_items_and_tables(blocks)
    parties, party_prov = _extract_parties(meta)
    amounts, amount_prov = _extract_amounts(blocks)
    dates, date_prov = _extract_dates(blocks, meta)

    attachments: List[Dict[str, Any]] = []
    if isinstance(meta.get("email_attachments"), list):
        attachments.extend(meta["email_attachments"])
    if isinstance(meta.get("zip_entries"), list):
        attachments.extend(meta["zip_entries"])

    provenance = {
        "parties": party_prov,
        "amounts": amount_prov,
        "dates": date_prov,
        "items": [
            {
                "block_index": item["block_index"],
                "label": _item_provenance_label(item),
            }
            for item in items
        ],
    }

    llm_hints = _build_llm_hints(meta, parties, amounts, tables, attachments, document.meta)
    llm_facets = build_llm_facets(
        blocks,
        meta,
        parties,
        amounts,
        dates,
        items,
        tables,
        attachments,
        document.meta,
        llm_hints,
    )

    unified = {
        "schema_version": "1.0",
        "doc_id": str(uuid.uuid5(uuid.NAMESPACE_URL, str(payload.source))),
        "doc_type": document.meta.get("type", payload.mime),
        "source": str(payload.source),
        "mime": payload.mime,
        "confidence": _avg_confidence(blocks),
        "text_blocks": text_blocks,
        "tables": tables,
        "items": items,
        "parties": parties,
        "amounts": amounts,
        "dates": dates,
        "attachments": attachments,
        "meta": meta,
        "provenance": provenance,
        "validation": _compile_validation(meta, blocks, payload),
        "llm_hints": llm_hints,
        "llm": llm_facets,
    }

    return unified

