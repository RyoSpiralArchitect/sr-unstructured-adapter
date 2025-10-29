"""Utilities for parsing and summarizing semi-structured log files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional

__all__ = ["LogEntry", "iter_log_entries", "parse_log_line", "summarize_log_text"]


_LOG_PATTERN = re.compile(
    r"^"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}"
    r"(?:[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?(?:Z|[+-]\d{2}:\d{2})?)?)"
    r"\s+"
    r"(?P<level>[A-Z]{3,9})"
    r"[:\s]+"
    r"(?P<message>.*)"
    r"$"
)


def _normalize_timestamp(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        return raw
    # Convert millisecond separator to ``.`` for ``fromisoformat`` compatibility.
    cleaned = cleaned.replace(",", ".")
    tzinfo = None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1]
        tzinfo = timezone.utc
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return raw
    if tzinfo is not None and parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tzinfo)
    return parsed.isoformat()


@dataclass
class LogEntry:
    """A structured representation of a single log line."""

    raw: str
    message: str
    level: Optional[str] = None
    timestamp: Optional[str] = None
    raw_timestamp: Optional[str] = None

    @property
    def has_structured_fields(self) -> bool:
        return bool(self.level or self.timestamp)


def parse_log_line(line: str) -> Optional[LogEntry]:
    stripped = line.strip()
    if not stripped:
        return None
    match = _LOG_PATTERN.match(stripped)
    if not match:
        return None
    raw_timestamp = match.group("timestamp")
    level = match.group("level")
    message = match.group("message").strip()
    normalized_ts = _normalize_timestamp(raw_timestamp)
    return LogEntry(
        raw=line.rstrip("\n"),
        message=message,
        level=level.upper() if level else None,
        timestamp=normalized_ts if normalized_ts else None,
        raw_timestamp=raw_timestamp,
    )


def iter_log_entries(lines: Iterable[str]) -> Iterator[LogEntry]:
    for line in lines:
        parsed = parse_log_line(line)
        if parsed is not None:
            yield parsed


def summarize_log_text(text: str) -> dict:
    """Return summary metadata for *text* if it resembles structured logs."""

    entries = list(iter_log_entries(text.splitlines()))
    if not entries:
        return {}

    levels = sorted({entry.level for entry in entries if entry.level})
    timestamps = [entry.timestamp for entry in entries if entry.timestamp]
    summary = {
        "log_line_count": len(entries),
        "log_levels": levels,
    }
    if timestamps:
        summary["log_first_timestamp"] = timestamps[0]
        summary["log_last_timestamp"] = timestamps[-1]
    return summary
