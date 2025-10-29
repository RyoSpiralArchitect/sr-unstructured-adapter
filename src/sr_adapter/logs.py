"""Utilities for parsing and summarizing semi-structured log files."""

from __future__ import annotations

from collections import Counter
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator, List, Optional, Sequence

__all__ = ["LogEntry", "iter_log_entries", "parse_log_line", "summarize_log_text"]


_LOG_PATTERN = re.compile(
    r"^\s*"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}"
    r"(?:[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})?)?)"
    r"[\sT]*"
    r"(?:\[(?P<bracket_level>[A-Za-z]{3,10})\]|(?P<level>[A-Za-z]{3,10}))"
    r"[\s:|-]+"
    r"(?P<message>.*)"
    r"$"
)

_LEVEL_NORMALISATION = {
    "WARNING": "WARN",
    "WARN": "WARN",
    "ERR": "ERROR",
    "ERROR": "ERROR",
    "SEVERE": "ERROR",
    "FATAL": "CRITICAL",
    "CRIT": "CRITICAL",
    "CRITICAL": "CRITICAL",
    "ALERT": "ALERT",
    "EMERG": "EMERGENCY",
    "EMERGENCY": "EMERGENCY",
}

_SEVERITY_PRIORITY = {
    "EMERGENCY": 0,
    "ALERT": 1,
    "CRITICAL": 2,
    "ERROR": 3,
    "WARN": 4,
    "NOTICE": 5,
    "INFO": 6,
    "DEBUG": 7,
    "TRACE": 8,
}


def _normalize_timestamp(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        return raw
    # Convert millisecond separator to ``.`` for ``fromisoformat`` compatibility.
    cleaned = cleaned.replace(",", ".")
    # Normalise numeric timezone offsets such as ``+0000`` to ``+00:00``.
    tz_match = re.search(r"([+-])(\d{2})(\d{2})$", cleaned)
    if tz_match and ":" not in cleaned[tz_match.start():]:
        cleaned = f"{cleaned[:tz_match.start()]}{tz_match.group(1)}{tz_match.group(2)}:{tz_match.group(3)}"
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
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    @property
    def has_structured_fields(self) -> bool:
        return bool(self.level or self.timestamp)


def _normalize_level(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    upper = raw.upper()
    return _LEVEL_NORMALISATION.get(upper, upper)


def parse_log_line(line: str) -> Optional[LogEntry]:
    stripped = line.strip()
    if not stripped:
        return None
    match = _LOG_PATTERN.match(stripped)
    if not match:
        return None
    raw_timestamp = match.group("timestamp")
    level = match.group("level") or match.group("bracket_level")
    message = match.group("message").rstrip()
    normalized_ts = _normalize_timestamp(raw_timestamp)
    return LogEntry(
        raw=line.rstrip("\n"),
        message=message,
        level=_normalize_level(level),
        timestamp=normalized_ts if normalized_ts else None,
        raw_timestamp=raw_timestamp,
    )


def _coalesce_entries(lines: Sequence[str]) -> List[LogEntry]:
    entries: List[LogEntry] = []
    current: Optional[LogEntry] = None
    for index, raw in enumerate(lines):
        parsed = parse_log_line(raw)
        if parsed is not None:
            if current is not None:
                current.end_line = index
                entries.append(current)
            parsed.start_line = index
            current = parsed
            continue

        if current is None:
            continue

        stripped = raw.strip()
        is_continuation = (
            not stripped
            or raw[:1].isspace()
            or stripped.startswith("Traceback")
            or stripped.startswith("Caused by")
            or stripped.startswith("During handling of the above exception")
        )
        if not is_continuation:
            current.end_line = index
            entries.append(current)
            current = None
            continue

        addition = raw.rstrip()
        if addition:
            if current.message:
                current.message = f"{current.message}\n{addition}"
            else:
                current.message = addition
        current.raw = f"{current.raw}\n{raw.rstrip()}" if current.raw else raw.rstrip()

    if current is not None:
        current.end_line = len(lines)
        entries.append(current)

    return entries


def iter_log_entries(lines: Iterable[str]) -> Iterator[LogEntry]:
    for entry in _coalesce_entries(list(lines)):
        yield entry


def summarize_log_text(text: str) -> dict:
    """Return summary metadata for *text* if it resembles structured logs."""

    lines = text.splitlines()
    entries = _coalesce_entries(lines)
    if not entries:
        return {}

    levels = [entry.level for entry in entries if entry.level]
    level_counts = Counter(levels)
    ordered_levels = sorted(
        level_counts,
        key=lambda lvl: (_SEVERITY_PRIORITY.get(lvl, 99), lvl),
    )

    timestamps = [entry.timestamp for entry in entries if entry.timestamp]
    has_multiline = any("\n" in entry.message for entry in entries if entry.message)

    summary = {
        "log_line_count": len(entries),
        "log_levels": ordered_levels,
        "log_level_counts": {level: level_counts[level] for level in ordered_levels},
    }
    if timestamps:
        summary["log_first_timestamp"] = timestamps[0]
        summary["log_last_timestamp"] = timestamps[-1]
    if has_multiline:
        summary["log_has_multiline_entries"] = True

    high_severity = sum(
        count for level, count in level_counts.items() if _SEVERITY_PRIORITY.get(level, 99) <= 3
    )
    summary["log_high_severity_count"] = high_severity

    examples = []
    for entry in entries[:3]:
        examples.append(
            {
                "level": entry.level,
                "timestamp": entry.timestamp or entry.raw_timestamp,
                "message": (entry.message or "").strip()[:200],
            }
        )
    if examples:
        summary["log_examples"] = examples
    return summary
