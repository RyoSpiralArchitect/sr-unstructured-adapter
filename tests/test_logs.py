from __future__ import annotations

from sr_adapter.logs import iter_log_entries, summarize_log_text


def test_iter_log_entries_coalesces_multiline_segments() -> None:
    lines = [
        "2024-01-01 10:00:00 ERROR failed hard",
        "    Traceback (most recent call last):",
        "    ValueError: boom",
        "noise that should stand alone",
        "2024-01-01 10:01:00 info recovered",
    ]

    entries = list(iter_log_entries(lines))

    assert len(entries) == 2
    assert entries[0].start_line == 0
    assert entries[0].end_line == 3
    assert "Traceback" in entries[0].message
    assert entries[1].start_line == 4
    assert entries[1].level == "INFO"


def test_summarize_log_text_rolls_up_counts() -> None:
    text = "\n".join(
        [
            "2024-01-01T09:00:00+0000 warn cache low",
            "2024-01-01T09:05:00+00:00 ERROR outage detected",
            "2024-01-01T09:10:00Z debug tracing",
        ]
    )

    summary = summarize_log_text(text)

    assert summary["log_line_count"] == 3
    assert summary["log_levels"] == ["ERROR", "WARN", "DEBUG"]
    assert summary["log_level_counts"] == {"ERROR": 1, "WARN": 1, "DEBUG": 1}
    assert summary["log_high_severity_count"] == 1
    assert summary["log_first_timestamp"].endswith("+00:00")
    assert summary["log_last_timestamp"].endswith("+00:00")
    assert summary["log_examples"][0]["message"].startswith("cache low")
