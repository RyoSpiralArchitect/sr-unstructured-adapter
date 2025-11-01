#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validate that the native runtime reported by the CLI is active."""

from __future__ import annotations

import json
import pathlib
import sys


def _usage() -> None:
    print("usage: assert_native_ready.py <status.json>", file=sys.stderr)


def _get_flag(data: dict, section: str, key: str) -> bool:
    section_data = data.get(section) or {}
    value = section_data.get(key)
    return bool(value)


def main() -> int:
    if len(sys.argv) != 2:
        _usage()
        return 2

    path = pathlib.Path(sys.argv[1])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"❌ status file not found: {path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"❌ failed to parse {path}: {exc}", file=sys.stderr)
        return 1

    text_enabled = _get_flag(payload, "text", "enabled")
    layout_enabled = _get_flag(payload, "layout", "enabled")

    if not (text_enabled or layout_enabled):
        print("❌ Native runtime not enabled (both text/layout disabled).", file=sys.stderr)
        return 1

    print(f"✅ Native runtime: text={text_enabled} layout={layout_enabled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
