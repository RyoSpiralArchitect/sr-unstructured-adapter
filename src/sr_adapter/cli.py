# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command line interface for the adapter pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import batch_convert
from .writer import write_jsonl


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert unstructured files into Blocks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert files to JSONL")
    convert_parser.add_argument("paths", nargs="+", help="Files or directories to convert")
    convert_parser.add_argument("--recipe", default="default", help="Recipe to apply")
    convert_parser.add_argument("--out", required=True, help="Destination JSONL file")
    convert_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable low confidence LLM escalation",
    )

    return parser.parse_args(argv)


def _expand_paths(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for entry in paths:
        path = Path(entry).expanduser()
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*")))
        else:
            expanded.append(path)
    return [path for path in expanded if path.is_file()]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "convert":
        files = _expand_paths(args.paths)
        documents = batch_convert(files, recipe=args.recipe, llm_ok=not args.no_llm)
        write_jsonl(documents, args.out)
        return 0
    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

