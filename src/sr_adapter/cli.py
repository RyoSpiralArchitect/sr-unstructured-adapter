"""Command line interface entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import convert, convert_many


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert files into normalized JSON blocks.")
    sub = parser.add_subparsers(dest="command", required=True)

    convert_parser = sub.add_parser("convert", help="Convert one or more files")
    convert_parser.add_argument("paths", nargs="+", help="Input file paths")
    convert_parser.add_argument("--recipe", default="default", help="Recipe name to apply")
    convert_parser.add_argument("--out", required=True, help="Destination JSONL file")
    convert_parser.add_argument(
        "--no-llm",
        dest="llm_ok",
        action="store_false",
        help="Disable low-confidence LLM escalation",
    )

    args = parser.parse_args(argv)
    if args.command == "convert":
        paths = [Path(p) for p in args.paths]
        if len(paths) == 1:
            convert(paths[0], args.recipe, args.out, llm_ok=args.llm_ok)
        else:
            convert_many(paths, args.recipe, args.out, llm_ok=args.llm_ok)
        return 0
    parser.error("Unknown command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
