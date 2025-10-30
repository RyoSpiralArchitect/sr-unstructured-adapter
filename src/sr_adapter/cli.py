"""Command line interface for the adapter pipeline."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Iterable, List

from .adapter import build_payload, to_unified_payload
from .pipeline import convert
from .writer import write_jsonl


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert unstructured files into Blocks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert files to JSONL")
    convert_parser.add_argument("paths", nargs="+", help="Files or directories to convert")
    convert_parser.add_argument("--recipe", default="default", help="Recipe to apply")
    convert_parser.add_argument("--out", default="-", help="Destination JSONL file or '-' for stdout")
    convert_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable low confidence LLM escalation",
    )
    convert_parser.add_argument(
        "--format",
        choices=("document", "payload", "unified"),
        default="document",
        help="Choose the output representation",
    )
    convert_parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop with a non-zero exit code if any file fails to convert",
    )

    return parser.parse_args(argv)


def _expand_paths(paths: Iterable[str]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for entry in paths:
        path = Path(entry).expanduser()
        if path.is_dir():
            candidates: Iterable[Path] = sorted(p for p in path.rglob("*") if p.is_file())
        else:
            candidates = [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            expanded.append(candidate)
    return expanded


def _handle_error(path: Path, exc: Exception) -> dict[str, object]:
    message = f"Failed to convert {path}: {exc}"
    print(message, file=sys.stderr)
    debug = "".join(traceback.format_exception(exc)).strip()
    return {
        "source": str(path),
        "ok": False,
        "error": message,
        "error_type": type(exc).__name__,
        "details": debug,
    }


def _convert_documents(paths: Iterable[Path], *, recipe: str, llm_ok: bool, format: str) -> List[object]:
    results: List[object] = []
    for path in paths:
        try:
            if format == "payload":
                results.append(build_payload(path).to_dict())
            elif format == "unified":
                results.append(to_unified_payload(path))
            else:
                results.append(convert(path, recipe=recipe, llm_ok=llm_ok))
        except Exception as exc:  # pragma: no cover - defensive; exercised via CLI test
            results.append(_handle_error(path, exc))
    return results


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "convert":
        files = _expand_paths(args.paths)
        if not files:
            print("No files to convert", file=sys.stderr)
            return 1
        documents = _convert_documents(
            files,
            recipe=args.recipe,
            llm_ok=not args.no_llm,
            format=args.format,
        )
        write_jsonl(documents, args.out)
        had_errors = any(isinstance(item, dict) and not item.get("ok", True) for item in documents)
        if had_errors and args.strict:
            return 1
        return 0
    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

