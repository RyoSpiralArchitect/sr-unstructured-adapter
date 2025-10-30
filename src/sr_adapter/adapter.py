"""Public interface for building unified payloads from files."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Iterable, Iterator, List

from .loaders import read_file_contents
from .metadata import collect_metadata
from .models import Payload
from .pipeline import convert
from .unify import build_unified_payload

mimetypes.init()


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if path.suffix.lower() == ".json":
        return "application/json"
    return mime or "application/octet-stream"


def build_payload(path: str | Path) -> Payload:
    """Return a :class:`Payload` for *path*."""

    normalized = _normalize_path(path)
    mime = _guess_mime(normalized)
    text, extra_meta = read_file_contents(normalized, mime)
    metadata = collect_metadata(normalized, text, mime, extra_meta)
    return Payload(source=normalized, mime=mime, text=text, meta=metadata)


def to_unified_payload(path: str | Path) -> dict:
    """Return a JSON-serialisable payload for *path*."""

    payload = build_payload(path)
    document = convert(path, recipe="default", llm_ok=False)
    return build_unified_payload(payload, document)


def stream_payloads(paths: Iterable[str | Path]) -> Iterator[Payload]:
    """Yield payloads for each path in *paths*."""

    for path in paths:
        yield build_payload(path)


def main(argv: List[str] | None = None) -> int:
    """Simple CLI entry-point used by ``python -m sr_adapter.adapter``."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Files to normalize")
    parser.add_argument(
        "--as-json-lines",
        action="store_true",
        help="Emit one JSON object per line instead of a list",
    )
    args = parser.parse_args(argv)

    payloads = [build_payload(path).to_dict() for path in args.paths]
    if args.as_json_lines:
        for payload in payloads:
            print(json.dumps(payload, ensure_ascii=False))
    else:
        print(json.dumps(payloads, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution convenience
    raise SystemExit(main())
