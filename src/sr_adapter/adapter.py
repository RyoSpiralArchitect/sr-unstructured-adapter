
"""Public interface for building unified payloads from files."""

from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

from .loaders import read_file_contents
from .metadata import collect_metadata
from .models import Payload

mimetypes.init()

# ---- tunables ---------------------------------------------------------------
DEFAULT_MAX_SIZE_MB = float(os.getenv("SR_ADAPTER_MAX_SIZE_MB", "200"))
# ----------------------------------------------------------------------------


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _guess_mime(path: Path) -> str:
    """Heuristic MIME guess with common overrides."""
    # explicit overrides by suffix
    suf = path.suffix.lower()
    if suf in {".json"}:      return "application/json"
    if suf in {".jsonl", ".ndjson"}: return "application/x-ndjson"
    if suf in {".md", ".markdown"}:  return "text/markdown"
    if suf in {".txt", ".log"}:      return "text/plain"
    if suf in {".yaml", ".yml"}:     return "application/yaml"
    if suf in {".toml"}:             return "application/toml"
    if suf in {".csv"}:              return "text/csv"
    if suf in {".tsv"}:              return "text/tab-separated-values"
    if suf in {".xml"}:              return "application/xml"
    if suf in {".rtf"}:              return "application/rtf"
    if suf in {".eml"}:              return "message/rfc822"
    if suf in {".msg"}:              return "application/vnd.ms-outlook"
    if suf in {".odt"}:              return "application/vnd.oasis.opendocument.text"
    if suf in {".ods"}:              return "application/vnd.oasis.opendocument.spreadsheet"
    if suf in {".odp"}:              return "application/vnd.oasis.opendocument.presentation"
    if suf in {".epub"}:             return "application/epub+zip"
    if suf in {".ics"}:              return "text/calendar"
    if suf in {".vcf"}:              return "text/vcard"
    if suf in {".docx"}:             return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suf in {".pptx"}:             return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if suf in {".xlsx"}:             return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if suf in {".pdf"}:              return "application/pdf"
    if suf in {".html", ".htm"}:     return "text/html"
    # fallback to stdlib
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    # optional: python-magic if present
    try:
        import magic  # type: ignore
        m = magic.from_file(str(path), mime=True)
        if isinstance(m, str) and m:
            return m
    except Exception:
        pass
    return "application/octet-stream"


def _stat_guard(path: Path, *, max_size_mb: float = DEFAULT_MAX_SIZE_MB) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Not a regular file: {path}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"File too large ({size_mb:.1f} MB > {max_size_mb} MB): {path}")


def build_payload(path: str | Path) -> Payload:
    """Return a :class:`Payload` for *path*."""
    normalized = _normalize_path(path)
    _stat_guard(normalized)
    mime = _guess_mime(normalized)
    text, extra_meta = read_file_contents(normalized, mime)
    metadata = collect_metadata(normalized, text, mime, extra_meta)
    return Payload(source=normalized, mime=mime, text=text, meta=metadata)


def to_unified_payload(path: str | Path) -> dict:
    """Return a JSON-serialisable payload for *path*."""
    return build_payload(path).to_dict()


def _iter_files(root_or_files: Sequence[str | Path], *, recursive: bool = False) -> Iterator[Path]:
    for item in root_or_files:
        p = _normalize_path(item)
        if p.is_dir():
            if recursive:
                yield from (q for q in p.rglob("*") if q.is_file())
            else:
                yield from (q for q in p.iterdir() if q.is_file())
        else:
            yield p


def stream_payloads(paths: Iterable[str | Path], *, recursive: bool = False) -> Iterator[Payload]:
    """Yield payloads for each path in *paths*.

    Note: exceptions are propagated. Wrap in try/except if you need partial success.
    """
    for p in _iter_files(list(paths), recursive=recursive):
        yield build_payload(p)


# ----------------------------- CLI ------------------------------------------

def _emit(obj: object, *, as_json_lines: bool) -> None:
    if as_json_lines:
        print(json.dumps(obj, ensure_ascii=False))
    else:
        print(json.dumps(obj, ensure_ascii=False, indent=2))


def main(argv: List[str] | None = None) -> int:
    """Simple CLI entry-point used by ``python -m sr_adapter.adapter``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Files or directories")
    parser.add_argument("--as-json-lines", action="store_true",
                        help="Emit one JSON object per line instead of a list")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recurse into directories")
    parser.add_argument("--parallel", "-j", type=int, default=0,
                        help="Threaded I/O (0=off, N=threads)")
    parser.add_argument("--ignore-glob", action="append", default=[],
                        help="Glob to ignore (can be repeated)")
    args = parser.parse_args(argv)

    files = [p for p in _iter_files(args.paths, recursive=args.recursive)]
    if args.ignore_glob:
        import fnmatch
        files = [p for p in files if not any(fnmatch.fnmatch(str(p), g) for g in args.ignore_glob)]

    # parallel I/O (safe for read-only operations)
    results: List[dict] = []
    errors: List[dict] = []

    def _task(p: Path) -> dict:
        try:
            return build_payload(p).to_dict()
        except Exception as e:
            return {"error": str(e), "path": str(p)}

    if args.parallel and args.parallel > 0:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(_task, p): p for p in files}
            for f in as_completed(futs):
                out = f.result()
                if "error" in out:
                    errors.append(out)
                else:
                    if args.as_json_lines:
                        _emit(out, as_json_lines=True)
                    else:
                        results.append(out)
    else:
        for p in files:
            out = _task(p)
            if "error" in out:
                errors.append(out)
            else:
                if args.as_json_lines:
                    _emit(out, as_json_lines=True)
                else:
                    results.append(out)

    if not args.as_json_lines:
        _emit(results, as_json_lines=False)

    # report errors to stderr as JSONL (非破壊)
    if errors:
        for e in errors:
            sys.stderr.write(json.dumps(e, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution convenience
    raise SystemExit(main())