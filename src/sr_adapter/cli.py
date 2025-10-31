# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command line interface for the adapter pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys

from .pipeline import batch_convert
from .writer import write_jsonl
from .drivers import DriverManager
from .normalizer import LLMNormalizer
from .recipe import load_recipe


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

    llm_parser = subparsers.add_parser("llm", help="LLM driver utilities")
    llm_subparsers = llm_parser.add_subparsers(dest="llm_command", required=True)

    list_parser = llm_subparsers.add_parser("list-tenants", help="List configured tenants")
    list_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Return success without printing when no tenants are configured",
    )

    run_parser = llm_subparsers.add_parser("run", help="Run a prompt against a tenant driver")
    run_parser.add_argument("--recipe", default="default", help="Recipe providing the LLM config")
    run_parser.add_argument(
        "--tenant",
        help="Tenant to use (defaults to recipe tenant or SR_ADAPTER_TENANT)",
    )
    run_parser.add_argument("--prompt", help="Prompt text (overrides stdin)")
    run_parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Read prompt text from the given file (overrides stdin)",
    )
    run_parser.add_argument(
        "--metadata",
        help="Optional JSON object passed through to the driver as metadata",
    )
    run_parser.add_argument(
        "--metadata-file",
        type=Path,
        help="Read JSON metadata from a file (mutually exclusive with --metadata)",
    )

    replay_parser = llm_subparsers.add_parser(
        "replay",
        help="Replay prompts from a JSONL dataset against a tenant driver",
    )
    replay_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a JSONL file containing prompts (fields: text/prompt, metadata)",
    )
    replay_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSONL destination for normalized responses",
    )
    replay_parser.add_argument("--recipe", default="default", help="Recipe providing the LLM config")
    replay_parser.add_argument(
        "--tenant",
        help="Tenant to use (defaults to recipe tenant or SR_ADAPTER_TENANT)",
    )
    replay_parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many successful prompts from the dataset",
    )
    replay_parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue replay when a record is invalid or the driver call fails",
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
    if args.command == "llm":
        return _handle_llm(args)
    raise ValueError(f"Unhandled command: {args.command}")


def _handle_llm(args: argparse.Namespace) -> int:
    manager = DriverManager()
    if args.llm_command == "list-tenants":
        tenants = manager.tenant_manager.list_tenants()
        if tenants:
            for tenant in tenants:
                print(tenant)
        elif not args.quiet:
            print("No tenants configured", file=sys.stderr)
        return 0
    if args.llm_command == "run":
        try:
            prompt = _resolve_prompt(args.prompt, args.prompt_file)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        recipe = load_recipe(args.recipe)
        llm_config = dict(recipe.llm or {})
        if not llm_config.get("enable"):
            print(f"Recipe '{recipe.name}' does not enable LLM escalation", file=sys.stderr)
            return 3
        tenant = args.tenant or llm_config.get("tenant") or manager.tenant_manager.get_default_tenant()
        try:
            metadata = _resolve_metadata(args.metadata, args.metadata_file)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 4
        driver = manager.get_driver(str(tenant), llm_config)
        try:
            raw = driver.generate(prompt, metadata=metadata)
        except Exception as exc:  # pragma: no cover - runtime failure path
            print(
                f"LLM driver '{driver.name}' failed for tenant '{tenant}': {exc}",
                file=sys.stderr,
            )
            return 10
        normalizer = LLMNormalizer()
        normalized = normalizer.normalize(driver.name, raw, prompt=prompt)
        json.dump(asdict(normalized), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.llm_command == "replay":
        dataset = args.input.expanduser()
        if not dataset.exists():
            print(f"Dataset '{dataset}' does not exist", file=sys.stderr)
            return 5
        if not dataset.is_file():
            print(f"Dataset '{dataset}' is not a file", file=sys.stderr)
            return 6
        recipe = load_recipe(args.recipe)
        llm_config = dict(recipe.llm or {})
        if not llm_config.get("enable"):
            print(f"Recipe '{recipe.name}' does not enable LLM escalation", file=sys.stderr)
            return 3
        tenant = args.tenant or llm_config.get("tenant") or manager.tenant_manager.get_default_tenant()
        driver = manager.get_driver(str(tenant), llm_config)
        normalizer = LLMNormalizer()
        output_handle = None
        if args.output:
            output_path = args.output.expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w", encoding="utf-8")
        stats = ReplayStats()
        try:
            with dataset.open("r", encoding="utf-8") as input_handle:
                for line_number, line in enumerate(input_handle, 1):
                    if args.limit is not None and stats.processed >= args.limit:
                        stats.limit_reached = True
                        break
                    stripped = line.strip()
                    if not stripped:
                        continue
                    stats.attempted += 1
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        message = f"Invalid JSON payload on line {line_number}: {exc}"
                        if args.skip_errors:
                            stats.errors.append(message)
                            continue
                        print(message, file=sys.stderr)
                        return 7
                    prompt = _extract_prompt(record)
                    if not prompt:
                        message = (
                            f"Record on line {line_number} is missing a prompt/text field"
                        )
                        if args.skip_errors:
                            stats.errors.append(message)
                            continue
                        print(message, file=sys.stderr)
                        return 8
                    metadata = record.get("metadata")
                    try:
                        raw = driver.generate(prompt, metadata=metadata)
                    except Exception as exc:  # pragma: no cover - runtime failure path
                        message = f"LLM driver '{driver.name}' failed on line {line_number}: {exc}"
                        if args.skip_errors:
                            stats.errors.append(message)
                            continue
                        print(message, file=sys.stderr)
                        return 10
                    normalized = normalizer.normalize(driver.name, raw, prompt=prompt)
                    payload = {
                        "record": record,
                        "response": asdict(normalized),
                        "tenant": tenant,
                        "driver": driver.name,
                    }
                    target = output_handle or sys.stdout
                    json.dump(payload, target, ensure_ascii=False)
                    target.write("\n")
                    stats.processed += 1
        finally:
            if output_handle:
                output_handle.close()
        if stats.processed == 0:
            print("No prompts were processed from the dataset", file=sys.stderr)
            return 9
        if args.skip_errors or args.limit is not None:
            _report_replay_summary(stats, limit=args.limit)
        if stats.errors:
            return 10
        return 0
    raise ValueError(f"Unhandled llm command: {args.llm_command}")


def _resolve_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt and prompt_file:
        raise ValueError("Provide either --prompt or --prompt-file, not both")
    if prompt:
        text = prompt
    elif prompt_file:
        text = prompt_file.expanduser().read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    text = text.strip("\n")
    if not text:
        raise ValueError("Prompt text is empty")
    return text


def _resolve_metadata(metadata: str | None, metadata_file: Path | None) -> dict | None:
    if metadata and metadata_file:
        raise ValueError("Provide either --metadata or --metadata-file, not both")
    if metadata_file:
        path = metadata_file.expanduser()
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Failed to read metadata file '{path}': {exc}") from exc
        try:
            return json.loads(contents) if contents.strip() else None
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid metadata JSON in file '{path}': {exc}") from exc
    if metadata:
        try:
            return json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid metadata JSON: {exc}") from exc
    return None


def _extract_prompt(record: dict | None) -> str | None:
    if not isinstance(record, dict):
        return None
    prompt = (
        record.get("prompt")
        or record.get("text")
        or record.get("input")
        or record.get("content")
    )
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return None


@dataclass
class ReplayStats:
    processed: int = 0
    attempted: int = 0
    errors: list[str] = field(default_factory=list)
    limit_reached: bool = False


def _report_replay_summary(stats: ReplayStats, *, limit: int | None, error_limit: int = 5) -> None:
    attempted_message = (
        f"{stats.processed} prompt(s) processed out of {stats.attempted} record(s)"
        if stats.attempted
        else f"{stats.processed} prompt(s) processed"
    )
    print(attempted_message, file=sys.stderr)
    if limit is not None and stats.limit_reached:
        print(f"Stopped after reaching limit of {limit} successful prompt(s)", file=sys.stderr)
    if stats.errors:
        print(f"{len(stats.errors)} record(s) failed:", file=sys.stderr)
        for message in stats.errors[:error_limit]:
            print(f"  - {message}", file=sys.stderr)
        remaining = len(stats.errors) - error_limit
        if remaining > 0:
            print(f"  ... {remaining} more", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

