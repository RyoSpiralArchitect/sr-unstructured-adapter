#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare pytest-benchmark runs against a baseline and enforce a threshold."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, Iterable


def _load_benchmarks(paths: Iterable[pathlib.Path]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"failed to parse benchmark file {path}: {exc}") from exc
        for entry in payload.get("benchmarks", []):
            name = entry.get("name") or entry.get("fullname") or entry.get("id")
            stats = entry.get("stats") or {}
            mean = stats.get("mean")
            if name is None or mean is None:
                continue
            try:
                results[name] = float(mean)
            except (TypeError, ValueError):
                raise RuntimeError(f"invalid mean value in {path} for benchmark {name}")
    return results


def _format_ratio(ratio: float) -> str:
    pct = ratio * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def _write_report(lines: Iterable[str], destination: pathlib.Path) -> None:
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=pathlib.Path)
    parser.add_argument("--current", required=True, nargs="+", type=pathlib.Path)
    parser.add_argument("--max-regression", type=float, default=0.10)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.max_regression < 0:
        parser.error("--max-regression must be non-negative")

    baseline = _load_benchmarks([args.baseline])
    if not baseline:
        print(f"⚠️ No baseline benchmarks found in {args.baseline}", file=sys.stderr)
        return 0

    current_paths = [path for path in args.current if path.exists()]
    if not current_paths:
        print("❌ No benchmark output files found", file=sys.stderr)
        return 1

    current = _load_benchmarks(current_paths)

    missing = sorted(set(baseline) - set(current))
    if missing:
        print(f"❌ Missing benchmarks in current run: {', '.join(missing)}", file=sys.stderr)
        return 1

    lines = [
        "Benchmark regression report",
        f"Baseline: {args.baseline}",
        "Current files:",
    ]
    lines.extend(f"  - {path}" for path in current_paths)
    lines.append(f"Allowed regression: {args.max_regression * 100:.1f}%")
    lines.append("")
    lines.append("Name | Baseline (s) | Current (s) | Delta | Status")
    lines.append("---- | ------------- | ----------- | ----- | ------")

    failed = False
    for name in sorted(baseline):
        base = baseline[name]
        curr = current[name]
        if base <= 0 or math.isnan(base):
            delta_ratio = math.inf
        else:
            delta_ratio = (curr - base) / base
        status = "PASS"
        if delta_ratio > args.max_regression:
            status = "FAIL"
            failed = True
        lines.append(
            f"{name} | {base:.6f} | {curr:.6f} | {_format_ratio(delta_ratio)} | {status}"
        )

    report_path = pathlib.Path("perf_report.txt")
    _write_report(lines, report_path)
    print("\n".join(lines))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
