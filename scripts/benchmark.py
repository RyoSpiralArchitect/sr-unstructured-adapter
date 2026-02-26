#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate an ablation report for the escalation confidence gate.

This script compares three knobs that strongly influence the unstructured→structured
pipeline quality/cost trade-off:

- use_structural_gate: apply the structural confidence threshold (max_confidence)
- use_semantic_gate: compute + apply deterministic semantic confidence gating
- use_aif: use the learned escalation meta-model vs. a naive gate

It produces JSON + Markdown reports and can optionally update README benchmarks
between marker comments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sr_adapter.escalation import get_escalation_policy, reset_escalation_policy
from sr_adapter.schema import Block
from sr_adapter.confidence import semantic_confidence, structural_confidence
from sr_adapter.semantic import annotate_semantic_confidence


README_MARKER_START = "<!-- BENCHMARK:START -->"
README_MARKER_END = "<!-- BENCHMARK:END -->"


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class PolicySpec:
    max_confidence: float | None = None
    max_semantic_confidence: float | None = None
    allow_types: tuple[str, ...] = ()
    limit: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PolicySpec":
        payload = dict(data or {})
        max_conf = _safe_float(payload.get("max_confidence"))
        max_sem = _safe_float(payload.get("max_semantic_confidence"))
        allow = payload.get("allow_types") or ()
        if isinstance(allow, str):
            allow_types = (allow,)
        else:
            allow_types = tuple(str(item) for item in allow if str(item).strip())
        limit = payload.get("limit")
        limit_value = int(limit) if isinstance(limit, int) or (isinstance(limit, str) and limit.strip().isdigit()) else None
        if isinstance(limit_value, int) and limit_value <= 0:
            limit_value = None
        return cls(
            max_confidence=max_conf,
            max_semantic_confidence=max_sem,
            allow_types=allow_types,
            limit=limit_value,
        )


@dataclass(frozen=True)
class Case:
    id: str
    blocks: tuple[Block, ...]
    policy: PolicySpec
    expected: tuple[int, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Case":
        case_id = str(data.get("id") or "").strip()
        if not case_id:
            raise ValueError("case.id must be non-empty")
        policy = PolicySpec.from_dict(data.get("policy"))
        expected_raw = data.get("expected") or []
        if not isinstance(expected_raw, list) or not all(isinstance(idx, int) for idx in expected_raw):
            raise ValueError(f"case.expected must be a list[int] (case={case_id})")
        blocks_raw = data.get("blocks") or []
        if not isinstance(blocks_raw, list) or not blocks_raw:
            raise ValueError(f"case.blocks must be a non-empty list (case={case_id})")
        blocks: list[Block] = []
        for item in blocks_raw:
            if not isinstance(item, Mapping):
                raise ValueError(f"case.blocks items must be objects (case={case_id})")
            block_type = str(item.get("type") or "paragraph")
            text = str(item.get("text") or "")
            conf = _safe_float(item.get("confidence"))
            attrs = item.get("attrs")
            attrs_payload = dict(attrs) if isinstance(attrs, Mapping) else {}
            blocks.append(
                Block(
                    type=block_type,  # type: ignore[arg-type]
                    text=text,
                    confidence=float(conf) if conf is not None else 0.5,
                    attrs=attrs_payload,
                )
            )
        return cls(
            id=case_id,
            blocks=tuple(blocks),
            policy=policy,
            expected=tuple(expected_raw),
        )


def _load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ValueError(f"Expected JSON object on line {line_no} in {path}")
        cases.append(Case.from_dict(data))
    if not cases:
        raise ValueError(f"No benchmark cases loaded from {path}")
    return cases


def _naive_select(
    blocks: Sequence[Block],
    *,
    max_confidence: float | None,
    max_semantic_confidence: float | None,
    allow_types: Sequence[str] = (),
    limit: int | None,
) -> list[int]:
    if isinstance(limit, int) and limit <= 0:
        return []

    allow_set = {t for t in allow_types or ()}
    enforce_types = bool(allow_set)

    candidates: list[tuple[int, float, bool]] = []
    semantic_forced: list[int] = []
    for idx, block in enumerate(blocks):
        if enforce_types and block.type not in allow_set:
            continue
        struct_score = structural_confidence(block)
        struct_low = max_confidence is not None and struct_score <= max_confidence
        semantic_score: float | None = None
        semantic_low = False
        if max_semantic_confidence is not None:
            semantic_score = semantic_confidence(block)
            if semantic_score is not None and semantic_score <= max_semantic_confidence:
                semantic_low = True
        if max_confidence is not None or max_semantic_confidence is not None:
            if not (struct_low or semantic_low):
                continue
        effective = 1.0
        if struct_low:
            effective = min(effective, float(struct_score))
        if semantic_low and semantic_score is not None:
            effective = min(effective, float(semantic_score))
        candidates.append((idx, effective, semantic_low))
        if semantic_low:
            semantic_forced.append(idx)

    selected: list[int] = []
    for index in semantic_forced:
        selected.append(index)
        if isinstance(limit, int) and limit > 0 and len(selected) >= limit:
            return selected

    remaining = [row for row in candidates if row[0] not in set(selected)]
    remaining.sort(key=lambda row: (row[1], row[0]))
    for idx, _, _ in remaining:
        selected.append(idx)
        if isinstance(limit, int) and limit > 0 and len(selected) >= limit:
            break
    return selected


@dataclass(frozen=True)
class RunConfig:
    use_structural_gate: bool
    use_semantic_gate: bool
    use_aif: bool

    @property
    def key(self) -> str:
        return (
            f"structural_gate={int(self.use_structural_gate)} "
            f"semantic_gate={int(self.use_semantic_gate)} "
            f"aif={int(self.use_aif)}"
        )


@dataclass
class CaseResult:
    id: str
    expected: tuple[int, ...]
    predicted: tuple[int, ...]
    latency_ms: float

    @property
    def tp(self) -> int:
        return len(set(self.expected).intersection(self.predicted))

    @property
    def fp(self) -> int:
        return len(set(self.predicted) - set(self.expected))

    @property
    def fn(self) -> int:
        return len(set(self.expected) - set(self.predicted))


def _score(results: Sequence[CaseResult]) -> dict[str, float]:
    tp = sum(item.tp for item in results)
    fp = sum(item.fp for item in results)
    fn = sum(item.fn for item in results)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    latencies = [item.latency_ms for item in results]
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "latency_ms_mean": float(statistics.mean(latencies)) if latencies else 0.0,
        "latency_ms_p50": float(_percentile(latencies, 50.0)),
        "latency_ms_p95": float(_percentile(latencies, 95.0)),
    }


def _format_md_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| use_structural_gate | use_semantic_gate | use_aif | precision | recall | F1 | mean ms | p50 ms | p95 ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {struct} | {sem} | {aif} | {p:.3f} | {r:.3f} | {f1:.3f} | {mean:.2f} | {p50:.2f} | {p95:.2f} |".format(
                struct=int(bool(row["use_structural_gate"])),
                sem=int(bool(row["use_semantic_gate"])),
                aif=int(bool(row["use_aif"])),
                p=float(row["precision"]),
                r=float(row["recall"]),
                f1=float(row["f1"]),
                mean=float(row["latency_ms_mean"]),
                p50=float(row["latency_ms_p50"]),
                p95=float(row["latency_ms_p95"]),
            )
        )
    return "\n".join(lines)


def _update_readme(readme_path: Path, snippet: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if README_MARKER_START not in text or README_MARKER_END not in text:
        raise ValueError(
            f"README markers not found. Add {README_MARKER_START} and {README_MARKER_END} to {readme_path}"
        )
    before, rest = text.split(README_MARKER_START, 1)
    _, after = rest.split(README_MARKER_END, 1)
    updated = f"{before}{README_MARKER_START}\n{snippet}\n{README_MARKER_END}{after}"
    readme_path.write_text(updated, encoding="utf-8")


def _run_one(
    cases: Sequence[Case],
    *,
    config: RunConfig,
) -> tuple[list[CaseResult], dict[str, float]]:
    reset_escalation_policy()
    engine = get_escalation_policy()

    case_results: list[CaseResult] = []
    for case in cases:
        blocks: list[Block] = [block for block in case.blocks]
        max_conf = case.policy.max_confidence if config.use_structural_gate else None
        max_sem = case.policy.max_semantic_confidence if config.use_semantic_gate else None
        start = time.perf_counter()
        if config.use_semantic_gate and max_sem is not None:
            blocks = list(annotate_semantic_confidence(blocks))
        if config.use_aif:
            selection = engine.evaluate(
                blocks,
                max_confidence=max_conf,
                max_semantic_confidence=max_sem,
                allow_types=case.policy.allow_types,
                limit=case.policy.limit,
            )
            predicted = tuple(int(idx) for idx in selection.indices)
        else:
            predicted = tuple(
                _naive_select(
                    blocks,
                    max_confidence=max_conf,
                    max_semantic_confidence=max_sem,
                    allow_types=case.policy.allow_types,
                    limit=case.policy.limit,
                )
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        case_results.append(
            CaseResult(
                id=case.id,
                expected=tuple(case.expected),
                predicted=predicted,
                latency_ms=float(latency_ms),
            )
        )

    score = _score(case_results)
    return case_results, score


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/escalation_benchmark.jsonl"),
        help="Benchmark dataset (JSONL)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("bench_report.json"),
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("bench_report.md"),
        help="Write Markdown report to this path",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Update README benchmark section between marker comments",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="README to update when --update-readme is provided",
    )
    parser.add_argument(
        "--use-encoder-context",
        type=str,
        default=None,
        help="Deprecated alias for --use-semantic-gate (true/false)",
    )
    parser.add_argument(
        "--use-structural-gate",
        type=str,
        default=None,
        help="When set, only run this value for use_structural_gate (true/false)",
    )
    parser.add_argument(
        "--use-semantic-gate",
        type=str,
        default=None,
        help="When set, only run this value for use_semantic_gate (true/false)",
    )
    parser.add_argument(
        "--use-aif",
        type=str,
        default=None,
        help="When set, only run this value for use_aif (true/false)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    dataset = args.dataset
    if not dataset.exists():
        print(f"❌ Dataset not found: {dataset}", file=sys.stderr)
        return 2

    cases = _load_cases(dataset)

    if args.use_encoder_context is not None and args.use_semantic_gate is not None:
        print("❌ Provide only one of --use-encoder-context and --use-semantic-gate", file=sys.stderr)
        return 2

    structural_filter = None if args.use_structural_gate is None else _parse_bool(args.use_structural_gate)
    semantic_filter = None
    if args.use_semantic_gate is not None:
        semantic_filter = _parse_bool(args.use_semantic_gate)
    elif args.use_encoder_context is not None:
        semantic_filter = _parse_bool(args.use_encoder_context)
    aif_filter = None if args.use_aif is None else _parse_bool(args.use_aif)

    configs: list[RunConfig] = []
    for structural in (False, True):
        if structural_filter is not None and structural != structural_filter:
            continue
        for semantic in (False, True):
            if semantic_filter is not None and semantic != semantic_filter:
                continue
            for aif in (False, True):
                if aif_filter is not None and aif != aif_filter:
                    continue
                configs.append(
                    RunConfig(
                        use_structural_gate=structural,
                        use_semantic_gate=semantic,
                        use_aif=aif,
                    )
                )

    rows: list[dict[str, Any]] = []
    combos: list[dict[str, Any]] = []

    for config in configs:
        case_results, score = _run_one(cases, config=config)
        row = {
            "use_structural_gate": config.use_structural_gate,
            "use_semantic_gate": config.use_semantic_gate,
            "use_aif": config.use_aif,
            **score,
        }
        rows.append(row)
        combos.append(
            {
                "config": dataclasses.asdict(config),
                "score": score,
                "cases": [dataclasses.asdict(item) for item in case_results],
            }
        )

    md_table = _format_md_table(rows)
    md_report = "\n".join(
        [
            "# Escalation policy ablation report",
            "",
            f"- Generated: {datetime.now(UTC).isoformat()}",
            f"- Dataset: `{dataset}` ({len(cases)} case(s))",
            f"- Python: {platform.python_version()}",
            f"- Platform: {platform.platform()}",
            "",
            md_table,
            "",
        ]
    )

    json_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset),
        "case_count": len(cases),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "combos": combos,
    }

    args.out_json.write_text(json.dumps(json_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(md_report, encoding="utf-8")
    print(md_report)

    if args.update_readme:
        snippet = "\n".join(
            [
                md_table,
                "",
                f"_Generated from `{dataset}` via `python scripts/benchmark.py --update-readme`._",
            ]
        )
        _update_readme(args.readme, snippet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
