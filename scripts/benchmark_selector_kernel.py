"""Paired selector timings with fixed corpora and exact output parity."""

import argparse
import hashlib
import json
import platform
import statistics
import time

from sr_adapter.native.selector import find_text_occurrence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = [
        ("overlapping_japanese", "あ" * 200000, "あ" * 2048, 50000),
        ("overlapping_ascii", "a" * 200000, "a" * 2048, 50000),
        (
            "ordinary_lines",
            "Review requires approval.\n" * 2000,
            "requires approval.",
            1200,
        ),
    ]
    start = time.perf_counter()
    find_text_occurrence("warm", "warm", 0, backend="native")
    cold = time.perf_counter() - start
    results = []
    for name, text, needle, occurrence in cases:
        times = {"python": [], "native": [], "auto": []}
        expected = find_text_occurrence(text, needle, occurrence, backend="python")
        for trial in range(7):
            for backend in (
                ("python", "native", "auto")
                if trial % 2 == 0
                else ("auto", "native", "python")
            ):
                start = time.perf_counter()
                value = find_text_occurrence(text, needle, occurrence, backend=backend)
                times[backend].append(time.perf_counter() - start)
                assert value == expected
        medians = {key: statistics.median(values) for key, values in times.items()}
        results.append(
            {
                "case": name,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "needle_sha256": hashlib.sha256(needle.encode()).hexdigest(),
                "text_chars": len(text),
                "needle_chars": len(needle),
                "occurrence": occurrence,
                "offset": expected,
                "seconds": times,
                "medians": medians,
                "python_over_auto_ratio": medians["python"] / medians["auto"],
            }
        )
    receipt = {
        "version": 1,
        "platform": platform.platform(),
        "trials": 7,
        "warm_or_compile_seconds": cold,
        "results": results,
        "scope": "Literal selector only. Repetitive stress cases are not representative document throughput.",
    }
    with open(args.out, "x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            [
                {
                    "case": r["case"],
                    "medians": r["medians"],
                    "ratio": r["python_over_auto_ratio"],
                }
                for r in results
            ]
        )
    )


if __name__ == "__main__":
    main()
