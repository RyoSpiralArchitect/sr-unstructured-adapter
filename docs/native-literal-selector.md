# Native literal selector

`resolve_text_selector` retains exact, overlapping, zero-based occurrence
matching and returns Python Unicode code-point offsets. It now delegates expensive
repeated searches to an optional standalone C++ KMP kernel. This supports evidence
anchoring without changing semantic proposal formats or normalizing source text.

Automatic selection requires at least 16,384 source characters, a 32-character
needle, and `(occurrence + 1) * len(needle) >= 1_000_000`. Short searches continue
through Python. These are conservative dispatch heuristics, not a learned policy.
The native path accepts at most 32 MiB of UTF-8 source and 1 MiB of UTF-8 needle.
Unsupported sizes, unpaired surrogates and unavailable compilers fall back to the
original Python search. `SR_ADAPTER_DISABLE_NATIVE_RUNTIME=1` disables native use.

The bundled source compiles into the existing external, source-keyed native cache.
The first eligible call can incur compilation time; no compiler is required to use
the adapter. This kernel has its own ABI and does not modify the text kernel ABI.
Embedded NULs, combining characters and supplementary-plane characters retain
their literal meaning. UTF-8 byte matching is converted to a code-point offset
before returning to Python, which checks that the returned substring matches.

## Validation and bounded performance observation

`tests/test_selector_kernel.py` compares Python and C++ on 1,500 seeded Unicode
cases plus overlapping Japanese, emoji, NUL, absent and empty-source cases. It
also checks native-unavailable and explicitly disabled fallback through the public
selector API.

Run the paired microbenchmark from the repository environment:

```bash
python scripts/benchmark_selector_kernel.py --out /tmp/selector-benchmark.json
```

The output includes input hashes, seven alternating-order trials, matching offsets,
raw times, medians and platform identity. A local macOS arm64 observation on
2026-11-13 produced the following warm-cache medians:

| Synthetic case | Python | Automatic dispatch |
| --- | ---: | ---: |
| 200,000 repeated Japanese characters; 2,048-character needle; occurrence 50,000 | 239.22 ms | 0.249 ms |
| 200,000 repeated ASCII characters; 2,048-character needle; occurrence 50,000 | 296.23 ms | 0.050 ms |
| Ordinary repeated lines; 18-character needle; occurrence 1,200 | 0.131 ms | 0.123 ms |

The first two cases deliberately stress repeated overlapping searches. They do
not measure document extraction, model calls, SQLite throughput or typical
end-to-end speed. The ordinary case remains on Python, and its small timing
difference is measurement noise rather than a claimed optimization. These are
warm-cache observations, not cold compilation timings.
