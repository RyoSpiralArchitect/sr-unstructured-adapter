# Cross-format ontology reuse and native text batches

The experiment fixes the model to `gpt-6-luna`. Two synthetic documents are
rendered as PDF, DOCX and PNG, giving six inputs with fixed source hashes. A calls
Atlas a loan-review system and assigns review to 審査部. B calls Atlas an audit
team and proposes 部長 review for the same target and amount. B also proposes an
emergency exception to A's execution prohibition. Its proposal is unapproved,
the disagreement unresolved, and the final approver's name absent.

## Recorded runs

| Stage | Requests | Result |
| --- | ---: | --- |
| Initial ingestion | 6 | 3 activated, 3 failed |
| Explicit v5 follow-up of failed inputs | 3 | 2 activated, 1 failed |
| Local recovery of retained v5 DOCX response | 0 | 1 activated with 2 unresolved role values |
| SQLite evidence reuse | 1 | 7 answers with inspected-reference citations |

Initial failures involved an empty vocabulary, an undeclared/incompatible term,
and date-only strings in fields requiring timezone-bearing timestamps. The
source-agnostic v5 generation profile addresses these structural constraints.
The retained v5 DOCX response selected `資料A` inside a paragraph containing
`資料B`, where that exact mention is absent. V5 records those two argument values
as `unknown` with `unresolved_text_selector`, retaining the original selectors,
response and claim evidence.

There were nine extraction requests plus one reuse request, with no automatic
retries. Follow-up was a separately recorded engineering stage after code changes.
The final six active entries contain **53 candidate claims**, 105 unique verified
text anchors, and two ungrounded selectors represented as unknown values.
Earlier failed receipts remain in the database. All active timestamp fields are
null, consistent with source dates that have no explicit timezone.

## Source readings and reuse

Five canonical body sentences are checked per input after removing whitespace
only. Both PDF/DOCX pairs contain all five. PNG readings contain 3/5 and 4/5
complete canonical sentences, respectively. Missing characters in words such as
`名称` and `審査担当` explain several mismatches. Page PNGs and original source
bytes remain recoverable from SQLite for visual checking.

Readback copied SQLite, inspected every active claim, verified source hashes and
anchors, and checked that reading left the database hash unchanged. Thirty-five
vocabulary IRIs occur in more than one entry. Collection, document and snapshot
references remain separate.

The fixed seven-question reuse run distinguished the meanings of Atlas,
retained 審査部 versus 部長 and the emergency condition, kept B's exception as an
unapproved proposal, selected no authoritative winner, and left the final
approver unknown. Every returned citation matched an inspected pinned reference.
Host-selected tool outputs supplied the context. The reuse call consumed 161,296
input and 4,727 output tokens. These are integration observations on this corpus.

Original inspection JSON occupied **1,938,206 bytes**. Lossless context packing
reduced it to **473,899 bytes**, about 75.5% smaller. Expansion restored every
original JSON value exactly. The stored ontologies and source readings retain
their strings and references.

## Native text kernel observation

The binding now reserves the valid input's byte length as output capacity and
normally invokes the bundled transform once. C++ whitespace processing uses one
normalization workspace. Expanding external libraries retain the size-query
fallback and existing C ABI.

Seven paired trials used identical fixed corpora on macOS arm64.

| Corpus | Blocks/batch | Batches/trial | Before median | After median | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short ASCII | 128 | 150 | 0.3272 ms | 0.2672 ms | 1.225× |
| Mixed Japanese | 128 | 50 | 1.2133 ms | 0.6569 ms | 1.847× |
| Larger mixed text | 16 | 10 | 7.8876 ms | 2.6651 ms | 2.960× |

Text, type and confidence matched in every measured case. Independent parity
probes covered 37,449 short exhaustive strings, 1,112,064 Unicode scalar
singletons and 20,000 structured random strings. Measurements cover native batch
latency. Output capacity reserves the input byte length, including whitespace
that may later disappear. Ontology evidence uses the source-reader path and
retains its exact text.

## Validation

The full suite passed 1,474 tests with one skipped; Ruff and source/wheel builds
passed. A clean installed-wheel readback outside the checkout restored all 53
inspections and source bytes, preserved the database hash and exactly expanded
the packed context while PyArrow and PDFium imports were blocked. The installed
native kernel also passed UTF-8 and embedded-NUL smoke cases. A fresh read-only
review found no actionable issues and passed 49 focused tests.

## Reproduce

```sh
# A font supporting Japanese is required. Preparation makes zero model calls.
python scripts/probe_cross_format_ontology.py prepare --out <new-directory> --font <font-file>
python scripts/probe_cross_format_ontology.py ingest --out <directory> --execute
python scripts/probe_cross_format_ontology.py reuse --out <directory> --execute
```

New preparations use v5 across formats. A legacy prepared run can explicitly
invoke `followup` once for pre-v5 failed inputs. `recover` reprocesses saved failed
responses locally. Submission markers prevent another paid attempt, and existing
request bytes are preserved. The corpus, protocol, requests, responses, failures,
tool trace and answers are retained locally under
`/Users/ryohiga/SpiralReality/artifacts/cross-format-ontology-20261103/`.
