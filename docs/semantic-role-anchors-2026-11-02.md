# Role anchor development observation — 2026-11-02

The new selector separates phrases inside one saved source unit. In a development
smoke using the same synthetic DOCX and PNG as the [previous observation](semantic-proposals-2026-10-28.md),
both post-fix GPT-5.5 responses selected `審査部` as the processing actor,
`融資システム` as the system, and the **second** `申請A` as the processed application.
These now survive as separate scalar anchor rows, with exact source IDs and offsets.

This is an inspected implementation example, not an extraction accuracy benchmark.
The inventories are still handwritten fixtures; the image call also received the
original PNG. There was no new raw-document extraction, held-out corpus, independent
review, or comparison with a raw VLM. No customer documents were used.

## Development sequence and retained failures

The two original source bundles were reused byte-for-byte. Requests, model
(`gpt-5.5`), expectations and source snapshots were saved before each phase.
The response model was `gpt-5.5-2026-04-23` in all four calls.

1. The first DOCX and image calls returned HTTP 200 and completed model responses,
   but both failed local proposal validation. The model used datatype abbreviations
   such as `xsd:decimal` and `xsd:date`, while the semantic contract requires full IRIs.
   Raw responses and failed receipts were retained; no completed bundle was emitted.
2. Version 2's wire schema was corrected to enumerate the seven supported datatype
   IRIs, matching the existing local validation. The source data, model and textual
   instructions were unchanged. This changes the output schema, so it is a separate
   development attempt, **not** a retry of the same frozen request.
3. One new call per format completed validation and storage. No further calls or
   response-dependent semantic prompt changes were made.

Overall: **4 submitted requests, 2 validation failures, 2 completed bundles,
0 automatic retries**. The [receipt](validation/semantic-role-anchors-2026-11-02/observation.json)
retains request/response hashes and outcomes from both phases. Successful results
must not be presented as four successful calls or as an independent evaluation.

## Observed post-fix output

| Observation | DOCX | Image |
| --- | --- | --- |
| Candidate claims | 8 | 7 |
| Source arguments with exact selections | 19 | 18 |
| Whole-record source arguments | 0 | 0 |
| Model-reported represented / context-only / unresolved units | 5 / 2 / 0 | 5 / 1 / 0 |
| Input / output tokens | 9,629 / 4,666 | 6,907 / 3,851 |
| SQLite inserted rows / repeated write | 153 / 0 | 119 / 0 |

The counts describe these responses, including any unclaimed expressions; they
are not accuracy, completeness or role-selection recall. A matching quote can
still identify the wrong phrase or interpret the role incorrectly.

Both responses use these positions in the same first paragraph/observation:

| Processing role | Selected text | Occurrence (zero-based) | Unicode range `[start,end)` |
| --- | --- | --- | --- |
| Actor | 審査部 | 0 | `[27,30)` |
| System | 融資システム | 0 | `[31,37)` |
| Object/application | 申請A | 1 | `[38,41)` |

The image also selects the first `申請A` at `[5,8)` for the approval condition.
The DOCX instead selects the broader `承認済みの申請A` phrase for that condition.
This illustrates that exact matching does not enforce equal semantic granularity.

Both portable bundles replayed from their saved inputs/request/response. The
`argument_anchors` rows matched SQLite readback, and repeat writes inserted zero
rows. The earlier version 1 DOCX/image bundles also replayed to their original
semantic snapshot IDs, without migration or reissuing model calls.

## Remaining interpretation limits

Inspection was performed by the coding assistant, not an independent or blinded
reviewer. Both outputs retain a required implication with approval and negated
emergency stop in the antecedent, denied-required and denied-prohibited prior
approval, the exact monetary value, date literals without an invented timezone,
an unknown owner-related claim, and separate reported viewpoints. However:

- The DOCX monetary ceiling is still a separate descriptive claim. It now names
  an application occurrence, but has no typed link binding it to the processing
  obligation. The image puts the amount directly on the processing expression.
- The period is still a separate claim with a textual reference/context on the
  obligation; no structured rule-to-period scope was added in this change.
- Both dissent links target the claim describing a person's application stance.
  The disputed rule content is not identified precisely. Consistent unspecified
  speaker attribution in these two responses does not establish general attribution
  correctness.
- Image anchors select saved label text. They do not locate individual glyphs in
  pixels, repair OCR, establish entity identity, or cover unseen diagram regions.

The next slice should make applicability/scope and reporter-versus-described-actor
relationships inspectable, while keeping these source anchors. All proposals
remain unreviewed; nothing was published to Fabric or Foundry.
