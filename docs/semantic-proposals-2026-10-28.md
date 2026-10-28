# Semantic proposal observation — 2026-10-28

GPT-5.5 produced a structurally valid semantic proposal from each of two synthetic
Japanese sources: a DOCX and a PNG carrying the same six paragraphs. Each response
contained seven claims. This is a small, inspected example of the proposal stage,
not an extraction accuracy benchmark or a comparison against a raw VLM.

## Inputs and execution

The probe used `scripts/probe_semantic_proposals.py`. Its source inventories were
handwritten fixtures: seven DOCX text units including the heading, and six image
observations excluding the heading. The image request also included the original
PNG. Source hashes, prompts, model and review expectations were fixed before calls.

The first credential returned HTTP 429 `credit_balance_exhausted` for both formats.
Those responses were retained. An explicitly selected, already available second
credential was used for one call per format, reusing the **same archived source
bundles and byte-identical requests**. No response-dependent prompt adjustment,
model change or automatic retry was made. Overall: four requests, two quota
failures, two completed responses; no source-extraction model calls.

| Observation | DOCX | Image |
| --- | --- | --- |
| Response model | `gpt-5.5-2026-04-23` | `gpt-5.5-2026-04-23` |
| Candidate claims | 7 | 7 |
| Model-reported represented units | 5 | 5 |
| Model-reported context-only units | 2 | 1 |
| Model-reported unresolved units | 0 | 0 |
| Input / output tokens | 9,157 / 4,109 | 6,436 / 4,485 |

These coverage numbers are the model's own accounting. Zero unresolved units
does not mean the interpretation is correct or complete. Portable replay and
SQLite readback were checked separately; they establish transport consistency.
The [observation receipt](validation/semantic-proposals-2026-10-28/observation.json)
records source/request/response hashes and those local checks. Raw synthetic
bundles and provider responses were retained locally, not committed here.

## Inspection against the frozen expectations

This inspection was performed by the coding assistant, not by an independent
human reviewer or a blinded judge. Candidates remain `unreviewed`.

| Frozen expectation | Observed representation | Limit of the observation |
| --- | --- | --- |
| Approval AND NOT emergency stop stays inside the obligation | Both have one `required` claim over an implication with a conjunctive antecedent and a negated stop condition. No separate processing claim. | Source arguments are still coarse paragraph/observation references. |
| Exact monetary ceiling and unit survive | Both retain decimal lexical `10000000.00` and currency `円`. | The DOCX limit is a separate atom in the consequent, without an explicit target argument; its association relies on the enclosing expression. |
| Not required is distinguished from prohibited | Both use `denied + required` and `denied + prohibited` claims for prior approval. | This checks this explicit example; it does not establish general handling of deontic language. |
| Unknown owner stays unknown | Both retain an `unknown` owner claim and an unknown department value. | No identity-resolution quality was measured. |
| Dates survive without an invented timezone | Both retain date literals and leave timezone-qualified validity null. | DOCX places the period in a separate claim; its connection to the obligation is only a free-text cross-reference. Image also cites the period unit on the obligation. |
| Viewpoints and dissent stay separate | Both keep separate application-stance and objection claims with a `challenges` link, without choosing a winner. | DOCX uses unspecified-speaker attribution; image attributes the objection to the audit role. This narrator-versus-actor distinction still needs review. |
| Quoted instructions stay source data | Both mark the quoted instruction as context and retain the processing conditions. | One synthetic injection-like sentence is not a security evaluation. |

## Concrete next improvements

1. **Finer role anchors.** Both outputs sometimes use the same whole paragraph or
   image observation as multiple role values, such as actor and object. Valid
   references do not distinguish the mentions inside that paragraph. Introduce
   source-bound mention/span selection before downstream entity joins; do not
   infer equality from identical source-record references or equal labels.
2. **Explicit rule-to-period scope.** Make applicability and scope relationships
   inspectable as structured links, instead of relying on a note such as “see c5”.
3. **Attribution boundaries.** Preserve who reports a statement separately from
   the person or department whose position is described. A descriptive report of
   an objection should not silently become that actor's verified utterance.
4. **Independent samples and review.** Broader documents, imperfect inventories,
   diagrams with omitted regions, and independent review are still needed before
   estimating accuracy. The present observation used text boxes with supplied
   readings, not an unseen free-form diagram benchmark.

No Fabric/Foundry resources were modified, and no generated claim was published,
approved, or promoted to a fact by this probe.
