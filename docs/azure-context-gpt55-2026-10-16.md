# Azure GPT-5.5 paired context probe — 2026-10-16

On these two fixed cases, **both normalized records and row contexts scored 24/24
for answer content and 24/24 for the existing assertion-ID-aware strict metric**.
There was no accuracy winner. Row contexts used 6.3% fewer reported input tokens;
cache counts differed, so this does not establish a cost or latency improvement.

| Metric | Normalized records | Row contexts |
| --- | ---: | ---: |
| Successful requests / planned | 4 / 4 | 4 / 4 |
| Answer content | 24 / 24 | 24 / 24 |
| Strict answer + assertion IDs | 24 / 24 | 24 / 24 |
| Input tokens | 42,694 | 40,002 |
| Cached input tokens (included above) | 20,480 | 10,496 |
| Output tokens | 1,370 | 1,489 |
| Reasoning tokens (included in output) | 128 | 245 |
| Total tokens | 44,064 | 41,491 |

The synthetic case has eight questions covering row/predicate association, a
known and unknown budget, reverse edges, unresolved identity, and an absent
project. The Digital Agency roles case has four questions covering three role
descriptions and an unsupported reporting relation. Each case was repeated twice
per arm, giving 24 answer slots per arm. These are repeated observations of two
fixed cases, not 24 independent documents.

## Frozen conditions and retained failure

The source inputs, system prompt, questions, gold, and scorer were copied without
change from the [existing v2 fixture](validation/foundry-context-2026-10-16/v2/frozen/spec.json).
Gold was used only for local scoring. Requests contained the fixed system text,
one arm's context, and question IDs/text. Replay reconstructs each request and
rejects even a rehashed request that differs from that contract.

Both arms used the same existing Azure GPT-5.5 deployment and project Responses
endpoint, `reasoning.effort=low`, `max_output_tokens=4096`, `store=false`, and
`text.format.type=json_object`. Temperature was omitted. All eight successful raw
responses reported `model=gpt-5.5`. The second repeat reversed both case and arm
order. There were no automatic retries, tools, image inputs, or external retrieval.

The initial v1 attempt received HTTP 400 because the project endpoint required an
explicit input item type. Its raw request/error and receipt are retained; the
remaining seven slots were not sent. V2 changed only the transport envelope to
`type=message` with `content=[{type:input_text,text:...}]`. The prompt text and all
experimental conditions were unchanged and that equality was checked for all
eight planned slots before interpreting results. V2 was separately frozen before
execution. In total there were **nine API attempts: one schema rejection and eight
completed model responses**. The rejection is not counted as a model answer.

- v1 spec SHA-256: `45108d72574046fece3448199eb36895d3adaa3948842672439da12cca6afa5b`
- v2 spec SHA-256: `d1e2eabce379495044770dbd51007f80231be6dc8b78f8251dfbeb00c037879b`
- [Manifest and claim boundaries](validation/azure-context-gpt55-2026-10-16/manifest.json)
- [v1 failed receipt](validation/azure-context-gpt55-2026-10-16/v1/run/receipt.json)
- [v2 completed receipt](validation/azure-context-gpt55-2026-10-16/v2/run/receipt.json)
- [v2 replay totals](validation/azure-context-gpt55-2026-10-16/v2/replay.json)
- [Transport-repair invariance](validation/azure-context-gpt55-2026-10-16/v2/transport-repair-invariance.json)

The original scorer checks status/value equality, unique known assertion IDs, the
presence of required supporting IDs, and absence of unexpected question answers.
It permits additional known assertion IDs; the strict score is not an exact-set
evidence metric or proof of complete citation coverage. Endpoint/resource IDs and
authorization credentials are excluded from the public artifacts. Raw response
objects and request bodies are retained with hashes.

## Replay and bounded reuse

Replay requires no Azure login and makes no model calls. Use each version's frozen
runner so later script edits cannot silently change the interpretation:

```bash
python docs/validation/azure-context-gpt55-2026-10-16/v2/frozen/azure_runner.py replay \
  --frozen docs/validation/azure-context-gpt55-2026-10-16/v2/frozen \
  --run docs/validation/azure-context-gpt55-2026-10-16/v2/run
```

For a new, explicitly authorized eight-request run:

```bash
pip install '.[foundry]'
python scripts/probe_azure_relation_context.py prepare \
  --source-frozen docs/validation/foundry-context-2026-10-16/v2/frozen \
  --endpoint "$AZURE_PROJECT_RESPONSES_ENDPOINT" --out artifacts/azure-probe/frozen
python scripts/probe_azure_relation_context.py run \
  --frozen artifacts/azure-probe/frozen --out artifacts/azure-probe/run \
  --endpoint "$AZURE_PROJECT_RESPONSES_ENDPOINT" --execute
```

The endpoint must be an explicit existing Azure project `/openai/v1/responses`
URL. `AzureCliCredential` obtains the `https://ai.azure.com/.default` token in
memory. Output directories are exclusive. A failed or incomplete response stops
the remaining requests; there is no loop to collect eight successes. Without
`--execute`, no credential is acquired and no request is sent.

This experiment measures **context consumption only**. It does not compare a VLM
reading page images with the adapter, test native MCP/Foundry IQ retrieval, or read
Fabric tables. Those are separate validation paths. The API surface is documented
in Microsoft's [project Responses reference](https://learn.microsoft.com/rest/api/aifoundry/project/responses).
