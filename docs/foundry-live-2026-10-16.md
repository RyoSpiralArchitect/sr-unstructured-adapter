# Foundry native MCP live validation — 2026-10-16

The adapter's eight relation cards were published to an existing Azure AI Search
service and connected to an ordinary Foundry Prompt Agent using the deployed
`gpt-5.5` model. Publication and readback completed. A real native MCP call and a
completed agent response correctly identified 朝凪's primary owner as 東部室 and
secondary owner as 西部室, while retaining 星凪's budget as unknown.

The input bundles are the same synthetic and Digital Agency role-table examples
used by the [Fabric validation](fabric-lakehouse-2026-10-16.md). Search and Fabric
receive projections of those verified packages. This is not a Fabric-to-Search
indexer, and the agent did not query the Fabric SQL endpoint.

[Sanitized live receipt](validation/foundry-live-2026-10-16/live-verification.json)
records model, version, completion status, usage, snapshot and private-receipt
hashes. Three checked facts establish bounded integration behavior, not accuracy
or equivalence to direct document vision. The separate
[paired GPT-5.5 context comparison](azure-context-gpt55-2026-10-16.md) retains the
fixed inputs, all responses, failed first transport attempt and offline scores.

## Service behavior retained in the implementation

- ARM persisted the project connection's `isSharedToAll` as `false`. Foundry export
  schema v2 uses that value; the recorded v1 artifacts remain verifiable.
- The selected Search service accepted `2026-08-01-preview` MCP and rejected
  `2026-04-01`. The successful endpoint version is recorded, not inferred from a
  general availability label.
- Foundry rejected request-time reasoning overrides when selecting an agent.
  Reasoning is part of the versioned Prompt Agent definition.

Full raw service receipts remain private because they include resource metadata.
No deployment keys, tenant identifiers or live endpoint URLs are published here.
