# Compare source-local claims

`compare_ontology_claims(database, left=<claim reference>, right=<claim reference>)`
returns two primary claims, their complete reachable expressions, differences in
status/modality/basis/attribution/context/time bounds, and shared labels with each
side's separate vocabulary declarations. It also includes losslessly packed full
inspections, preserving connected dissent, role anchors and source context.

The primary comparison uses only expressions reachable from each selected claim;
linked claims remain supporting context. Vocabulary comparisons are bounded to
1,000 same-label pairs. Cross-document labels and IRIs do not establish identity.
Different field values do not establish logical contradiction. Same-document
snapshot changes are reported as `same_entry_versions`; priority stays unresolved.
Draft/current distinctions remain in the original claims/context for review.

`review_ontology_comparison(comparison, category=..., explanation=...)` attaches
an explicitly unverified interpretation, bound to the comparison hash and both
references. Categories are `different_sense`, `different_condition`,
`different_authority_or_revision`, `compatible` and `uncertain`. This record does
not change either ontology, assert identity or select an authoritative winner.

`OntologyTools` exposes `compare_ontology_claims` with a shared allowlisted
`collection` plus `left_name`, `left_snapshot_id`, `left_id` and corresponding
`right_*` fields. The tool returns bounded comparison data and exact inspection
actions; inspect both source references before citing. The host API additionally
supports different collections, subject to the embedding application's access
checks. Retrieval itself remains read-only and makes no model calls.

Tests cover explicit dissent, distinct primary versus connected scope, changed
versions/definitions/context, allowlists, response budgets and database immutability.
The comparison describes stored interpretations; it is not automatic semantic
conflict detection or an authority resolver.
