# Summary: model_trust

## Final state

Ollama accepted as a trusted model source, comparable in trustworthiness to HuggingFace (per 4503203471). No schema changes required: `model_sources` is already provider-agnostic, and `model_hash` (SHA256 of weights) serves as the deduplication ground truth regardless of origin. Implementation impact is additive only — `ollama` added as a new `source_types` entry; `source_metadata` JSONB stores `{"ollama_tag": "...", "ollama_digest": "..."}` for Ollama-sourced rows.

## Notable / unusual decisions

- Ollama treated as a redistribution layer over HuggingFace, not an independent trust root — rationale: Ollama Library models trace provenance back to HF or original creators, making them equivalent in tier to HF creator-published models (per 4503203471). Downstream: trust-tier logic need not treat the two differently.
- `model_hash` designated explicit trust anchor, not source URL or tag — rationale: community/unverified models from either platform lack guaranteed provenance; hash is the only stable signal (per 4503203471). Downstream: any future trust-tier feature must key off hash, not provider.
- Ollama digest (per-pull) accepted as a secondary pre-check before full `model_hash` computation — rationale: reduces redundant hashing cost on repeated pulls of the same tag (per 4503203471).

## Open / unresolved

- Trust-tier taxonomy for community/unverified models (from Ollama or HF) was flagged but not designed: no schema column, no tier values, no enforcement rule settled in this thread.
  - address: #14, https://github.com/Rethunk-AI/bakeoff/issues/14#issuecomment-4503203471

## Cross-topic links

- `model_sources` table and `source_types` lookup — shared surface with any topic covering model ingestion or deduplication.
- `model_hash` deduplication workflow — referenced as already established; see related schema-spec tickets covering hash/merge behavior.
