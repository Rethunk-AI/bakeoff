# Summary: model_trust

## Final state

Ollama is an **acceptable model source**, roughly equivalent in trustworthiness to HuggingFace. Decision reached in #14 (per 4503203471).

Source tier outcome:
- **Ollama Library (curated)** — same trust tier as HuggingFace creator-published models. Provenance is traceable back to HF or original creator; weights are the same GGUF files, just repackaged.
- **Community models (Ollama or HF, unverified)** — lower tier; no guaranteed provenance chain.

Schema impact: no structural changes required. `ollama` added as a row in the `source_types` lookup table alongside `huggingface`, `direct_url`, `local_file`. The `model_sources` table is already provider-agnostic.

`source_metadata` JSONB for Ollama entries stores `{"ollama_tag": "llama3:8b-q4_K_M", "ollama_digest": "..."}`. The Ollama digest serves as a secondary hash check before the full `model_hash` (SHA256 of weights) computation.

**Trust anchor:** `model_hash` (SHA256 of weights) is the deduplication and trust ground truth regardless of source. An Ollama-sourced model with the same `model_hash` as an HF-sourced model is detected as a duplicate and merged.

## Notable / unusual decisions

- **`model_hash` as primary trust anchor, not source URL** — source metadata (URL, tag, digest) is informational; SHA256 of weights is authoritative. This means source provenance is tracked but does not gate deduplication or result attribution.
- **Ollama digest as secondary check** — Ollama exposes a per-pull digest that can pre-validate before the more expensive full `model_hash` computation. Stored in `source_metadata` JSONB for Ollama entries.

## Open / unresolved

- **Formal "trust tier" field** — Bastion noted that if the system ever tracks a trust tier explicitly, Ollama Library and HF creator-published models share a tier, with unverified community models lower. No tier field was added to the schema in this discussion; remains a potential future addition.
- **Source tier enumeration** — exact list of `source_types` seed rows and their tier assignments not finalized in this thread.

## Cross-topic links

- `model_sources.source_type` → `source_types` lookup table (where `ollama` row is added)
- `model_sources.model_hash` — deduplication key; ties to the broader model identity schema (discussed in #12, run_model_metrics)
