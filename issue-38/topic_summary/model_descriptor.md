# Summary: model_descriptor

## Final state

### `models` table (settled schema as of `dac26d5`)

| Column | SQL Type | Notes |
|---|---|---|
| `model_id` | `UUID PRIMARY KEY` | UUID5 — primary: `UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash)`; provisional: `UUID5(BAKEOFF_MODEL_NAMESPACE, source_url \| parameter_count_b \| model_source_size)` (per 4522610944, 4524304026, 4525059885) |
| `name` | `TEXT NOT NULL` | |
| `creator_id` | `UUID NOT NULL REFERENCES creators` | UUID5 from `UUID5(BAKEOFF_CREATOR_NAMESPACE, homepage)` (per 4525059885) |
| `model_hash` | `TEXT UNIQUE` | SHA256 of model source file; deduplication ground truth (per 4503203471) |
| `parameter_count_b` | `FLOAT` | Hard field — validated against weights at ingest |
| `active_parameter_count_b` | `FLOAT` | Hard field |
| `architecture_id` | `INT NOT NULL REFERENCES model_architectures` | Lookup FK; text in disk file (per 4525059885) |
| `file_format_id` | `INT NOT NULL REFERENCES model_file_formats` | Lookup FK; text in disk file (per 4525059885) |
| `quantization_id` | `INT NOT NULL REFERENCES quantization_methods` | Lookup FK; text in disk file (per 4519072195) |
| `context_length_default` | `INT` | |
| `context_length_min` | `INT` | |
| `context_length_max` | `INT` | |
| `release_date` | `DATE` | From upstream model card |
| `version` | `TEXT` | |
| `description` | `TEXT` | |
| `predecessor_model_id` | `UUID REFERENCES models` | UUID of prior version's record (per 4533385174, 4533408702) |
| `model_source_mtime` | `TIMESTAMPTZ` | Optional; mirrors disk descriptor field |
| `model_source_size` | `BIGINT` | Used in provisional UUID and cache-drift detection |
| `provisional` | `BOOLEAN` | True until model_hash-based UUID is confirmed |
| `created_at` | `TIMESTAMPTZ` | DB trigger-managed row audit |
| `updated_at` | `TIMESTAMPTZ` | DB trigger-managed row audit |

`min_vram_mb`/`min_vram_gb` deliberately absent — calculated at claim time from `active_parameter_count_b × vram_multiplier × 1.15` (per 4513822952, 4525059885).

### `model_sources` table

| Column | SQL Type | Notes |
|---|---|---|
| `source_id` | PK (SERIAL or UUID) | |
| `model_id` | `UUID NOT NULL REFERENCES models` | |
| `source_type` | `TEXT NOT NULL REFERENCES source_types` | Includes `ollama` (added per 4505741990) |
| `source_metadata` | `JSONB NULLABLE` | Flat structure — provider identity + stats co-located, no sub-objects (per 4505741990, 4506245788) |
| `updated` | `TIMESTAMPTZ NULLABLE` | Single timestamp anchor for all data in the row including model_hash; multi-query consistency protocol applies (per 4506245788) |

### `quantization_methods` lookup table

| Column | SQL Type | Notes |
|---|---|---|
| `quantization_id` | `SERIAL PRIMARY KEY` | |
| `name` | `TEXT NOT NULL UNIQUE` | e.g. `"fp32"`, `"q4_k_m"` |
| `vram_multiplier` | `DECIMAL NOT NULL` | Bytes per active parameter; not nullable — no ELSE fallback (per 4516038912, 4519072195) |
| `description` | `TEXT` | |

27 rows seeded from llama.cpp `ggml-common.h` block layout math; stored in `schema/seeds/quantization_methods.json` (per 4519570310, 4533311354).

### `model_architectures` lookup table

| Column | SQL Type | Notes |
|---|---|---|
| `architecture_id` | `SERIAL PRIMARY KEY` | |
| `name` | `TEXT NOT NULL UNIQUE` | Dense / MoE / SSM / Hybrid |
| `description` | `TEXT` | |

Seeded in `schema/seeds/model_architectures.json` (per 4533311354).

### `model_file_formats` lookup table

| Column | SQL Type | Notes |
|---|---|---|
| `file_format_id` | `SERIAL PRIMARY KEY` | |
| `name` | `TEXT NOT NULL UNIQUE` | GGUF / SafeTensors / PyTorch / ONNX / ExLlamaV2 / MLX |
| `description` | `TEXT` | |

Seeded in `schema/seeds/model_file_formats.json` (per 4533311354).

### Model disk file (`models/<uuid>.json`) — settled format

```json
{
  "schema_version": 1,
  "model_id": "<uuid-v5>",
  "name": "...",
  "architecture": "Dense",
  "parameter_count_b": 8.0,
  "active_parameter_count_b": 8.0,
  "quantization": "q4_k_m",
  "file_format": "GGUF",
  "model_hash": "<sha256>",
  "model_source_mtime": "<iso8601>",
  "model_source_size": 4924194816,
  "context_length": {"default": 4096, "min": 1, "max": 4096},
  "release_date": "<iso8601-date>",
  "version": "...",
  "description": "...",
  "predecessor_uuid": null,
  "creator": {"uuid": "...", "name": "...", "display_name": "...", "homepage": "...", "service_identifiers": {}},
  "sources": [{"source_type": "...", "url": "...", "source_metadata": {}, "updated": "..."}],
  "created_at": "<iso8601>",
  "updated_at": "<iso8601>"
}
```

`schema_version` is a simple incrementing integer (per 4522798455, 4524243217). `creator` and `sources` are embedded objects — not separate files (per 4524243217, 4570345289). `created_at`/`updated_at` describe the disk record lifecycle, not the model weights (per 4525059885).

Stub format: only `schema_version`, `stub: true`, and `source_url` required; pipeline resolves all fields, renames file to `<uuid>.json` (per 4533408702).

### Implementation status

Implemented in `bench/descriptor.py` + `bench/store.py`, merged `0d0288e` (per 4570546812). 413 tests green at close.

---

## Notable / unusual decisions

- **Stats merged into `source_metadata` JSONB, no separate stats columns.** Initial proposal added `source_stats` + `stats_fetched_at` as typed columns; @gissf1 collapsed them into the existing `source_metadata` blob (per 4505728892). Trade-off: staleness queries become JSONB path expressions rather than indexed column comparisons. Resolved by adding `updated` as a native column for the full-row timestamp anchor (per 4505899850, 4506245788) — downstream benefit: `WHERE updated < NOW() - INTERVAL '30 days'` is indexable.

- **Multi-query consistency protocol.** All source metadata queries for a single row must complete within a 1-minute window; `updated` = earliest returned timestamp. If any query exceeds 1 minute, retry all queries; accept second round only if values match first round. Hard limit 5 minutes total (per 4506245788). Rationale: `model_hash` is scoped to the `updated` timestamp, so the acquisition window must be bounded and coherent.

- **`quantization_methods` as lookup table instead of CASE expression.** The original claim query embedded a `CASE m.quantization WHEN ... END` block. @gissf1 requested a lookup table so the VRAM multiplier has a single source of truth, the UI filter bar can derive from it, and the schema can extend without a CASE rewrite (per 4516038912). Downstream: removes ELSE fallback risk; `vram_multiplier NOT NULL` enforces completeness.

- **UUID5 deterministic model_id.** Using `UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash)` as primary ID means the same model always gets the same ID on any host without a DB round-trip. Two-tier design: provisional UUID from `(source_url | parameter_count_b | model_source_size)` before weights are pulled; promoted to hash-based UUID at weight-pull time (per 4522610944, 4524304026). Namespace UUID committed as a project constant in `bench/constants.py`.

- **FK auto-add policy: reject, do not auto-insert.** When an ingested disk file carries a text value (architecture, file_format, quantization) with no matching lookup row, the ingest is rejected and held in `pending_review`. Admin must manually add the lookup entry and re-run ingest. No automated additions (per 4525059885). Rationale: unreviewed quantization entries corrupt VRAM estimates and UI filter bar populations.

- **Conflict detection hard/soft field split.** On hash match with a conflicting incoming record: soft fields (`description`, `creator.display_name`, `creator.homepage`) accepted silently; hard fields (`parameter_count_b`, `active_parameter_count_b`, `context_length_*`, `architecture`, `file_format`, `quantization`) validated against the weights before accepting a correction; disk file never deleted on rejection (per 4533310100). Multi-hard-field diff with changed `version` surfaces a warning and prompts operator to create a new record with a new UUID (per 4533385174, 4533408702).

- **`min_vram_mb` deliberately not stored.** Calculated at claim time; storing it creates a staleness risk and a derived-data redundancy. Both DB table and disk file omit it (per 4513796211, 4525059885).

- **Config file is `config.yaml` in project root**, not a new `bakeoff.yaml`. Runner and queue knobs (`file_mtime_min_age_seconds: 30`, `poll_interval_seconds: 60`, etc.) merge into the existing `config.yaml` (per 4524200643, 4524304026).

- **File mtime guard set to 30 seconds** (not 5 seconds). ext4 default `commit=5s` means a 5-second guard is insufficient when `relatime`/`lazytime` mount options are in play. 30 seconds provides comfortable margin; configurable (per 4519072195, 4522549514).

---

## Open / unresolved

- **`schema_version` format — integer vs. date-based scheme not fully closed.** @gissf1 proposed a YYYYMMDDNN date-based format as an option; Bastion recommended simple integer increment; @gissf1 replied "Agreed to simple integer increment" (per 4524243217). The integer form was implemented (per 4570546812). Status: converged on integer, but the date-based option was raised and formally acknowledged before agreement — no lingering dispute.

  None — converged.

- **`creators.creator_id` UUID composition — secondary identifier not finalized.** Bastion proposed `UUID5(BAKEOFF_CREATOR_NAMESPACE, homepage)` with `UUID5(BAKEOFF_CREATOR_NAMESPACE, display_name)` as a fallback when no homepage. @gissf1 asked whether the source site's own user ID could be used to generate the creator UUID (per 4524573993). The homepage-primary approach was proceeded with in `dac26d5`, but the question of whether a source-site user ID should be incorporated was not explicitly closed.
  - address: #13, https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524573993

- **Provisional UUID promotion — coalescing race and multi-source ordering not fully specified.** @gissf1 asked: how is a conflict between a primary and provisional disk file detected? What happens when the same model is available from multiple sources and provisionals coalesce? What if a provisional for a new source is added after the primary already exists (per 4524573993)? Bastion provided a detailed coalescing policy (per 4525059885), but @gissf1's final reply on that comment only agreed with the explanation and added the conflict detection extension — the multi-provisional race (two provisionals resolving to the same hash simultaneously on different hosts) was not explicitly addressed.
  - address: #13, https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524573993

- **`models/pending/` directory for stub isolation — design option, not decision.** Bastion noted that a `models/pending/` subdirectory could isolate stubs from settled records if naming collisions become a concern, but flagged this as a design option to be resolved when the ingestion pipeline is specced (per 4533408702). Not decided.
  - address: #13, https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4533408702

- **Embedded GGUF metadata cross-check (runner startup validation) — deferred to P2.** @gissf1 accepted deferral; flagged as P2 in the close-out (per 4570345289). Not yet scoped.
  - address: #15, https://github.com/Rethunk-AI/bakeoff/issues/15#issuecomment-4570345289

- **URL-submit ingestion handler — deferred to P2.** The agreed ingestion path (URL → metadata pull → validate → store) is the target architecture (per 4522737381, 4522798455), but the implementation was explicitly excluded from the closed scope. Manual/admin JSON + stub pipeline only in `0d0288e`.
  - address: #15, https://github.com/Rethunk-AI/bakeoff/issues/15#issuecomment-4570345289

- **`quantization_methods` seed multipliers — pending cross-check confirmation.** Bastion flagged that the initial seed listed `0.45` for `q4_k_m` in the example description but the seed SQL used `0.563`; committed to cross-checking all multipliers against llama.cpp MODELS.md and a reference model file size (per 4519570310, 4524200643). @gissf1 acknowledged but did not have domain expertise to confirm independently. Cross-check result was not posted in the thread.
  - address: #13, https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4519570310

---

## Cross-topic links

- **`quantization_methods`** — FK from `models.quantization_id`; seed data also used by the runner in standalone mode and the bakeoff-results filter bar dropdown. Links to `run_queue` topic (claim query VRAM filter depends on this table).
- **`model_architectures`** — FK from `models.architecture_id`; bakeoff-results filter bar. New architecture values require admin review before insertion.
- **`model_file_formats`** — FK from `models.file_format_id`; bakeoff-results filter bar.
- **`creators`** — FK from `models.creator_id`; UUID5-keyed; embedded in model disk file.
- **`model_sources`** — 1-to-many from `models`; embedded in model disk file under `sources` key. `source_types` lookup table governs `source_type` values (including `ollama`).
- **`runners`** / **`run_queue`** — VRAM claim query joins `models` + `quantization_methods`; `run_queue.source_file` points to the model disk file path. Links to `run_queue` topic.
- **hardware tables (#17–#21)** — preserved intact in `dac26d5`; disk persistence for hardware tables spun off to #22.
- **result signing / verification** — mentioned as a dependency for standalone runner output (per 4519072195); deferred to a separate thread (bakeoff#16 handles signing identity in `runners`).
- **scoring architecture / prompt descriptor** — prompt descriptor file format explicitly deferred; links to scoring topic.
