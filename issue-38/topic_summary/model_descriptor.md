# Summary: model_descriptor

## Final state

### Model disk file format (`models/<uuid>.json`)

Settled as JSON with integer `schema_version` (starting at `1`). YAML rejected for data files (frontend reads JSON natively; YAML stays for human-edited `config.yaml` only). Implemented in `bench/descriptor.py` + `bench/store.py`, merged `0d0288e` (per 4570546812).

| Field | Type in disk file | Notes |
|-------|------------------|-------|
| `schema_version` | integer | Hard-gated; loader rejects unsupported values |
| `uuid` | UUID5 string | Primary: `UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash)`; provisional: `UUID5(BAKEOFF_MODEL_NAMESPACE, source_url + "\|" + param_count + "\|" + source_size)` |
| `name` | string | |
| `creator` | embedded object | `{uuid, name, display_name, homepage, service_identifiers}` — embedded from `creators` table |
| `model_hash` | string | SHA256 of model source file; nullable until weights pulled |
| `parameter_count_b` | float | |
| `active_parameter_count_b` | float | |
| `architecture` | string (text) | FK to `model_architectures` lookup at ingest |
| `file_format` | string (text) | FK to `model_file_formats` lookup at ingest |
| `quantization` | string (text) | FK to `quantization_methods` lookup at ingest |
| `context_length` | nested object | `{default, min, max}` — maps to 3 INT cols in DB |
| `release_date` | ISO8601 date | |
| `version` | string | |
| `description` | string | |
| `predecessor_uuid` | UUID5 string | Maps to `predecessor_model_id UUID FK` in DB |
| `model_source_mtime` | ISO8601 timestamp | mtime of weights file on disk |
| `model_source_size` | integer (bytes) | Size of weights file on disk |
| `sources` | array of objects | Embedded `model_sources` entries: `{source_type, url, source_metadata, updated}` |
| `created_at` | ISO8601 timestamp | Disk-record lifecycle (when this file was first written) |
| `updated_at` | ISO8601 timestamp | Disk-record lifecycle (last modified) |

`min_vram_mb` / `min_vram_gb` explicitly removed — calculated at claim time, not stored.

### Disk layout

```
<BAKEOFF_DATA_DIR>/
  models/
    <uuid>.json        ← persistent model records
  run_queue/
    <uuid>.json        ← ephemeral job submissions (exception to UUID-filename convention)
```

`BAKEOFF_DATA_DIR` is env-configurable. `run_queue` is the only ephemeral exception; all other tables follow directory-per-table / UUID-filename pattern (per 4522798455, 4524243217).

### `quantization_methods` table (+ seed data)

```sql
CREATE TABLE quantization_methods (
    quantization_id  SERIAL   PRIMARY KEY,
    name             TEXT     NOT NULL UNIQUE,
    vram_multiplier  DECIMAL  NOT NULL,   -- bytes per active parameter
    description      TEXT
);
```

27 rows seeded from GGUF/llama.cpp block-size constants. Representative values: `fp32=4.000`, `fp16/bf16=2.000`, `q8_0=1.063`, `q4_k_m=0.563`, `q2_k=0.352`, `iq1_s=0.188`. Shipped as `schema/seeds/quantization_methods.json`; DB seeded from same file used by standalone runner (per 4524573993 + 4524304026). Note: initial `0.45` for `q4_k_m` in the 221322Z proposal was corrected to `0.563` in the seed SQL comment at 4519570310.

`models.quantization` (TEXT) → `models.quantization_id INT NOT NULL REFERENCES quantization_methods(quantization_id)`.

VRAM estimate (used in claim query, not stored):
```sql
CEIL(m.active_parameter_count_b * qm.vram_multiplier * 1.15) <= $runner_vram_gb
```
15% overhead for KV cache + activations.

### Additional lookup tables (adopted per 4525059885, shipped in `dac26d5`)

**`model_architectures`**: `Dense`, `MoE`, `SSM`, `Hybrid` — seeded in `schema/seeds/model_architectures.json`.

**`model_file_formats`**: `GGUF`, `SafeTensors`, `PyTorch`, `ONNX`, `ExLlamaV2`, `MLX` — seeded in `schema/seeds/model_file_formats.json`.

`models.architecture` and `models.file_format` TEXT columns → FK INT columns referencing these tables. Disk file continues to use text strings; resolved to FK at ingest.

### `model_id` / `creator_id` — deterministic UUID5

`models.model_id` → `UUID PRIMARY KEY` (replaces SERIAL). Namespace constant `BAKEOFF_MODEL_NAMESPACE` committed in `bench/constants.py` (per 4524304026).

Two-tier strategy (per 4524304026, confirmed 4524573993):
- **Primary** (hash known): `UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash)`
- **Provisional** (hash not yet computed): `UUID5(BAKEOFF_MODEL_NAMESPACE, source_url + "|" + parameter_count_b + "|" + model_source_size)`
- Provisional promoted to primary when weights are pulled; disk file renamed to `<primary-uuid>.json`

`creators.creator_id` → UUID5 similarly; namespace `BAKEOFF_CREATOR_NAMESPACE` (distinct constant). Composition: `UUID5(BAKEOFF_CREATOR_NAMESPACE, homepage)`; fallback to `display_name` with `provisional: true`.

### Ingest pipeline

Submission path: URL-submit → upstream metadata pull (no weight download yet) → validate → store (DB + disk file). No manual JSON submission except admin/seed. No CI-generated descriptors. Embedded-GGUF cross-check deferred to P2 (per 4522737381, 4522798455, 4570345289).

**Stub file format** (per 4533408702):
```json
{
  "schema_version": 1,
  "stub": true,
  "source_url": "https://huggingface.co/...",
  "name": "optional human name",
  "predecessor_uuid": null
}
```
Pipeline detects `stub: true` or missing `model_id`, resolves from source URL, computes hash, populates all fields, renames file.

### Cache validation policy (per 4524304026 — strict default)

| Condition | Action |
|-----------|--------|
| `model_source_size` changed | INVALID — re-ingest required |
| mtime changed, size unchanged | Recompute hash, compare; if content unchanged, update mtime only |
| mtime changed, content unchanged | Update `model_source_mtime` in disk file, no re-ingest |

Queue worker defaults to `--strict-cache`; standalone emits warning.

### Conflict detection at ingest (per 4533310100, 4533385174, 4533408702)

| Scenario | Action |
|----------|--------|
| Hash match, soft fields differ (`description`, creator contact) | Accept update |
| Hash match, hard field matches weights but DB differs | Accept correction |
| Hash match, hard field does not match weights | Reject source, clear error, keep disk file |
| Hash match, multiple hard fields + version changed | Surface warning, block source, keep disk file, prompt for new record |
| Hash match, new source URL | Add `model_sources` row only |
| Version change → new weights | New hash → new UUID; auto-set `predecessor_model_id` from old record |

Disk file is never deleted on rejection — submitter retains file to correct and resubmit.

FK auto-add policy (per 4525059885): unrecognized text values for architecture/file_format/quantization → reject ingest, emit admin alert, hold in `pending_review` state. No automated lookup-table insertion.

### `model_sources` table (`source_metadata` design)

```sql
-- Final columns (per 4506245788)
source_metadata   JSONB NULLABLE    -- flat structure: provider identity + stats, no sub-objects
updated           TIMESTAMPTZ       -- single anchor for all fields in row including model_hash
```

Flat `source_metadata` examples:
- HuggingFace: `{"hf_commit": "abc123", "downloads": 50000, "likes": 1200}`
- Ollama: `{"ollama_tag": "llama3:8b-q4_K_M", "ollama_digest": "sha256:...", "pulls": 3000}`

No `stats` sub-object. No timestamp key inside the blob. `updated` is a native TIMESTAMPTZ column for indexed staleness queries.

`ollama` added as `source_types` entry alongside `huggingface`, `direct_url`, `local_file` (per 4505741990). `model_hash` is deduplication ground truth regardless of source.

Multi-query consistency protocol (per 4506245788): all queries for one row complete within 1-minute window; `updated` = earliest returned timestamp; if window exceeded, retry all; second round must match first or discard and restart; hard 5-minute total cap.

## Notable / unusual decisions

- **`stats_fetched_at` inside blob vs. native column** — @gissf1 corrected the initial Bastion proposal (per 4505899850): stats stay flat in `source_metadata`, timestamp becomes a native `updated` column. Reason: single JSONB blob reduces parse surface; `updated` as a real column enables indexed staleness queries without path expressions. Downstream: staleness refresh uses `WHERE updated < NOW() - INTERVAL '30 days'` directly.

- **`vram_multiplier` in a lookup table, not a CASE expression** — initial design used a CASE expression inline in the claim query (per 4513822952). @gissf1 proposed a lookup table (per 4516038912); adopted in 4519072195. Reason: extensible without query change, eliminates ELSE fallback risk, serves as the single data source for the bakeoff-results UI filter bar dropdown. Downstream: adding a new quantization format requires only a new seed row + admin approval, no code change.

- **`q4_k_m` multiplier corrected from `0.45` to `0.563`** — the 221322Z design comment listed 0.45 as an example; the seed SQL at 4519570310 corrected to 0.563 based on actual 4.5 bits/weight block arithmetic. The discrepancy was noted inline. Downstream: claim query VRAM estimates use 0.563; using 0.45 would have over-allocated runners.

- **Provisional UUID promoted at hash-computation time** — two-tier UUID strategy handles the race between submission and weight-download. The provisional ID lets the system register a model from metadata alone (before large weight download), then atomically promote to the hash-anchored primary UUID when weights are available. Disk file is renamed, not recreated. Downstream: no orphaned records; multi-source coalescing is automatic.

- **`model_source_mtime` + `model_source_size` as cache-validation fields** — @gissf1 proposed these (per 4522549514) to detect drift between cached weights and disk file without requiring a full re-hash on every startup. The size check is free (stat syscall); mtime change triggers a hash recompute only when size is unchanged. Downstream: prevents stale-cache silent failures in long-lived runner deployments.

- **`schema_version` integer increment over YYYYMMDDNN** — @gissf1 proposed date-based versioning; Bastion argued against it (per 4522798455): date-based couples schema to calendar, adds parsing complexity, provides no validation advantage a schema registry can't give more cleanly. YYYYMMDDNN is a release artifact naming convention, not a format version identifier. Settled: simple integer.

- **`runners.status` TEXT + CHECK (deferred ENUM)** — ENUM rejected during active schema churn (per 4522610944). ALTER TYPE for ENUM is transactional in PG 12+ but changing value set order still requires type rebuild. TEXT + CHECK is one-line migration during design phase. Deferred P2 cleanup comment added to `schema.sql`.

## Open / unresolved

- **URL-submit ingest handler not yet implemented** — the disk format and ingest contract are fixed; the actual handler (URL → metadata pull → validate → store) is a P2 implementation task. Embedded-GGUF metadata cross-check is explicitly deferred to P2 (per 4570345289).

- **`quantization_methods` multiplier verification** — @gissf1 acknowledged the values look reasonable but noted it is not their expertise (per 4522549514). Bastion committed to cross-checking against llama.cpp MODELS.md and known-size model file sizes within 1% tolerance. No confirmation posted in this thread before close.

- **`models/pending/` directory for stub isolation** — noted as a design option (per 4533408702) to prevent naming collisions between stubs and settled records. Not adopted, not rejected; flagged for consideration when ingestion pipeline is specced.

- **Prompt descriptor file format** — explicitly deferred to the scoring architecture thread in all relevant responses. No fields defined here.

- **Result signing/verification scheme** — deferred to a separate thread (`#16` referenced). Not designed in this topic.

## Cross-topic links

- **`quantization_methods`** — FK target for `models.quantization_id`; also feeds the bakeoff-results UI filter bar dropdown. Seed data in `schema/seeds/quantization_methods.json`. See also: hardware topic (VRAM claim query uses this table).
- **`model_architectures`** — FK target for `models.architecture_id`; seed in `schema/seeds/model_architectures.json`.
- **`model_file_formats`** — FK target for `models.file_format_id`; seed in `schema/seeds/model_file_formats.json`.
- **`model_sources`** — embedded as `sources: [{...}]` in model disk file; FK `models.model_id` in DB.
- **`creators`** — embedded as `creator: {...}` in model disk file; FK `models.creator_id` in DB.
- **`run_queue`** — consumes model disk files at job submission; `source_file` column tracks DR artefact path. See run_queue topic.
- **`runners`** — claim query joins `quantization_methods` via `models`; `runners.runner_id` FK in `runs`. See runners topic.
- **Hardware disk persistence** — spun off to `#22`; not covered here.
- **Signing/verification** — `#16`; not covered here.
- **Scoring architecture / prompt descriptors** — deferred to its own thread; not covered here.
