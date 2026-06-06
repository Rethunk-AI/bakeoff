# Summary: persistence_layout

## Final state

### Directory-per-table layout under `BAKEOFF_DATA_DIR`

The canonical filesystem layout is (per 4522798455, ratified 4524243217):

```
<BAKEOFF_DATA_DIR>/
  seeds/
    source_types.json
    task_categories.json
    quantization_methods.json
    model_architectures.json
    model_file_formats.json
    interface_types.json          ← shipped 3b0bce4
  models/
    <uuid>.json
  tasks/
    <natural_key_hash>.json
  prompts/
    <content_sha256>.json
  runners/
    <runner_id>.json
  runs/
    <run_id>.json
```

`run_queue` is the deliberate ephemeral exception — it holds transient job-submission descriptors, not persistent records. The `run_queue/` directory stores job descriptor files but they are not UUID-keyed persistent records (per 4522798455); they move to `queue/completed/` on `COMPLETE`.

### File identity scheme

| Table | Filename key |
|-------|-------------|
| `models` | UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash) — primary; or provisional UUID5(namespace, source_url + param_count_b + model_source_size) before weights pulled |
| `tasks` | SHA256 of canonical path relative to prompts root (e.g. `baseline/bash_scripting`) — idempotent across re-scans |
| `prompts` | `content_sha256` of prompt text (alias prompts: SHA256(path + original_hash)) |
| `runs` | Client-generated UUID (stored as `run_id`) |
| `runners` | Stable agent ID (hostname-based, suffix-incremented for multi-process) |

### Embedding map (per 4524304061, confirmed 4524243217)

Table-per-file unless noted:
- `creators` — embedded as `creator` object inside `models/<uuid>.json` (1-to-1 at write time)
- `model_sources` — embedded as `sources` array inside `models/<uuid>.json` (1-to-many)
- `run_model_metrics` — embedded as `metrics` array inside `runs/<run_id>.json` (1-to-many; self-contained for distribution, per 4524200643 §6)

### Disk-file schema fields vs DB divergences (per 4524304026 §2)

- FK columns in DB collapse to embedded objects or UUID references in disk files
- Disk files gain `schema_version` (integer, `1` from day one), `created_at`, `updated_at` as record-lifecycle audit fields — independent copies of what the DB tracks via trigger
- Data-carrying field names and types are identical between disk and DB in all cases
- `context_length_default/min/max` is nested as `context_length: {default, min, max}` in disk files; three columns in DB
- `predecessor_model_id` FK in DB becomes `predecessor_uuid` UUID reference in disk file

### `source_file` column on `run_queue`

`run_queue.source_file TEXT` (per 4513822952 §7) stores the path to the originating descriptor file. DR path: scan `queue/pending/`, check `run_model_metrics` for results — if present, move to `queue/completed/`; if absent, re-enqueue PENDING (per 4513822952 §7, confirmed 4522549514).

### `run_queue` disk two-directory layout

```
queue/
  pending/    # job descriptor files — DR source of truth
  completed/  # moved here on COMPLETE; prevents re-enqueue on DR
```

New-file mtime gate: default 30 seconds before read; configurable via `config.yaml` under `queue.file_mtime_min_age_seconds` (per 4519072195 §3, confirmed 4522549514).

### `BAKEOFF_DATA_DIR` env var

Path is env-configurable. No hardcoded path anywhere in the codebase (per 4522798455).

### `created_at` / `updated_at` in disk files

These describe the lifecycle of the disk record itself — when first written to this bakeoff instance, and when last modified (metadata refresh, field correction, source addition). They are not upstream publication dates. The DB holds authoritative copies; disk files carry independent copies so the file is self-contained without a DB round-trip (per 4525059995).

### Standalone runner default

Standalone mode (no DB) is the default. Queue worker mode is opt-in via `--queue` flag. Invariant recorded in AGENTS.md. The inference core is shared; queue integration is a thin wrapper (per 4519072195 §4, confirmed 4522549514).

Runner reads model properties from the model disk file at startup. No DB lookup required for VRAM estimate in standalone mode — the bundled `seeds/quantization_methods.json` provides the `vram_multiplier` table locally (per 4519072195 §5, confirmed 4522549514).

### `schema_version` in disk files

Simple integer increment (`schema_version: 1`). Not date-based. A loader that encounters an unsupported version rejects with a clear error — clean migration path (per 4522798455, confirmed 4524243217).

### Seed files

All admin-controlled lookup tables have seed JSON files under `seeds/`. The DB is seeded from these files; the runner reads them directly in standalone mode. No duplicate maintenance between DB and runner (per 4519072195 §1 and §6, confirmed 4522549514).

Current seed files shipped (per 4533311354):
- `seeds/quantization_methods.json` — 27 rows with `vram_multiplier`
- `seeds/model_architectures.json` — 4 rows
- `seeds/model_file_formats.json` — 6 rows
- `seeds/interface_types.json` — shipped 3b0bce4

### Implementation status

`bench/queue.py` and `bench/store.py` shipped on main (`0d0288e`) per 4570546649 / 4570546812. `claim()` is race-safe via rename-as-mutex. Spec in `specs/disk-persistence-layer/`. 38 new tests; full suite 413 green.

---

## Notable / unusual decisions

- **UUID5 for model identity, two-tier** — primary UUID = UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash) once weights are pulled; provisional UUID = UUID5(namespace, source_url + param_count_b + model_source_size) before hash is known. Provisional UUID is promoted at hash-computation time (per 4524304026 §3, 4524200643 §3). Downstream use: cross-system ID consistency without a DB sequence; model_id in disk file and DB are identical by construction.

- **`tasks` keyed by canonical path hash, not SERIAL** — `natural_key_hash` = SHA256(canonical relative path). SERIAL IDs are regeneration-unsafe; a re-scan would break FK references in `run_model_metrics` (per 4497170538 §4). The hash is idempotent — same file always gets the same DB ID across re-scans.

- **`content_sha256` UNIQUE on `prompts`, alias indirection** — intentional duplicate prompts are represented by YAML stub files (`type: alias, source: ../other/file.md`). Scanner computes content_sha256 for an alias as SHA256(path + original_hash) — deterministic, unique, distinguishes intentional from unintentional duplicates. Unintentional duplicates (non-alias with colliding hash) are hard-blocked at import and flagged to the administrator (per 4503203398, confirmed 4501587412).

- **`run_model_metrics` embedded in run disk file, not its own file** — self-contained result envelope for distribution; no external DB dependency to read output files. Deduplication happens at ingest, not at write time (per 4524200643 §6, confirmed 4522610944 §6).

- **`creators` embedded in model disk file, not a separate file** — 1-to-1 at write time. Keeps the model file self-contained. DB has a `creators` table for queryability; disk file embeds the object inline (per 4524304061).

- **`model_source_mtime` and `model_source_size` in disk file** — runner statics the model source file at startup and compares mtime + size against descriptor. Mismatch policy: size changed → INVALID (re-ingest required); mtime changed, size unchanged → recompute hash, compare; mtime changed, content unchanged → update mtime in disk file only, no re-ingest. Strict cache validation is the default (per 4524304026 answers, confirmed 4524200643 §3 addendum).

- **File mtime min-age gate for new file detection** — 30 seconds (configurable). Addresses ext4 `commit=5s` window plus NFS write-behind and lazytime mount scenarios. Prevents reading partially written job descriptors (per 4519072195 §3, confirmed 4522549514).

- **`run_queue` is the only ephemeral exception to the directory-per-table pattern** — job files are self-contained submissions with a lifecycle (created → consumed → archived) fundamentally different from persistent records. All other tables with persistent data follow the UUID-filename pattern (per 4522798455, confirmed 4524243217).

---

## Open / unresolved

- **Hardware tables disk persistence** — `system_hardware`, `system_software`, `gpu_hardware`, `system_gpu_link`, `run_hardware_metrics` disk layout confirmed in design (Q2–Q5 of bakeoff#22 closed per 4574341430) but the Phase-2 UUID-migration framework that underpins versioned hardware UUIDs is still in design in **bakeoff#25** (spun off from bakeoff#22). The `uuid_namespace` per-table constant design was corrected (per-table fixed namespace, not per-version) but bakeoff#25 is not closed.
  - Trade-off: per-version namespaces force UUID recomputation for all records on every schema bump (uuid_migrations explosion); per-table namespaces limit migration entries to records whose UUID-input fields actually changed.

- **`runners.status` TEXT + CHECK vs ENUM** — deferred to Phase 2 once schema stabilizes. A `TODO(P2)` comment is placed in `schema.sql`. Trade-off: ENUM is cleaner for closed value sets but requires `ALTER TYPE ... ADD VALUE` to extend; TEXT + CHECK is one-line to iterate during active design (per 4519072195 §4, deferred confirmed 4524304026 §4, 4524200643 §4).

- **`schema_tables.uuid_namespace` storage** — should per-table UUID namespaces live in `schema_tables` DB table (and a backing seed file) or as Go constants only? DB storage allows tooling to look up namespaces without code changes; Go constants keep source of truth in code. Not yet confirmed (per 4573134793, 4572098058).

- **`schema_hops` table shape and name** — the migration-hop registry (formerly `uuid_migrations`, now proposed as `schema_hops` without UUID columns) is designed but not confirmed. `gissf1` asked whether `table_migrations` or integration into `schema_versions` is preferred (per 4573134793, 4572098058). Carries into bakeoff#25.

- **Prompt descriptor file format** — deferred to the scoring architecture thread; no disk file format defined yet for `prompts` (per 4524304026 open question 4, 4519072195 open question 4).

- **Result signing/verification** — scheme undefined; carries to a separate thread (per 4519072195 open question 1, 4522610944 §3).

---

## Cross-topic links

- **schema_versions / migration framework** (bakeoff#25) — the disk-persistence layer's `schema_version` field in every disk file and the UUID5 namespace-per-table identity scheme both depend on the migration framework currently in design in bakeoff#25. The `schema_tables` seed file (storing `uuid_namespace` per table) is a direct dependency on that thread's outcome.
- **run_queue / queue design** (bakeoff#13) — `source_file` column, `retry_after` claim gate, `queue/pending/` + `queue/completed/` two-directory DR layout, and the standalone-vs-queue-worker split all originated in bakeoff#13 and are reflected here as settled.
- **model descriptor / capability ingestion** (bakeoff#15, closed) — `models/<uuid>.json` format, `schema_version` integer, `creator` + `sources` embedding, `model_hash` + `model_source_mtime` + `model_source_size` fields, and UUID5 provisional ID generation all settled in bakeoff#15.
- **hardware schema** (bakeoff#17–#22) — `interface_types.json` seed shipped; full hardware disk persistence (system_hardware, gpu_hardware, run_hardware_metrics) confirmed in design but depends on bakeoff#25 migration framework.
- **tasks / prompts** (bakeoff#12) — `tasks/<natural_key_hash>.json` and `prompts/<content_sha256>.json` identity schemes derive directly from the idempotent-ID and alias-hash decisions in bakeoff#12.
