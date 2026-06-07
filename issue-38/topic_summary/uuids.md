# Summary: uuids

## Final state

### `models.model_id` — UUID v5 (deterministic)
Settled: `models.model_id` is `UUID PRIMARY KEY`, replacing `SERIAL`. Generated via UUID v5 using a fixed project-constant namespace (`BAKEOFF_MODEL_NAMESPACE`) committed to `bench/constants.py` (per 4524304026).

Two-tier generation strategy (per 4524304026, confirmed 4524573993):
- **Primary:** `UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash)` — authoritative once weights are hashed.
- **Provisional:** `UUID5(BAKEOFF_MODEL_NAMESPACE, source_url + "|" + str(parameter_count_b) + "|" + str(model_source_size))` — used before hash is computed; promoted to primary at weight-pull time.

`model_source_mtime` excluded from UUID components as insufficiently stable (per 4524304026). `min_vram_mb` removed from both DB table and disk file — calculated from parameter counts at claim time (per 4525059885).

### `creators.creator_id` — UUID v5
`creator_id` changed from serial integer to `UUID PRIMARY KEY`. Composition: `UUID5(BAKEOFF_CREATOR_NAMESPACE, homepage)` where homepage is the canonical URL; fallback to `UUID5(BAKEOFF_CREATOR_NAMESPACE, display_name)` with `provisional: true` when no homepage is present (per 4525059885). `BAKEOFF_CREATOR_NAMESPACE` is a second project-constant UUID, distinct from `BAKEOFF_MODEL_NAMESPACE`, committed to `bench/constants.py`.

### `system_hardware` / `gpu_hardware` — UUID identity
Hardware fingerprint UUIDs (per 4533647917, 4533933819, 4550408728, 4552726518):

**system_hardware_id UUID input fields (final Phase 1):**
| Field | In UUID |
|---|---|
| `cpu_model` | YES |
| `cpu_threads` | YES |
| `cpu_base_clock_mhz` | YES |
| `cpu_peak_clock_mhz` | YES |
| `ram_gb` | YES |
| `motherboard` | YES |
| `memory_speed_mhz` | YES |
| `memory_channels` | YES |
| `memory_interleave_profile` | YES |
| `bios_notes.bar_size_mb` | YES |
| `cpu_cores` | NO (implied by cpu_model) |
| `bios_notes.SMT_enabled` | NO (expressed by cpu_threads) |
| `bios_notes.above_4g_decoding` | NO |
| `bios_notes.iommu_enabled` | NO |
| `bios_notes.power_limit_w` | NO (metadata only; Phase 2 candidate) |
| `bios_notes.pcie_gen_override` | NOT STORED (covered by measured link width) |

**gpu_hardware UUID input fields (final Phase 1):**
| Field | In UUID |
|---|---|
| `pci_vendor_id` | YES |
| `pci_device_id` | YES |
| `pci_subsystem_vendor_id` | YES |
| `pci_subsystem_device_id` | YES |
| `vram_mb` | YES |
| `vram_type` (FK to seeded `vram_types`) | YES |
| `slot_index` | NO (position only; GPU identity is slot-independent) |
| `gpu_architecture` (FK) | NO (categorization; implied by PCI IDs) |
| tflops fields | NO (performance metrics) |
| `tdp_w` | NO (spec metadata) |
| `memory_bandwidth_peak_gb_s` | NO (performance metric) |
| `compute_units` | NO (dropped; implied by PCI IDs + arch) |

### UUID namespace design (per-table, not per-version)
After extended iteration (4568864985, 4572098058, 4573134793), the settled principle is **per-table namespace, not per-version**: `schema_tables.uuid_namespace` is a fixed UUID per table, set once and never changed. This prevents uuid_migrations from growing proportional to (all records × version count). Namespace stored in both DB (`schema_tables`) and backing disk file (per 4572098058).

Version-based namespace variants (one per schema version) were proposed and then rejected (4572098058) due to forcing UUID recomputation for every record on every schema bump.

### Migration-related tables (as agreed at thread close)

**`schema_versions`** (per 4559571210, changes JSONB dropped at 4559373879 direction):
```sql
CREATE TABLE schema_versions (
    schema_version_id       SERIAL PRIMARY KEY,
    description             TEXT NOT NULL,
    schema_migration_script TEXT,   -- DDL; idempotent; NULL = no-op
    record_migration_script TEXT,   -- per-record transform at ingest; NULL = identity
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**`schema_tables`** (per 4552726518, 4568864985):
```sql
CREATE TABLE schema_tables (
    table_id              SERIAL PRIMARY KEY,
    table_name            TEXT NOT NULL UNIQUE,
    added_schema_version  INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    last_modified_version INTEGER REFERENCES schema_versions(schema_version_id),
    uses_uuid_identity    BOOLEAN NOT NULL DEFAULT FALSE,
    uuid_namespace        UUID   -- per-table fixed namespace; source of truth for UUID v5 generation
);
```

**`uuid_migrations`** / **`table_migrations`**: Still in design revision at thread close. Name "table_migrations" preferred by @gissf1 over "schema_hops" (per 4574170103). @gissf1 requested the fields be merged into `schema_versions` rather than kept as a separate table, and directed that no implementation proceed until a new thread (#25) resolves the exact merged shape (per 4574170103, 4574254854).

Old_uuid/new_uuid columns: removed from the proposed table by 4573134793 — FK resolution is computable from `schema_version_id` on each record plus the per-table namespace, not stored lookup.

Scripting language for migration scripts: **Go templates + sprig**, with three custom function categories (record field access, table lookups, idempotent DDL helpers). Lua and CEL were considered and rejected or deferred (per 4554570690). Phase 1: stub signatures only; Phase 2: full implementation.

### `runs.run_id` — UUID natural key
`run_id` is `UUID PRIMARY KEY`, client-generated before DB write. UUIDs chosen over serial integers specifically to prevent enumeration of run IDs by external actors (per 4494171918).

### Hardware disk file layout
Hardware specs are embedded inline in the run disk file (not a separate per-hardware file requiring a DB round-trip) — air-gap compatible, self-contained. `hardware_id` UUID computed deterministically at ingest from embedded fields; not stored in the disk file (per 4533647917). Hardware snapshot once per process start, not per run (per 4533647917).

---

## Notable / unusual decisions

- **Two-tier provisional UUID for models** — allows model registration from a URL stub before weights are downloaded; promotional to hash-based UUID is automatic at weight-pull. Ensures distributed ID generation without DB round-trip while preserving deduplication integrity (per 4524304026, 4525059885). Downstream: conflict detection matrix handles soft fields (accept update silently) vs hard fields (re-derive from weights before accepting), with "do not delete disk file on rejection" as an explicit safety guarantee (per 4533208484, 4533310100).

- **UUID v5 with per-table fixed namespace (not per-version)** — unusual over the common approach of random UUID v4. Chosen so same canonical inputs always produce the same UUID on any host without a DB round-trip, enabling distributed ingestion without collision risk. Per-version namespaces were explored and rejected because they force mass UUID recomputation on every schema bump (per 4572098058, 4573134793). Downstream: `schema_tables.uuid_namespace` becomes a critical seed artifact.

- **`min_vram_mb` removed from schema** — initially proposed as a stored field, then explicitly dropped because it is derivable: `active_parameter_count_b × vram_multiplier × 1.15` at claim time (per 4525059885). Avoids storing a computed value that could drift from the calculation.

- **`bios_notes` as structured key-value JSON with UUID whitelist** — BIOS settings changed from a free-form string to a typed JSON object. Only a specific whitelist of keys is included in UUID hashing (currently only `bar_size_mb`); other keys are stored as metadata only. Whitelist is versioned; new keys trigger a schema version increment and migration records for existing hardware (per 4534654458, 4552726518, 4554570690).

- **Go templates + sprig for migration scripting** — instead of a standard migration tool (Liquibase, sqitch) or embedding a scripting language (Lua, JS), Go's standard `text/template` plus the sprig function library was chosen. Rationale: no new language dependency, idiomatic for a Go project, sandboxable, covers field transformation, substring extraction, and hash function changes. Custom functions registered for record field access, table lookups, and idempotent DDL helpers (per 4554570690, 4559373879). Phase 1 stubs only.

- **`allow_migration` boolean gate on migration records** — migration is not automatic on schema upgrade; it requires explicit operator confirmation per table transition. Prevents silent data corruption from unverified migration scripts (per 4559373879, 4554570690). Name settled over alternatives `is_migrating`, `migration_started`, `begin_migrating` (per 4553785030, 4559373879).

- **`tflops_source` as FK to `tflops_sources` lookup table** — manufacturer-supplied vs measured throughput numbers are tracked per record; the source table itself has a `url_template TEXT` field where a static URL prefix indicates a verbatim URL and any other prefix indicates a Go template expression (consistent with the scripting language choice). Seed ID 1 = unknown/unverified; "Rethunk measured" added when the project ships its own benchmarks (per 4552726518, 4554217644).

---

## Open / unresolved

- **`table_migrations` final schema / merge into `schema_versions`** — @gissf1 directed that `schema_hops`/`table_migrations` fields be merged into `schema_versions` rather than kept as a separate table, and that `schema_version_history` be eliminated with its fields moved onto `schema_tables`. The exact merged structure was not resolved — @gissf1 stated "I think this is becoming a mess" and directed the conversation to a new thread (#25) before any design was approved. The UUID namespace strategy ("UUID changes to any table can simply be migrated to a new table using a new UUID namespace") was also flagged as an open design question in that thread, not confirmed (per 4574170103, 4574254854).
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574170103>

- **`schema_tables.uuid_namespace` — DB vs Go constants** — at end of thread, storing namespaces in `schema_tables` and backing disk files was confirmed (per 4572098058). However the exact storage mechanism (column type, seeding process, relationship to backing file format) was not fully specified before the migration discussion was handed off to #25.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4572098058>

- **GPU UUID field set — completeness** — @gissf1 noted "we probably need more fields here too" for GPU hardware (per 4533894066) and the extended field set was proposed (per 4533933819). The question of which additional GPU fields (beyond PCI IDs + vram_mb + vram_type) belong in the UUID was not fully closed; `slot_index` was excluded from UUID but flagged as a carry-forward item for multi-GPU inference scenarios. No explicit "GPU UUID field list is final" confirmation was issued.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533894066>

- **`bios_notes` UUID whitelist completeness** — the whitelist was iterated to `bar_size_mb` (YES) only in Phase 1 (per 4554570690). @gissf1 noted other bus clock rates, multipliers, and widths should be tracked "as we notice them in results" (per 4550408728). No formal closure on what constitutes the complete Phase 1 whitelist beyond the confirmed entries.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728>

- **Multi-hop uuid_migrations chaining vs version-based matching** — after the direction to remove old_uuid/new_uuid from uuid_migrations (per 4572098058) and adopt version-based FK resolution, the exact mechanics of the chaining protocol (how many hops, when to flag for operator vs auto-resolve) were not confirmed before the thread closed. This is a first item for #25.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4572098058>

---

## Cross-topic links

- **hardware** (#17–#21): `system_hardware`, `gpu_hardware`, `system_gpu_link`, `run_hardware_metrics` schemas are referenced here as the UUID-identity tables for hardware fingerprinting. Hardware schema was locked in those tickets before this thread's UUID design began.
- **models/runs** (#12, #13, #15): `models.model_id`, `creators.creator_id`, `runs.run_id` UUID schemes were established in those tickets; this topic extends the discussion to hardware UUIDs and the migration framework.
- **migration framework** (#25): All unresolved questions about `schema_versions`, `table_migrations`, `schema_tables.uuid_namespace`, and the Go template scripting engine were explicitly handed off to #25 at thread close (per 4574254854).
- **schema_tables namespace backing files** (#13): @gissf1 noted that `schema_tables.uuid_namespace` backing files "likely affects the work done for #13" (per 4572098058) — the disk persistence design from #13 must accommodate these namespace seed files.
