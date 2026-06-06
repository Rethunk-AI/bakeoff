# Summary: uuids

## Final state

### run_id — UUID, client-generated

`runs.run_id UUID PRIMARY KEY`. Generated client-side before any DB write; globally unique without a DB sequence; prevents enumeration by outsiders. Natural key — no surrogate needed (per 4494171918).

### model_id — UUID v5, deterministic, two-tier

`models.model_id UUID PRIMARY KEY` (replaced SERIAL). Primary UUID: `UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash)` — authoritative once weights are pulled. Provisional UUID: `UUID5(BAKEOFF_MODEL_NAMESPACE, source_url + "|" + str(parameter_count_b) + "|" + str(model_source_size))` — used before hash is computed; promoted to primary at hash-computation time (per 4524304026, 4524573993).

UUID composition includes `name`, `architecture`, `quantization`, `active_parameter_count_b`, `model_source_size`. `model_source_mtime` excluded — not stable for identity (per 4524304026).

### creator_id — UUID v5 from homepage

`creators.creator_id UUID PK`. Composition: `UUID5(BAKEOFF_CREATOR_NAMESPACE, homepage)`. Fallback when no homepage: `UUID5(BAKEOFF_CREATOR_NAMESPACE, display_name)` with `provisional: true` (per 4525059885). Two separate namespace constants — `BAKEOFF_MODEL_NAMESPACE` and `BAKEOFF_CREATOR_NAMESPACE` — committed to `bench/constants.py` (shipped in `dac26d5`, per 4533311354).

### system_id — stable per-host UUID

`system_hardware.system_id UUID NOT NULL UNIQUE`. Generated once via `uuid.uuid4()` at first runner invocation; persisted to `~/.config/bakeoff/system_id`; never regenerated. Stable across runs and across different submitting accounts on the same machine. Combined with `publisher_id` and `pci.device_id` for result-fabrication detection (per 4467940739, issue #19).

### hardware_id — deterministic from embedded snapshot fields

Not stored in disk file; derived at ingest time from the embedded hardware snapshot. UUID input fields (per 4533933819, 4552726518, 4553785030):

**System:**
- `cpu_model`, `cpu_threads`, `cpu_base_clock_mhz`, `cpu_peak_clock_mhz`, `ram_gb`, `motherboard`
- `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile`
- `bios_notes.bar_size_mb` (only UUID-input bios key; Phase 1 final per 4559571210)

Fields excluded from UUID: `cpu_cores` (implied by `cpu_model`), `bios_notes.above_4g_decoding`, `bios_notes.SMT_enabled` (expressed by `cpu_threads`), `bios_notes.iommu_enabled`, `bios_notes.pcie_gen_override` (not stored at all), `bios_notes.power_limit_w` (metadata only).

**GPU:**
- `pci_vendor_id`, `pci_device_id`, `pci_subsystem_vendor_id`, `pci_subsystem_device_id`, `vram_mb`, `vram_type` (FK to seeded lookup)
- `slot_index` excluded — moving a GPU to a different slot does not change the GPU's identity; slot tracked in `system_gpu_link` (per 4534654458)
- All performance metrics (tflops, bandwidth, TDP) excluded from UUID (per 4550408728)

Hardware snapshot is embedded inline in the run disk file (portable, air-gap safe); hardware_id UUID is an ingest artifact, not a stored field (per 4533647917).

### Per-table UUID namespace (fixed, not per-version)

`schema_tables.uuid_namespace UUID` — one fixed namespace per table, set at table creation, never changed. UUID v5 = `uuid5(TABLE_NAMESPACE, canonical_fields_json)`. Per-version namespaces were rejected: they force UUID recomputation for every record on every schema bump, causing unbounded `uuid_migrations` growth (per 4573134793).

### FK type conventions

All UUID-based PKs use `UUID` column type in DB. Non-UUID lookup tables (model_architectures, model_file_formats, quantization_methods, interface_type, vram_types, gpu_architectures, tflops_sources, task_categories) use `SERIAL PRIMARY KEY`. The `predecessor_model_id` FK on `models` is `UUID FK → models`.

---

## Notable / unusual decisions

- **Two-tier model_id (provisional → primary):** Allows distributed model registration without weights being available. Provisional UUID from URL + parameter_count_b + model_source_size is promotable at hash-computation time. Multiple provisionals for the same model coalesce automatically when the first one is hashed (per 4524304026, 4525059885). Downstream use: enables stub disk files with only `source_url` — ingest pipeline resolves all fields and renames the file.

- **hardware_id derived at ingest, not stored:** Disk files carry raw hardware fields; the UUID is computed on import. This keeps run files self-contained and air-gap compatible — no DB or UUID library needed on the runner (per 4533647917). Downstream use: dedup cache at `hardware-snapshot/<hardware_id>.json` written by ingest pipeline; runner never touches it.

- **system_id separate from hardware_id:** `system_id` (stable per-host UUID in local config) identifies the physical machine across accounts and runs. `hardware_id` is derived from the full hardware fingerprint including BIOS settings and GPU config. They serve different purposes: `system_id` is an anti-fraud anchor; `hardware_id` is a benchmark dedup key (per 4467940739, issue #19).

- **bios_notes as structured key-value JSON:** Changed from free-form string after recognizing that structured keys allow selective UUID inclusion. Only keys on a whitelist enter the UUID hash; others stored as metadata. Whitelist updated via schema version; new keys trigger migration records for affected hardware entries (per 4534654458).

- **UUID conflict detection matrix (model ingest):** Hard fields (`parameter_count_b`, `context_length_*`, `architecture`, `file_format`, `quantization`) validated against derived-from-weights values before accepting updates. Soft fields (`description`, `creator.display_name`, `creator.homepage`) accepted unconditionally. Disk file never deleted on rejection — submitter retains data for correction and resubmit (per 4533208484, 4533310100).

- **Namespace constants committed to `bench/constants.py`:** Two UUIDs — `BAKEOFF_MODEL_NAMESPACE` and `BAKEOFF_CREATOR_NAMESPACE` — generated once and committed as project constants (shipped `dac26d5`). Ensures same model registered independently on two hosts produces the same `model_id`. Downstream use: model disk file and DB row share identical UUID by construction; no DB round-trip needed to generate a model_id.

- **runners.status TEXT+CHECK, not ENUM:** Chosen during active schema evolution phase; ENUM requires `ALTER TYPE` to add values, TEXT+CHECK is a one-line migration. Carries `TODO(P2): migrate to ENUM` comment in `schema.sql` (per 4522610944, 4524304026).

---

## Open / unresolved

- **table_migrations / schema_versions merger — unresolved.** The migration framework design was explicitly handed off to a new issue (bakeoff#25) at the close of #22 (per 4574170103, 4574254854). At closure of #22, gissf1 had not approved the table shape. The following sub-questions remain open in #25:
  - Whether `table_migrations` is a standalone table or merged into `schema_versions`
  - Whether `schema_version_history` is eliminated by adding `schema_version_id` + `applied_at` directly to `schema_tables`
  - Exact merge structure (one row per table per version? JSONB array? other?)

- **UUID namespace strategy for schema version changes — unresolved.** gissf1's final position (4574170103): "UUID changes to any table can simply be migrated to a new table using a new UUID namespace." Bastion proposed per-table fixed namespaces (not per-version). These are architecturally different; gissf1 was not yet settled. Carried to #25 as first open design question (per 4574254854).

- **FK resolution across tables after UUID migration — not fully confirmed.** Bastion proposed lazy resolution (encounter old UUID in FK → look up uuid_migrations → use new_uuid) plus batch FK-column update pass in schema_migration_script. gissf1 accepted the approach in principle but UUID/migration table shape was still in flux at handoff (per 4568864985, 4572098058).

- **Go templates + sprig custom function plan — Phase 2 stub only.** Scripting language confirmed as Go templates + sprig (over Lua, Jinja2, AWK). Three custom function categories agreed: record field access, table lookups, idempotent DDL helpers. Implementation deferred to Phase 2; only stub signatures needed in Phase 1 (per 4554570690, 4559571210, 4568864985).

- **schema_versions.changes JSONB — dropped**, but gissf1's direction was to make `migration_script` self-documenting. Two separate script fields (`schema_migration_script`, `record_migration_script`) confirmed as the right split (per 4559571210, 4568864985). This part is settled; the broader framework shape is not.

- **GPU UUID field list — `bar_size_mb` confirmed as sole bios_notes UUID input**, but gissf1 noted additional GPU fields may be needed beyond the PCI IDs + vram_mb + vram_type set without specifying what they are (per 4550408728). Left open pending actual GPU data accumulation.

---

## Cross-topic links

- **gpu_hardware** — GPU UUID input fields (`pci_vendor_id`, `pci_device_id`, `pci_subsystem_vendor_id`, `pci_subsystem_device_id`, `vram_mb`, `vram_type`) live on `gpu_hardware`; vram_type is a FK to a seeded lookup table that is part of gpu_hardware's topic scope. `gpu_hardware.gpu_hardware_id` is a UUID PK.

- **system_hardware** — `system_hardware.system_id` (stable per-host UUID) and the full system UUID input field set live here. `hardware_id` is derived from system_hardware + gpu_hardware at ingest.

- **system_gpu_link** — carries `slot_index`, which is excluded from GPU UUID for identity reasons; it is relational data about a GPU's position in a system, not part of GPU identity.

- **run_hardware_metrics** — references `system_gpu_link` compound key; `run_id` is a UUID FK → `runs`. Full hardware context traversal: `run_hardware_metrics → system_gpu_link → system_hardware + gpu_hardware + interface_type`.

- **runs** (run_model_metrics topic) — `runs.run_id UUID PRIMARY KEY` is the anchor FK for both `run_model_metrics` and `run_hardware_metrics`. `runs.publisher_id` is a non-UUID text key; `runs.runner_version` is a semver/commit string.

- **models** (model topic) — `models.model_id UUID PK` (UUID v5), `models.predecessor_model_id UUID FK → models`, `models.creator_id UUID FK → creators`. The UUID generation strategy and conflict detection matrix are the primary content of this topic.

- **schema_versions / table_migrations** (migration framework topic, bakeoff#25) — `schema_tables.uuid_namespace` is the per-table UUID v5 namespace constant; stored in DB and backed by a seed file per #13 dependency note (per 4572098058). The full migration framework design lives in #25.
