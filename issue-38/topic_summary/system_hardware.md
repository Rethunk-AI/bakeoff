# Summary: system_hardware

## Final state

### `system_hardware` table

| Column | SQL type | Constraint | Notes |
|--------|----------|------------|-------|
| `system_hardware_id` | `SERIAL` | `PRIMARY KEY` | Surrogate PK |
| `system_id` | `UUID` | `NOT NULL UNIQUE` | Stable per-host UUID; generated via `uuid.uuid4()` at first run, persisted to `~/.config/bakeoff/system_id`; never regenerated; v5 namespace per schema version for self-identifying UUIDs (per 4568864985) |
| `publisher_id` | `TEXT` | `NOT NULL` | Submitting user/account from runner config |
| `cpu_model` | `TEXT` | `NULLABLE` | From `/proc/cpuinfo` `model name`; `platform.processor()` fallback |
| `cpu_threads` | `INT` | `NULLABLE` | Logical thread count; BIOS-configurable (HT/SMT), so not implied by `cpu_model` alone (per 4533894066) |
| `cpu_base_clock_mhz` | `INT` | `NULLABLE` | From `/proc/cpuinfo` or `dmidecode`; detects fixed-clock / underclocked configs |
| `cpu_peak_clock_mhz` | `INT` | `NULLABLE` | Detects boost/OC configs |
| `ram_total_gb` | `FLOAT` | `NULLABLE` | `psutil.virtual_memory().total / (1024**3)` |
| `motherboard` | `TEXT` | `NULLABLE` | Make/model; from `dmidecode` |
| `memory_speed_mhz` | `INT` | `NULLABLE` | Active clock (not SPD rated) |
| `memory_channels` | `INT` | `NULLABLE` | Active channel count; BIOS-configurable |
| `memory_interleave_profile` | `TEXT` | `NULLABLE` | XMP / EXPO / DOCP / `manual` |
| `bios_notes` | `JSONB` | `NULLABLE` | Structured key-value BIOS settings; subset whitelisted for UUID input (see below) |
| `schema_version_id` | `INTEGER` | `REFERENCES schema_versions` | Added to all UUID-identity tables for migration tracking (per 4568864985) |

`cpu_cores` was explicitly dropped — implied by `cpu_model` (per 4533894066). `os` and `kernel_version` were moved to the separate `system_software` table (per #19 body).

**`bios_notes` UUID whitelist (Phase 1 final, per 4559571210):**

| Key | UUID input? |
|-----|-------------|
| `bar_size_mb` | YES — BAR aperture affects VRAM access throughput |
| `SMT_enabled` | NO — expressed by `cpu_threads` |
| `above_4g_decoding` | NO |
| `iommu_enabled` | NO — security boundary |
| `power_limit_w` | NO — metadata only; transient; Phase 2 candidate |
| `pcie_gen_override` | NOT STORED — fully removed; covered by measured link rate in `system_gpu_link` |

**Deduplication:** upsert on `system_id` (unique constraint); same machine across runs reuses the existing row.

**UUID strategy:** `system_id` = `uuid5(NAMESPACE_VN, canonical_fields_json)` where `NAMESPACE_VN = uuid5(UUID_NIL, "bakeoff.hardware_uuid.vN")`. Version embedded in namespace so UUID is self-identifying without a version tag on the record (per 4568864985). Fields included in UUID input: `cpu_model`, `cpu_threads`, `cpu_base_clock_mhz`, `cpu_peak_clock_mhz`, `ram_gb`, `motherboard`, `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile`, `bar_size_mb` (from `bios_notes`), plus per-slot GPU tuple list from `system_gpu_link` (sorted by `slot_index`).

---

### `system_software` table

| Column | SQL type | Constraint | Notes |
|--------|----------|------------|-------|
| `system_software_id` | `SERIAL` | `PRIMARY KEY` | |
| `os` | `TEXT` | `NULLABLE` | `platform.platform()` or `distro.name(pretty=True)` |
| `kernel_version` | `TEXT` | `NULLABLE` | `platform.release()` |
| `python_version` | `TEXT` | `NULLABLE` | `platform.python_version()` |
| `gpu_driver_version` | `TEXT` | `NULLABLE` | `nvidia-smi --query-gpu=driver_version` |
| `cuda_version` | `TEXT` | `NULLABLE` | `nvidia-smi` header or `torch.version.cuda`; null for ROCm/CPU-only |
| `rocm_version` | `TEXT` | `NULLABLE` | `rocm-smi --showversion`; null for CUDA runners |
| `runner_version` | `TEXT` | `NULLABLE` | Harness `__version__` or `git describe --tags --always` |

**Deduplication:** NOT deduplicated — each run inserts a fresh row (environment snapshot). Optional hash dedup deferred to Phase 2.

---

### `system_gpu_link` table (join table — many-to-many)

| Column | SQL type | Constraint | Notes |
|--------|----------|------------|-------|
| `system_hardware_id` | `INT` | `NOT NULL REFERENCES system_hardware` | FK to host |
| `slot_index` | `INT` | `NOT NULL` | GPU index from nvidia-smi; part of PK |
| — | — | `PRIMARY KEY (system_hardware_id, slot_index)` | Slot is a fixed property of the system |
| `gpu_hardware_id` | `INT` | `NOT NULL REFERENCES gpu_hardware` | FK to GPU model record |
| `slot_native_interface_type_id` | `INT` | `NULLABLE REFERENCES interface_type` | Motherboard slot's rated max spec |
| `actual_interface_type_id` | `INT` | `NULLABLE REFERENCES interface_type` | Negotiated running state (what both sides agreed to at link-up) |

Secondary index on `gpu_hardware_id` for "all systems running GPU model X" queries.

`is_pcie_slot_limited` is NOT stored — derived as `slot_native_interface_type_id != actual_interface_type_id` at query/display time (per #20).

NVLink deferred to Phase 2 — no NVLink columns in this table.

**Deduplication:** upsert on `(system_hardware_id, slot_index)`; update `gpu_hardware_id` and interface FKs if GPU in slot changes.

---

### `run_hardware_metrics` FK additions (per #21)

```sql
ALTER TABLE run_hardware_metrics
    ADD COLUMN system_hardware_id INT NULLABLE,
    ADD COLUMN slot_index         INT NULLABLE,
    ADD COLUMN system_software_id INT NULLABLE REFERENCES system_software,
    ADD CONSTRAINT fk_run_hardware_metrics_system_gpu_link
        FOREIGN KEY (system_hardware_id, slot_index)
        REFERENCES system_gpu_link (system_hardware_id, slot_index);
```

Nullable on first addition; NOT NULL once backfill confirmed clean. Links each run to its full hardware context via a 3-way join through `system_gpu_link` → `gpu_hardware` + `system_hardware`.

---

### Schema versioning / migration tables (per #22 thread)

```sql
CREATE TABLE schema_versions (
    schema_version_id       SERIAL PRIMARY KEY,
    description             TEXT NOT NULL,
    schema_migration_script TEXT,   -- Go template + sprig; DDL; runs once per upgrade; idempotent
    record_migration_script TEXT,   -- Go template + sprig; per-record transform at ingest
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE schema_version_history (
    schema_version_id INTEGER NOT NULL REFERENCES schema_versions PRIMARY KEY,
    applied_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE schema_tables (
    table_id              SERIAL PRIMARY KEY,
    table_name            TEXT NOT NULL UNIQUE,
    uuid_namespace_v1     UUID NOT NULL,   -- base namespace for UUID v5 generation
    uses_uuid_identity    BOOLEAN NOT NULL DEFAULT FALSE,
    added_schema_version  INTEGER NOT NULL REFERENCES schema_versions,
    last_modified_version INTEGER REFERENCES schema_versions
);

CREATE TABLE uuid_migrations (
    migration_id  SERIAL PRIMARY KEY,
    entity_type   TEXT NOT NULL,   -- 'system_hardware', 'gpu_hardware', etc.
    old_uuid      UUID NOT NULL,
    new_uuid      UUID NOT NULL,
    from_version  INTEGER NOT NULL REFERENCES schema_versions,
    to_version    INTEGER NOT NULL REFERENCES schema_versions,
    allow_migration BOOLEAN NOT NULL DEFAULT FALSE,
    migrated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (entity_type, old_uuid)
);
```

One row per hop per table in `uuid_migrations` (v1→v2 and v2→v3 are separate rows). Pipeline chains hops in order (per 4559373879). `new_uuid` present in final consolidated dump (4568864985); `allow_migration` field name confirmed (per 4553530231 + 4559571210).

`schema_tables.uuid_namespace_v1` storage — open: DB column vs Go constants only (per 4568864985 final question, not yet confirmed by @gissf1).

---

## Notable / unusual decisions

- **`cpu_cores` dropped** — gissf1 directed removal (per 4533894066): implied by `cpu_model`, so storing it would be redundant. Only `cpu_threads` is kept because HT/SMT can be disabled in BIOS, making thread count independently observable.

- **`system_software` separate from `system_hardware`** — software environment changes between runs (driver updates, kernel upgrades) while hardware stays fixed. Separating them avoids re-stamping hardware identity for every software change. `run_hardware_metrics` FK to `system_software` captures the snapshot active during that specific run.

- **Option C (join table) chosen over Option A/B for GPU-system relationship** (per 4470208621) — Option A (FK on `system_hardware`) breaks multi-GPU systems. Option B (FK on `gpu_hardware`) prevents multiple users with the same GPU model from sharing one `gpu_hardware` row. Join table is the only structure that handles both many GPUs per system and many systems per GPU model at community scale.

- **Interface data lives on `system_gpu_link`, not on either entity** — PCIe gen/width is a property of the relationship between a GPU and a slot, not of either independently. Placing it on the join table is architecturally correct and enables bottleneck detection (slot limiting GPU or GPU limiting slot) as a derived display annotation.

- **`actual_interface_type_id` renamed from `slot_actual_interface_type_id`** (per 4474333279) — "actual" belongs to neither side; it is what both negotiated at link-up. The rename signals this joint ownership.

- **`gpu_native_interface_type_id` moved from `system_gpu_link` to `gpu_hardware`** (per 4474333279) — it is a fixed property of the GPU card (what interface spec the card itself supports), not of the slot relationship.

- **UUID v5 namespace per schema version** (per 4568864985) — each schema version defines `NAMESPACE_VN = uuid5(UUID_NIL, "bakeoff.hardware_uuid.vN")`. Hardware fingerprint UUID = `uuid5(NAMESPACE_VN, fields_json)`. Two properties: (1) UUID is self-identifying without needing a stored version tag; (2) same hardware fingerprint in v1 and v2 produces different UUIDs, so old and new records cannot collide.

- **`bios_notes` as structured JSONB with UUID whitelist, not free-form string** (per 4533647917 + 4534654458) — allows extending BIOS tracking without schema migration, while giving a stable, deterministic input to UUID computation via a keyed whitelist. Keys not in the whitelist are stored but never hashed.

- **`bar_size_mb` as UUID input instead of `resizable_bar` boolean** (per 4552726518 / confirmed 4553530231) — boolean granularity is wrong; the actual BAR aperture size (e.g., 256 MB vs 16 GB) is what materially differentiates throughput. `resizable_bar` bool replaced by `bar_size_mb` integer.

- **Go templates + sprig as migration scripting language** (per 4554570690) — chosen over Lua (4553785030) after gissf1 pushed back on new language dependency. `text/template` is Go stdlib; sprig covers field transformation, hash functions, substring, mapping. Custom functions registered for record field access, table lookups, and idempotent DDL helpers.

- **Two separate migration script fields (`schema_migration_script`, `record_migration_script`)** (per 4559571210 + 4568864985) — DDL runs once per DB upgrade; record transform runs per-record at ingest. Mixing them would put DDL in a per-record hot path at volume, which is unsafe.

---

## Open / unresolved

- **`schema_tables.uuid_namespace_v1` storage location** (per 4568864985 final question): store namespace in DB column (allows tooling to look up without code changes) vs define as Go constants only (simpler, source of truth in code). @gissf1 had not yet responded to this question as of the last comment.

- **`pci_vendor_id`, `pci_subsystem_vendor_id` on `gpu_hardware`** (per 4629938143): gissf1's issue #38 lists these as expected fields. Bastion's audit noted they appear in the #22 discussion but are not in the #18 spec as authored. Needs @gissf1 to confirm which issue/comment ratified them, or confirm they should be added in the schema patch.

- **`run_hardware_metrics` completeness** (per 4629938143 / issue #38 body): gissf1 stated #21 does not cover all changes discussed near end of #8. Specific missing fields or constraints not yet enumerated by @gissf1. Bastion awaiting that list before treating #21 as closed.

- **`tflops_fp32` and `tflops_bf16` in current schema** (per 4629938143): these columns exist in `schema.sql` but were not in the #18 spec as written. Bastion asked: retain or drop? No answer yet.

- **`vram_type` as TEXT vs FK to lookup table**: gissf1 directed that it should be a seeded lookup table FK (per 4534370703). The #18 spec as authored used TEXT. Bastion's audit flagged this discrepancy in 4629938143 but did not receive final direction on the FK vs TEXT resolution.

- **Many other gap items from #38 / 4629938143**: schema audit is pending @gissf1 sign-off on the consolidated patch DDL before any migration runs. Multiple tables (interface_type seed data, gpu_hardware columns, system_hardware/system_software presence at all) flagged as unimplemented vs. spec.

---

## Cross-topic links

- `system_gpu_link.gpu_hardware_id` → **gpu_hardware** topic (FK to GPU model record; defines interface and compute spec)
- `system_gpu_link.slot_native_interface_type_id`, `actual_interface_type_id` → **interface_type** lookup table (PCIe gen/width matrix, SXM, NVLink variants)
- `run_hardware_metrics.(system_hardware_id, slot_index)` → **system_gpu_link** (compound FK; ties run to specific GPU-in-slot-on-host context)
- `run_hardware_metrics.system_software_id` → **system_software** table (software environment snapshot for that run)
- `schema_versions`, `uuid_migrations`, `schema_tables`, `schema_version_history` → cross-cutting migration infrastructure; applies to `gpu_hardware` and any other UUID-identity table, not just `system_hardware`
