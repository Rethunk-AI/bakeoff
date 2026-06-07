# Summary: system_hardware

## Final state

Five normalized tables plus three migration-infrastructure tables, as settled by end of #22 thread.

### `interface_type`
| column | type | constraints |
|---|---|---|
| `interface_type_id` | SERIAL | PRIMARY KEY |
| `bandwidth_peak_gb_s` | FLOAT | NOT NULL |
| `description` | TEXT | NOT NULL |
| `interface_family` | TEXT | NULLABLE |
| `transfer_rate` | INT | NULLABLE (PCIe GT/s; null for non-PCIe) |
| `lane_count` | INT | NULLABLE (PCIe width; null for non-PCIe) |

Seeded via `seeds/interface_types.json` (per 4533634217, 4533647917).

### `gpu_hardware`
| column | type | constraints |
|---|---|---|
| `gpu_hardware_id` | SERIAL | PRIMARY KEY |
| `gpu_name` | TEXT | NOT NULL |
| `pci_vendor_id` | TEXT | NULLABLE |
| `pci_device_id` | TEXT | NULLABLE |
| `pci_subsystem_vendor_id` | TEXT | NULLABLE |
| `pci_subsystem_device_id` | TEXT | NULLABLE |
| `vram_total_mb` | INT | NULLABLE |
| `vram_type` | FK → `vram_types` | NULLABLE (seeded lookup) |
| `memory_bus_width_bits` | INT | NULLABLE |
| `memory_bandwidth_peak_gb_s` | FLOAT | NULLABLE |
| `clock_memory_mhz` | INT | NULLABLE |
| `clock_graphics_boost_mhz` | INT | NULLABLE |
| `peak_tflops_fp16` | FLOAT | NULLABLE |
| `tdp_watts` | INT | NULLABLE |
| `gpu_native_interface_type_id` | INT | NULLABLE FK → `interface_type` |
| `gpu_architecture_id` | FK → `gpu_architectures` | NULLABLE (categorization only; not in UUID) |

tflops values live in a related table `gpu_tflops { gpu_hardware_id, compute_format_id FK → compute_formats, tflops_value, tflops_source_id FK → tflops_sources }` (per 4534654458). `compute_units` column dropped — implied by PCI IDs (per 4552726518).

### `system_hardware`
| column | type | constraints |
|---|---|---|
| `system_hardware_id` | SERIAL | PRIMARY KEY |
| `system_id` | UUID | NOT NULL UNIQUE (stable per-host, generated once at first run) |
| `publisher_id` | TEXT | NOT NULL |
| `cpu_model` | TEXT | NULLABLE |
| `cpu_threads` | INT | NULLABLE |
| `cpu_base_clock_mhz` | INT | NULLABLE |
| `cpu_peak_clock_mhz` | INT | NULLABLE |
| `ram_total_gb` | FLOAT | NULLABLE |
| `motherboard` | TEXT | NULLABLE |
| `memory_speed_mhz` | INT | NULLABLE |
| `memory_channels` | INT | NULLABLE |
| `memory_interleave_profile` | TEXT | NULLABLE |
| `bios_notes` | JSONB | NULLABLE (structured key-value) |

OS, kernel, driver fields moved to `system_software` (per #19 body). `cpu_cores` dropped — implied by `cpu_model` (per 4533894066).

### `system_software`
| column | type | constraints |
|---|---|---|
| `system_software_id` | SERIAL | PRIMARY KEY |
| `os` | TEXT | NULLABLE |
| `kernel_version` | TEXT | NULLABLE |
| `python_version` | TEXT | NULLABLE |
| `gpu_driver_version` | TEXT | NULLABLE |
| `cuda_version` | TEXT | NULLABLE |
| `rocm_version` | TEXT | NULLABLE |
| `runner_version` | TEXT | NULLABLE |

Inserts a fresh row per run (environment snapshot, not deduplicated across runs) (per #19 body).

### `system_gpu_link`
| column | type | constraints |
|---|---|---|
| `system_hardware_id` | INT | NOT NULL FK → `system_hardware` |
| `slot_index` | INT | NOT NULL |
| PK | — | `(system_hardware_id, slot_index)` |
| `gpu_hardware_id` | INT | NOT NULL FK → `gpu_hardware` (indexed) |
| `slot_native_interface_type_id` | INT | NULLABLE FK → `interface_type` |
| `actual_interface_type_id` | INT | NULLABLE FK → `interface_type` |

`is_pcie_slot_limited` not stored — derivable from `slot_native != actual` (per #20 body). NVLink deferred to Phase 2 (per #20 body).

### `run_hardware_metrics`
| column | type | constraints |
|---|---|---|
| `run_id` | TEXT | NOT NULL |
| `system_hardware_id` | INT | NULLABLE → NOT NULL after backfill |
| `slot_index` | INT | NULLABLE → NOT NULL after backfill |
| FK | — | `(system_hardware_id, slot_index)` → `system_gpu_link` |
| `system_software_id` | INT | NULLABLE FK → `system_software` |
| `wall_clock_seconds` | FLOAT | NULLABLE |
| `time_to_first_token_ms` | FLOAT | NULLABLE |
| `tokens_per_second` | FLOAT | NULLABLE |
| `peak_vram_mb` | FLOAT | NULLABLE |
| `gpu_sm_utilization_pct` | FLOAT | NULLABLE |
| `tflops_utilization_pct` | FLOAT | NULLABLE |
| `cpu_cycles_elapsed` | BIGINT | NULLABLE |
| `cpu_time_user_ms` | FLOAT | NULLABLE |
| `cpu_time_sys_ms` | FLOAT | NULLABLE |
| `gpu_wall_time_ms` | FLOAT | NULLABLE (optional, runner-dependent) |

### Migration infrastructure tables

**`schema_versions`** (per 4559571210, 4568864985):
```sql
CREATE TABLE schema_versions (
    schema_version_id       SERIAL PRIMARY KEY,
    description             TEXT NOT NULL,
    schema_migration_script TEXT,   -- DDL; runs once per upgrade; idempotent; NULL = no-op
    record_migration_script TEXT,   -- per-record transform at ingest; NULL = identity
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```
`changes` JSONB column dropped — `migration_script` is self-documenting (per 4559373879, 4559571210). Scripting language: Go templates + sprig + three custom function categories (record field access, table lookups, idempotent DDL helpers); Lua considered and rejected (per 4554570690).

**`schema_version_history`** (per 4568864985):
```sql
CREATE TABLE schema_version_history (
    schema_version_id INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    applied_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (schema_version_id)
);
```

**`uuid_migrations`** (final form per 4568864985):
```sql
CREATE TABLE uuid_migrations (
    migration_id  SERIAL PRIMARY KEY,
    entity_type   TEXT NOT NULL,
    old_uuid      UUID NOT NULL,
    new_uuid      UUID NOT NULL,
    from_version  INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    to_version    INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    migrated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (entity_type, old_uuid)
);
```

**`schema_tables`** (per 4568864985):
```sql
CREATE TABLE schema_tables (
    table_id           SERIAL PRIMARY KEY,
    table_name         TEXT NOT NULL UNIQUE,
    uuid_namespace_v1  UUID NOT NULL
);
```

`hardware_id_migrations` → renamed `uuid_migrations` (generic, not hardware-specific) (per 4550408728). `schema_versions` generalized from `hardware_schema_versions` (per 4550408728).

### `hardware_id` UUID field sets (Phase 1)

**system_hardware UUID inputs** (per 4533933819, 4550408728, 4553785030):
`cpu_model`, `cpu_threads`, `cpu_base_clock_mhz`, `cpu_peak_clock_mhz`, `ram_gb`, `motherboard`, `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile`, `bios_notes.bar_size_mb` (only UUID-whitelisted key).

**gpu_hardware UUID inputs** (per 4534654458, 4552726518):
`pci_vendor_id`, `pci_device_id`, `pci_subsystem_vendor_id`, `pci_subsystem_device_id`, `vram_mb`, `vram_type` (FK). All tflops/bandwidth/TDP/architecture fields excluded from UUID (per 4534370703).

**UUID generation scheme**: UUID v5 namespaces per schema version — `uuid5(NAMESPACE_VN, canonical_fields_json)` (per 4568864985). Enables self-identification of version without disk file metadata. `uuid_migrations` retains FK bridge role only.

### Disk persistence

- Hardware specs embedded inline in run disk file (self-contained, air-gap compatible) (per 4533647917 confirming option C+A).
- `hardware_id` UUID computed at ingest from embedded fields, not stored in disk file (per 4533647917).
- Hardware snapshot dedup cache: single `hardware-snapshot/<uuid>.json` combining system + GPU list; per-GPU UUID files deferred to Phase 2 (per 4533933819).
- Run hardware context embedded in `runs/<run_id>.json` as a `hardware:` block; sparse/reference format Phase 2 (per 4533634217, 4533647917).
- Hardware snapshot taken once at process start; reused across runs in same session (per 4533647917 confirming option A).

---

## Notable / unusual decisions

- **Three-way normalized split** (`gpu_hardware` / `system_hardware` / `system_gpu_link`) instead of embedding GPU data in system row — driven by community benchmarking: many users submitting RTX 4090 results should share one `gpu_hardware` row, not duplicate it. Multi-GPU systems require one slot row each (per 4470208621).

- **`system_gpu_link` PK is `(system_hardware_id, slot_index)` not a surrogate** — the slot is a fixed physical property of the system; which GPU occupies it is the data (per #20 body). Enables upsert when GPU changes without new PK.

- **`is_pcie_slot_limited` not stored** — fully derivable from `slot_native != actual`; storing it would create stale-flag risk (per #20 body).

- **`cost_usd` not stored** — derived at display time from energy specs × duration or provider pricing × tokens, depending on local vs API mode (per 4462460688).

- **`memory_bandwidth_peak_gb_s` stored (not derived)** — formula requires `vram_type` to select DDR factor; storing avoids re-implementing that at every display callsite (per 4469686066).

- **`gpu_native_interface_type_id` on `gpu_hardware`, not `system_gpu_link`** — it describes a fixed property of the card (native interface spec), not the slot relationship (per 4474333279).

- **bios_notes UUID whitelist — only `bar_size_mb`** — `SMT_enabled` expressed by `cpu_threads`; `pcie_gen_override` fully removed (covered by measured link rate in join table); `power_limit_w` stored as metadata only (transient; Phase 2 candidate); `above_4g_decoding` and `iommu_enabled` excluded as non-performance (per 4559571210, 4554570690).

- **UUID v5 namespaces per schema version** — hardware UUIDs are self-identifying: `uuid5(NAMESPACE_VN, fields)` differs across versions even for same hardware fingerprint; eliminates need for version tag on the UUID itself (per 4568864985, @AlbinoGeek proposal accepted).

- **Go templates + sprig + custom functions, not Lua** — Lua was proposed (4553785030) then rejected in favor of Go templates as no new language dependency, idiomatic for Go project, well-tested security profile (Helm/Kubernetes) (per 4554570690).

- **Two separate script fields** (`schema_migration_script` + `record_migration_script`) on `schema_versions` — DDL runs once per upgrade; record transforms run per-record at ingest; mixing them would put DDL inside a per-record hot path (per 4559571210).

- **`uuid_migrations` holds one row per hop** (v1→v2, v2→v3 = two rows), not one pointing to latest — cleaner audit trail; each hop's script self-contained and tested independently (per 4559373879 confirmed, 4559571210).

- **tflops via related table `gpu_tflops`** rather than per-format columns on `gpu_hardware` — extensible to new compute formats (fp8, int4, etc.) without schema migration; `tflops_source_id` FK to `tflops_sources` table which carries URL template + `contacts JSONB` (per 4534654458, 4552726518, 4553785030).

- **`compute_units` dropped** — implied by PCI IDs + architecture; no independent signal for test surface (per 4552726518, per @gissf1 direction 4534370703).

---

## Open / unresolved

1. **GPU UUID field set incomplete** — @gissf1 noted "we probably need more fields here too" for GPU hardware beyond the PCI IDs + vram_mb + vram_type set. No specific additional fields agreed. GPU performance metrics (tflops, bandwidth, TDP) explicitly excluded from UUID but additional identity fields TBD.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728>

2. **BIOS UUID whitelist additions** — thread confirmed only `bar_size_mb` for Phase 1. @gissf1 noted tracking "other bus clock rates, multipliers, widths, etc." but no specific additional keys were agreed. Whitelist is open-ended for Phase 2 additions via schema_version increment.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728>

3. **`uuid_migrations.entity_type` as TEXT vs FK to `schema_tables`** — Bastion proposed FK to `schema_tables(table_id)` (per 4552726518); final consolidated dump in 4568864985 reverted to `entity_type TEXT NOT NULL`. The two representations are inconsistent within the thread; never explicitly reconciled.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4568864985>

4. **`schema_tables.uuid_namespace_v1` — DB storage vs Go constants** — Bastion raised this explicitly at end of final comment and did not receive a direction: "should `schema_tables.uuid_namespace_v1` be stored in DB (as above), or defined as Go constants only?"
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4568864985>

5. **`tflops_sources.contacts` multi-contact structure** — JSONB array of `{type, value}` pairs accepted for Phase 1; full contacts lookup table deferred to Phase 2. No confirmation from @gissf1 on the JSONB contacts shape (it remained in Bastion's open-items list at 4553785030 with status "awaiting confirmation").
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553785030>

6. **EAV approach for BIOS settings** — @gissf1 proposed a lookup table for BIOS settings/hardware variants to avoid duplicating large text records per BIOS variant; Bastion tagged as Phase 2 and @gissf1 agreed, but no Phase 2 design was started.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728>

7. **Multi-processor systems** — @gissf1 raised handling of multi-processor hosts; deferred as "not urgent enough" with no design proposed.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728>

8. **`power_limit_w` UUID inclusion** — stored as metadata in `bios_notes`, explicitly confirmed not-UUID for Phase 1 (per 4559571210); Phase 2 candidate if results data shows divergence, but no threshold or trigger criterion defined.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4559373879>

---

## Cross-topic links

- **`interface_type`** — FK target for `gpu_hardware.gpu_native_interface_type_id`, `system_gpu_link.slot_native_interface_type_id`, `system_gpu_link.actual_interface_type_id`. Introduced in #17 (referenced throughout #8 and #22).
- **`gpu_hardware`** — FK target for `system_gpu_link.gpu_hardware_id`. Introduced in #18.
- **`system_gpu_link`** — FK target for `run_hardware_metrics.(system_hardware_id, slot_index)`. Introduced in #20.
- **`system_software`** — FK target for `run_hardware_metrics.system_software_id`. Introduced in #19.
- **`run_hardware_metrics`** — references `run_id` which ties into the `runs` / `run_model_metrics` tables (disk persistence pattern established in #15; per #22 body).
- **`gpu_tflops`** — FK to `compute_formats` (seeded lookup) and `tflops_sources` (introduced in #22 discussion).
- **`gpu_hardware`** — FK to `vram_types` (seeded lookup, per 4534370703), `gpu_architectures` (seeded lookup, per 4534654458).
- **`schema_versions` / `schema_tables` / `uuid_migrations`** — cross-cutting migration infrastructure; applies to any UUID-identity table in the schema, not hardware-specific.
