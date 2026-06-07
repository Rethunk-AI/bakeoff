# Summary: gpu_architecture

## Final state

All gpu_architecture entries are from #22. Schema is partially settled with significant open questions remaining on the `uuid_migrations` table design and several GPU field details.

**`gpu_hardware` table** (fields settled by end of thread):

| Column | SQL type | Notes |
|---|---|---|
| `pci_vendor_id` | TEXT | UUID input; identity anchor |
| `pci_device_id` | TEXT | UUID input; identity anchor |
| `pci_subsystem_vendor_id` | TEXT | UUID input; board-level OEM differentiation (per 4534654458) |
| `pci_subsystem_device_id` | TEXT | UUID input; board-level OEM differentiation (per 4534654458) |
| `vram_mb` | INTEGER | UUID input |
| `vram_type_id` | INTEGER FK → `vram_types` | UUID input; seeded lookup (per 4534654458) |
| `gpu_architecture_id` | INTEGER FK → `gpu_architectures` | NOT in UUID; categorization FK only (per 4534370703, 4534654458) |
| `compute_units` | INTEGER | NOT in UUID; implied by PCI IDs + arch; rename to `parallel_execution_units` under consideration |
| `memory_bandwidth_peak_gb_s` | NUMERIC | NOT in UUID; performance metric; unit standardized to GB/s (per 4534654458) |
| `tdp_w` | INTEGER | NOT in UUID; spec metadata; reliability flagged as open question |

**`gpu_tflops` table** (replacing per-format columns):

| Column | SQL type | Notes |
|---|---|---|
| `gpu_hardware_id` | UUID FK → `gpu_hardware` | PK component |
| `compute_format_id` | INTEGER FK → `compute_formats` | PK component; seeded: fp16, fp32, bf16, int8, fp8, etc. |
| `tflops_value` | NUMERIC | |
| `tflops_source` | TEXT | `'manufacturer'` \| `'measured'` (per 4534654458) |

**`uuid_migrations` table** (generalized, partially settled):

| Column | SQL type | Notes |
|---|---|---|
| `entity_type` | TEXT NOT NULL | `'system_hardware'` \| `'gpu_hardware'` \| etc. |
| `old_id` | UUID NOT NULL | |
| `schema_version` | INTEGER NOT NULL FK → `hardware_schema_versions` | replaces free-text reason (per 4534654458) |
| `verification_field` | TEXT | |
| `verification_source` | TEXT | `'disk_file'` \| `'hardware_probe'` \| `'manual'` |
| `verified` | BOOLEAN NOT NULL DEFAULT FALSE | |
| `migrated_at` | TIMESTAMPTZ NOT NULL DEFAULT NOW() | |
| `new_id` | UUID nullable? | **DISPUTED** — see Open / unresolved |

**`hardware_schema_versions` table** (settled):

| Column | SQL type | Notes |
|---|---|---|
| `schema_version` | INTEGER PRIMARY KEY | |
| `description` | TEXT NOT NULL | |
| `uuid_fields_added` | TEXT[] | |
| `uuid_fields_removed` | TEXT[] | |
| `released_at` | TIMESTAMPTZ NOT NULL DEFAULT NOW() | |

**`system_gpu_link` join table** (referenced, not fully specified here): holds `slot_index` between `system_hardware` and `gpu_hardware`; slot excluded from GPU UUID (per 4534654458).

**Seeded lookup tables confirmed:** `vram_types`, `gpu_architectures`, `compute_formats`.

**`bios_notes` field** (on `system_hardware`, not gpu — included for UUID context): changed from free-form string to structured JSON key-value; a whitelist of keys is included in UUID hash in stable sorted order. Initial whitelist: `SMT_enabled` (YES), `resizable_bar` (YES), `above_4g_decoding` (YES), `fTPM_enabled` (NO), `secure_boot` (NO) (per 4534654458).

## Notable / unusual decisions

- **`cpu_cores` removed from system hardware UUID** — implied by `cpu_model`; tracking it would create false distinctness between identical CPUs (per 4533894066, 4533933819). Downstream: reduces UUID collision from model-implied fields.
- **`slot_index` excluded from GPU UUID** — physical slot reassignment does not change GPU identity; tracked only in the join table. Avoids UUID churn on hardware reorganization (per 4534654458).
- **`gpu_architecture` demoted to categorization FK, not UUID input** — implied by PCI IDs; including it would be redundant and add a derived field to the hash (per 4534370703, 4534654458). Downstream: results-page grouping only.
- **Per-format tflops columns replaced by `gpu_tflops` child table** — extensible without schema migration per new compute format; avoids NULL-heavy wide row (per 4534654458).
- **`tflops_source` flag distinguishes manufacturer vs measured** — enables the system to accumulate measured values over time and eventually surface empirical throughput ratings distinct from spec sheets (per 4534370703, 4534654458). Downstream: statistical aggregation on measured-only rows.
- **`uuid_migrations` generalized** — renamed from `hardware_id_migrations`; `entity_type` column makes it reusable for any UUID-keyed entity, not just system hardware (per 4534370703, 4534654458).
- **`migration_reason` normalized to FK** — free-text reason removed; all records affected by one schema change share a single `hardware_schema_versions` row, eliminating repetition (per 4534370703, 4534654458).
- **`bios_notes` whitelist-filtered into UUID** — structured JSON with key whitelist; unknown keys stored but not hashed. Schema version increment required to add new keys to hash (per 4534370703, 4534654458).

## Open / unresolved

- **`uuid_migrations.new_id` — store or derive?** Bastion proposed omitting `new_id` (derive at ingest via re-probe + current field set). @gissf1 raised that a schema change adding an ENUM field produces multiple possible new UUIDs, making storage of a fixed `new_id` infeasible. Bastion responded with a derivation protocol but explicitly asked: "do you want a stored (but nullable) `new_id` column for caching once first resolution occurs?" No answer in thread.
  - Trade-offs: stored nullable `new_id` enables O(1) repeat lookup after first resolution; pure derivation avoids stale cache but requires re-read of disk file every ingest. Unresolved.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534654458>

- **`uuid_migrations` PK** — @gissf1 noted that `(old_id, new_id)` as PK is wrong when new_id is not fixed; `old_id` alone may also be insufficient if the same old UUID maps to multiple possible new UUIDs depending on resolution path. No alternative PK proposed or confirmed.
  - Trade-offs: composite PK needs stable `new_id`; single-column PK loses ability to cache multiple resolution paths; surrogate PK adds indirection. Unresolved.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534370703>

- **`tdp_w` reliability** — @gissf1 asked how reliable TDP is and how closely it correlates to actual wattage. Bastion did not answer. Whether to include it in the schema at all, or annotate it as a spec-only estimate, is unresolved.
  - Trade-offs: TDP is a manufacturer spec and frequently diverges from actual power draw; including it without a source flag may mislead result consumers. Unresolved.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534370703>

- **`compute_units` rename** — Bastion offered renaming to `parallel_execution_units` but did not confirm. @gissf1's original comment indicated the `compute_units` label was unclear ("I don't understand what this is from your explanation"). No explicit confirmation of final column name.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534370703>

- **`bios_notes` UUID whitelist** — Bastion proposed the initial whitelist and asked "Does this whitelist look right?" No confirmation from @gissf1 in thread.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534654458>

- **GPU UUID field list** — Bastion presented the updated field list and asked for `slot_index` exclusion confirmation: "Confirm?" No explicit confirmation in thread.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534654458>

## Cross-topic links

- **system_hardware** — `gpu_hardware` is externalized from `system_hardware` and referenced via `system_gpu_link` join table; `bios_notes` and system UUID fields live on `system_hardware`.
- **uuid_migrations / hardware_schema_versions** — shared migration infrastructure applies to both `system_hardware` and `gpu_hardware` entity types; schema version table is shared.
- **vram_types**, **gpu_architectures**, **compute_formats** — seeded lookup tables; dependency on seed data topic if one exists.
