# Summary: tflops_source

## Final state

### Decision: `tflops_source` is a separate lookup table (`tflops_sources`), not a column

`tflops_fp16` (and other per-format TFLOPS values) are NOT stored as flat columns on `gpu_hardware`. Instead, a related table `gpu_tflops` links each GPU to per-format throughput values, with a FK to `tflops_sources` for provenance. (per 4534654458, confirmed 4553530231)

### `tflops_sources` table (settled shape, pending scripting/contact open items)

| Column | Type | Notes |
|--------|------|-------|
| `source_id` | SERIAL PRIMARY KEY | |
| `name` | TEXT NOT NULL | e.g. "unknown/unverified", "Rethunk measured", manufacturer name |
| `contacts` | JSONB | array of `{type, value}` objects; multiple contact methods/contacts per source; full contacts table deferred P2 |
| `url_template` | TEXT | static URL or simple token substitution (`{pci_device_id}`, `{pci_vendor_id}`, `{gpu_model}`); complex URL construction uses `url_script` |
| `url_script` | TEXT | Lua (or chosen scripting language); takes precedence when non-null; handles capitalization normalization, slug mapping, cross-field construction |

Seed rows:
- ID 1: `'unknown/unverified'` — default when provenance is not established
- ID 2: `'Rethunk measured'` — added when Rethunk ships its own benchmark suite

### `gpu_tflops` table

| Column | Type | Notes |
|--------|------|-------|
| `gpu_hardware_id` | FK to gpu_hardware | |
| `compute_format_id` | FK to compute_formats | fp16, fp32, bf16, int8, fp8, etc.; seeded lookup; extensible without schema migration |
| `tflops_value` | FLOAT (or DECIMAL) | |
| `tflops_source_id` | FK to tflops_sources | provenance per value |

This replaces the flat `peak_tflops_fp16`, `tflops_fp32`, `tflops_bf16` columns that appeared in earlier schema stubs.

### `_TFLOPS_TABLE` in harness (existing implementation)

- Already implemented in the runner harness before #38 was filed.
- Used during `gpu_hardware` auto-detection at startup.
- `peak_tflops_fp16` is looked up from `_TFLOPS_TABLE` keyed on `gpu_name`; result is `None` if the GPU is not in the table.
- This lookup populates `gpu_hardware.peak_tflops_fp16` during auto-detection (per #18 spec / issue body).
- `peak_tflops_fp16` still exists as a column on `gpu_hardware` in the #18 spec — whether it is superseded by the `gpu_tflops` table or retained as a convenience denormalization is an open question (see below).

### `gpu_hardware` table — TFLOPS-relevant columns (as of #38 audit)

Current `schema.sql` contains `tflops_fp16` (renamed to `peak_tflops_fp16` per #38 audit), `tflops_fp32`, `tflops_bf16` as flat columns. Per the discussion, these should be migrated to `gpu_tflops` + `compute_formats`. Status as of #38 filing: `tflops_source` table absent from schema; flat columns present but named incorrectly. (per 4629938143)

### "None if not in table" rule

Explicit policy for `_TFLOPS_TABLE` lookup: if the GPU's name is not found in the lookup table, `peak_tflops_fp16` is `None` (null). No fallback estimate, no guessing. This prevents silently using a wrong value during VRAM/throughput calculations. (per issue #18 body)

---

## Notable / unusual decisions

- **Manufacturer values flagged, not rejected** — gissf1's direction is to accept manufacturer-supplied TFLOPS as a seeded starting point but mark them with `tflops_source = 'manufacturer'` rather than treat them as ground truth. The intent is to eventually replace manufacturer values with Rethunk's own measured values once sufficient validated results accumulate on a given GPU. The `tflops_sources` table's `'Rethunk measured'` seed row is pre-reserved for this. (per 4534370703)

- **Per-format TFLOPS in a related table, not flat columns** — flat `tflops_fp16` / `tflops_fp32` / `tflops_bf16` columns were in the initial schema; gissf1 rejected them as a pattern that requires a schema migration every time a new compute format appears (fp8, int4, etc.). A `gpu_tflops` + `compute_formats` relational design is extensible via data insert. (per 4534370703, 4534654458)

- **`url_script` alongside `url_template`** — simple `{token}` substitution in `url_template` is insufficient for real manufacturer spec pages (slug normalization, architecture-based path segments, subsystem-ID mapping). Adding a parallel `url_script TEXT` (Lua or chosen scripting language) handles the complex cases; `url_template` is retained for simple static cases. Evaluation order: `url_script` takes precedence when non-null. (per 4553785030, 4554217644)

- **`contacts` as JSONB array** — a flat `contact_url TEXT` field cannot represent multiple contact methods (email, phone, web form) or multiple contacts per source (manager, engineer, PM). JSONB array of `{type, value}` objects handles both without a join table for Phase 1; full contacts relational table is deferred P2. (per 4553785030, confirmed 4554217644)

- **`tflops_utilization_pct` as a derived metric** — separate from stored TFLOPS, the harness computes `theoretical_tflops_utilization` = (tokens_completion / gpu_wall_time_s) / model_flops_per_token / gpu_peak_tflops. This normalizes "what fraction of the GPU's theoretical peak was actually used" across hardware generations. It is the most hardware-invariant cross-model comparison axis proposed in #8. (per 4447581353)

- **GPU performance metrics excluded from UUID** — `peak_tflops_fp16`, `memory_bandwidth_peak_gb_s`, `tdp_w`, and all `gpu_tflops` values are explicitly excluded from the `gpu_hardware` UUID hash. These are measurements/specs, not identity fields. (per 4534370703, confirmed 4534654458)

---

## Open / unresolved

- **`peak_tflops_fp16` on `gpu_hardware` — retain as denorm or remove?** — the #18 spec carries it as a direct column; `_TFLOPS_TABLE` populates it at auto-detection. If `gpu_tflops` is the canonical store, `peak_tflops_fp16` on `gpu_hardware` is either a convenience denormalization or dead weight. Not explicitly settled. Bastion's #38 audit comment (4629938143) notes the column naming drift (`tflops_fp16` → `peak_tflops_fp16`) but does not address whether the column should be removed after `gpu_tflops` migration.

- **`tflops_fp32` and `tflops_bf16` on `gpu_hardware` — retain or drop?** — these are in the current schema stub but not in the #18 spec as authored. gissf1 asked (per 4629938143 audit question): "retain or drop?" Not answered in the thread.

- **`tflops_sources.url_script` scripting language** — gissf1 approved having a scripting/templating field but directed that whatever language is chosen for `schema_versions.migration_script` should be used consistently. The language itself (Lua vs. Go templates vs. Gonja/Pongo2 vs. CEL) is an open decision in the schema_versions thread, not yet resolved. `url_script` is blocked on that decision. (per 4554217644)

- **`tflops_sources` contact fields — multiple contacts** — JSONB contacts array approved for Phase 1. Whether individual contacts in the JSONB can reference external contact records (for cross-source reuse) is deferred P2. (per 4554217644)

- **`_TFLOPS_TABLE` completeness and validation** — Bastion proposed cross-checking multipliers against public model file sizes (within 1% tolerance). No confirmation appears in the thread; gissf1 flagged this as outside their expertise but expressed trust in the approach. The table's actual contents and accuracy are unverified in the discussion record.

- **`tflops_source` table absent from `schema.sql` as of #38** — gissf1 opened #38 specifically noting the table is missing from the current schema. Bastion's #38 response (4629938143) confirmed it was not found in the schema or in the reviewed issues, and asked for the issue/comment reference. This is an implementation gap, not a design gap — the design is settled, the DDL has not been written. (per 4629938143)

---

## Cross-topic links

- **`gpu_hardware`** — `gpu_tflops.gpu_hardware_id` FK to `gpu_hardware`; `peak_tflops_fp16` column on `gpu_hardware` populated from `_TFLOPS_TABLE` during auto-detection (per #18). The `pci_vendor_id`, `pci_device_id`, `pci_subsystem_vendor_id`, `pci_subsystem_device_id` fields on `gpu_hardware` are the canonical identity/dedup key for GPU model records, which also anchor the `tflops_sources.url_template`/`url_script` substitution tokens.

- **`compute_formats`** — seeded lookup table for TFLOPS precision formats (fp16, fp32, bf16, int8, fp8); FK target for `gpu_tflops.compute_format_id`. Shared with `run_hardware_metrics` if that table also carries per-format utilization fields.

- **`run_hardware_metrics`** (#8, #21) — `tflops_utilization_pct` and related per-run GPU metrics are stored here, not in `gpu_hardware` or `gpu_tflops`. `gpu_tflops` stores static spec values; `run_hardware_metrics` stores measured per-run actuals. #38 notes this table is not fully correct and #21 may not cover all end-of-#8 discussion items — flagged as requiring review.

- **`quantization_methods`** (scheduling_queue topic) — VRAM calculation in the `run_queue` claim query uses `quantization_methods.vram_multiplier`; TFLOPS utilization in `run_hardware_metrics` uses `gpu_tflops` values from this topic. Both feed from `models.quantization_id` FK.

- **`schema_versions` / `uuid_migrations`** — `tflops_sources.url_script` language is blocked on the scripting language decision made in that discussion thread (Lua vs. Go templates vs. CEL).
