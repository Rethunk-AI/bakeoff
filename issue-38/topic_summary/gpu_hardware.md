# Summary: gpu_hardware

## Final state

Schema evolved across #8, #13, #18–#22 into a normalized multi-table model. Final settled tables as of end of thread:

### `interface_type`
| Column | Type | Notes |
|---|---|---|
| `interface_type_id` | `SERIAL PRIMARY KEY` | |
| `bandwidth_peak_gb_s` | `FLOAT NOT NULL` | |
| `description` | `TEXT NOT NULL` | e.g. "PCIe 4.0 x16", "SXM5" |
| `interface_family` | `TEXT NULLABLE` | "PCIe", "SXM", "CXL", "NVLink", "USB", "OCuLink" |
| `transfer_rate` | `INT NULLABLE` | PCIe GT/s per lane; null for non-PCIe |
| `lane_count` | `INT NULLABLE` | PCIe lane width; null for non-PCIe |

Seeded via `seeds/interface_types.json`; admin-review-gated for new rows (per 4533647917).

### `gpu_hardware`
| Column | Type | Notes |
|---|---|---|
| `gpu_hardware_id` | `SERIAL PRIMARY KEY` | |
| `gpu_name` | `TEXT NOT NULL` | |
| `pci_device_id` | `TEXT NULLABLE` | |
| `pci_sub_device_id` | `TEXT NULLABLE` | board-partner variant |
| `pci_subsystem_vendor_id` | `TEXT NULLABLE` | added per 4534654458 |
| `pci_subsystem_device_id` | `TEXT NULLABLE` | added per 4534654458 |
| `vram_total_mb` | `INT NULLABLE` | |
| `vram_type` | `INT FK → vram_types` | seeded lookup; per 4534370703 / 4534654458 |
| `memory_bus_width_bits` | `INT NULLABLE` | |
| `memory_bandwidth_peak_gb_s` | `FLOAT NULLABLE` | stored, not derived (per 4469686066) |
| `clock_memory_mhz` | `INT NULLABLE` | |
| `clock_graphics_boost_mhz` | `INT NULLABLE` | |
| `peak_tflops_fp16` | `FLOAT NULLABLE` | existing field; tflops_fp16 removed as duplicate (per 4534370703) |
| `tdp_watts` | `INT NULLABLE` | |
| `gpu_native_interface_type_id` | `INT NULLABLE FK → interface_type` | card's rated spec; moved here from system_gpu_link (per 4474333279) |
| `gpu_architecture_id` | `INT FK → gpu_architectures` | seeded lookup; categorization only, not in UUID (per 4534654458) |

Per-format tflops moved to separate table `gpu_tflops { gpu_hardware_id, compute_format_id FK → compute_formats, tflops_value, tflops_source_id FK → tflops_sources }` (per 4534654458). `compute_units` dropped as implied by PCI IDs + architecture (per 4550408728 / 4552726518).

Dedup key: `(pci_device_id, pci_sub_device_id)` when available; fall back to `gpu_name` normalization. Model-level record — multiple users with same GPU share one row (per 4470208621).

### `system_hardware`
| Column | Type | Notes |
|---|---|---|
| `system_hardware_id` | `SERIAL PRIMARY KEY` | |
| `system_id` | `UUID NOT NULL UNIQUE` | stable per-host, generated at first run |
| `publisher_id` | `TEXT NOT NULL` | submitting user/account |
| `cpu_model` | `TEXT NULLABLE` | |
| `cpu_threads` | `INT NULLABLE` | BIOS-configurable; `cpu_cores` dropped as implied by cpu_model (per 4533894066 / 4533933819) |
| `cpu_base_clock_mhz` | `INT NULLABLE` | detects underclocking (per 4533933819) |
| `cpu_peak_clock_mhz` | `INT NULLABLE` | detects OC (per 4533933819) |
| `ram_gb` | `FLOAT NULLABLE` | |
| `motherboard` | `TEXT NULLABLE` | |
| `memory_speed_mhz` | `INT NULLABLE` | active clock |
| `memory_channels` | `INT NULLABLE` | BIOS-configurable |
| `memory_interleave_profile` | `TEXT NULLABLE` | XMP/EXPO/DOCP/manual |
| `bios_notes` | `JSONB NULLABLE` | structured key-value; whitelist subset hashed into UUID (per 4534654458) |
| `os` | `TEXT NULLABLE` | |
| `kernel_version` | `TEXT NULLABLE` | |

`motherboard`/CPU split into separate tables deferred to Phase 2; single `system_hardware` record confirmed for Phase 1 (per 4550408728 / 4552726518). Multi-processor systems also deferred (per 4552726518).

### `system_gpu_link`
| Column | Type | Notes |
|---|---|---|
| `system_hardware_id` | `INT NOT NULL FK → system_hardware` | PK component |
| `slot_index` | `INT NOT NULL` | PK component |
| `gpu_hardware_id` | `INT NOT NULL FK → gpu_hardware` | indexed |
| `slot_native_interface_type_id` | `INT NULLABLE FK → interface_type` | motherboard slot rated max |
| `actual_interface_type_id` | `INT NULLABLE FK → interface_type` | negotiated running state |

`is_pcie_slot_limited` not stored — derived as `slot_native != actual` at query time (per #20 body). NVLink deferred to Phase 2 (per #20 body).

### `run_hardware_metrics`
| Column | Type | Notes |
|---|---|---|
| `run_id` | `TEXT NOT NULL` | references run |
| `system_hardware_id` | `INT NULLABLE` | compound FK to system_gpu_link |
| `slot_index` | `INT NULLABLE` | compound FK to system_gpu_link |
| `system_software_id` | `INT NULLABLE FK → system_software` | |
| `wall_clock_seconds` | `FLOAT NULLABLE` | |
| `time_to_first_token_ms` | `FLOAT NULLABLE` | |
| `tokens_per_second` | `FLOAT NULLABLE` | |
| `peak_vram_mb` | `FLOAT NULLABLE` | |
| `gpu_sm_utilization_pct` | `FLOAT NULLABLE` | |
| `tflops_utilization_pct` | `FLOAT NULLABLE` | |
| `cpu_cycles_elapsed` | `BIGINT NULLABLE` | optional |
| `cpu_time_user_ms` | `FLOAT NULLABLE` | optional |
| `cpu_time_sys_ms` | `FLOAT NULLABLE` | optional |
| `gpu_wall_time_ms` | `FLOAT NULLABLE` | optional, runner-dependent |

### Supporting lookup/infrastructure tables (settled by end of thread)
- `quantization_methods { quantization_id PK, name TEXT UNIQUE, vram_multiplier DECIMAL NOT NULL, description TEXT }` — full seed SQL posted (per 4519570310); `models.quantization TEXT` → FK (per 4519072195)
- `runners { runner_id TEXT PK, hostname, process_id, effective_user, last_heartbeat TIMESTAMPTZ, status CHECK('ACTIVE','IDLE','DEAD'), started_at TIMESTAMPTZ }` — hardware fields dropped, runner declares capability at claim time (per 4513822952)
- `run_queue` additions: `source_file TEXT`, `retry_after TIMESTAMPTZ`, `priority INT` with increment `5 * attempt_count` per retry, `max_attempts = 5` (per 4513822952)
- `vram_types` — seeded lookup for GDDR6, GDDR6X, HBM2e, HBM3, LPDDR5X, etc. (per 4534370703 / 4534654458)
- `gpu_architectures` — seeded FK lookup for GPU microarchitecture categorization (per 4534654458)
- `gpu_tflops { gpu_hardware_id, compute_format_id FK → compute_formats, tflops_value FLOAT, tflops_source_id FK → tflops_sources }` (per 4534654458)
- `tflops_sources { source_id SERIAL PK, name, contact_url, url_template }` — ID 1 = unknown, ID 2 = Rethunk measured (per 4552726518)
- `schema_versions { schema_version_id SERIAL PK, description TEXT, changes JSONB }` — generalized (renamed from `hardware_schema_versions`); JSONB changes vocabulary includes `uuid_fields_added/removed`, `tables_added/modified`, `encoding_changes` (per 4550408728 / 4552726518)
- `schema_tables { table_id SERIAL PK, table_name TEXT UNIQUE, added_schema_version INT FK, last_modified_version INT FK, uses_uuid_identity BOOLEAN }` (per 4552726518)
- `uuid_migrations { entity_type INT FK → schema_tables, old_id UUID, schema_version INT FK, verification_field TEXT, verification_source TEXT, verified BOOLEAN, migrated_at TIMESTAMPTZ }` — `new_id` not stored, computed at migration time; `entity_type` FK to `schema_tables` (per 4550408728 / 4552726518)

### Disk file layout (settled by #22 thread)
- Hardware specs embedded inline in run disk file (`runs/<run_id>.json`) — portable, air-gap compatible (per 4533634217 / 4533647917)
- `hardware_id` UUID computed deterministically at ingest from system + GPU fields, never stored in disk file (per 4533647917)
- `hardware-snapshot/<hardware_id>.json` written as dedup cache at ingest time — combined system + `gpus: [...]` array (Option A; per 4533647917)
- GPU `gpus` array: inline spec for unknown GPUs (Phase 1); reference by `gpu_hardware_id` for known GPUs deferred to Phase 2 (per 4533894066 / 4533933819)
- `hardware-migrations/<old_uuid>.json` disk mirror of `uuid_migrations` entries
- Lookup tables seeded via `seeds/*.json` files (interface_types, quantization_methods, model_architectures, model_file_formats, vram_types, gpu_architectures, compute_formats, tflops_sources)
- `interface_type` seed file: `seeds/interface_types.json` confirmed (per 4533634217)

### `hardware_id` UUID input fields (system)
`cpu_model`, `cpu_threads`, `cpu_base_clock_mhz`, `cpu_peak_clock_mhz`, `ram_gb`, `motherboard`, `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile`, `bios_notes` whitelist keys: `bar_size_mb`, `pcie_gen_override`, `power_limit_w` (per 4550408728 / 4552726518). `cpu_cores` excluded (implied by model). `SMT_enabled`, `resizable_bar` bool, `above_4g_decoding` excluded.

### `gpu_hardware_id` UUID input fields
`pci_vendor_id`, `pci_device_id`, `pci_subsystem_vendor_id`, `pci_subsystem_device_id`, `vram_mb`, `vram_type` (FK). `slot_index` excluded (per 4534654458 / 4550408728). Performance metrics (tflops, bandwidth, TDP, compute_units) excluded from UUID.

### Runner / queue design
- Standalone mode default; queue worker mode opt-in via `--queue` flag (per 4516038912 / 4519072195)
- VRAM claim filter: `CEIL(active_parameter_count_b × vram_multiplier × 1.15) <= $runner_vram_gb` via JOIN to `quantization_methods` (per 4519072195)
- `retry_after TIMESTAMPTZ` replaces `updated_at` arithmetic in claim query (per 4513822952)
- Dependency ordering: scores are absolute; priority-only ordering, no enforced baseline dependency (per 4513796211 / 4513822952)
- File mtime delay before read: 30 seconds default, configurable (per 4519072195)
- DR: scan `queue/pending/`, check `run_model_metrics` for valid results, re-enqueue absent; `queue/completed/` on success (per 4513822952)
- Reaper: embedded probabilistic (10% after each job outcome); extract to daemon only if contention warrants (per 4519072195)
- Runner idle: sleep to `MIN(retry_after)` of eligible FAILED items rather than busy-poll (per 4519072195)
- `runs.runner_id TEXT FK → runners` added (per 4513822952)

### `cost_usd`
Not stored. Derived at display time from `energy_wh_total × kwh_rate` (local) or provider pricing × token counts (API) (per 4462460688, confirmed in implementation 4466000757).

---

## Notable / unusual decisions

- **`memory_bandwidth_peak_gb_s` stored, not derived** — formula requires `vram_type` to select DDR factor; storing avoids re-implementing at every display callsite (per 4469686066). Consistent `_gb_s` suffix standardized throughout (per 4534370703 / 4534654458).

- **Three-entity split (`gpu_hardware` / `system_hardware` / `system_gpu_link`)** rather than embedding GPU specs in host record — motivated by community benchmarking at scale: many users with same GPU model share one `gpu_hardware` row; interface data (PCIe gen/width) belongs on the relationship, not either entity (per 4470208621). Enables bottleneck detection: `slot_native != actual` flags degraded slot without storing a separate boolean.

- **`gpu_native_interface_type_id` lives on `gpu_hardware`, not `system_gpu_link`** — it is a fixed property of the card, not the relationship (per 4474333279 per @gissf1 direction).

- **Limitation attribution display format** — actual link type shown first, constraining device(s) named in parenthetical; both devices named when each constrains on a different axis (e.g. gen vs. lane width) (per 4474333279).

- **`cpu_cores` excluded from `hardware_id` UUID** — implied by `cpu_model`; `cpu_threads` retained because BIOS HT/SMT settings can change it independently (per 4533894066 / 4533933819).

- **`slot_index` excluded from `gpu_hardware_id` UUID** — moving a GPU to a different slot doesn't change the hardware identity; slot tracking belongs in `system_gpu_link` (per 4534654458, confirmed 4550408728).

- **`compute_units` dropped entirely** — implied by PCI IDs + architecture; no independent signal for the test surface (per 4550408728 / 4552726518).

- **`uuid_migrations.new_id` not stored** — computing it at migration time avoids a stale-cache problem when the UUID field set evolves again; new UUID is always re-derived from disk file + field set (per 4550408728 / 4552726518).

- **`tflops_source` as FK to `tflops_sources` lookup with URL template substitution** — a single template row can reference all GPU variants via `{pci_device_id}` token; no per-GPU source entries needed (per 4552726518).

- **BIOS settings EAV approach deferred to Phase 2** — structured `bios_notes` JSONB + UUID whitelist achieves Phase 1 deduplication goal without EAV join complexity; EAV gates on Phase 2 when BIOS variant proliferation is visible in data (per 4552726518).

- **Quantization_methods lookup table adopted over inline CASE expression** — eliminates fragile duplicated CASE in claim query; single source for bakeoff-results filter bar dropdown; seed multipliers derived from GGUF/llama.cpp block layout math (per 4516038912 / 4519072195; seed SQL at 4519570310). Note: `q4_k_m` multiplier corrected from 0.45 (design discussion) to 0.563 (calculated from block layout) in seed data.

---

## Open / unresolved

1. **`bios_notes` UUID whitelist completeness** — thread converged `bar_size_mb`, `pcie_gen_override`, `power_limit_w` as UUID inputs. Open question remains: are there motherboard-specific bus width overrides (BIOS forcing x8 for an x16 slot) to add, or are those fully captured by the join table's measured link width? @gissf1's 4550408728 comment narrowed the list but did not explicitly close this question; 4552726518 left it as an open item.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4552726518>

2. **`schema_versions.changes` JSONB vocabulary completeness** — proposed keys (`uuid_fields_added/removed`, `tables_added/modified`, `encoding_changes`) left open with "any change type not covered?" at end of 4552726518. @gissf1 did not respond to confirm.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4552726518>

3. **`schema_tables.uses_uuid_identity` flag adequacy** — Bastion proposed this as the discriminator for UUID-migration-eligible tables; @gissf1's response in 4550408728 did not explicitly confirm. Left as open item in 4552726518.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4552726518>

4. **`uuid_migrations` — verification_field / verification_source necessity** — @gissf1 asked "why do we have verification_field and source?" and "what is the purpose of the verified field?" in 4550408728. Bastion explained in 4552726518 but @gissf1 did not confirm acceptance; thread ends there.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728>

5. **GPU `hardware_id` UUID field list completeness** — 4534654458 closed first pass; 4550408728 said "we probably need more fields here too" for GPU fields but only resolved specific items. 4552726518 asked for confirmation on extended bios_notes list but thread ends without final sign-off from @gissf1.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728>

6. **`quantization_methods` seed multipliers** — seed SQL posted at 4519570310 with explicit request: "@gissf1 — please flag any multipliers that look wrong." No confirmation received in thread; multipliers remain unratified.
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4519570310>

7. **Model capability ingestion path** (`min_vram_gb` calculated, but `active_parameter_count_b` / `quantization` source) — left open in 4513822952 Q3: "where does min_vram_gb / param_count_b come from? Model card at submission time, CI artifact, or runner autodiscovery?" Deferred to separate thread in 4516038912 comment; no resolution in scope.
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4513822952>

8. **Result signing/verification scheme** — explicitly deferred: "separate thread when ready" (per 4519072195 / 4525059885). Not discussed further.
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4519072195>

---

## Cross-topic links

- **`interface_type`** (topic: interface_type / #17) — FK from `gpu_hardware.gpu_native_interface_type_id`, `system_gpu_link.slot_native_interface_type_id`, `system_gpu_link.actual_interface_type_id`
- **`system_hardware`** (topic: system_hardware / #19) — PK referenced by `system_gpu_link` and `run_hardware_metrics`
- **`system_software`** (topic: system_software / #19) — FK from `run_hardware_metrics.system_software_id`
- **`runs`** (topic: runs / #15 / #13) — `run_hardware_metrics.run_id` references runs; `runs.runner_id` FK to `runners`
- **`models`** (topic: models / #13 / #15) — `models.quantization_id` FK to `quantization_methods`; `models.architecture_id` FK to `model_architectures`; `models.file_format_id` FK to `model_file_formats`
- **`run_queue`** (topic: run_queue / #13) — claim query JOINs `quantization_methods` via `models.quantization_id` for VRAM filter
- **`schema_versions` / `schema_tables`** — cross-cutting infrastructure; applies to all topics with UUID identity
