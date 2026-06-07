# Summary: system_gpu_link

## Final state

Four-table hardware schema, ratified at end of #8 thread and formalized in sub-issues #17–21. Full SQL as locked:

### `interface_type`
| Column | SQL type | Notes |
|---|---|---|
| `interface_type_id` | `SERIAL` PK | |
| `bandwidth_peak_gb_s` | `FLOAT NOT NULL` | primary filterable metric |
| `description` | `TEXT NOT NULL` | e.g. "PCIe 4.0 x16", "SXM5", "Thunderbolt 4" |
| `interface_family` | `TEXT NULLABLE` | "PCIe", "SXM", "CXL", "NVLink", "USB", "OCuLink" |
| `transfer_rate` | `INT NULLABLE` | PCIe GT/s per lane; null for non-PCIe |
| `lane_count` | `INT NULLABLE` | PCIe lane width; null for non-PCIe |

Seed file: `seeds/interface_types.json` (confirmed in #22, per 4533647917).

### `gpu_hardware`
| Column | SQL type | Notes |
|---|---|---|
| `gpu_hardware_id` | `SERIAL` PK | |
| `gpu_name` | `TEXT NOT NULL` | |
| `pci_device_id` | `TEXT NULLABLE` | |
| `pci_sub_device_id` | `TEXT NULLABLE` | board partner variant |
| `vram_total_mb` | `INT NULLABLE` | |
| `vram_type` | `TEXT NULLABLE` | "GDDR6X", "HBM2e", etc. |
| `memory_bus_width_bits` | `INT NULLABLE` | |
| `memory_bandwidth_peak_gb_s` | `FLOAT NULLABLE` | stored (not derived) to avoid re-implementing DDR factor at every callsite |
| `clock_memory_mhz` | `INT NULLABLE` | |
| `clock_graphics_boost_mhz` | `INT NULLABLE` | |
| `peak_tflops_fp16` | `FLOAT NULLABLE` | |
| `tdp_watts` | `INT NULLABLE` | |
| `gpu_native_interface_type_id` | `INT NULLABLE` FK → `interface_type` | GPU card's rated spec (moved here from `system_gpu_link` per 4472607438) |

### `system_hardware`
| Column | SQL type | Notes |
|---|---|---|
| `system_hardware_id` | `SERIAL` PK | |
| `system_id` | `UUID NOT NULL UNIQUE` | stable per-host UUID, generated at first run |
| `publisher_id` | `TEXT NOT NULL` | submitting user/account |
| `cpu_model` | `TEXT NULLABLE` | |
| `ram_total_gb` | `FLOAT NULLABLE` | |
| `os` | `TEXT NULLABLE` | |
| `kernel_version` | `TEXT NULLABLE` | |

### `system_gpu_link`
| Column | SQL type | Notes |
|---|---|---|
| `system_hardware_id` | `INT NOT NULL` FK → `system_hardware` | PK component |
| `slot_index` | `INT NOT NULL` | PK component |
| `gpu_hardware_id` | `INT NOT NULL` FK → `gpu_hardware` | data, not PK; indexed |
| `slot_native_interface_type_id` | `INT NULLABLE` FK → `interface_type` | motherboard slot's rated max |
| `actual_interface_type_id` | `INT NULLABLE` FK → `interface_type` | negotiated running state (belongs to neither side alone) |

PK: `(system_hardware_id, slot_index)`. Index on `gpu_hardware_id`.

### `run_hardware_metrics`
| Column | SQL type | Notes |
|---|---|---|
| `run_id` | `TEXT NOT NULL` | |
| `system_hardware_id` | `INT NOT NULL` | FK component → `system_gpu_link` |
| `slot_index` | `INT NOT NULL` | FK component → `system_gpu_link` |
| `wall_clock_seconds` | `FLOAT NULLABLE` | |
| `time_to_first_token_ms` | `FLOAT NULLABLE` | |
| `tokens_per_second` | `FLOAT NULLABLE` | |
| `peak_vram_mb` | `FLOAT NULLABLE` | |
| `gpu_sm_utilization_pct` | `FLOAT NULLABLE` | |
| `tflops_utilization_pct` | `FLOAT NULLABLE` | |
| `cpu_cycles_elapsed` | `BIGINT NULLABLE` | |
| `cpu_time_user_ms` | `FLOAT NULLABLE` | |
| `cpu_time_sys_ms` | `FLOAT NULLABLE` | |
| `gpu_wall_time_ms` | `FLOAT NULLABLE` | runner-dependent |

Compound FK `(system_hardware_id, slot_index)` → `system_gpu_link`.

---

## Notable / unusual decisions

- **Join table over direct FK** — Options A (system→GPU FK) and B (GPU→system FK) both rejected (per 4470208621). Option A breaks multi-GPU; Option B inverts semantics and defeats normalization. `system_gpu_link` as many-to-many join is load-bearing because many community submitters sharing the same GPU model is the expected steady state, not edge case.

- **PK is `(system_hardware_id, slot_index)`, not including `gpu_hardware_id`** — a slot is a fixed property of the system; the GPU in it is data (per 4472501126). The prior `(system_hardware_id, gpu_hardware_id, slot_index)` PK would have prevented two identical GPU models occupying different slots in the same system.

- **`is_slot_limited` dropped entirely** — derivable as `slot_native_interface_type_id != actual_interface_type_id`; PostgreSQL GENERATED columns can't use subqueries; trigger-based storage unjustified. Compute in queries/views (per 4472501126).

- **`gpu_native_interface_type_id` lives on `gpu_hardware`, not `system_gpu_link`** — GPU's rated spec is a fixed property of the card, not the slot relationship (per 4472607438). `slot_native_interface_type_id` stays on the link to represent the motherboard slot's spec independently.

- **`actual_interface_type_id` renamed from `slot_actual_interface_type_id`** — "actual" belongs to neither device alone; it is hardware-negotiated at link-up (per 4472607438 / 4474333279).

- **Runtime concatenation for degraded-link descriptions, not exhaustive pre-seeded rows** — number of native/actual pairing combinations is large; pre-seeding every downgrade state is impractical. Display layer constructs the string from both `interface_type.description` fields; `bandwidth_peak_gb_s` from `actual_interface_type_id` is authoritative for filtering (per 4472501126).

- **Attribution direction: actual type first, constraining device parenthetical second** — e.g. "PCIe 3.0 x8 link (limited by PCIe 4.0 x8 system slot and PCIe 3.0 x16 GPU interface)" when both axes degrade. Requires `transfer_rate` and `lane_count` on `interface_type` for per-axis comparison; `bandwidth_peak_gb_s` alone collapses two variables (per 4474333279).

- **`memory_bandwidth_peak_gb_s` stored, not derived** — storing avoids re-implementing the DDR-factor selection (GDDR vs HBM) at every display callsite. Unusual choice given derivability; rationale is callsite simplicity (per 4469686066).

- **NVLink deferred, namespace reserved** — NVLink changes GPU-to-GPU topology in ways `system_gpu_link` cannot model (unified memory pool). Requires separate `gpu_gpu_link` table. Phase 1: no NVLink columns anywhere; `interface_family = "NVLink"` reserved in `interface_type` as named constant only (per 4470666757).

- **Hardware snapshot disk identity: deterministic UUID at ingest, not stored by runner** — hardware specs embedded inline in run disk file (portable, air-gap compatible); `hardware_id` UUID derived from those embedded fields at ingest time. Runner never computes or stores the UUID (per 4533647917, Q2).

- **Hardware snapshot once per process start, not per run** — consumer hardware cannot hot-swap GPUs mid-session; eGPU hot-swap during a benchmark is not a real scenario; PCIe/SXM/NVLink cannot hot-swap (per 4533647917, Q5).

---

## Open / unresolved

1. **BIOS UUID field list not confirmed** — Bastion proposed `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile` as UUID inputs; `power_limit_w` and `pcie_gen_override` as non-UUID stored fields. @gissf1's reply (4553530231) addressed `pcie_gen_override` (covered by link table) and `power_limit_w` (not needed for phase 1 UUID) but did not explicitly confirm the three proposed UUID fields. The set of BIOS settings that contribute to `hardware_id` UUID generation remains unconfirmed.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533647917>

2. **`schema_versions` / UUID migration scripting language** — @gissf1 rejected the simple `verification_field`/`verification_source` mapping approach as insufficient for complex migrations (multi-source fields, substrings, remapping, hash function changes). Raised whether YAML/Jinja2, AWK, Lua, or another language should be used; concern about security and file-count explosion for external storage. No resolution reached.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553530231>

3. **`tflops_sources.contact_url` multi-contact structure** — @gissf1 flagged that a single URL field cannot represent multiple contact methods per source or multiple contacts per source (manager + engineer + PM). No alternative structure proposed or accepted.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553530231>

4. **`tflops_sources.url_template` complexity** — @gissf1 noted `url_template` may require GPU architecture fields, capitalization transforms, slug mappings, and substrings — related to the uuid_migrations scripting language open item above. No resolution.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553530231>

5. **`hardware-snapshot/<uuid>.json` single-file layout (Q3) not explicitly confirmed** — Bastion recommended Option A (single combined file: system fields + embedded `gpus: [...]` array); @gissf1's reply did not explicitly confirm Q3. Option B (separate `gpu_hardware/<id>.json` files) deferred to phase 2 by Bastion's recommendation but not ratified.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533647917>

---

## Cross-topic links

- **`interface_type`** — FK target for `system_gpu_link.slot_native_interface_type_id`, `system_gpu_link.actual_interface_type_id`, and `gpu_hardware.gpu_native_interface_type_id`. Implemented in #17.
- **`gpu_hardware`** — FK target for `system_gpu_link.gpu_hardware_id`. Implemented in #18.
- **`system_hardware`** — FK target for `system_gpu_link.system_hardware_id`. Implemented in #19.
- **`system_software`** — referenced by `run_hardware_metrics.system_software_id` (wired in #21); `system_software` schema lives in #19 topic.
- **`run_hardware_metrics`** — holds compound FK into `system_gpu_link`; full hardware context for any run requires three-way join: `run_hardware_metrics → system_gpu_link → system_hardware + gpu_hardware + interface_type`. Implemented in #21.
- **`run_model_metrics`** — separate table (`run_id` PK, model/task/prompt/score fields); shares `run_id` with `run_hardware_metrics` but is not an FK dependency of `system_gpu_link`.
- **Disk persistence pattern** — follows pattern established in #15 (models/tasks/prompts/runs disk layout). Hardware disk format scoped in #22.
