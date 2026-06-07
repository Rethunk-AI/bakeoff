# Summary: interface_type

## Final state

`interface_type` is a normalized lookup table for GPU bus/slot types, proposed in (per 4470606665) and schema-finalized in (per 4474333279). Issue #17 carried the ratified DDL to implementation; Q1 of #22 confirmed seed-file delivery on `main` (commit `3b0bce4`) (per 4570546987).

**Resolved final schema (`interface_type`):**

| Column | SQL type | Constraints |
|---|---|---|
| `interface_type_id` | `SERIAL` | PRIMARY KEY |
| `bandwidth_peak_gb_s` | `FLOAT` | NOT NULL |
| `description` | `TEXT` | NOT NULL — human-readable, e.g. "PCIe 4.0 x16", "SXM5" |
| `interface_family` | `TEXT` | NULLABLE — grouping: "PCIe", "SXM", "CXL", "NVLink", "USB", "OCuLink" |
| `transfer_rate` | `INT` | NULLABLE — PCIe only: GT/s per lane; null for non-PCIe |
| `lane_count` | `INT` | NULLABLE — PCIe only: lane width; null for non-PCIe |

No FK dependencies. Referenced by `gpu_hardware` (`gpu_native_interface_type_id`) and `system_gpu_link` (`slot_native_interface_type_id`, `actual_interface_type_id`).

Seed file: `schema/seeds/interface_types.json` — rows carry `name` only (no `description` column in the shipped seed; thread note in 4570546987 states rows carry `name` only, deviating from the DDL above; see Open items).

Pre-seeded coverage: PCIe Gen 1–5 × x1/x4/x8/x16; SXM2/SXM4/SXM5; NVLink 2.0/3.0/4.0 (reserved, no FK references); Thunderbolt 3/4; OCuLink 2.0. No pre-seeded degraded-state rows — display layer constructs degradation strings at runtime.

## Notable / unusual decisions

- **Bandwidth-first generic schema** — field names do not assume PCIe. `transfer_rate`/`lane_count` are nullable PCIe-specific fields; `bandwidth_peak_gb_s` is the universal comparable. Rationale: future-proofs for SXM, CXL, OCuLink, USB4 without schema changes (per 4470666757).

- **Degraded states NOT pre-seeded** — all possible native/actual pairings were rejected as impractical to enumerate. Runtime concatenation from two `description` fields instead: `"[actual.description] (limited by [slot_native.description] system slot and [gpu_native.description] GPU interface)"` (per 4472501126, 4472607438). Bandwidth authoritative number comes from `actual_interface_type_id.bandwidth_peak_gb_s`.

- **`is_slot_limited` dropped** — derivable as `slot_native_interface_type_id != actual_interface_type_id`; PostgreSQL `GENERATED ALWAYS AS` cannot use subquery expressions, and a trigger was judged not worth the complexity (per 4472501126).

- **NVLink reserved but inert in Phase 1** — `interface_family = "NVLink"` rows seeded; no table carries an FK to them yet. Rationale: NVLink changes GPU-to-GPU topology fundamentally (unified memory pool), warranting a separate `gpu_gpu_link` table when needed. Deferred to Phase 2 (per 4470666757).

- **`memory_bandwidth_peak_gb_s` stored, not derived** — even though it is computable from `clock_memory_mhz`, `memory_bus_width_bits`, and `vram_type`, it is stored on `gpu_hardware` to avoid re-implementing the DDR-factor selection at every display callsite (per 4469686066).

- **Attribution direction in degradation string** — actual type stated first; constraining device(s) named in parenthetical. When both sides constrain on different axes (mixed gen/width downgrade), both are named (per 4472607438).

## Open / unresolved

1. **Seed file `name`-only vs full DDL** — #22 comment 4570546987 states shipped rows carry `name` only, but the ratified DDL (and #17 acceptance criteria) requires `bandwidth_peak_gb_s`, `description`, `interface_family`, `transfer_rate`, `lane_count`. It is unclear whether the seed file seeds all columns or only `name`, and whether the `description` column exists in the shipped table. This was not resolved in the thread.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4570546987>

2. **Q2 — Hardware snapshot UUID field list** — @gissf1 agreed in principle to a deterministic UUID derived from hardware fingerprint fields (option A + C hybrid) but requested the full field list before confirming which fields to include, and flagged additional BIOS performance-affecting settings (wait states, memory interleaving) as candidates. No resolution in thread.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533634217>

3. **Q3 — gpu_hardware / system_gpu_link disk layout** — @gissf1 leaned toward option B (separate `gpu_hardware/<id>.json` files) but asked for elaboration on option A before deciding. No final ruling.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533634217>

4. **Q5 — Hardware snapshot timing (per-session vs per-run)** — @gissf1 accepted option A (once per process start) for consumer PCIe hardware, but flagged the answer as hardware-dependent and raised network-connected GPU hot-swap as an open question. No firm policy set.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533634217>

5. **Phase-2 migration framework** — referenced in 4570546987 as still in design, no further detail in thread.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4570546987>

## Cross-topic links

- **`gpu_hardware`** — carries `gpu_native_interface_type_id FK → interface_type`; `interface_type` must exist before `gpu_hardware` (implements #18, depends on #17).
- **`system_gpu_link`** — carries `slot_native_interface_type_id` and `actual_interface_type_id FK → interface_type`; implements #20, depends on #17.
- **`run_hardware_metrics`** — references `system_gpu_link` compound FK `(system_hardware_id, slot_index)`; full context join traverses `interface_type` twice (native + actual); implements #21.
- **`system_hardware`** / **`system_software`** — referenced in #22 disk-persistence discussion; design not finalized (topics: system_hardware, system_software).
- **disk persistence pattern** — #22 follows pattern established in #15 (models/tasks/prompts/runs); #15 is the upstream dependency for disk-file conventions applied here.
