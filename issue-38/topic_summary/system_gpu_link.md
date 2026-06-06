# Summary: system_gpu_link

## Final state

Join table linking `system_hardware` ↔ `gpu_hardware` in a many-to-many relationship. Carries slot identity and interface state. Ratified across #8 comments through #20 issue body.

```sql
CREATE TABLE system_gpu_link (
    system_hardware_id              INT NOT NULL REFERENCES system_hardware,
    slot_index                      INT NOT NULL,
    PRIMARY KEY (system_hardware_id, slot_index),

    gpu_hardware_id                 INT NOT NULL REFERENCES gpu_hardware,
    slot_native_interface_type_id   INT NULLABLE REFERENCES interface_type,  -- motherboard slot's rated max
    actual_interface_type_id        INT NULLABLE REFERENCES interface_type    -- negotiated running state
);

CREATE INDEX ON system_gpu_link (gpu_hardware_id);
```

`gpu_hardware.gpu_native_interface_type_id` carries the GPU card's own rated spec (moved off `system_gpu_link` per 4472607438).

`is_slot_limited` is **not stored** — derivable as `slot_native_interface_type_id != actual_interface_type_id`; no trigger complexity warranted (per 4472501126).

Degraded-state description is runtime-constructed: `"[actual.description] (limited by [slot_native.description] / [gpu_native.description])"`. Pre-seeded exhaustive degraded rows rejected as impractical (per 4472501126). Multi-axis PCIe attribution rules (which device is constraining) defined in the full schema dump (per 4474333279).

`run_hardware_metrics` references this table via compound FK `(system_hardware_id, slot_index)`. Forward-compatible with a future `run_gpu_usage` join table for multi-GPU runs without schema changes.

## Notable / unusual decisions

- **PK is `(system_hardware_id, slot_index)`, not `(system_hardware_id, gpu_hardware_id, slot_index)`** — a slot is a fixed physical property of the motherboard; the GPU occupying it is data about that slot. The three-column PK would incorrectly prevent two identical GPU models from occupying separate slots in the same system (per 4472501126).
- **GPU's native interface type lives on `gpu_hardware`, not `system_gpu_link`** — the GPU card's rated spec is a fixed property of the card, not of a particular slot relationship. Moving it here was a schema correction from an earlier draft (per 4472607438). Downstream: leaderboard limitation attribution requires joining both `gpu_hardware.gpu_native_interface_type_id` and `system_gpu_link.slot_native_interface_type_id`.
- **`actual_interface_type_id` named "actual", not "slot_actual"** — the negotiated link state is synchronized across both hardware parties (slot + GPU), so attributing it to "slot" was semantically wrong (per 4472607438).
- **Standalone index on `gpu_hardware_id`** — not covered by the PK index; needed for the "all systems running GPU model X" query pattern expected at community-benchmarking scale.
- **NVLink deferred to Phase 2** — Phase 1 reserves `interface_family = "NVLink"` in `interface_type` as a named constant but adds no NVLink columns. A `gpu_gpu_link` table is the correct Phase 2 structure; NVLink fundamentally changes GPU-to-GPU topology and does not fit `system_gpu_link` semantics (per 4470666757).
- **Multi-GPU and multi-system distribution deferred** — future extension is `run_gpu_usage (run_id, system_hardware_id, slot_index)` referencing this table's PK; no schema changes to existing tables required (per 4472501126).

## Open / unresolved

- **`pci_vendor_id` / `pci_subsystem_vendor_id` on `gpu_hardware`** — @gissf1 referenced these fields in issue #38 (4629938143) but they are not in the #18 spec as authored; Bastion requested a citation or confirmation that they should be added. Not yet resolved as of the last entry in this topic.
- **`tflops_fp32` / `tflops_bf16` in current `gpu_hardware` schema** — present in the live `schema.sql` but absent from the spec; Bastion asked retain-or-drop, no answer recorded in this topic's thread.
- **`vram_type` as TEXT vs FK** — #18 spec has it as `TEXT NULLABLE`; @gissf1's #38 message lists it as a FK field. Clarification was requested (4629938143) but not resolved within this topic.
- **`tflops_source` table** — referenced by @gissf1 in #38 as expected; not found in any prior issue or schema. Origin issue not yet identified; Bastion requested a reference.
- **Disk persistence format (issue #22)** — hardware snapshot UUID input field list (BIOS fields: `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile`) was proposed but @gissf1's response in 4553530231 raised questions about `pcie_gen_override` and `power_limit_w` without a definitive final field list. The combined inline+dedup disk layout (Q2 = A+C hybrid, Q3 = Option A) was confirmed, but the exact UUID input whitelist remains partially open.

## Cross-topic links

- `system_gpu_link.gpu_hardware_id` → **gpu_hardware** (topic: gpu_hardware / issue #18)
- `system_gpu_link.system_hardware_id` → **system_hardware** (topic: system_hardware / issue #19)
- `system_gpu_link.slot_native_interface_type_id` → **interface_type** (issue #17)
- `system_gpu_link.actual_interface_type_id` → **interface_type** (issue #17)
- `gpu_hardware.gpu_native_interface_type_id` → **interface_type** (issue #17) — third interface FK used in attribution logic
- `run_hardware_metrics.(system_hardware_id, slot_index)` → **system_gpu_link** (topic: run_model_metrics / issue #21)
