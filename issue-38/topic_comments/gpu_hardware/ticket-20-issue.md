---
ticket: 20
type: issue-body
author: AlbinoGeek
posted: 2026-05-22T19:52:27Z
topic: gpu_hardware
url: https://github.com/Rethunk-AI/bakeoff/issues/20
title: schema: implement system_gpu_link join table with PCIe detection
---

**Parent:** #8 — Additional Performance Metrics
**Depends on:** #17 (`interface_type`), #18 (`gpu_hardware`), #19 (`system_hardware`)

Implement the `system_gpu_link` join table as ratified in #8. This table models the relationship between a physical host and a GPU in a specific slot, including the interface type information.

## Schema

```sql
CREATE TABLE system_gpu_link (
    system_hardware_id             INT NOT NULL REFERENCES system_hardware,
    slot_index                     INT NOT NULL,
    PRIMARY KEY (system_hardware_id, slot_index),

    gpu_hardware_id                INT NOT NULL REFERENCES gpu_hardware,
    slot_native_interface_type_id  INT NULLABLE REFERENCES interface_type,  -- motherboard slot's rated max
    actual_interface_type_id       INT NULLABLE REFERENCES interface_type   -- negotiated running state
);

CREATE INDEX ON system_gpu_link (gpu_hardware_id);
```

## Key design decisions (from #8 thread)

- PK is `(system_hardware_id, slot_index)` — a slot is a fixed property of the system. The GPU occupying it is data, not part of the key. Two identical GPUs in the same system each get their own slot row.
- `gpu_hardware_id` is an FK carrying the GPU model identity, indexed for efficient lookups of "all systems running GPU model X".
- `is_slot_limited` is **not stored** — derivable as `slot_native_interface_type_id != actual_interface_type_id`. Compute in queries and views.
- NVLink deferred to Phase 2 — no NVLink columns.

## Auto-detection requirements

Harness populates `system_gpu_link` from nvidia-smi at startup:

- **`slot_index`** — GPU index from nvidia-smi (`--id=0`, `--id=1`, etc.). Multi-GPU systems get one row per GPU.
- **`gpu_hardware_id`** — FK to the `gpu_hardware` row detected/created in #18
- **`slot_native_interface_type_id`** — FK to `interface_type` matching `pcie.link.gen.max × pcie.link.width.max` from `nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max`
- **`actual_interface_type_id`** — FK to `interface_type` matching `pcie.link.gen.current × pcie.link.width.current` from `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`

When `slot_native != actual`, the slot is running at reduced capability. The display layer generates the limitation description at runtime from both interface_type rows' `description` fields.

## Deduplication / upsert

`(system_hardware_id, slot_index)` uniquely identifies a slot. On subsequent runs, upsert on this PK — update `gpu_hardware_id` and interface FKs if the GPU in the slot has changed.

## Acceptance criteria

- [ ] Migration creates `system_gpu_link` with correct PK, FKs, and index on `gpu_hardware_id`
- [ ] Harness auto-populates one row per GPU at startup (supports multi-GPU systems)
- [ ] `slot_native_interface_type_id` and `actual_interface_type_id` are resolved from seeded `interface_type` rows
- [ ] Upsert behavior: existing slot rows updated if GPU changes, not duplicated
- [ ] Migration is reversible

— Bastion
