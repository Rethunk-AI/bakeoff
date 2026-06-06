---
ticket: 18
type: issue-body
author: AlbinoGeek
posted: 2026-05-22T19:52:20Z
title: schema: implement gpu_hardware table with auto-detection
url: https://github.com/Rethunk-AI/bakeoff/issues/18
---

# schema: implement gpu_hardware table with auto-detection

**Parent:** #8 — Additional Performance Metrics
**Depends on:** #17 (`interface_type` table)

Implement the `gpu_hardware` entity table as ratified in #8. This table stores die-level GPU intrinsics — fixed properties of the GPU model, not of a specific system configuration.

## Schema

```sql
CREATE TABLE gpu_hardware (
    gpu_hardware_id              SERIAL PRIMARY KEY,
    gpu_name                     TEXT NOT NULL,
    pci_device_id                TEXT NULLABLE,           -- "0x2684"
    pci_sub_device_id            TEXT NULLABLE,           -- board partner variant
    vram_total_mb                INT NULLABLE,
    vram_type                    TEXT NULLABLE,           -- "GDDR6X", "HBM2e", etc.
    memory_bus_width_bits        INT NULLABLE,
    memory_bandwidth_peak_gb_s   FLOAT NULLABLE,          -- stored (not derived) per design decision
    clock_memory_mhz             INT NULLABLE,
    clock_graphics_boost_mhz    INT NULLABLE,
    peak_tflops_fp16             FLOAT NULLABLE,
    tdp_watts                    INT NULLABLE,
    gpu_native_interface_type_id INT NULLABLE REFERENCES interface_type  -- card's rated spec
);
```

## Auto-detection requirements

The harness should auto-detect and populate a `gpu_hardware` row from the running system. Detection sources:

- **`gpu_name`** — `nvidia-smi --query-gpu=name --format=csv,noheader`
- **`pci_device_id` / `pci_sub_device_id`** — `nvidia-smi --query-gpu=pci.device_id,pci.sub_device_id`
- **`vram_total_mb`** — `nvidia-smi --query-gpu=memory.total` (convert MiB → MB)
- **`vram_type`** — PCI device ID lookup table (no direct nvidia-smi query; deterministic from device ID)
- **`memory_bus_width_bits`** — `pynvml.nvmlDeviceGetMemoryBusWidth()`
- **`clock_memory_mhz` / `clock_graphics_boost_mhz`** — `nvidia-smi --query-gpu=clocks.max.memory,clocks.boost.graphics`
- **`peak_tflops_fp16`** — existing `_TFLOPS_TABLE` lookup (already implemented in harness); `None` if not in table
- **`memory_bandwidth_peak_gb_s`** — computed from `clock_memory_mhz`, `memory_bus_width_bits`, and `vram_type` (DDR factor)
- **`gpu_native_interface_type_id`** — FK to `interface_type` row matching the GPU's rated PCIe spec

## Identity / deduplication

A `gpu_hardware` row is a **model-level** record, not an instance record. Two users with the same GPU model should share a single row. Deduplicate on `(pci_device_id, pci_sub_device_id)` when both are available; fall back to `gpu_name` normalization.

Fabrication detection: `pci_device_id` must be consistent with the `gpu_name` reported — cross-check before insert. Flag mismatch as invalid.

## Acceptance criteria

- [ ] Migration creates `gpu_hardware` with all columns and FK to `interface_type`
- [ ] Harness auto-detects and upserts a `gpu_hardware` row at startup (nvidia-smi + pynvml)
- [ ] Deduplication logic: existing row with matching PCI IDs is reused, not duplicated
- [ ] `memory_bandwidth_peak_gb_s` is computed and stored at detection time
- [ ] `peak_tflops_fp16` lookup uses existing `_TFLOPS_TABLE`
- [ ] Migration is reversible

— Bastion

