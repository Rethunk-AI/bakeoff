# Summary: gpu_architecture

## Final state

`gpu_architecture` is resolved as a **FK to a seeded lookup table**, not a raw string column on `gpu_hardware`. Agreed in #22 (per 4534654458).

```sql
-- Lookup table (seeded)
-- Exact DDL not yet drafted; conceptually:
CREATE TABLE gpu_architectures (
    gpu_architecture_id  SERIAL PRIMARY KEY,
    name                 TEXT NOT NULL  -- e.g. "Ampere", "Ada Lovelace", "RDNA 3", "Hopper", "Blackwell"
    -- additional fields TBD
);

-- gpu_hardware carries:
gpu_architecture_id  INT NULLABLE REFERENCES gpu_architectures
```

Purpose: results-page **categorization only**. Not included in the GPU UUID hash — architecture is implied by `pci_vendor_id` + `pci_device_id` and is therefore redundant as an identity input.

Agreed GPU UUID inputs (per 4534654458):

| Field | UUID input? |
|-------|-------------|
| `pci_vendor_id` | YES |
| `pci_device_id` | YES |
| `pci_subsystem_vendor_id` | YES |
| `pci_subsystem_device_id` | YES |
| `vram_mb` | YES |
| `vram_type` (FK) | YES |
| `gpu_architecture` (FK) | NO — categorization, implied by PCI IDs |
| `compute_units` | NO — implied by PCI IDs + arch |
| Performance metrics (tflops, bandwidth, TDP) | NO |

`slot_index` also excluded from UUID — slot position belongs in `system_gpu_link`, not the GPU identity record (per 4534654458; confirmation from @gissf1 pending).

`vram_type` resolved as a **seeded FK lookup table** (`vram_types`): GDDR6, GDDR6X, HBM2e, HBM3, LPDDR5X, etc. Included in UUID — memory type materially affects bandwidth and inference latency profile (per 4534654458).

`pci_subsystem_vendor_id` + `pci_subsystem_device_id` added to capture board-level OEM differentiation (non-reference memory configs, modified clocks, alternate thermal designs) (per 4534654458).

## Notable / unusual decisions

- **`gpu_architecture` as FK, not enum or string** — enables grouping results by generation (Ampere/Hopper/Ada/RDNA 3/Blackwell) without touching GPU records when new architectures appear; seeded table is the extension point.
- **Architecture excluded from UUID** — PCI vendor+device IDs already uniquely identify architecture; including it would be redundant and create stale UUIDs if the lookup table is refined. Categorization value is preserved via the FK without affecting identity.
- **`compute_units` excluded from UUID** — SMs/CUs/EUs are implied by PCI ID + architecture. Field retained on the table for reference but not an identity anchor. Name `compute_units` is architecture-neutral but was flagged as potentially ambiguous (per 4534654458 noted `parallel_execution_units` as an alternative — not yet resolved).
- **`tflops` moved to related table `gpu_tflops`** — per-format columns (`tflops_fp16`, `tflops_fp32`, `tflops_bf16`) replaced by `gpu_tflops { gpu_hardware_id, compute_format_id, tflops_value, tflops_source }` with `compute_format_id FK → compute_formats`. Extensible to fp8, int8, etc. without schema migration per new format (per 4534654458).
- **`tflops_source` flag** — `'manufacturer'` | `'measured'`. Captures that manufacturer TDP/tflops specs may not reflect actual inference performance; once enough validated runs exist for a GPU, a `'measured'` value can be computed and stored alongside.

## Open / unresolved

- **`slot_index` in UUID** — Bastion recommended excluding it (GPU identity should not change when moved to a different PCIe slot); explicit @gissf1 confirmation not recorded in thread (per 4534654458 asked "Confirm?").
- **`compute_units` naming** — `compute_units` vs. `parallel_execution_units` left unresolved in thread.
- **`gpu_architectures` table DDL** — agreed as a seeded lookup table but exact columns not specified.
- **`vram_types` table DDL** — agreed as a seeded lookup table; exact columns and seed values not specified.
- **`compute_formats` table DDL** — agreed as a seeded lookup for tflops format types; not specified.
- **Implementation gap** — #38 (per 4629938143) confirms `gpu_hardware` in live `schema.sql` is a reduced stub missing `gpu_architecture` FK, PCI fields, `vram_type` FK, and others. Remediation pending @gissf1 sign-off.

## Cross-topic links

- `gpu_hardware.gpu_architecture_id` → `gpu_architectures` (seeded lookup; this topic)
- `gpu_hardware.vram_type` → `vram_types` (seeded lookup; adjacent to this topic)
- `gpu_hardware` ← `system_gpu_link` (FK; hardware placement context)
- `gpu_tflops.gpu_hardware_id` → `gpu_hardware` (related table for per-format tflops)
- `gpu_tflops.compute_format_id` → `compute_formats` (seeded lookup)
- `system_hardware` — houses the UUID migration table (`uuid_migrations`) and `hardware_schema_versions` that govern GPU UUID evolution
