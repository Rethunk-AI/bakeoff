# Summary: gpu_hardware

## Final state

**Table: `gpu_hardware`** — die-level GPU intrinsics; one row per GPU model, not per physical instance. Deduplication key: `(pci_device_id, pci_sub_device_id)` when both are available; fall back to normalized `gpu_name`.

```sql
CREATE TABLE gpu_hardware (
    gpu_hardware_id                  SERIAL PRIMARY KEY,
    gpu_name                         TEXT NOT NULL,                   -- renamed from gpu_model (per 4629938143)
    pci_device_id                    TEXT NULLABLE,                   -- "0x2684"; confirmed from #22 thread
    pci_sub_device_id                TEXT NULLABLE,                   -- board partner variant
    pci_vendor_id                    TEXT NULLABLE,                   -- e.g. "10de" = NVIDIA (added in #22)
    pci_subsystem_vendor_id          TEXT NULLABLE,                   -- board OEM identity (added in #22)
    pci_subsystem_device_id          TEXT NULLABLE,                   -- board variant (added in #22)
    vram_total_mb                    INT NULLABLE,                    -- renamed from vram_mb (per 4629938143)
    vram_type                        TEXT NULLABLE,                   -- OPEN: FK vs inline TEXT; see below
    memory_bus_width_bits            INT NULLABLE,
    memory_bandwidth_peak_gb_s       FLOAT NULLABLE,                  -- stored, not derived (per 4469686066)
    clock_memory_mhz                 INT NULLABLE,
    clock_graphics_boost_mhz         INT NULLABLE,
    peak_tflops_fp16                 FLOAT NULLABLE,                  -- renamed from tflops_fp16 (per 4629938143)
    tdp_watts                        INT NULLABLE,
    gpu_native_interface_type_id     INT NULLABLE REFERENCES interface_type,  -- renamed from interface_type_id (per 4629938143)
    gpu_architecture_id              INT NULLABLE REFERENCES gpu_architectures  -- FK; categorization only, not UUID
);
```

**Column renames confirmed** (per 4629938143):
- `gpu_model` → `gpu_name`
- `vram_mb` → `vram_total_mb`
- `tflops_fp16` → `peak_tflops_fp16`
- `interface_type_id` → `gpu_native_interface_type_id`

**`vram_type` folded into `gpu_hardware`** — @gissf1 directed in 4636026879 that `vram_type` is not a separate topic; it belongs here. Whether it is a FK to a seeded `vram_types` lookup table or a bare TEXT field remains open (see Open section).

**TFLOPs stored via related table** (per #22 thread, ~4534654458):
```sql
CREATE TABLE gpu_tflops (
    gpu_hardware_id    INT NOT NULL REFERENCES gpu_hardware,
    compute_format_id  INT NOT NULL REFERENCES compute_formats,   -- fp16, fp32, bf16, int8, fp8, etc.
    tflops_value       FLOAT NOT NULL,
    tflops_source_id   INT NOT NULL REFERENCES tflops_sources,
    PRIMARY KEY (gpu_hardware_id, compute_format_id)
);
```

`compute_formats` and `tflops_sources` are seeded lookup tables. `tflops_sources` seed: ID 1 = 'unknown/unverified', ID 2 = 'Rethunk measured'; supports `url_template` with `{pci_device_id}` / `{gpu_model}` substitution.

**Related lookup tables:**
- `gpu_architectures (architecture_id, name, description)` — seeded; FK used for results-page categorization only
- `compute_formats (compute_format_id, name, description)` — seeded; fp16, fp32, bf16, int8, fp8, etc.
- `tflops_sources (source_id, name, contact_url, url_template)` — seeded
- `vram_types` — seeded if FK approach is adopted (open item)

**Auto-detection sources** (per #18 issue body):
- `gpu_name`: `nvidia-smi --query-gpu=name`
- `pci_device_id` / `pci_sub_device_id`: `nvidia-smi --query-gpu=pci.device_id,pci.sub_device_id`
- `vram_total_mb`: `nvidia-smi --query-gpu=memory.total` (MiB → MB conversion)
- `vram_type`: PCI device ID lookup table (no direct nvidia-smi query; deterministic from device ID)
- `memory_bus_width_bits`: `pynvml.nvmlDeviceGetMemoryBusWidth()`
- `clock_memory_mhz` / `clock_graphics_boost_mhz`: `nvidia-smi --query-gpu=clocks.max.memory,clocks.boost.graphics`
- `peak_tflops_fp16`: existing `_TFLOPS_TABLE` lookup in harness
- `memory_bandwidth_peak_gb_s`: computed at detection time from `clock_memory_mhz` + `memory_bus_width_bits` + `vram_type` DDR factor; stored (not re-derived at display callsites)
- `gpu_native_interface_type_id`: FK to `interface_type` row matching rated PCIe spec

## Notable / unusual decisions

- **`memory_bandwidth_peak_gb_s` is stored, not derived** — formula requires `vram_type` to select DDR factor (GDDR uses ×2, HBM uses ×1); replicating that logic at every display callsite was deemed more error-prone than storing the computed value once (per 4469686066). Downstream: queries can use it directly without re-implementing DDR/HBM branching.

- **`gpu_hardware` is model-level, not instance-level** — two users with the same GPU model share one row. Fabrication detection: `pci_device_id` is cross-checked against `gpu_name` before upsert; mismatch is flagged as invalid (per #18 issue body). Downstream: normalization prevents redundant rows per submitter; community benchmarking at scale is the explicit target.

- **`gpu_native_interface_type_id` lives on `gpu_hardware`, not `system_gpu_link`** — @gissf1 directed in 4472607438 that the GPU's native link spec is a fixed property of the card; the join table carries `slot_native_interface_type_id` (motherboard slot's rated max) and `actual_interface_type_id` (negotiated running state). Downstream: attribution logic for "PCIe 3.0 x16 GPU limiting PCIe 4.0 x16 slot" (or vice versa) compares the three interface_type rows at display time without touching `gpu_hardware`.

- **`tflops_fp32` / `tflops_bf16` dropped from direct columns → `gpu_tflops` table** — @gissf1 rejected per-format columns in 4534370703 ("I don't like listing out every tflops format as a new field"); replaced by a join table keyed on `compute_format_id`. Extensible to fp8, int8, etc. without schema migration. Downstream: filter bar on results page reads `compute_formats` seed table for dropdown.

- **`compute_units` dropped** — @gissf1 ruled in 4550408728 it is implied by PCI IDs + architecture and has no independent signal for the test surface.

- **`pci_vendor_id` provenance is unclear** — see Open section.

## Open / unresolved

- **`tflops_fp32` and `tflops_bf16` in current `schema.sql` — retain or drop?** Per 4629938143, Bastion flagged these two columns as not in the #18 spec and asked for direction. No @gissf1 response on this specific question yet. Trade-off: dropping them means all tflops data migrates to `gpu_tflops`; retaining them creates inconsistency (fp16 is in a separate table, fp32/bf16 are direct columns). Likely should drop, but requires explicit confirmation.

- **`vram_type` — FK to seeded `vram_types` table vs. bare TEXT column** — @gissf1 stated in 4534370703 "vram_type should probably be a seeded lookup table"; @gissf1 also stated in 4636026879 that `vram_type` folds into `gpu_hardware` (not a separate topic). The #18 spec authors it as `TEXT NULLABLE`. No final schema DDL shows it as a FK. Awaiting explicit confirmation of FK vs TEXT. If FK: a `vram_types (vram_type_id, name, description)` table is needed.

- **`pci_vendor_id`, `pci_subsystem_vendor_id`, `pci_subsystem_device_id` provenance** — per 4629938143, Bastion noted these three fields were not in the #18 spec as authored and asked @gissf1 to confirm the issue/comment where they were agreed. The fields were discussed in the #22 thread (4533933819 proposed them; 4534370703 confirmed them for that context), but whether they belong in the `gpu_hardware` spec vs. the system/UUID context is not yet formally ratified for issue #18. Carry as expected additions pending that confirmation.

- **`tflops_source` table spec origin** — per 4629938143, Bastion could not locate `tflops_source` in the reviewed issue history and asked @gissf1 to point to the reference. The full `tflops_sources` design appears in #22 thread (4534654458 / 4550408728), not in #18 directly. Whether this makes it in-scope for `gpu_hardware` schema or a separate spec issue is not yet confirmed.

- **BIOS `bios_notes` UUID whitelist** — final key set (`bar_size_mb`, `pcie_gen_override`, `power_limit_w`) was proposed in 4552726518 with open questions to @gissf1; no confirmation received in the captured thread. Not part of `gpu_hardware` directly, but affects hardware UUID computation which references GPU fields.

## Cross-topic links

- **`interface_type`** — `gpu_hardware.gpu_native_interface_type_id` is a FK to `interface_type.interface_type_id`. `interface_type` also receives FKs from `system_gpu_link` (`slot_native_interface_type_id`, `actual_interface_type_id`). Both sides must reference the same seeded lookup. Note: `interface_type.transfer_rate` should be renamed `lane_transfer_rate` per @gissf1 direction in #38 issue body.

- **`system_gpu_link`** — join table between `system_hardware` and `gpu_hardware`. Carries `slot_index`, `slot_native_interface_type_id`, `actual_interface_type_id`. `run_hardware_metrics` references `system_gpu_link` via compound FK `(system_hardware_id, slot_index)`.

- **`run_hardware_metrics`** — per-run measurements reference `system_gpu_link` compound PK; does not reference `gpu_hardware` directly (goes through the join table). Fields: `peak_vram_mb`, `gpu_sm_utilization_pct`, `tflops_utilization_pct`, etc.

- **`gpu_architectures`** — `gpu_hardware.gpu_architecture_id` FK; seeded lookup for results-page categorization.

- **`compute_formats`** — FK target for `gpu_tflops.compute_format_id`.

- **`tflops_sources`** — FK target for `gpu_tflops.tflops_source_id`.

- **`vram_types`** (if FK approach confirmed) — FK target for `gpu_hardware.vram_type_id`.

- **`uuids` topic** — GPU fields included in the hardware UUID computation: `pci_vendor_id`, `pci_device_id`, `pci_subsystem_vendor_id`, `pci_subsystem_device_id`, `vram_total_mb`, `vram_type`. GPU performance metrics (`tflops`, `memory_bandwidth_peak_gb_s`, `tdp_watts`) are explicitly excluded from UUID input (per 4534370703). See `uuids` topic for full hash field set and `uuid_migrations` table design.
