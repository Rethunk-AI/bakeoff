# Summary: interface_type

## Final state

### Table: `interface_type`

| Column | SQL Type | Constraints | Notes |
|---|---|---|---|
| `interface_type_id` | `SERIAL` | PRIMARY KEY | |
| `bandwidth_peak_gb_s` | `FLOAT` | NOT NULL | Universal comparable field across all interface families |
| `description` | `TEXT` | NOT NULL | Human-readable label: "PCIe 4.0 x16", "SXM5", "Thunderbolt 4" |
| `interface_family` | `TEXT` | NULLABLE | Grouping enum: "PCIe", "SXM", "CXL", "NVLink", "USB", "OCuLink" |
| `lane_transfer_rate` | `INT` | NULLABLE | PCIe: GT/s per lane (e.g. 16 for Gen 4); null for non-PCIe (per #38 rename from `transfer_rate`) |
| `lane_count` | `INT` | NULLABLE | PCIe: lane width (e.g. 16); null for non-PCIe |

Column rename `transfer_rate` → `lane_transfer_rate` confirmed in issue #38 body (per 4629938143).

### Seed data scope

Pre-seeded rows required (per #17 / #38):

- PCIe Gen 1–5 × x1 / x4 / x8 / x16 (bandwidth computed from `lane_transfer_rate × lane_count × 2 / 8`)
- SXM2, SXM4, SXM5
- NVLink 2.0 / 3.0 / 4.0 — `interface_family = "NVLink"`, reserved but no FK references in Phase 1
- OCuLink 2.0
- Thunderbolt 3, Thunderbolt 4

No pre-seeded rows for degraded states — display layer constructs degradation descriptions at runtime from the two FK rows (`slot_native` and `actual`) on `system_gpu_link`.

Current live schema has only `interface_type_id` + `name`; all columns above are missing from the deployed schema (confirmed in 4629938143). Seed data is a 10-row stub only.

## Notable / unusual decisions

- **Bandwidth-first, PCIe-agnostic field set** — gissf1 explicitly rejected encoding PCIe generation/width as dedicated columns; the only universal invariant across PCIe, SXM, NVLink, USB, OCuLink is bandwidth. `lane_transfer_rate` and `lane_count` are nullable extras for PCIe-specific attribution logic, not load-bearing keys (per 4470606665).

- **`description` as free-text attribution carrier** — rather than building degradation state into stored rows (e.g. a pre-seeded "PCIe 3.0 x16 degraded from PCIe 4.0 x16" row), degradation description is assembled at runtime by the display layer: `"[actual.description] (limited by [slot_native.description] and/or [gpu_native.description])"`. This avoids a combinatorial explosion of pre-seeded rows for every possible native/actual pairing (per 4472501126, confirmed 4472607438).

- **`lane_transfer_rate` + `lane_count` nullable pair** — added late in the thread (per 4474333279) specifically to support per-axis attribution (gen vs. lane width can degrade independently). Example: PCIe 3.0 x16 GPU in PCIe 4.0 x8 slot degrades on both axes simultaneously; bandwidth alone cannot distinguish which device is the constraint. Non-PCIe interfaces leave both null; `bandwidth_peak_gb_s` is the sole comparison field.

- **`interface_family` as grouping only** — not a PK component, not an enum type in the DB. String value allows future families to be added without a migration.

- **NVLink reserved, not wired** — NVLink rows seeded with `interface_family = "NVLink"` in Phase 1 but no table holds an FK to them yet. Full NVLink topology modelling (GPU-to-GPU, multi-card unified memory) deferred to Phase 2; reserved namespace prevents churn (per 4470666757).

- **`memory_bandwidth_peak_gb_s` stored, not derived** — on `gpu_hardware` (not `interface_type`), the peak memory bandwidth is stored rather than derived at display time. Reason: the derivation formula requires `vram_type` to select the DDR factor (GDDR uses ×2, HBM uses ×1); storing avoids re-implementing that logic at every callsite (per 4469686066).

## Open / unresolved

- **`vram_type` as TEXT vs FK** — in #38 body, gissf1 references a `vram_type` FK table; the #18 spec has it as `TEXT`. Bastion's 4629938143 explicitly flagged this as an open clarification: "is `vram_type` meant to be a free-text field or a FK to a separate lookup table?" — no resolution yet.

- **`tflops_source` table** — gissf1 referenced this in #38 body; Bastion confirmed it is absent from the current schema and from all prior issue discussions and requested the issue/comment where it was specified. No resolution in this thread.

- **`pci_vendor_id` / `pci_subsystem_vendor_id` on `gpu_hardware`** — gissf1 listed these as expected fields in #38 body; Bastion noted they are not in the #18 spec and requested the source comment. No resolution.

- **`tflops_fp32` / `tflops_bf16` on `gpu_hardware`** — present in the current live schema but not in the #18 spec. Bastion asked: retain or drop? No answer yet.

- **Q2–Q5 from #22 (disk persistence layout)** — snapshot identity UUID fields, GPU hardware disk layout (inline vs. separate files), run_hardware_metrics disk relationship, and runner snapshot timing are all unresolved pending gissf1 reply. Q1 (seed file pattern) was shipped on `main` (`3b0bce4`) with a stub that omits the full column set; that stub itself needs patching once #38 gaps are resolved.

## Cross-topic links

- **`system_gpu_link`** holds two FKs → `interface_type`:
  - `slot_native_interface_type_id` — motherboard slot's rated maximum
  - `actual_interface_type_id` — negotiated running state at link-up
  Both are currently missing from the live `system_gpu_link` table (per 4629938143).

- **`gpu_hardware`** holds one FK → `interface_type`:
  - `gpu_native_interface_type_id` — GPU card's rated interface spec (moved here from `system_gpu_link` per 4472607438 / 4474333279)
  Also missing from live schema.

- **`run_hardware_metrics`** references `system_gpu_link (system_hardware_id, slot_index)` compound FK, giving full hardware context via: `run_hardware_metrics → system_gpu_link → system_hardware + gpu_hardware + interface_type (×2)`.
