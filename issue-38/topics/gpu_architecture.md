# Topic: gpu_architecture

Consolidated chat history (chronological, ascending comment-id). 4 entries. Verbatim quotes; attribution in each header.

## #22 comment 4533894066 — @gissf1 — 2026-05-25T11:29:48Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533894066>

RE: Q2:
- system fields:
  - cpu_cores is implied by cpu_model, and therefore likely redundant.
  - cpu_threads can be influenced by BIOS settings (en/disable hyper-threading, core count limits, etc), therefore is still useful to track.
  - CPU base/peak clock speeds might be useful to have to detect CPUs with forced fixed clocks or overclocking. This data might also be useful in the hardware table if not already present.
  - agreed on adding memory information and bios_notes to table and it probably could be condensed in a more packed format when used within the UUID.
  - There are probably other BIOS settings to include, but we can add them and update records as we notice them in results.
    - That said, maybe it also makes sense to have a "replacement list" somewhere - that when an old UUID is replaced by a new one (because we added a new field to the UUID hash) that we can update old records we encounter later to use the new UUID without a complicated process.  This should be simply the old UUID and new UUID, but maybe we need to indicate why or what field is required to verify it.  If we try to look up a UUID and can't find it in its proper table, we can try looking it up in this table to see if we can find a replacement to use in its place.  This would also allow older test results to apply to the correct hardware records after we add the appropriate fields to the record.  Ideally we could tell whether the results were done on the proper hardware configuration to be valid; for example if we come up with a new setting to include in the hash, say some new BIOS flag, we could check the results file data to determine whether the results were generated with the flag or without, to properly update the records.  If not, perhaps we can have some process to re-validate just the hardware ID information on the source system to determine the value and update the record without needing to redo the testing entirely.  This is just an idea, but I think it could be useful for forward migration of data.
- GPU fields:
  - I think this is a good start, but we probably need more fields here too.

RE: Q3: I expect there are many more motherboard + RAM + CPU configurations than GPU models.  There will be a high level of redundancy in the GPU hardware because of that.  I think externalizing the GPU hardware info into a separate file and using that UUID within the system hardware table is the most efficient mechanism for storage.  That said, I do think there will be cases where the GPU is unknown and we don't have a record for it yet.  I think the right answer is accepting multiple formats, starting with option A and then moving to option B as our database of hardware improves to the point that we have known records in the seed data already that users can reference.  The difference is just changing the `gpus: [...]` array so records for known GPUs become something like `{slot_index: 0, gpu_hardware_id: "123456789"}` instead of all the redundant details as in the example you presented.  I agree that since we need to have more existing records before that's a concern, and sparse format is something slated for a later, it makes sense to put developing that into a later phase.  Agreed with postponing for now until we have basic reporting working.

RE: open items for confirmation: I think I addressed your concerns above, but let me know if you need additional clarification/confirmation.

---

## #22 comment 4533933819 — @AlbinoGeek — 2026-05-25T11:35:33Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533933819>

**Bastion response — 251134ZMAY26**

@gissf1 — all points received. Addressing in order.

---

**Q2 — System fields for hardware_id UUID (revised)**

| Field | UUID input? | Notes |
|-------|-------------|-------|
| `cpu_model` | YES | identity anchor |
| `cpu_cores` | NO | implied by cpu_model — removed per your direction |
| `cpu_threads` | YES | BIOS-configurable (HT/SMT en/disable, core limits) |
| `cpu_base_clock_mhz` | YES | detects fixed-clock or underclocked configs |
| `cpu_peak_clock_mhz` | YES | detects boost/OC configs |
| `ram_gb` | YES | |
| `motherboard` | YES | |
| `memory_speed_mhz` | YES | active clock, not SPD rated |
| `memory_channels` | YES | BIOS-configurable |
| `memory_interleave_profile` | YES | XMP/EXPO/DOCP/manual |
| `bios_notes` | NO — stored only | free-form, not stable for dedup |

`cpu_base_clock_mhz` and `cpu_peak_clock_mhz` auto-populated from `/proc/cpuinfo` or `dmidecode`; runner captures both at snapshot time.

---

**UUID migration table — confirmed and proposed schema**

The migration table is the right answer for forward schema evolution. When the UUID input field set changes, old records carry a stale UUID that won't match new ingest. The table bridges them.

Proposed disk file: `hardware-migrations/<old_uuid>.json`

```json
{
  "schema_version": 1,
  "old_hardware_id": "<old-uuid>",
  "new_hardware_id": "<new-uuid>",
  "migration_reason": "added cpu_base_clock_mhz, cpu_peak_clock_mhz to UUID input",
  "verification_field": "cpu_base_clock_mhz",
  "verification_source": "disk_file",
  "verified": true
}
```

DB table:

```sql
CREATE TABLE hardware_id_migrations (
    old_hardware_id   UUID NOT NULL,
    new_hardware_id   UUID NOT NULL REFERENCES system_hardware(system_hardware_id),
    migration_reason  TEXT NOT NULL,
    verification_field TEXT,          -- which disk field confirms the migration
    verification_source TEXT,         -- 'disk_file' | 'hardware_probe' | 'manual'
    verified          BOOLEAN NOT NULL DEFAULT FALSE,
    migrated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (old_hardware_id, new_hardware_id)
);
```

Lookup protocol at ingest: UUID not found in `system_hardware` → check `hardware_id_migrations` for `old_hardware_id` match → if found and `verified = true`, use `new_hardware_id`; if found and `verified = false`, flag for operator review; if not found, treat as new hardware record.

Re-validation without full retest: runner can be invoked with `--probe-hardware-only` flag (Phase 2) to re-snapshot hardware fields and emit a new UUID without running benchmarks. The migration entry is created automatically; `verified` flips to true once the new field value is confirmed present in the disk file.

---

**Q3 — Multi-format GPU array: confirmed**

Agreed on the graceful progression. The `gpus` array in hardware snapshots and run files accepts two entry shapes:

- **Inline** (unknown GPU or no seed record yet):
  ```json
  {"slot_index": 0, "gpu_model": "AMD Radeon RX 7900 XTX", "vram_mb": 24576, "interface": "PCIe 4.0 x16"}
  ```
- **Reference** (GPU known in seed/DB):
  ```json
  {"slot_index": 0, "gpu_hardware_id": "<gpu-uuid>"}
  ```

Phase 1: inline only. Phase 2: validator accepts references; ingest resolves them. Sparse format deferred as agreed.

---

**GPU fields — proposed extended set**

You noted more GPU fields are needed. Proposed additions for the GPU record (inline or seed):

| Field | Description |
|-------|-------------|
| `gpu_architecture` | Microarchitecture name (RDNA 3, Ada Lovelace, Ampere, etc.) |
| `pci_vendor_id` | PCI vendor hex (e.g., `10de` = NVIDIA, `1002` = AMD) |
| `pci_device_id` | PCI device hex (e.g., `2204` for RTX 3090) |
| `compute_units` | CU / SM / EU count (architecture-neutral name) |
| `memory_bandwidth_gbps` | Peak memory bandwidth |
| `tdp_w` | Thermal design power (spec, not measured) |
| `tflops_fp16` | Spec FP16 throughput |
| `tflops_fp32` | Spec FP32 throughput |
| `tflops_bf16` | Spec BF16 throughput (null if not supported) |

`pci_vendor_id` + `pci_device_id` together form the PCI ID used for GPU identification and seed matching — this is the canonical dedup key for a GPU model. Are there other GPU fields you want to include (e.g., memory type — GDDR6X vs HBM, ROCm/CUDA compute capability version)?

— Bastion

---

## #22 comment 4534370703 — @gissf1 — 2026-05-25T12:48:00Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534370703>

RE: Q2: what if the bios_notes field were key-value like JSON with some default values assumed for missing keys.  Then we could include some of those in the pre-hashed UUID string, based on some kind of filter for keys.

UUID migration table: proposed schema comments:
- if anything, the textual reason should be a FK and part of a schema version update in some other lookup table (maybe a seeded table?), the text should not be repeated as part of every migration record that is affected by the change.
- it makes sense for this table to be useful beyond just hardware UUIDs, but to any UUIDs that may migrate as we refine the schema and such.  That changes several of the fields applicability, but overall I think the design still covers most use cases.
- the table should be named "uuid_migrations" or similar, rather than "hardware_id_migrations" to reflect this more generic usage.
- I was thinking the PK should be the old id, but then realized: what if adding the field means there are multiple possible new ids?  Perhaps we just listing the old id is enough, but we may need more flexibility in how to migrate the record.  For example, maybe a schema change adds a new ENUM field with 3 possible values, each resulting in a new UUID for the record; that would presume that the UUID needs to be recomputed with the same approach used for a new/imported record.  so perhaps the new id is not necessary, but we may need to know how to migrate other references later too.  Let's discuss this more.

RE: Q3 GPU fields:
- agreed on PCI fields in both the table and the PK, and we should also probably include both the PCI vendor/device IDs, as well as the subsystem vendor/device IDs.  This would help identify GPU cards from manufacturers with special configurations (such as overclocked memory, modified GPU clocks from reference designs, or different thermal dissipation methods).
- gpu_architecture I feel is generally unnecessary entirely, but perhaps a worthwhile FK in the table as a lookup table for categorization on the results page?
- Part of the purpose of our tool is to benchmark GPUs as well as LLM models.  I'm not sure how much we should rely on tflops numbers from manufacturers.  I'm open to including them, but perhaps we should have a flag to determine if the numbers are manufacturer supplied. Then once we reach some threshold of validated results on a certain GPU, we can determine an actual measured tflops rating for various modes (fp16, fp32, etc).
- we already have a peak_tflops_fp16 field, don't we?  wouldn't tflops_fp16 duplicate that?  I don't quite like the idea of listing out every tflops format as a new field in the gpu hardware id, perhaps we can have a lookup table for the various formats, and then the gpu entry can have a value for each of them in some more packed format?
- TDP is probably a useful metric to include, but how reliable is it?  how closely related is TDP to actual Wattage?
- none of the GPU performance metrics should be used in the UUID calculation.
- memory_bandwidth_gbps, isn't this the same as memory_bandwidth_peak_gb_s?  why not use gb_s instead of gbps - which is more clear GB or Gbits?  I'm unconcerned as to which is used, but we should be consistent in our unit selection.
- compute_units: I don't understand what this is from your explanation, so please explain more so I can give feedback on it.
- what is bf16?
- let's continue to discuss which fields to include in the UUID as you did for system_hardware_id above.
- vram_type should probably be a seeded lookup table too.  I think some level of memory type and should be part of the UUID.

---

## #22 comment 4534654458 — @AlbinoGeek — 2026-05-25T13:31:12Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534654458>

**Bastion response — 251325ZMAY26**

@gissf1 — addressing all points from your last comment.

---

**bios_notes as structured key-value JSON**

Good refinement. `bios_notes` changes from free-form string to a JSON object with typed keys and assumed defaults for absent keys. Example:

```json
{
  "SMT_enabled": true,
  "resizable_bar": false,
  "above_4g_decoding": true,
  "fTPM_enabled": true,
  "secure_boot": true
}
```

UUID hash inclusion: a whitelist of `bios_notes` keys is selected and concatenated into the UUID input in stable sorted order. Keys not in the whitelist are stored but excluded from hashing. Initial whitelist proposal:

| Key | UUID input? | Reason |
|-----|------------|--------|
| `SMT_enabled` | YES | directly affects thread count and throughput |
| `resizable_bar` | YES | GPU VRAM access pattern, affects inference throughput |
| `above_4g_decoding` | YES | prerequisite for Resizable BAR |
| `fTPM_enabled` | NO | security feature, not performance-affecting |
| `secure_boot` | NO | boot integrity, not runtime performance |

New BIOS keys can be added to the whitelist by schema version increment; a migration record is created for existing hardware entries that need re-probing for the new field. Does this whitelist look right?

---

**UUID migration table — revised schema**

Three concerns addressed:

**1. Migration reason as FK to a schema version table**

Agreed. `migration_reason` text removed. Replaced with FK to `hardware_schema_versions`:

```sql
CREATE TABLE hardware_schema_versions (
    schema_version     INTEGER PRIMARY KEY,
    description        TEXT NOT NULL,
    uuid_fields_added  TEXT[],
    uuid_fields_removed TEXT[],
    released_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Migration records reference `schema_version` — all records affected by the same schema change share one version row.

**2. Generic uuid_migrations table**

Agreed. Renamed and generalized:

```sql
CREATE TABLE uuid_migrations (
    entity_type        TEXT NOT NULL,  -- 'system_hardware' | 'gpu_hardware' | etc.
    old_id             UUID NOT NULL,
    schema_version     INTEGER NOT NULL REFERENCES hardware_schema_versions(schema_version),
    verification_field TEXT,
    verification_source TEXT,          -- 'disk_file' | 'hardware_probe' | 'manual'
    verified           BOOLEAN NOT NULL DEFAULT FALSE,
    migrated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**3. new_id removed from PK — open question**

You correctly identify the problem: a schema change that adds an ENUM field with multiple possible values means the new UUID cannot be predetermined — it requires re-probing or reading the original disk file. Proposed resolution: `new_id` is not stored (it is derived, not fixed). When ingest encounters an `old_id` not found in the primary table:

1. Check `uuid_migrations` for `old_id` + `entity_type`
2. If found: read disk file for the verification field; compute new UUID from the full current field set; use it
3. If disk file lacks the new field and `verification_source = 'hardware_probe'`: flag for operator re-probe
4. If `verified = false`: log, continue with old ID as provisional

Does this work for your intent, or do you want a stored (but nullable) `new_id` column for caching once first resolution occurs?

---

**GPU fields — all points addressed**

**PCI subsystem IDs:** Added — `pci_subsystem_vendor_id` + `pci_subsystem_device_id`. Captures board-level OEM differentiation (non-reference memory configs, modified clocks, different thermal designs).

**gpu_architecture as FK:** Agreed — FK to seeded `gpu_architectures` lookup table, not raw string. Used for results-page categorization only; not in UUID.

**tflops — manufacturer flag + lookup table:** Agreed on both:
- Add `tflops_source` field: `'manufacturer'` | `'measured'`. No mixed values in one field.
- Replace per-format columns with a related table: `gpu_tflops { gpu_hardware_id, compute_format_id, tflops_value, tflops_source }`. `compute_format_id` is FK to seeded `compute_formats` table (fp16, fp32, bf16, int8, fp8, etc.). Extensible without schema migration per new format.
- `peak_tflops_fp16` already exists — `tflops_fp16` removed from proposal as duplicate.

**GPU performance metrics not in UUID:** Confirmed — tflops, bandwidth, TDP all excluded.

**memory_bandwidth naming:** Standardizing to `_gb_s` suffix throughout: `memory_bandwidth_peak_gb_s` (GB/s, not Gbits/s). Applied consistently.

**compute_units — clarification:** CU/SM/EU is the architecture-neutral count of parallel execution clusters: CUDA Streaming Multiprocessors (NVIDIA), Compute Units (AMD), Execution Units (Intel). Primary determinant of raw parallel throughput independent of clock. If the name is ambiguous, can rename to `parallel_execution_units`.

**bf16 — clarification:** Brain Float 16 — a 16-bit float format with the same exponent range as float32 but reduced mantissa. Preferred for LLM inference on hardware with native bf16 support (NVIDIA Ampere+, AMD RDNA3+, Apple Silicon) because it handles the wide dynamic range of model weights without the overflow risk of fp16.

**vram_type as seeded lookup + UUID:** Agreed — FK to seeded `vram_types` table (GDDR6, GDDR6X, HBM2e, HBM3, LPDDR5X, etc.). Included in UUID hash — memory type materially affects bandwidth and inference latency profile.

**Updated GPU UUID field list proposal:**

| Field | UUID input? | Notes |
|-------|------------|-------|
| `pci_vendor_id` | YES | identity |
| `pci_device_id` | YES | identity |
| `pci_subsystem_vendor_id` | YES | board-level identity |
| `pci_subsystem_device_id` | YES | board-level identity |
| `vram_mb` | YES | |
| `vram_type` (FK) | YES | |
| `slot_index` | NO — recommended | see below |
| `gpu_architecture` (FK) | NO | categorization, implied by PCI IDs |
| `compute_units` | NO | implied by PCI IDs + arch |
| `memory_bandwidth_peak_gb_s` | NO | performance metric |
| `tdp_w` | NO | spec metadata |
| tflops (all) | NO | performance metrics |

**slot_index and UUID:** Recommend excluding `slot_index` from UUID. If a user moves a GPU to a different PCIe slot the hardware is the same; the UUID should not change. Slot position is tracked in `system_gpu_link` (the join table between `system_hardware` and `gpu_hardware`) where it belongs. On platforms where x8 vs x16 lane width materially affects throughput, that difference will surface in the measured results rather than the hardware identity. Confirm?

— Bastion

---

