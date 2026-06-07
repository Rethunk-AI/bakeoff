# Topic: tflops_source

Consolidated chat history (chronological, ascending comment-id). 10 entries. Verbatim quotes; attribution in each header. **#38 thread excluded from this variant.**

## Source entries (provenance TOC)

Entries used to build this topic and its summary. (Not migrated into `topic_summary/`; audit reference only.)

| ticket | entry | author | posted | url |
|---|---|---|---|---|
| #8 | comment 4447581353 | @AlbinoGeek | 2026-05-14T04:39:42Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4447581353> |
| #18 | issue body | @AlbinoGeek | 2026-05-22T19:52:20Z | <https://github.com/Rethunk-AI/bakeoff/issues/18> |
| #22 | comment 4533933819 | @AlbinoGeek | 2026-05-25T11:35:33Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533933819> |
| #22 | comment 4534370703 | @gissf1 | 2026-05-25T12:48:00Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534370703> |
| #22 | comment 4534654458 | @AlbinoGeek | 2026-05-25T13:31:12Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534654458> |
| #22 | comment 4550408728 | @gissf1 | 2026-05-27T01:19:24Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728> |
| #22 | comment 4552726518 | @AlbinoGeek | 2026-05-27T08:15:50Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4552726518> |
| #22 | comment 4553530231 | @gissf1 | 2026-05-27T10:04:55Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553530231> |
| #22 | comment 4553785030 | @AlbinoGeek | 2026-05-27T10:41:42Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553785030> |
| #22 | comment 4554217644 | @gissf1 | 2026-05-27T11:46:49Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4554217644> |

---

## Consolidated thread

### #8 comment 4447581353 — @AlbinoGeek — 2026-05-14T04:39:42Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4447581353>

**Bastion response — 140449ZMAY26**

Commander,

Valid critique. "Tokens" is an abstraction over highly variable computation depending on tokenizer, context length, and model architecture. Not a reliable hardware-correlated metric across diverse model families. Addressing your specific proposals:

---

**RDTSC / hardware counter approach**

Feasible on x86. `rdtscp` (serializing variant) gives cycle-accurate timestamps bracketing execution with no perf hooks and near-zero overhead. On Linux, accessible via `CLOCK_MONOTONIC_RAW` syscall (nanosecond precision) or direct inline assembly. For local inference runners (llama.cpp, Ollama, vllm), we can record counter-start before inference call and counter-end on final token receipt. The delta captures CPU cycles consumed by the inference process — not the GPU directly, but measurable.

For GPU-side cycle counts: CUDA provides `cudaEvent_t` elapsed time (GPU wall time in milliseconds) via `cudaEventRecord` + `cudaEventElapsedTime`. No invasive hooks — it's a first-class CUDA API. Equivalent for ROCm: `hipEventElapsedTime`. This gives GPU-side execution time with ~microsecond precision.

Both are non-invasive and read existing hardware counters. No tracing hooks, no instrumentation overhead.

---

**Concrete hardware-correlated metrics**

You are correct that we can compute better proxies than raw token counts. Proposed additions to the measurement schema:

| Metric | Source | Notes |
|--------|--------|-------|
| `cpu_cycles_elapsed` | RDTSC delta (rdtscp start/end bracketing inference) | x86 only; arm64 has `PMCCNTR_EL0` equivalent |
| `gpu_wall_time_ms` | CUDA Event API (or ROCm equivalent) | GPU-side execution time; excludes CPU-GPU transfer overhead |
| `peak_vram_mb` | NVML `nvmlDeviceGetMemoryInfo` sampled during inference | Reflects actual memory pressure |
| `gpu_sm_utilization_pct` | NVML `nvmlDeviceGetUtilizationRates` sampled 10Hz | Average SM utilization during inference window |
| `memory_bandwidth_gb_per_s` | Theoretical: computed from VRAM clock × bus width; actual: requires vendor tools | Computed estimate is sufficient for cross-hardware normalization |
| `theoretical_tflops_utilization` | `tokens_completion / gpu_wall_time_s / model_flops_per_token / gpu_peak_tflops` | Model FLOPs per forward pass estimated from architecture (layers × hidden dim² × 2) |

The `theoretical_tflops_utilization` figure is the most hardware-invariant: it expresses "what fraction of this GPU's theoretical compute capacity was the model consuming?" This normalizes across hardware generations.

---

**Cross-architecture estimation**

For architecture-variant models (MoE vs dense, GQA vs MHA, speculative decoding), raw token counts diverge meaningfully from compute consumed. Better normalization axis: FLOPs per forward pass, which is estimable from published architecture specs:

```
FLOPs per token ≈ 2 × num_params (dense)
FLOPs per token ≈ 2 × num_active_params (MoE)
```

This lets us express "effective compute per output token" normalized to hardware TFLOPs — comparable across diverse architectures even when tokenizers differ.

---

**Schema additions (revised):**

```json
{
  "cpu_cycles_elapsed": 14200000000,
  "gpu_wall_time_ms": 3240,
  "peak_vram_mb": 6144,
  "gpu_sm_utilization_pct": 87.3,
  "flops_per_token_theoretical": 8000000000,
  "tflops_utilization_pct": 71.2,
  "tokens_per_second": 39.5
}
```

These fields are optional (not all runners expose all counters) — schema should mark them nullable with source noted.

Standing by for direction on which subset to prioritize for initial implementation.

— Bastion

---

### issue #18 body — @AlbinoGeek — 2026-05-22T19:52:20Z
<https://github.com/Rethunk-AI/bakeoff/issues/18>

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

---

### #22 comment 4533933819 — @AlbinoGeek — 2026-05-25T11:35:33Z
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

### #22 comment 4534370703 — @gissf1 — 2026-05-25T12:48:00Z
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

### #22 comment 4534654458 — @AlbinoGeek — 2026-05-25T13:31:12Z
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

### #22 comment 4550408728 — @gissf1 — 2026-05-27T01:19:24Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728>

RE: bios_notes:
- I don't think "above 4G decoding" needs to be in the UUID input
- I don't think "resizable bar" is the right thing to be in the UUID, but rather the BAR size we use probably should be.
- I don't think SMT enabled/disabled status needs to be in the UUID; I think CPU thread count and CPU model should make that clear.
- we should track BIOS settings for other bus clock rates, multipliers, widths, etc.; anything not already tracked in (or encompassed by) another field that could affect transfer rates and overall performance of our tests.  For example, we are already going to track PCIe rates to the GPU in the join table, so there is no reason to track those settings.  Similarly, we are already tracking CPU clock speed, so there is no need to track an FSB clock and multiplier.
- does it make sense to separate the motherboard from the CPU into separate tables to reduce redundancy?  I'm thinking not since our tests will be GPU focused, but maybe?  If so, we would probably have to do something similar to the link table where we list linking capabilities and actual link rates and CPU interface becomes just another interface in the interface_types list.
- How would we handle multi-processor systems?  Perhaps this isn't an urgent enough question to address now though, since we have more important questions to focus on.
- Now that I'm thinking about it, just like we did with PCIe link interface information, maybe we should create a lookup table for various hardware adjustments and settings, like BIOS settings.  I'm thinking it would be nice if each motherboard can have just 1 base hardware record that is shared by all the setting variants that board can have, so only the settings change and without duplicating the larger text records. Then again, maybe I'm overthinking this for the current phase.

RE: hardware_schema_versions:
- this should be generalized as well to be used by any schema changes to any table
- therefore it should also be renamed "schema_versions"
- schema_version_id makes sense since this table will be seeded and serial
- instead of listing UUID fields added and removed, I think it makes more sense to have a text field that can be used to describe the changes in an machine readable format that can describe any number of changes.  UUID changes are important, but also adding tables or changes in how data is encoded somewhere might be a more complex change than just adding/removing a field from a UUID.  Is there some standard modeling format for DB schema changes that we can build from here?

RE: uuid_migrations:
- why do we have verification field and source?
- what is the purpose of the "verified" field?
- do we need entity_type?  I feel like that should be a FK to a lookup table rather than a text field.  The lookup table would be indexed and basically list the tables in the schema, along with a schema_version_id FK to the schema change that added the table and another to the schema_version_id that last modified it.  Maybe name the table schema_tables?
- in my mind, the purpose of this table is to allow us to recognize old UUIDs (from any table) and be able to migrate them to the new UUID with less effort.  I feel this is important if we plan to have (hopefully) millions of hardware submissions and need to track that data for a long time without data loss and across many generations of hardware changes.

RE: new_id removed from PK question: I think that proposition generally sounds like the correct process, but, as mentioned above, I question the necessity of the entity_type field to do so.  Perhaps we can include a method within the schema_versions table to explain where to get the new field's information, if it's possible.  Otherwise, the default (when information is not present) would be to log and fall back to operator flagging.  Ideally they would be able to implement some method of automation though because with thousands of hardware records, manually resolving changes by hand would be unmanageable at scale.  The new_id field is definitely not wanted in the table though.

RE: GPU fields:
- tflops_source should probably be another FK to a lookup table, which references a list of provided sources with contact information and/or public URLs when using manufacturer's values.  ID 0 (or 1 if zero is not available) in that lookup table should be the "unknown" value.  When we are ready to provide these metrics, we can add ourselves as a tflops_source in this table.  The records in the sources table should probably have some way to mangle the URL when needed to reference multiple GPUs from a single URL with (for example) the model number or architecture mapped as part of the URL.
- I don't think compute_units is a necessary field in the database; it will be implied by whatever manufacturer and architecture the GPU is based on, and possibly the software stack in use, but most importantly it doesn't matter to us what they want to call their units.  Unless you can explain why it's necessary or useful to us, I would say drop it.
- bf16 - I don't see why rescaling a float32 to 16 bit is much different than a bf16, but thank you for explaining that, and for the sake of completeness, it makes sense to include it.
- slot_index and UUID - while I agree that moving slots doesn't change the GPU itself, and for now I'm ok with not including the slot_index field, if we reach a point where hardware can distribute inference across multiple GPUs, we may need to know the list of GPUs (multiple/plural) used.

---

### #22 comment 4552726518 — @AlbinoGeek — 2026-05-27T08:15:50Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4552726518>

**Bastion response — 270812ZMAY26**

@gissf1 — addressing all points from your latest comment.

---

**bios_notes UUID whitelist — revised**

| Key | UUID input? | Change | Reason |
|-----|------------|--------|--------|
| `SMT_enabled` | NO | REMOVED | Already expressed by `cpu_threads`; no independent signal |
| `resizable_bar` (bool) | NO | REPLACED | Wrong granularity |
| `bar_size_mb` | YES | NEW | Actual BAR aperture size is what affects GPU VRAM access patterns; a machine with BAR at 256 MB vs 16 GB has meaningfully different throughput |
| `above_4g_decoding` | NO | REMOVED per your direction | |

On additional bus settings — candidates not already covered by existing fields (PCIe rates in join table, CPU clocks and memory clock/channels/interleave already tracked):

| Candidate key | UUID input? | Notes |
|---------------|------------|-------|
| `pcie_gen_override` | YES | BIOS-forced PCIe Gen 3 on Gen 4 platform affects throughput in ways the join table's measured rate won't capture until probed |
| `power_limit_w` | YES | Enforced platform TDP cap directly changes benchmark headroom |
| `iommu_enabled` | NO | Security boundary, not performance |

Open question: are there motherboard-specific bus width overrides (e.g., BIOS forcing x8 for a slot that supports x16) you want here, or do you consider those fully captured by the join table's measured link width?

---

**Motherboard/CPU table separation — deferred**

Agreed: single `system_hardware` record (CPU + motherboard + RAM) is the right shape for Phase 1. Tests are GPU-focused; the system record is the environment anchor, not a first-class benchmark subject. Multi-processor systems deferred as well — flag when first encountered in results data.

**BIOS settings as lookup table (EAV approach)**

Not overthinking it — the idea is valid and has a name: Entity-Attribute-Value (EAV) for hardware configuration. Bastion's recommendation: **Phase 2 enhancement, not Phase 1 scope.** The structured `bios_notes` JSON + UUID whitelist achieves the deduplication goal for Phase 1 without EAV join complexity. When hardware records accumulate at scale and BIOS variant proliferation becomes visible in the data, that's the signal to gate Phase 2 EAV work. Mark as Phase 2 candidate.

---

**schema_versions — generalized**

All three points accepted:

- Rename to `schema_versions`, `schema_version_id` as serial PK
- Generic — applies to any table, not just hardware UUIDs
- Machine-readable change format: recommendation is **JSONB `changes` column with a defined internal vocabulary**, not adoption of an external DSL (Liquibase/sqitch solve a different problem — DDL, not semantic field-set changes relevant to UUID recomputation). Proposed vocabulary:

```json
{
  "uuid_fields_added": ["cpu_base_clock_mhz"],
  "uuid_fields_removed": [],
  "tables_added": ["gpu_tflops"],
  "tables_modified": [{"table": "system_hardware", "columns_added": ["power_limit_w"]}],
  "encoding_changes": []
}
```

Keys are optional — omit when empty. Extensible without altering the table definition. Is there a specific change type you're anticipating that doesn't fit this vocabulary?

---

**uuid_migrations — explaining verification_field, verification_source, verified**

**Why verification_field and verification_source?**
When ingest encounters an old UUID it doesn't recognize, it needs to know: (a) which field in the disk file holds the value that produces the corrected UUID, and (b) where that value can be found. Without those two pointers, automation cannot recompute the new UUID — it falls back to operator flagging for every record. With them, the migration pipeline self-resolves disk-file-resolvable migrations automatically and only escalates records where the value isn't in the disk file (requiring a hardware re-probe).

**Why verified?**
Prevents silent data corruption. A migration entry exists when the schema change is declared — but the new UUID hasn't been confirmed for that specific hardware record yet. `verified = false` = "migration logic says the old ID maps here, but resolution hasn't been confirmed." `verified = true` = pipeline confirmed the computed new UUID resolves to a valid record. Without the flag, a bad migration silently routes results to the wrong hardware record.

**entity_type as FK to schema_tables — accepted**

Agreed. Proposed `schema_tables`:

```sql
CREATE TABLE schema_tables (
    table_id              SERIAL PRIMARY KEY,
    table_name            TEXT NOT NULL UNIQUE,
    added_schema_version  INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    last_modified_version INTEGER REFERENCES schema_versions(schema_version_id),
    uses_uuid_identity    BOOLEAN NOT NULL DEFAULT FALSE
);
```

`uses_uuid_identity` flags tables where UUID migration tracking applies. `uuid_migrations` then FKs to `schema_tables.table_id` instead of carrying raw text. Seeded and extended at each schema version that adds a new table.

`new_id` — confirmed removed. Resolution is always computed at migration time; storing a pre-computed value creates a stale-cache problem when the UUID field set evolves again.

On automation at scale: agreed that manual resolution for thousands of records is unmanageable. The proposed pipeline (verification_field + verification_source) is the automation hook — it tells the resolver exactly where to get the missing value. For schema changes that add a new ENUM field with multiple possible values, `verification_source = 'hardware_probe'` flags those for automated re-probe rather than operator intervention. The `schema_versions.changes` JSONB can carry a `resolution_strategy` key to tell the pipeline which path to take per change type.

---

**GPU fields — all points**

**tflops_source as FK to lookup table:** Accepted.

```sql
CREATE TABLE tflops_sources (
    source_id    SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    contact_url  TEXT,
    url_template TEXT   -- e.g. 'https://manufacturer.com/gpu/{pci_device_id}/specs'
);
-- Seed ID 1 = 'unknown/unverified'
-- Seed ID 2 = 'Rethunk measured' (added when we ship our own benchmarks)
```

URL template supports `{pci_device_id}`, `{pci_vendor_id}`, `{gpu_model}` substitution tokens — no hardcoded GPU-specific entries needed in the source table itself.

**compute_units — dropped.** Agreed: implied by PCI IDs + architecture. No independent signal for our test surface.

**bf16 — confirmed included** in `gpu_tflops` via `compute_formats` FK.

**slot_index and multi-GPU inference:** Noted as carry-forward. `system_gpu_link` already carries `slot_index` — when multi-GPU inference support lands, the run file references `hardware_id` + a list of `slot_index` values from that join table. No change to UUID scheme required; the extension point is already present.

---

**Open items for your confirmation:**

1. `bar_size_mb` as UUID input instead of `resizable_bar` bool — correct replacement?
2. `pcie_gen_override` + `power_limit_w` as additional UUID-input bios_notes keys — adjust the list?
3. JSONB `changes` vocabulary on `schema_versions` — any change type not covered by the proposed keys?
4. `schema_tables.uses_uuid_identity` flag for UUID-migration-eligible tables — adequate, or different discriminator needed?

— Bastion

---

### #22 comment 4553530231 — @gissf1 — 2026-05-27T10:04:55Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553530231>

RE: bios_notes UUID whitelist:
- pcie_gen_override - wouldn't this show up as if the board is limiting the GPU's PCIe rate?  If so, I don't see a need to include it since it's encompassed in the interface link table.  In any case, I don't think it's something to include in the GPU UUID.  I'm not sure about how it fits into the system_hardware table though.
- power_limit_w	- this is surely useful to store, but I'm not sure it's needed in the UUID for phase 1.
- Your "Open question": I believe the join table's measured link width should fully encompass any motherboard specific bus width overrides.  Do you have any example hardware situations or circumstances that you think it might not appropriately represent?

RE: BIOS settings as lookup table (EAV approach):  Agreed, phase 2.

RE: schema_versions: a change I anticipate that doesn't fit the example would be adding or modifying a field where data needs to be mapped from some other place or data:
- another field
- text blob containing output from commands like `lspci`, `cat /proc/cpuinfo`, or `dmidecode`
- a substring of another source
- recomputing a field value with different mapping functions
- changing the sources
- changing the hash function (for example, maybe we will change from sha256 to sha512 hashing)
- adding a new field based on a mapping of an old field
- a combination or manipulation of the above (for example, mapping a substring of another field to a set of values, which are then concatenated with another substring from another field.)

I feel like this whole thing almost needs to be a programming language.  Would it make sense to use something like YAML's parser (Jinja2?) to acquire data and specify the details?  Maybe AWK or LUA?  I feel like JavaScript might be too insecure for this use case, but do you have another suggestion that is effective to store in a database to reference the table data, processing efficient and limits security risk?  or maybe we don't need to store it in the database if we can find a way to store it externally in files that isn't going to become a nightmarish number of files over time.

RE: uuid_migrations: I don't think verification_field and verification_source give you what you need.  As described above, we don't know ahead of time what data we might need to compute the field data, and it may be multiple sources or a complex interaction.  I think the correct action is to use the calculation programming code as described above to generate the new field values from any existing data, or explain how to acquire that data externally for automation to retrieve it.  We also may add a new field that is not present in the existing data.

I'm ok with the verified field to indicate that we have confirmed the UUID migration record is valid to begin migrating records automatically, rather than simply flagging them.  Perhaps you can pick a more descriptive name though?  I'm thinking something like `is_migrating`, `allow_auto_migration`, `begin_migrating`, or `migration_started`, but I'm not sold on any of those, so I'm open to more suggestions.

RE: proposed schema_tables: Approved.

RE: schema_tables verification_source / resolution_strategy: I still don't think that is the right approach.  I think the complexity of the changes could require more fields and manipulation than the simple mapping you are implying in your example.

RE: GPU fields: tflops_sources:
- contact_url: I assume this could be things like `mailto:` or `tel:` links, but what if we want to store multiple contact methods (for example, mailing address + email + phone number + website contact form address) for a single source's contact?  or, similarly, what if we have multiple contacts for a single source (for example: a manager, an engineer, and a project manager)?
- url_template: what if this needs something like gpu_architecture?  or some manipulation like capitalization changes, substrings, and/or a mapping to certain slugs?  I feel like this is related to the uuid_migrations scripting language requirements.

RE: Open items for your confirmation:
- bar_size_mb - Confirmed.
- schema_tables.uses_uuid_identity flag: I think this is adequate for determining which tables have UUIDs and which have another form of PK.
- all others, see notes above.

---

### #22 comment 4553785030 — @AlbinoGeek — 2026-05-27T10:41:42Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553785030>

**Bastion response — 271035ZMAY26**

@gissf1 — addressing all points in order.

---

**bios_notes UUID whitelist — revised**

On `pcie_gen_override`: Agreed. The join table's measured link width captures the actual throughput impact at probe time — a BIOS-forced Gen 3 on a Gen 4 platform shows as x16 Gen 3 in the measured fields. No independent UUID signal. Remove from UUID; store in `bios_notes` as metadata only.

On `power_limit_w`: Agreed — useful metadata, defer from UUID for Phase 1. Store in `bios_notes`, not UUID input.

On motherboard bus width overrides: Agreed — join table covers it. Open question closed.

Updated `bios_notes` UUID whitelist:

| Key | UUID input? | Notes |
|-----|------------|-------|
| `bar_size_mb` | YES | Actual BAR aperture affects VRAM access throughput |
| `above_4g_decoding` | NO | — |
| `SMT_enabled` | NO | Expressed by `cpu_threads` |
| `pcie_gen_override` | NO | Metadata only; covered by measured link width |
| `power_limit_w` | NO | Metadata only; Phase 1 deferral |
| `iommu_enabled` | NO | Security boundary |

---

**schema_versions — scripting language for complex migrations**

Agreed. A fixed JSONB vocabulary is insufficient for the cases you describe (cross-field mapping, substring extraction, hash function changes, external data sources). A scripting language is the right call.

Recommendation: **Lua** (Go binding: [`gopher-lua`](https://github.com/yuin/gopher-lua)).

Rationale:
- Embeds in Go with no native/CGO dependency
- Sandboxable by default — remove `io`, `os`, `debug` libs at load time
- Expressive enough for field computation, substring extraction, lookup tables, and mapping functions
- Stores as TEXT in the DB — no external file proliferation
- Readable; deterministic; minimal attack surface compared to JS

Jinja2: Python-specific dependency, not viable. AWK: good for text parsing, poor at structured data mapping. JavaScript: agreed, V8 embedding risk too large. Lua hits the right balance.

Alternative: **CEL (Common Expression Language)** — designed for safe expression evaluation in infrastructure (Kubernetes admission policies). More restrictive than Lua (no loops, no mutation), which rules it out for the complex cases you describe. Phase 2 candidate if Lua proves heavier than needed.

On storage: DB `TEXT` column preferred over external files. The migration script must be co-located with the `schema_version` record it describes — external files introduce version drift and file-count problems you already identified. One `migration_script TEXT` column per schema_versions row solves this cleanly.

---

**uuid_migrations — revised approach**

Accepted. `verification_field` and `verification_source` are replaced by a general `resolution_script` (Lua). Revised definition:

```sql
CREATE TABLE uuid_migrations (
    migration_id        SERIAL PRIMARY KEY,
    old_uuid            TEXT NOT NULL,
    entity_type         INTEGER NOT NULL REFERENCES schema_tables(table_id),
    from_version        INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    to_version          INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    resolution_script   TEXT,         -- Lua; receives current record, returns new field values
    allow_auto_migrate  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

`resolution_script`: NULL when the migration is a simple field rename (handled by `schema_versions.changes` JSONB alone). Non-null for complex transformations. The script receives the full current record and returns the new field values needed for UUID recomputation — it can pull from any field, substring, mapping table, or external probe instruction embedded in the script body.

On `allow_auto_migrate`: this name is the clearest of your candidates. `is_migrating` implies in-progress state. `migration_started` implies past. `begin_migrating` is imperative-voice. `allow_auto_migrate` is a gate: "confirmed valid, automation may proceed." Bastion recommendation unless you prefer one of the others.

---

**schema_tables — resolution_strategy removed**

Agreed. `resolution_strategy` on `schema_tables` was insufficient for the complexity you described. The `resolution_script` on `uuid_migrations` handles per-record execution. `schema_tables` remains a registry only — no execution path needed there.

---

**tflops_sources — contact_url and url_template**

On multiple contacts per source: Replace `contact_url TEXT` with JSONB for Phase 1:

```sql
contacts JSONB  -- [{type: "email", value: "..."}, {type: "url", value: "..."}, {type: "tel", value: "..."}, ...]
```

This handles multiple contact methods and multiple contacts per source without a join. Full contacts lookup table is Phase 2 when contact reuse across sources becomes relevant.

On `url_template` with complex substitutions: Agreed — simple `{pci_device_id}` tokens won't cover capitalization normalization, slug mappings, or cross-field construction. This is the same scripting language problem. Proposal:

- Keep `url_template TEXT` for simple token substitution (`{pci_device_id}`, `{pci_vendor_id}`, `{gpu_model}`)
- Add `url_script TEXT` (Lua) alongside it — non-null when complex URL construction is required
- Evaluation order: `url_script` takes precedence when non-null; fall back to `url_template`

This avoids breaking the simple case while accommodating complex slug mappings and architecture-based path construction.

---

**Open items — updated status**

1. `bar_size_mb` as UUID input: **Confirmed** (your confirmation above)
2. `pcie_gen_override`: Removed from UUID (metadata only) — per this comment
3. `power_limit_w`: Metadata only, not UUID — per this comment
4. `schema_versions` Lua scripting approach: awaiting your direction
5. `allow_auto_migrate` naming: awaiting your confirmation
6. `uuid_migrations resolution_script` approach: awaiting your confirmation
7. `schema_tables.uses_uuid_identity`: **Confirmed** (your confirmation above)
8. `tflops_sources contacts JSONB`: awaiting your confirmation
9. `tflops_sources url_script` column: awaiting your confirmation

— Bastion

---

### #22 comment 4554217644 — @gissf1 — 2026-05-27T11:46:49Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4554217644>

RE: pcie_gen_override: I don't think that it even matters enough to store the setting, because its effects will be visible in the PCIe interface link table and therefore the setting will not impact performance any differently than if the actual link rate were degraded for some other reason.

RE: schema_versions scripting language: Is LUA a better option than libraries like Gonja, Pongo2, sprig, or just using Go templates?  There may be others as well.  I don't mind LUA if it's clearly the best option, but I want to be sure before considering introducing a new language dependency into the project.  You mentioned CEL, I could see that also working given the right CEL environment setup and functions, but ideally we should aim to minimize "rolling our own" on this if something else does the job for us; This will maximize the available features and reduce development and testing time, not to mention having less bugs and security risks due to higher code visibility / exposure.

RE: schema_versions storage: migration_script TEXT field in table makes sense to me; approved.

RE: uuid_migrations:
- maybe we should just rename entity_type to table_id?  That may be more clear.
- resolution_script field should be in schema_versions, not here.
- I'm not sure we need both of from_version and to_version in here; maybe neither?
- I don't know if we need schema_versions.changes the way we described before if we have a full scripting language in resolution_script.  What was our latest definition of schema_versions so I can refine both of these as needed, since they are related.
- I think resolution_script should get a better name.  `migration_script`?  or maybe we need a `record_migration_script` and a `schema_migration_script`?  Maybe the schema_migration_script can include code to iterate over records to do database record upgrades as well as the actual schema changes?
- allow_auto_migrate: I think this name makes sense, but maybe just `allow_migration` is good enough?  I'm open to suggestions on this as well.

RE: tflops_sources:
- contacts: JSONB is acceptable for now.  I'm not sure if we need an independent contacts table, but we can defer that decision to phase 2, as this will work for now.
- url_template / url_script: I think we should have a simple "url_template" field which can hold a static URL reference for now, and anything more complicated can be filtered out later.  We can differentiate static strings by looking at their prefix against known URL types: ("http://", "https://", "file://", etc), and we can assume anything else is an expression in whatever scripting (or templating) language we decide to use in schema_versions (so we are consistent).

---

