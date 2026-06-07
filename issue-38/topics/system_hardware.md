# Topic: system_hardware

Consolidated chat history (chronological, ascending comment-id). 24 entries. Verbatim quotes; attribution in each header. **#38 thread excluded from this variant.**

## Source entries (provenance TOC)

Entries used to build this topic and its summary. (Not migrated into `topic_summary/`; audit reference only.)

| ticket | entry | author | posted | url |
|---|---|---|---|---|
| #8 | comment 4462460688 | @AlbinoGeek | 2026-05-15T18:52:20Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4462460688> |
| #8 | comment 4469626687 | @gissf1 | 2026-05-17T06:39:58Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4469626687> |
| #8 | comment 4469686066 | @AlbinoGeek | 2026-05-17T07:07:49Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4469686066> |
| #8 | comment 4470208621 | @AlbinoGeek | 2026-05-17T09:53:11Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4470208621> |
| #8 | comment 4474333279 | @AlbinoGeek | 2026-05-18T04:25:19Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4474333279> |
| #19 | issue body | @AlbinoGeek | 2026-05-22T19:52:23Z | <https://github.com/Rethunk-AI/bakeoff/issues/19> |
| #20 | issue body | @AlbinoGeek | 2026-05-22T19:52:27Z | <https://github.com/Rethunk-AI/bakeoff/issues/20> |
| #21 | issue body | @AlbinoGeek | 2026-05-22T19:52:31Z | <https://github.com/Rethunk-AI/bakeoff/issues/21> |
| #8 | comment 4522303209 | @AlbinoGeek | 2026-05-22T19:53:06Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4522303209> |
| #22 | issue body | @AlbinoGeek | 2026-05-25T09:53:53Z | <https://github.com/Rethunk-AI/bakeoff/issues/22> |
| #22 | comment 4533634217 | @gissf1 | 2026-05-25T10:44:24Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533634217> |
| #22 | comment 4533647917 | @AlbinoGeek | 2026-05-25T10:46:41Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533647917> |
| #22 | comment 4533894066 | @gissf1 | 2026-05-25T11:29:48Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533894066> |
| #22 | comment 4533933819 | @AlbinoGeek | 2026-05-25T11:35:33Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533933819> |
| #22 | comment 4534370703 | @gissf1 | 2026-05-25T12:48:00Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534370703> |
| #22 | comment 4534654458 | @AlbinoGeek | 2026-05-25T13:31:12Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4534654458> |
| #22 | comment 4550408728 | @gissf1 | 2026-05-27T01:19:24Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4550408728> |
| #22 | comment 4552726518 | @AlbinoGeek | 2026-05-27T08:15:50Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4552726518> |
| #22 | comment 4553530231 | @gissf1 | 2026-05-27T10:04:55Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553530231> |
| #22 | comment 4553785030 | @AlbinoGeek | 2026-05-27T10:41:42Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4553785030> |
| #22 | comment 4554570690 | @AlbinoGeek | 2026-05-27T12:38:26Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4554570690> |
| #22 | comment 4559373879 | @gissf1 | 2026-05-27T23:01:35Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4559373879> |
| #22 | comment 4559571210 | @AlbinoGeek | 2026-05-27T23:38:56Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4559571210> |
| #22 | comment 4568864985 | @AlbinoGeek | 2026-05-28T22:31:07Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4568864985> |

---

## Consolidated thread

### #8 comment 4462460688 — @AlbinoGeek — 2026-05-15T18:52:20Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4462460688>

**Bastion response — 151851ZMAY26**

**Wall clock — agreed.** `clock_gettime(CLOCK_MONOTONIC)` is simpler, universally available, and externally consistent. `gpu_wall_time_ms` removed from required fields. CUDA Event API remains an optional runner-side metric for implementations that support it, but wall clock is the primary latency measure.

**CPU time** — retained as optional. `getrusage()` gives `ru_utime` (user) + `ru_stime` (sys) at near-zero overhead. Useful signal for detecting driver-side CPU overhead. Not a primary metric.

**Revised required schema fields:**

```json
{
  "wall_clock_seconds": 12.4,
  "time_to_first_token_ms": 340,
  "tokens_per_second": 39.5,
  "peak_vram_mb": 6144,
  "gpu_sm_utilization_pct": 87.3,
  "tflops_utilization_pct": 71.2,
  "hardware_id": "rtx-4090-24gb"
}
```

Optional (runner-dependent): `cpu_cycles_elapsed`, `gpu_wall_time_ms`, `cpu_time_user_ms`, `cpu_time_sys_ms`.

`cost_usd` — agreed, derived value. Removed from storage. Computed at display time from hardware energy specs × run duration OR provider pricing × token counts, depending on inference mode (local vs API).

---

**Three-category schema structure — strong proposal**

Your categorization maps cleanly onto a normalized data model:

| Category | Table | Contents |
|----------|-------|----------|
| Invariant hardware specs | `hardware_specs` | peak_tflops, memory_bandwidth_gb_s, vram_capacity_mb, tdp_watts — stored once per hardware_id |
| Per-run hardware metrics | `run_hardware_metrics` | wall_clock_seconds, time_to_first_token_ms, cpu/gpu utilization, peak_vram_mb_used, tokens_per_second, cpu_cycles_elapsed |
| Per-run model metrics | `run_model_metrics` | test scores, pass/fail rates, tflops_utilization_pct, flops_per_token_theoretical |

Separation benefits: hardware specs recorded once, referenced across all runs on the same hardware. Reduces redundancy. Enables hardware-normalized leaderboard views without storing derived fields.

Cross-reference: bakeoff-results#12 will receive updated schema guidance reflecting this structure.

— Bastion

---

### #8 comment 4469626687 — @gissf1 — 2026-05-17T06:39:58Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4469626687>

Sounds good.  Does it make sense to include vram memory bus width and channel count too so we can calculate theoretical peak memory bandwidth?  or is there another way to acquire the peak memory bandwidth rate (MB/s)?

It may also matter how the GPU interfaces with the rest of the system, so whether thats PCI Express Gen 1 x1 link vs a Gen 3 x16 link is a huge differnence.  And there are other interfaces than PCIe also; I doubt AGP is a contender here, but I know there are networked GPUs, CXL interfaces, SXM, OCuLink, and I suppose even USB4, or Thunderbolt.

Maybe it makes sense to have a "GPU hardware" schema for the GPU itself and then properties in the system object schema for how it is interconnected and identifying critical parts of the rest of the system?  That could also allow us to separate the specs of the system (CPU, RAM, motherboard chipset, etc) from the GPU.  the GPU will most likely be the primary performance contributor in most cases, but perhaps someone is using an RTX 4090 24GB on an old raspberry pi 2, which will likely limit the performance of the 4090 to some extent.

---

### #8 comment 4469686066 — @AlbinoGeek — 2026-05-17T07:07:49Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4469686066>

**Bastion response — 170707ZMAY26**

Both proposals are sound. Addressing each, then a schema draft.

---

**VRAM bus width and channel count — feasible, recommended**

nvidia-smi does not expose bus width as a structured query field. However, `pynvml.nvmlDeviceGetMemoryBusWidth()` returns it directly (integer, bits). Channel count is derivable: a 320-bit bus is 10 × 32-bit channels; a 256-bit bus is 8 × 32-bit channels; HBM stacks use 1024 bits per stack × N stacks. No need to store channel count separately — derive it at display time from bus width and memory type.

**Theoretical peak bandwidth formula:**

```
# GDDR (DDR factor = 2):
bandwidth_GB_s = (memory_clock_mhz × 2 × memory_bus_width_bits / 8) / 1000

# HBM (effective clock, not DDR — factor = 1):
bandwidth_GB_s = (memory_clock_mhz × memory_bus_width_bits / 8) / 1000
```

With `memory_bus_width_bits` and the existing `clock_memory_mhz` + `vram_type` fields, peak bandwidth is fully derivable at display time. No need to store it — computed and presented in the leaderboard view.

---

**GPU interface — detectable where possible, declared elsewhere**

nvidia-smi exposes PCIe link properties directly:

```
--query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max
```

`link.gen.current` vs `link.gen.max` detects PCIe slot downgrade (a Gen 4 card in a Gen 3 slot is an immediate flag — measurable bottleneck). `width.current` vs `width.max` detects mechanical lane limitation (x16 card in x4 slot).

NVLink: detectable via `nvmlDeviceGetNvLinkState()` — presence and link count.

SXM form factor: identifiable from GPU name string (H100 SXM, A100 SXM5). No separate query needed — the PCI device ID lookup already implies form factor for known GPUs.

CXL, OCuLink, USB4/Thunderbolt: not programmatically detectable from within the driver. These require user declaration in the config. Recommendation: accept an optional `gpu_interface_override` field in the runner config for non-PCIe topologies.

---

**Schema split — strong agree**

Your three-entity model maps cleanly:

| Schema | Contents | Detection |
|--------|----------|-----------|
| `gpu_hardware` | Die-level intrinsics: compute, VRAM type/size/bus width, theoretical bandwidth, TDP, clock profile | Auto via nvidia-smi + pynvml |
| `system_hardware` | CPU model, RAM capacity/speed, chipset, OS, kernel | `/proc/cpuinfo`, `psutil`, `dmidecode` (optional) |
| `gpu_system_interface` | PCIe gen/lanes (current + max), NVLink count, form factor, interface override | Auto where detectable; declared for exotic interconnects |

The RPi example is the correct motivating case. If PCIe gen or lane count are bottlenecked, wall_clock_seconds will diverge from `tflops_utilization_pct` in a predictable direction. Separating interface from die specs lets us flag that divergence as a known artifact of the measurement configuration.

---

**Proposed `gpu_hardware` additions:**

```json
{
  "memory_bus_width_bits": 320,
  "memory_bandwidth_peak_gb_s": 1008.0
}
```

`memory_bandwidth_peak_gb_s` stored (not derived at display time) because the formula requires `vram_type` to select the DDR factor — storage is simpler than re-implementing that at every display callsite.

**Proposed `gpu_system_interface` record:**

```json
{
  "pcie_gen_max": 4,
  "pcie_gen_current": 4,
  "pcie_width_max": 16,
  "pcie_width_current": 16,
  "is_pcie_slot_limited": false,
  "nvlink_count": 0,
  "form_factor": "PCIe",
  "interface_override": null
}
```

`is_pcie_slot_limited` = true when `gen_current < gen_max` OR `width_current < width_max`. Flagged in leaderboard display.

---

Will open sub-issues for the additions above if directed. If NOMAD approves the three-schema split as the target structure, recommend updating bakeoff-results#12 to reflect that structure before any further implementation work lands.

— Bastion

---

### #8 comment 4470208621 — @AlbinoGeek — 2026-05-17T09:53:11Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4470208621>

**Bastion response — 171005ZMAY26**

Merge of `gpu_system_interface` into `system_hardware` is correct. Interface data describes the relationship between a GPU and a host — it belongs in the link, not either entity. Addressing the three relationship options:

---

**Option A — `system_hardware.gpu_hardware_id` (FK, one GPU per system)**

Pros: simplest schema, simplest queries.

Cons: assumes one GPU per system. Breaks for multi-GPU workstations, dual-slot setups, server nodes with 4× or 8× GPUs. Also conflates the GPU *model* with a specific physical instance in a specific machine — `gpu_hardware` is intended as a model-level record, not an instance record. Two RTX 4090 cards in different machines should reference the same `gpu_hardware` row.

Not recommended. Adequate only if bakeoff is restricted to single-GPU consumer hardware forever.

---

**Option B — `gpu_hardware.system_hardware_id` (FK, GPU belongs to system)**

Wrong direction semantically. A GPU model is not "owned by" a system. This structure prevents the same GPU model from being referenced by multiple systems — every community member with an RTX 4090 would need a separate `gpu_hardware` row with identical specs. Defeats normalization entirely.

Not recommended.

---

**Option C — Join table `system_gpu_link` (many-to-many)**

`system_hardware` and `gpu_hardware` remain independent entity tables. `system_gpu_link` carries both the relationship and the interface fields (PCIe gen/width, NVLink count, form factor, interface override). This is where the "merged" interface data lives — it belongs on the relationship, not on either entity.

```
system_hardware       system_gpu_link              gpu_hardware
───────────────       ───────────────────────────  ────────────────
system_id (PK)  ←──  system_hardware_id (FK)  ──→ gpu_hardware_id (PK)
cpu_model             gpu_hardware_id (FK)          gpu_name
ram_total_gb          pcie_gen_current              vram_total_mb
os                    pcie_gen_max                  memory_bus_width_bits
...                   pcie_width_current            peak_tflops
                      pcie_width_max                ...
                      is_pcie_slot_limited
                      nvlink_count
                      form_factor
                      interface_override
                      slot_index  ← slot 0, slot 1 for multi-GPU
```

Pros:
- Multiple GPUs per system (multi-GPU workstation, server rack)
- Multiple systems per GPU model (community benchmarking at scale — many users with RTX 4090)
- Interface data (PCIe gen/lanes) lives on the relationship where it belongs
- `run_hardware_metrics` records reference `system_gpu_link.id` — captures both which GPU *and* which system/slot produced the run

Cons: join required for any query spanning GPU and system data. Acceptable — this is a known, bounded query pattern.

---

**Recommendation: Option C**

Two reasons the join table is load-bearing here:

1. Community benchmarking is the explicit use case. Multiple submitters with the same GPU model is not hypothetical — it is the expected steady state once the tool is in broader use. Normalization prevents storing redundant `gpu_hardware` rows per user.

2. Interface data (PCIe bottleneck detection) is only meaningful on the relationship. A standalone `system_hardware` table that embeds PCIe link data implies a system has exactly one GPU slot, which is architecturally incorrect.

Final proposed table structure:
- `gpu_hardware` — GPU intrinsics (die, VRAM, compute, thermals)
- `system_hardware` — host intrinsics (CPU, RAM, chipset, OS, `system_id`)
- `system_gpu_link` — relationship + interface (PCIe, NVLink, form factor, slot index)
- `run_hardware_metrics` references `system_gpu_link.id` + `gpu_hardware_id` — captures full context per run

Standing by for direction. Will open sub-issues to implement once schema is ratified.

— Bastion

---

### #8 comment 4474333279 — @AlbinoGeek — 2026-05-18T04:25:19Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4474333279>

**Bastion response — 180502ZMAY26**

All three points addressed. Schema corrections applied, attribution logic defined, full dump follows.

---

**Field corrections — confirmed**

`gpu_native_interface_type_id` moves from `system_gpu_link` to `gpu_hardware`. This is the right home — it describes a fixed property of the GPU card, not the relationship between card and slot.

`system_gpu_link` now carries:
- `slot_native_interface_type_id` — the motherboard slot's rated spec (what that physical slot can do at maximum)
- `actual_interface_type_id` — negotiated running state, synchronized across slot and GPU by hardware at link-up

Rename from `slot_actual_interface_type_id` to `actual_interface_type_id` accepted. "Actual" is accurate — it belongs to neither side alone; it is what both sides agreed to.

---

**Limitation attribution logic**

Determining which device is constraining requires comparing both axes (generation and lane width) independently, not just `bandwidth_peak_gb_s`. Bandwidth collapses two independent variables into one number — two interfaces can share the same bandwidth but from different gen/width combinations, both degrading the other on different axes. Example: PCIe 3.0 x16 GPU in a PCIe 4.0 x8 slot — `actual` degrades to PCIe 3.0 x8; both devices constrain on one axis each.

Recommendation: add `pcie_gen` and `pcie_width` (or their generic equivalents — `transfer_rate` and `lane_count`) as nullable integer fields to `interface_type`. Non-PCIe interfaces leave these null; comparison is then on `bandwidth_peak_gb_s` only. For PCIe, both fields are populated and the comparison is per-axis.

Attribution rules for display:

| Condition | Description format |
|-----------|-------------------|
| `slot_native.bandwidth < gpu_native.bandwidth` | `"[actual] link (PCIe [slot_native] system slot limiting PCIe [gpu_native] GPU interface)"` |
| `gpu_native.bandwidth < slot_native.bandwidth` | `"[actual] link (PCIe [gpu_native] GPU interface limiting PCIe [slot_native] system slot)"` |
| Both constrain on different axes | `"[actual] link (limited by PCIe [slot_native] system slot and PCIe [gpu_native] GPU interface)"` |
| native = actual on both sides | No annotation — no degradation |

Display order: actual type first, attribution parenthetical second. Matches your stated preference.

For the mixed case (PCIe 3.0 x16 GPU in PCIe 4.0 x8 slot → actual PCIe 3.0 x8): `"PCIe 3.0 x8 link (limited by PCIe 4.0 x8 system slot and PCIe 3.0 x16 GPU interface)"` — both constraining devices named explicitly.

---

**Full schema dump**

```
interface_type
──────────────────────────────────────────────────────────
interface_type_id       SERIAL PRIMARY KEY
bandwidth_peak_gb_s     FLOAT NOT NULL
description             TEXT NOT NULL           -- "PCIe 4.0 x16", "SXM5", "Thunderbolt 4"
interface_family        TEXT NULLABLE           -- "PCIe", "SXM", "CXL", "NVLink", "USB", "OCuLink"
transfer_rate           INT NULLABLE            -- PCIe: GT/s per lane (e.g. 16 for Gen4); null for non-PCIe
lane_count              INT NULLABLE            -- PCIe: lane width (e.g. 16); null for non-PCIe
```

```
gpu_hardware
──────────────────────────────────────────────────────────
gpu_hardware_id                  SERIAL PRIMARY KEY
gpu_name                         TEXT NOT NULL
pci_device_id                    TEXT NULLABLE            -- "0x2684"
pci_sub_device_id                TEXT NULLABLE            -- board partner variant
vram_total_mb                    INT NULLABLE
vram_type                        TEXT NULLABLE            -- "GDDR6X", "HBM2e", etc.
memory_bus_width_bits            INT NULLABLE
memory_bandwidth_peak_gb_s       FLOAT NULLABLE
clock_memory_mhz                 INT NULLABLE
clock_graphics_boost_mhz         INT NULLABLE
peak_tflops_fp16                 FLOAT NULLABLE
tdp_watts                        INT NULLABLE
gpu_native_interface_type_id     INT NULLABLE REFERENCES interface_type  -- card's rated spec
```

```
system_hardware
──────────────────────────────────────────────────────────
system_hardware_id      SERIAL PRIMARY KEY
system_id               UUID NOT NULL UNIQUE     -- stable per-host UUID (generated at first run)
publisher_id            TEXT NOT NULL            -- submitting user/account
cpu_model               TEXT NULLABLE
ram_total_gb            FLOAT NULLABLE
os                      TEXT NULLABLE
kernel_version          TEXT NULLABLE
```

```
system_gpu_link
──────────────────────────────────────────────────────────
system_hardware_id              INT NOT NULL REFERENCES system_hardware
slot_index                      INT NOT NULL
PRIMARY KEY (system_hardware_id, slot_index)

gpu_hardware_id                 INT NOT NULL REFERENCES gpu_hardware
slot_native_interface_type_id   INT NULLABLE REFERENCES interface_type  -- slot's rated max
actual_interface_type_id        INT NULLABLE REFERENCES interface_type  -- negotiated running state

INDEX: (gpu_hardware_id)
```

```
run_hardware_metrics
──────────────────────────────────────────────────────────
run_id                          TEXT NOT NULL              -- references run record
system_hardware_id              INT NOT NULL
slot_index                      INT NOT NULL
FOREIGN KEY (system_hardware_id, slot_index) REFERENCES system_gpu_link

wall_clock_seconds              FLOAT NULLABLE
time_to_first_token_ms          FLOAT NULLABLE
tokens_per_second               FLOAT NULLABLE
peak_vram_mb                    FLOAT NULLABLE
gpu_sm_utilization_pct          FLOAT NULLABLE
tflops_utilization_pct          FLOAT NULLABLE
cpu_cycles_elapsed              BIGINT NULLABLE
cpu_time_user_ms                FLOAT NULLABLE
cpu_time_sys_ms                 FLOAT NULLABLE
gpu_wall_time_ms                FLOAT NULLABLE            -- optional, runner-dependent
```

```
run_model_metrics
──────────────────────────────────────────────────────────
run_id                          TEXT NOT NULL PRIMARY KEY
model_id                        TEXT NOT NULL
task_id                         TEXT NOT NULL
prompt_id                       TEXT NOT NULL
score                           FLOAT NULLABLE
pass_fail                       BOOLEAN NULLABLE
flops_per_token_theoretical     BIGINT NULLABLE
```

---

Omitted from dump (out of scope for this design thread): `hardware_specs` seed data, `interface_type` pre-seeded rows, `run_model_metrics` index strategy. Those are implementation details, not schema design questions.

Multi-GPU and multi-system distribution — deferred per earlier agreement. The `run_hardware_metrics` compound FK on `(system_hardware_id, slot_index)` is forward-compatible with a `run_gpu_usage` join table when that work is scoped.

Standing by for ratification or final concerns.

— Bastion

---

### issue #19 body — @AlbinoGeek — 2026-05-22T19:52:23Z
<https://github.com/Rethunk-AI/bakeoff/issues/19>

**Parent:** #8 — Additional Performance Metrics

Implement `system_hardware` and `system_software` tables as ratified in #8. These are separate entities: hardware describes the fixed physical host; software describes the runtime environment active during a specific run.

## Schemas

```sql
CREATE TABLE system_hardware (
    system_hardware_id  SERIAL PRIMARY KEY,
    system_id           UUID NOT NULL UNIQUE,  -- stable per-host UUID, generated once at first run
    publisher_id        TEXT NOT NULL,          -- submitting user/account
    cpu_model           TEXT NULLABLE,
    ram_total_gb        FLOAT NULLABLE
);

CREATE TABLE system_software (
    system_software_id  SERIAL PRIMARY KEY,
    os                  TEXT NULLABLE,           -- "Ubuntu 24.04.2 LTS"
    kernel_version      TEXT NULLABLE,           -- "6.8.0-57-generic"
    python_version      TEXT NULLABLE,           -- "3.12.3"
    gpu_driver_version  TEXT NULLABLE,           -- nvidia-smi Driver Version field
    cuda_version        TEXT NULLABLE,           -- null for ROCm/CPU-only runners
    rocm_version        TEXT NULLABLE,           -- null for CUDA runners
    runner_version      TEXT NULLABLE            -- bakeoff harness commit hash or semver
);
```

## Auto-detection requirements

### `system_hardware`

- **`system_id`** — stable UUID stored in local config (e.g., `~/.config/bakeoff/system_id`). Generated with `uuid.uuid4()` on first run; never regenerated. This identifies the physical machine across runs even when submitted by different accounts.
- **`publisher_id`** — from existing runner config (already tracked in harness)
- **`cpu_model`** — `/proc/cpuinfo` → `model name` field (Linux); `platform.processor()` fallback
- **`ram_total_gb`** — `psutil.virtual_memory().total / (1024**3)`

### `system_software`

- **`os`** — `platform.platform()` or `distro.name(pretty=True)` for Linux distributions
- **`kernel_version`** — `platform.release()`
- **`python_version`** — `platform.python_version()`
- **`gpu_driver_version`** — `nvidia-smi --query-gpu=driver_version --format=csv,noheader`
- **`cuda_version`** — `nvidia-smi` → CUDA Version field in header; or `torch.version.cuda` if torch available
- **`rocm_version`** — `rocm-smi --showversion` output; null if unavailable
- **`runner_version`** — harness `__version__` constant or `git describe --tags --always` at build time

## Deduplication

`system_hardware` deduplicates on `system_id` (unique constraint). Same machine across runs reuses the existing row.

`system_software` does **not** deduplicate — the same software environment may be recorded multiple times as a new row; `run_hardware_metrics` references whichever row was active during that run. Deduplication by full column hash is optional optimization for Phase 2.

## Acceptance criteria

- [ ] Both migrations create tables with correct types and constraints
- [ ] `system_id` UUID is generated on first run and persisted to local config; stable across subsequent runs on the same machine
- [ ] Harness auto-populates both tables at startup
- [ ] `system_hardware` upserts on `system_id` (do not create duplicate rows for the same machine)
- [ ] `system_software` inserts a new row per run (environment snapshot, not deduplicated)
- [ ] Both migrations are reversible

— Bastion

---

### issue #20 body — @AlbinoGeek — 2026-05-22T19:52:27Z
<https://github.com/Rethunk-AI/bakeoff/issues/20>

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

---

### issue #21 body — @AlbinoGeek — 2026-05-22T19:52:31Z
<https://github.com/Rethunk-AI/bakeoff/issues/21>

**Parent:** #8 — Additional Performance Metrics
**Depends on:** #20 (`system_gpu_link`), #19 (`system_software`)

Update `run_hardware_metrics` to reference `system_gpu_link` (via compound FK) and `system_software`. This wires the per-run hardware measurements to the full hardware context established by the prior sub-issues.

## Schema change

Add two FK references to the existing `run_hardware_metrics` table:

```sql
ALTER TABLE run_hardware_metrics
    ADD COLUMN system_hardware_id INT NULLABLE,
    ADD COLUMN slot_index         INT NULLABLE,
    ADD COLUMN system_software_id INT NULLABLE REFERENCES system_software,
    ADD CONSTRAINT fk_run_hardware_metrics_system_gpu_link
        FOREIGN KEY (system_hardware_id, slot_index)
        REFERENCES system_gpu_link (system_hardware_id, slot_index);
```

Nullable on first addition to avoid breaking existing rows; set NOT NULL once backfill is confirmed clean or if starting fresh.

## Run context binding

At run time, the harness records:
- `(system_hardware_id, slot_index)` — identifies the `system_gpu_link` row active for this run (which GPU in which slot on which host)
- `system_software_id` — the `system_software` row created at startup for this run (software environment snapshot)

Both must be populated before inserting any `run_hardware_metrics` rows.

## Query path for full hardware context

A three-way join from any `run_hardware_metrics` row gives complete hardware context:

```sql
SELECT r.*, sg.*, gw.*, sh.*, ss.*, it_native.description, it_actual.description
FROM run_hardware_metrics r
JOIN system_gpu_link sg ON (r.system_hardware_id, r.slot_index) = (sg.system_hardware_id, sg.slot_index)
JOIN gpu_hardware gw ON sg.gpu_hardware_id = gw.gpu_hardware_id
JOIN system_hardware sh ON sg.system_hardware_id = sh.system_hardware_id
JOIN system_software ss ON r.system_software_id = ss.system_software_id
LEFT JOIN interface_type it_native ON sg.slot_native_interface_type_id = it_native.interface_type_id
LEFT JOIN interface_type it_actual ON sg.actual_interface_type_id = it_actual.interface_type_id;
```

## Multi-GPU future compatibility

This design is forward-compatible with a future `run_gpu_usage` join table for multi-GPU runs. When that is scoped, `run_gpu_usage` will reference `(system_hardware_id, slot_index)` per GPU used per run; `run_hardware_metrics` will reference `run_gpu_usage.run_id` instead of carrying the compound FK directly. No schema changes to the other tables required.

## Acceptance criteria

- [ ] Migration adds `system_hardware_id`, `slot_index`, and `system_software_id` to `run_hardware_metrics`
- [ ] FK constraint to `system_gpu_link` is enforced
- [ ] Harness populates both FKs on every run record insert
- [ ] Existing test suite passes (no regression from FK addition)
- [ ] Migration is reversible

— Bastion

---

### #8 comment 4522303209 — @AlbinoGeek — 2026-05-22T19:53:06Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4522303209>

**Bastion response — 221947ZMAY26**

Sub-issues opened. Implementation dependency order:

| Issue | Title | Depends on |
|-------|-------|-----------|
| #17 | `interface_type` lookup table + seed data | — |
| #18 | `gpu_hardware` table + auto-detection | #17 |
| #19 | `system_hardware` + `system_software` tables | — |
| #20 | `system_gpu_link` join table + PCIe detection | #17, #18, #19 |
| #21 | `run_hardware_metrics` FK wiring | #20, #19 |

#17 and #19 have no dependencies and can proceed in parallel. #18 requires #17 first. #20 requires #17 + #18 + #19. #21 is the terminal step.

Standing by for implementation direction or priority assignment.

— Bastion

---

### issue #22 body — @AlbinoGeek — 2026-05-25T09:53:53Z
<https://github.com/Rethunk-AI/bakeoff/issues/22>

## Context

Hardware schema locked in #17 (#18, #19, #20, #21). This thread analyses which hardware tables warrant disk files, what the layout should be, and how hardware records relate to run disk files.

Follows the disk persistence pattern established in #15 for models/tasks/prompts/runs: disk files carry `schema_version`, `created_at`, `updated_at`; FK columns collapse to embedded objects or UUID/ID references.

---

## Tables in scope

| Table | Issue | Purpose |
|-------|-------|---------|
| `interface_type` | #17 | Lookup — GPU bus type enum |
| `system_hardware` | #19 | Host CPU/RAM snapshot per run |
| `system_software` | #19 | Host OS/driver snapshot per run |
| `gpu_hardware` | #18 | Individual GPU slot spec |
| `system_gpu_link` | #20 | Many-to-many: GPUs in a system snapshot |
| `run_hardware_metrics` | #21 | Per-run hardware context (links run → hw snapshot) |

---

## Open questions

**Q1 — Lookup table disk files**

`interface_type` is a small admin-controlled enum (10 rows). Should it have a seed JSON file (like `seeds/quantization_methods.json`) or is it embedded in schema.sql only?

Proposed: seed file `seeds/interface_types.json`, parallel to existing seeds. Admin-review-gated like model_architectures/model_file_formats.

**Q2 — Hardware snapshot identity**

`system_hardware` and `system_software` rows are generated at run time by the runner probing the host machine. No UUID — they use SERIAL PKs.

Options for disk identity:
- A. Derive a deterministic UUID from the hardware fingerprint fields (cpu_model + cpu_cores + ram_gb + motherboard hash) — allows dedup across runs without DB
- B. Use DB-assigned SERIAL ID embedded in the run disk file — simpler, requires DB round-trip before file is self-contained
- C. No separate disk file for hardware — embed the hardware snapshot inline in the run disk file

**Q3 — gpu_hardware + system_gpu_link disk layout**

A system snapshot involves one or more GPU slots. Should the disk representation be:
- A. A single `hardware-snapshot/<uuid>.json` file containing both system_hardware fields and an embedded `gpus: [...]` array (inlined from system_gpu_link + gpu_hardware) — single file per snapshot
- B. Separate `gpu_hardware/<id>.json` files referenced by ID from the snapshot — allows GPU spec dedup across snapshots

**Q4 — run_hardware_metrics relationship to run disk file**

`run_hardware_metrics` is 1:1 with `runs` (one row per run). Should the hardware context be:
- A. Embedded inline in `runs/<run_id>.json` as a `hardware:` block — avoids a separate file, run file is self-contained
- B. A separate `run-hardware/<run_id>.json` file — parallel to the run, referenced by run_id

**Q5 — Runner hardware reporting timing**

Hardware snapshot is captured by the runner at claim time or job start. Should the runner:
- A. Create/update the hardware snapshot once per process start (reuse across runs in same session)
- B. Snapshot hardware fresh per run (detects GPU slot changes mid-session)

---

Tagging @gissf1 for design input per the pattern established in #15.

— Bastion

---

### #22 comment 4533634217 — @gissf1 — 2026-05-25T10:44:24Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533634217>

Q1: Agree with proposal; generally I think seeds are the right answer for these lookup tables, as it keeps the data organized and is low maintenance to add new records as the technology evolves a bit every few years.

Q2: Option B is completely wrong - the user may wish to do a run in an air-gapped environment, or similarly disconnected from real-time internet access.  A and C are both partially right.  I think the specifications from the user's machine (as defined in option C) should be part of the results disk file, but since option A is how the UUID is generated universally, it can be computed from those results programmatically.  or am I misunderstanding something here?

Also, there may be additional fields we want to include in the UUID, but I don't have the field list in scope mentally (if you want to discuss them in more detail, we can, but I would need you to provide the field list to identity what to include).  We may also want to include some crucial settings (along with listing them in a description), for example wait states, memory ganging/interleaving or other performance- or functionality- affecting settings in the BIOS setup.

Q3: Primarily we would want to dedupe hardware records (since there are likely to be many).  I think option B makes sense for the default, but that information should be embedded into a results file too, unless it is a known GPU type that is already known to be present (perhaps if it is already in a seed file).  Can you elaborate on option A to be sure I understand your definition before dismissing it entirely?

Q4: I think both make some sense, but in an effort to keep run results portable, I think embedding hardware info into the file (option A) makes the most sense by default.  It can be deduped at ingestion time.  If the user knows their hardware results are already present upstream, I suppose they could use a "sparse format" which limits the results data size using a reference to the hardware UUIDs instead, but that can be a later enhancement.

Q5: I doubt anyone is changing GPUs mid-session on consumer hardware.  Maybe this should be determined per-hardware depending on capabilities, but I think for our primary use case, option A is acceptable.  Is there any known hardware interfaces capable of hot-swapping GPUs like that?  I'm not aware of any?  Maybe if they were network-connected instead of PCIe-connected to the host system?  I don't want to waste time on impossible scenarios, but I also want to ensure accurate results.  That's why I think this answer is hardware dependent.

---

### #22 comment 4533647917 — @AlbinoGeek — 2026-05-25T10:46:41Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533647917>

**Bastion response — 251044ZMAY26**

@gissf1 — synthesizing all five answers.

---

**Q1 — interface_type seed file: CONFIRMED**

Agreed. `seeds/interface_types.json` will be added alongside the other lookup seed files. Admin-review-gated, same pattern as `model_architectures` and `model_file_formats`.

---

**Q2 — Hardware snapshot identity: A + C combined**

Confirmed reading. The two options are not mutually exclusive:

- Hardware specs are embedded inline in the results disk file (option C basis — portable, air-gap compatible, no round-trip required)
- `hardware_id` UUID is computed deterministically from those embedded specs at ingest time (option A UUID generation) — never stored in the disk file, derived on demand

This means disk files are fully self-contained; the UUID is an ingestion artifact, not a field the runner needs to compute or store.

**Proposed field list for `hardware_id` UUID input:**

System fields:
- `cpu_model` — CPU make/model string
- `cpu_cores` — physical core count
- `cpu_threads` — logical thread count
- `ram_gb` — total RAM in GB
- `motherboard` — motherboard make/model

GPU fields (sorted list of tuples, one per slot):
- `gpu_model` — GPU make/model string
- `vram_mb` — VRAM in MB
- `slot_index` — PCIe slot position

**BIOS / performance-affecting settings** — you raise a valid point. These affect benchmark outcomes and should be part of the UUID if they materially differentiate two otherwise-identical machines. Proposed additional fields (all optional, null if not set or auto):

| Field | Description |
|-------|-------------|
| `memory_speed_mhz` | Active memory clock rate |
| `memory_channels` | Active memory channel count (1 = single, 2 = dual, 4 = quad) |
| `memory_interleave_profile` | XMP / EXPO / DOCP profile name or `manual` |
| `bios_notes` | Free-form for other performance-affecting settings (e.g., fTPM off, SMT off) |

`memory_speed_mhz`, `memory_channels`, `memory_interleave_profile` — include in UUID input. `bios_notes` — stored in snapshot but excluded from UUID (free-form field not stable for dedup). Does this field list look right, or are there other BIOS settings you want to capture or include in UUID generation?

---

**Q3 — GPU/system snapshot disk layout: clarifying Option A, then recommendation**

Option A as proposed: a single `hardware-snapshot/<uuid>.json` that merges system fields and GPU list into one document — no separate per-GPU files at the disk level. Example:

```json
{
  "schema_version": 1,
  "hardware_id": "<uuid5>",
  "cpu_model": "AMD Ryzen 9 7950X",
  "cpu_cores": 16,
  "cpu_threads": 32,
  "ram_gb": 64.0,
  "motherboard": "ASUS ROG Crosshair X670E",
  "gpus": [
    {"slot_index": 0, "gpu_model": "AMD Radeon RX 7900 XTX", "vram_mb": 24576, "interface": "PCIe 4.0 x16"}
  ]
}
```

Option B would split that into `gpu_hardware/<gpu_uuid>.json` and a separate system snapshot referencing GPU UUIDs.

**Recommendation given Q4 (run file embeds full hardware block):** Option A is the right shape for the `hardware-snapshot/` dedup cache — it mirrors exactly what gets embedded in the run file. At ingest, the pipeline:
1. Extracts the hardware block from the run file
2. Computes `hardware_id` and per-GPU UUIDs
3. Writes `hardware-snapshot/<hardware_id>.json` if not already present (dedup cache)
4. Upserts DB rows for `system_hardware`, `gpu_hardware`, `system_gpu_link`

Separate `gpu_hardware/<gpu_uuid>.json` files (Option B) add a level of indirection that only pays off when many snapshots share the same GPU and you want to avoid re-embedding the spec. That optimization can land as the dedup enhancement in Phase 2 alongside the sparse run format.

---

**Q4 — run_hardware_metrics / run disk file: CONFIRMED**

Option A confirmed. Run file embeds full hardware block inline. Portable, self-contained, deduplicable at ingest. Sparse/reference format deferred to Phase 2.

---

**Q5 — Runner hardware snapshot timing: CONFIRMED with caveat**

Option A confirmed — snapshot once at process start, reuse across runs in same session. Consumer hardware does not hot-swap GPUs mid-session.

On interfaces: Thunderbolt 3/4, USB4, and eGPU enclosures are theoretically hot-swap capable (OS can detach/reattach the device). In practice, running a GPU benchmark mid-swap is not a real scenario. NVLink/SXM topologies cannot hot-swap. PCIe cannot hot-swap without platform-specific BIOS support.

Recommendation: Option A for Phase 1. If an operator is running on an eGPU enclosure and wants per-run snapshotting, they can restart the runner process. No need for a per-run snapshot path in the initial implementation.

---

**Open items for your confirmation:**

1. BIOS field list above — is `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile` the right set for UUID inclusion? Any fields to add or remove?
2. Option A hardware-snapshot layout — does the single combined file (system + GPUs embedded) make sense as the disk dedup format, given Option B is deferred?

— Bastion

---

### #22 comment 4533894066 — @gissf1 — 2026-05-25T11:29:48Z
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

### #22 comment 4554570690 — @AlbinoGeek — 2026-05-27T12:38:26Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4554570690>

**Bastion response — 271235ZMAY26**

@gissf1 — addressing all points.

---

**pcie_gen_override — removed from bios_notes entirely**

Agreed. Join table's measured link rate is the authoritative signal; a BIOS-forced Gen 3 flag adds no independent information and isn't worth storing even as metadata. Fully removed.

Updated `bios_notes` UUID whitelist — Phase 1 final:

| Key | UUID input? | Notes |
|-----|------------|-------|
| `bar_size_mb` | YES | BAR aperture size affects VRAM access throughput |
| `above_4g_decoding` | NO | — |
| `SMT_enabled` | NO | Expressed by `cpu_threads` |
| `iommu_enabled` | NO | Security boundary only |

`pcie_gen_override` and `power_limit_w` — not stored.

---

**Scripting language — revised recommendation**

Agreed on pushing back. Lua introduces a new language dependency and requires more "rolling our own" scaffolding. Revised recommendation: **Go templates + sprig**.

Rationale:
- `text/template` is Go standard library — no new language dep, no new language for Go devs to learn
- sprig adds ~100 functions: string manipulation (trim, substr, replace, regex), math, crypto, maps/slices — covers field transformations, hash function changes, value mapping
- `range` is supported: iteration over records or lists in the script is possible
- Custom functions registerable: any edge case not covered by sprig can be added
- Consistent across schema_versions and url_template (one language, one evaluation engine — per your direction below)
- Widely deployed (Helm, Kubernetes): well-tested, security profile understood, documentation extensive

CEL: reserved as Phase 2 candidate if strict sandboxing is required. Weaker for template-style URL construction and lacks loops in base form.

Gonja / Pongo2: viable if Jinja2 syntax is preferred, but Go templates are more idiomatic for a Go project. Gonja is also a transitive dep rather than a standard lib.

---

**schema_versions — current agreed definition**

Per your request, consolidated state:

```sql
CREATE TABLE schema_versions (
    schema_version_id SERIAL PRIMARY KEY,
    description       TEXT NOT NULL,     -- human-readable label for this version change
    changes           JSONB,             -- structured audit log: fields/tables added/removed/modified
    migration_script  TEXT,              -- Go template + sprig; NULL for trivial renames
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

`changes` and `migration_script` serve different roles:
- `changes` JSONB: machine-parseable audit trail and pipeline signal — declares WHAT changed. Simple migrations only need this.
- `migration_script` TEXT: executable Go template for complex transformations — declares HOW to migrate. NULL when `changes` alone is sufficient.

If you prefer to drop `changes` and make `migration_script` self-documenting (the script body IS the record of what changed and how), that's workable. Awaiting your direction on which to keep.

---

**uuid_migrations — revised**

Moving `migration_script` to `schema_versions` reshapes this table significantly:

```sql
CREATE TABLE uuid_migrations (
    migration_id    SERIAL PRIMARY KEY,
    old_uuid        TEXT NOT NULL,
    table_id        INTEGER NOT NULL REFERENCES schema_tables(table_id),
    target_version  INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    allow_migration BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Changes from prior definition:
- `entity_type` → `table_id`: cleaner FK, more explicit.
- `from_version` removed: implied by "any record not yet at target_version"; the script in schema_versions handles the transformation regardless of source version.
- `to_version` → `target_version`: direction clearer.
- `resolution_script` removed: lives in `schema_versions.migration_script` now.
- `allow_auto_migrate` → `allow_migration`: shortened, sufficient.

Pipeline flow: old UUID encountered → look up `uuid_migrations` by `old_uuid` + `table_id` → find `target_version` → execute `schema_versions[target_version].migration_script` against the record → gate on `allow_migration` before auto-running.

One open question: if a UUID has traversed multiple hops (v1→v2→v3), does `uuid_migrations` hold one row per hop, or one row pointing to the latest target? Bastion recommendation: one row per hop — cleaner audit trail, each hop's script is self-contained. Pipeline chains them. Confirm or adjust.

---

**url_template — single field, prefix detection**

Agreed. Single `url_template TEXT` field. Evaluation:
- Matches known URL prefix (`http://`, `https://`, `file://`, etc.) → static URL, use verbatim
- Otherwise → evaluate as Go template expression (consistent with `schema_versions.migration_script`)

`url_script` column dropped. Cleaner.

---

**Open items for your confirmation**

1. `bios_notes` — `pcie_gen_override` and `power_limit_w` fully removed: confirmed above, or adjust?
2. Go templates + sprig as scripting language: confirm, or prefer Gonja/CEL?
3. `schema_versions.changes` JSONB — keep alongside `migration_script` as audit trail, or drop in favor of self-documenting script body?
4. `uuid_migrations` one row per hop vs one row pointing to latest version: confirm one-per-hop?
5. `allow_migration` as final field name: confirm?

— Bastion

---

### #22 comment 4559373879 — @gissf1 — 2026-05-27T23:01:35Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4559373879>

RE: bios_notes:
- pcie_gen_override removal approved.
- I'm unsure if power_limit_w fits the exact same situation.  The system may be using more complex mechanisms than PCIe restrictions to limit power usage.  Also, what if the restrictions only affect limited scenarios like when running on emergency battery power or when extreme thermal limits are reached;  I think the correct approach is to track this in metadata if enabled and present, but we should not use it in the UUID yet.

RE: Scripting language: I think Go Templates + Sprig is a valid solution, but we will need to add several types of functions for the following purposes:
- reference keys in the current record (the one with the old uuid), possibly with record data coming from a file, but the template should be agnostic to the source (handle the record data the same no matter the source), whether DB or file)
- do lookups in other tables based on various criteria
- do schema changes to tables - if the table itself is in an older format, this will be a single call explaining what change is being made to the table. This likely will involve SQL-like operations, such as ALTER TABLE or RENAME TABLE style changes.

The goal would be for these to be atomic and idempotent so we don't have to manage state tracking.  If a table is already of a newer version, the schema changes should detect that and not modify them again, but still perform record upgrades for the "current record".

I think this is more of a future plan until we need this functionality for something, but I think planning a bit ahead is essential to creating a good design.  We may not need this for Phase 1, but we will surely need it for Phase 2 because we want to have database schema change logs planned before any result records become public.

RE: schema_versions: I think the `schema_versions.changes` field is redundant since the purpose of migration_script is to migrate the schema (and possibly data), which is all that would need to be done.  If there is some simpler schema changes or data migration that can be automated with standard functions, we can create external helper functions to turn the migration_script into a one-liner (reduces database size in the long term, and code refinements mean even past schema version changes benefit from future code improvements to the external helpers.  For example, perhaps we create a "schema_rename_table_field(source, dest)" method, in which case the call in the migration_script field is literally a one-liner; data migrations can be done implicitly as well within that function.  If we call it from a context where a record needs to be converted, but the database itself is already converted, it can possibly handle that conversion as well.

RE: uuid_migrations open question: Confirmed; one row per hop makes the most sense to me.  This gives us a method for importing even the oldest records into the latest database going through the required hops to convert it step by step in a well-tested process; no risk of divergence or migration errors due to untested code paths stored in the database.

If at some point we want to change how this works (perhaps using some table version semantics, different storage mechanism, or basically anything else), we can simply revise the code that handles the migration and adjust this table accordingly.  We could even remove the UUID matching and simply match based on table_id and the record's schema_version to do the migration.  The idea in including the UUID was to more easily detect records that have been already migrated, but perhaps it's unnecessary and overly voluminous with the migration_script logic?  I'd like to hear your thoughts on the design.

On this, you didn't address my question: do we need a separate record_migration_script and a schema_migration_script? or should the schema_migration_script include code to iterate over records to do database record upgrades as well as the actual schema changes?  I think it may make sense to have them separately, but as speculated above, this might be something to handle within helper functions.

RE: allow_migration: this field name is approved.

---

### #22 comment 4559571210 — @AlbinoGeek — 2026-05-27T23:38:56Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4559571210>

**Bastion response — 272335ZMAY26**

@gissf1 — addressing all points.

---

**bios_notes — power_limit_w**

`pcie_gen_override` and `power_limit_w` are different cases:

- `pcie_gen_override`: fully removed. Join table's measured link rate is authoritative; the setting adds no independent information.
- `power_limit_w`: keep as optional metadata in `bios_notes`, **excluded from UUID for Phase 1.** Your instinct is correct — power limits can be transient (thermal headroom, battery mode) and do not represent a stable identity dimension. When accumulated results show that a power-limited configuration materially diverges from an unlimited one in benchmark outcomes, Phase 2 can add it to the UUID whitelist with a migration record. Not a UUID field yet; not removed entirely.

Updated `bios_notes` UUID whitelist — Phase 1 final:

| Key | UUID input? | Notes |
|-----|------------|-------|
| `bar_size_mb` | YES | BAR aperture size affects VRAM access throughput |
| `above_4g_decoding` | NO | — |
| `SMT_enabled` | NO | Expressed by `cpu_threads` |
| `iommu_enabled` | NO | Security boundary only |
| `power_limit_w` | NO | Metadata only; transient; Phase 2 candidate |
| `pcie_gen_override` | NOT STORED | Fully removed; covered by measured link rate |

---

**Go templates + sprig — custom function plan**

Confirmed. Three custom function categories registered against the Go template engine:

**1. Record field access**

`record_field <field>` — reads a field from the current record being migrated. Source-agnostic: the template receives a normalized Go map regardless of whether the data originated from a disk file or a DB row.

**2. Table lookups**

`lookup_table <table_name> <where_field> <where_value>` — returns a matching row from the named table as a Go map. Covers FK resolution and cross-table mapping during migration.

**3. Schema DDL helpers (idempotent by design)**

| Function | Behavior |
|----------|----------|
| `add_column <table> <col> <type>` | No-op if column already exists |
| `rename_column <table> <old> <new>` | No-op if new name already exists |
| `add_table <table> <sql>` | No-op if table already exists |

Each helper checks current DB state before acting. Same script run twice → same result. Schema scripts run in a transaction; any step fails → full version DDL rolls back.

Phase 1 scope: stub function signatures now, implement in Phase 2. `schema_versions` table exists in Phase 1; migration scripts are not needed until the first schema evolution.

---

**schema_versions.changes — dropped**

Confirmed. `migration_script` with the helper function approach is self-documenting. `changes` JSONB column removed. Final definition:

```sql
CREATE TABLE schema_versions (
    schema_version_id       SERIAL PRIMARY KEY,
    description             TEXT NOT NULL,
    schema_migration_script TEXT,   -- DDL; runs once per DB upgrade; idempotent; NULL = no-op
    record_migration_script TEXT,   -- per-record transformation; runs at ingest for old records
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

(Note: two script fields — see below.)

---

**record_migration_script vs schema_migration_script — two fields**

Recommendation: **two separate fields on `schema_versions`**, not one combined script.

| | `schema_migration_script` | `record_migration_script` |
|--|--------------------------|--------------------------|
| **Runs** | Once per DB upgrade | Per-record at ingest when old version detected |
| **Scale** | Single execution | Potentially millions of calls |
| **Contains** | DDL (idempotent helpers) | Field transformation only — no DDL |
| **Input** | DB state | One record map |
| **Output** | Modified schema | Transformed field values |

Mixing both into one script requires context-awareness ("am I migrating schema or a record?") and means DDL runs inside a per-record hot path — unsafe at volume. Separation keeps both simple, independently testable, and callable in the right execution context.

Helper functions cover the one-liner vision: `schema_rename_field(source, dest)` in the schema script, with an equivalent record-level helper for the record script where needed. Code refinements to those helpers retroactively improve all past schema versions.

---

**uuid_migrations — version-based matching**

On your open question: Bastion recommendation is **version-based matching — drop per-record UUID rows**.

Instead of `uuid_migrations` holding one row per old UUID, each UUID-identity table carries a `schema_version_id` column. At ingest, the pipeline reads the record's `schema_version_id`, compares to current, and chains migration hops from `schema_versions` directly. This eliminates the UUID-per-record ledger entirely and scales to millions of records without a proportional row count in `uuid_migrations`.

Revised `uuid_migrations`:

```sql
CREATE TABLE uuid_migrations (
    migration_id    SERIAL PRIMARY KEY,
    table_id        INTEGER NOT NULL REFERENCES schema_tables(table_id),
    from_version    INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    to_version      INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    allow_migration BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

No `old_uuid` column. `uuid_migrations` is a registry of version transitions — one row per hop per table, not one row per migrated record. `allow_migration` gates auto-migration for all records at that version transition.

**Detecting already-migrated records:** the record's own `schema_version_id` tells you. If `schema_version_id = current`, no migration needed. No per-record UUID lookup required.

**One row per hop — confirmed.** Chained hops (v1→v2→v3) produce two rows, each with its own script. Pipeline chains them in order. Each hop's script is self-contained and tested independently; no risk of untested code paths for multi-generation migrations.

**Consequence:** adds `schema_version_id INTEGER REFERENCES schema_versions` to every UUID-identity table. Small cost; also useful for data provenance and audit. Awaiting your direction on whether this embedding is acceptable, or whether you prefer a different discriminator.

---

**Open items — updated status**

| # | Item | Status |
|---|------|--------|
| 1 | `pcie_gen_override` fully removed | Confirmed |
| 2 | `power_limit_w` metadata only, not UUID, Phase 1 defer | Bastion recommendation this comment |
| 3 | Go templates + sprig + custom fns — stub Phase 1, implement Phase 2 | Awaiting direction |
| 4 | `schema_versions.changes` JSONB dropped | Confirmed this comment |
| 5 | Two separate script fields (`schema_migration_script`, `record_migration_script`) | Awaiting direction |
| 6 | `uuid_migrations` version-based matching — `schema_version_id` on entity tables | Awaiting direction |
| 7 | `allow_migration` field name | Confirmed |

— Bastion

---

### #22 comment 4568864985 — @AlbinoGeek — 2026-05-28T22:31:07Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4568864985>

**Bastion response — 281811ZMAY26**

@gissf1 — addressing all points in order.

---

**bios_notes — power_limit_w: CONFIRMED**

`power_limit_w` locked as metadata only, not in UUID. Consistent with prior session lock (271235ZMAY26). Phase 2 candidate if accumulated results show power-limited configs diverge materially from unlimited ones.

---

**Go templates + sprig — DDL helper idempotency for remove-then-add**

The `add_column` helper is no-op if column already exists. The concern is: if column X is removed in version N and added back (possibly with a new type) in version M (M > N), does idempotency hold across crash-recovery re-runs?

Answer: yes, with symmetric no-op design across all DDL helpers:

| Helper | Idempotency rule |
|--------|-----------------|
| `add_column <table> <col> <type>` | No-op if column already exists |
| `drop_column <table> <col>` | No-op if column does not exist |
| `rename_column <table> <old> <new>` | No-op if `new` already exists and `old` is absent |
| `add_table <table> <sql>` | No-op if table already exists |
| `drop_table <table>` | No-op if table does not exist |

Scripts run in strict version order; the version tracking table ensures each script runs exactly once (except on crash recovery re-run, where the no-op rules make re-execution safe). The "remove X in vN, add X back in vM" case is entirely handled by sequencing: vN's `drop_column X` removes it, vM's `add_column X <new_type>` adds it back. No collision.

On `add_column` vs type change: if a column exists with the wrong type, `add_column` skips it silently. Type changes require an explicit `alter_column_type` helper (to be added to the function set). This makes the DDL intent unambiguous: `add_column` is append-only; `alter_column_type` is deliberate mutation.

Phase 1 scope confirmed: stub function signatures now. Implementation in Phase 2. `schema_versions` table present in Phase 1; first migration scripts appear when schema evolves.

---

**record_migration_script — per-record timing and UUID change**

**Timing:** `record_migration_script` runs at ingest time, per old-format record, lazily. When an old disk file is ingested (schema_version_id < current), the pipeline applies the script to the record before writing to DB. Schema migration (DDL) runs first, once, via `schema_migration_script`. Record migration runs after, per record, at ingest.

**For complex DDL migrations** where in-place lazy migration is infeasible (e.g., column type changes affecting all rows), your proposed pattern is correct and now canonical:

1. Write all affected records back to backing disk files using the old format
2. Apply DDL to the empty table (`schema_migration_script`)
3. Re-import all records through `record_migration_script` into the updated table

This is the batch path. Simple migrations (renames, type widening, removing NOT NULL, adding nullable columns) apply directly via `schema_migration_script` alone; no record re-export needed.

**If record_migration_script changes UUID-input fields:** the migration pipeline contract is:
1. Compute `old_uuid` from original disk file fields (before script runs)
2. Apply `record_migration_script`
3. Compute `new_uuid` from transformed fields (using current schema namespace — see below)
4. If `old_uuid ≠ new_uuid`: write entry to `uuid_migrations` automatically

This is enforced by the pipeline, not by the script author. Migration scripts declare field transformations; the pipeline handles UUID bookkeeping.

Two separate fields (`schema_migration_script` + `record_migration_script`) confirmed correct.

---

**Self-identifying UUIDs via UUID v5 namespaces — YES, feasible**

The idea is sound. Implementation: each schema version defines a fixed UUID v5 namespace derived from a version string:

```
NAMESPACE_V1 = uuid5(UUID_NIL, "bakeoff.hardware_uuid.v1")
NAMESPACE_V2 = uuid5(UUID_NIL, "bakeoff.hardware_uuid.v2")
```

Hardware fingerprint UUID for version N = `uuid5(NAMESPACE_VN, canonical_fields_json)`.

Two properties this gives:
1. **Version identification:** given an unknown UUID and a set of known namespaces, you can determine which schema version generated it by re-deriving it against each namespace. No separate version tag on the UUID is needed.
2. **Collision avoidance:** same hardware fingerprint in v1 and v2 produces different UUIDs, so there is no ambiguity.

In practice, the disk file's `schema_version_id` already tells you which namespace to use — you don't need to brute-force-test. But the namespace embedding means the UUID is still self-identifying if the disk file metadata is lost or unavailable.

**Does this eliminate uuid_migrations?** Partially. For disk-file-to-DB ingest, you can always recompute the new UUID from the disk file + current namespace — no uuid_migrations lookup needed. But uuid_migrations is still needed for **FK resolution across tables**: if table A holds a FK to table B using a v1 UUID, and B migrated to v2, the FK value in A is stale. uuid_migrations bridges old_uuid → new_uuid for those cross-table references. The table is not eliminated, but its role is narrowed to FK resolution rather than version identification.

Recommendation: adopt UUID v5 namespaces per your idea. uuid_migrations retains its FK bridge role only.

---

**uuid_migrations — FK concerns**

Both concerns addressed:

**1. How to update FKs in other tables pointing to old UUID values?**

Lazy resolution — no eager FK cascade. The pipeline resolves old UUIDs to new at ingest time:
- Encounter old UUID in FK field → look up `uuid_migrations` (old_uuid, entity_type) → use new_uuid
- For EXISTING DB records with stale FK values: a one-time batch migration pass at schema upgrade time iterates each FK column, resolves via uuid_migrations, and writes the current UUID. This runs once per upgrade as part of the schema_migration_script.

**2. How to recognize that a FK's UUID is an old version vs current?**

With UUID v5 namespaces: test the UUID against `NAMESPACE_VN` for each known version. If it matches an older namespace, it is an old UUID. If it matches the current namespace, it is current.

Without namespace testing: uuid_migrations lookup. If found in uuid_migrations as old_uuid → stale. If not found there AND found in the primary table → current.

**Remaining confirmations:**

- `schema_version_id` in each disk file record: CONFIRMED
- DB records at rest assumed current version (no per-row storage): CONFIRMED
- During migration, version tracked by record location (old table buffer vs new table): CONFIRMED
- Initial user-created records: assume current version unless explicit version tag present: CONFIRMED

---

**Consolidated migration-related table schemas (per your request)**

```sql
-- Records schema change history; scripts are Go template + sprig
CREATE TABLE schema_versions (
    schema_version_id       SERIAL PRIMARY KEY,
    description             TEXT NOT NULL,
    schema_migration_script TEXT,   -- DDL; runs once per upgrade; idempotent; NULL = no-op
    record_migration_script TEXT,   -- per-record transform; runs at ingest for old records; NULL = identity
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tracks which versions have been applied to this DB instance
CREATE TABLE schema_version_history (
    schema_version_id INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    applied_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (schema_version_id)
);

-- Bridges old UUIDs to current UUIDs for FK resolution across tables
CREATE TABLE uuid_migrations (
    migration_id  SERIAL PRIMARY KEY,
    entity_type   TEXT NOT NULL,   -- e.g. 'system_hardware', 'gpu_hardware'
    old_uuid      UUID NOT NULL,
    new_uuid      UUID NOT NULL,
    from_version  INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    to_version    INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    migrated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (entity_type, old_uuid)
);

-- Canonical list of tables with UUID-based identity (for entity_type FK)
CREATE TABLE schema_tables (
    table_id    SERIAL PRIMARY KEY,
    table_name  TEXT NOT NULL UNIQUE,
    uuid_namespace_v1 UUID NOT NULL  -- base namespace for UUID v5 generation
);
```

Open question remaining: should `schema_tables.uuid_namespace_v1` be stored in DB (as above), or defined as Go constants only and never in the DB? DB storage allows tooling to look up namespaces without code changes; Go constants are simpler and keep the source of truth in code.

— Bastion // J-2/J-3 // 281811ZMAY26

---

