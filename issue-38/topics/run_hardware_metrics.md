# Topic: run_hardware_metrics

Consolidated chat history (chronological, ascending comment-id). 20 entries. Verbatim quotes; attribution in each header. **#38 thread excluded from this variant.**

## Source entries (provenance TOC)

Entries used to build this topic and its summary. (Not migrated into `topic_summary/`; audit reference only.)

| ticket | entry | author | posted | url |
|---|---|---|---|---|
| #8 | issue body | @gissf1 | 2026-05-13T20:53:09Z | <https://github.com/Rethunk-AI/bakeoff/issues/8> |
| #8 | comment 4445642971 | @AlbinoGeek | 2026-05-13T22:19:56Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4445642971> |
| #8 | comment 4446310454 | @gissf1 | 2026-05-14T00:36:34Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4446310454> |
| #8 | comment 4447581353 | @AlbinoGeek | 2026-05-14T04:39:42Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4447581353> |
| #8 | comment 4450406819 | @gissf1 | 2026-05-14T11:49:57Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4450406819> |
| #8 | comment 4462460688 | @AlbinoGeek | 2026-05-15T18:52:20Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4462460688> |
| #8 | comment 4465943831 | @AlbinoGeek | 2026-05-16T06:09:45Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4465943831> |
| #8 | comment 4466000757 | @AlbinoGeek | 2026-05-16T06:27:00Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4466000757> |
| #8 | comment 4474333279 | @AlbinoGeek | 2026-05-18T04:25:19Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4474333279> |
| #8 | comment 4476713600 | @gissf1 | 2026-05-18T10:23:03Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4476713600> |
| #8 | comment 4480962683 | @AlbinoGeek | 2026-05-18T19:00:40Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4480962683> |
| #8 | comment 4482149275 | @gissf1 | 2026-05-18T20:59:13Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4482149275> |
| #8 | comment 4482502460 | @AlbinoGeek | 2026-05-18T21:40:59Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4482502460> |
| #8 | comment 4486952413 | @AlbinoGeek | 2026-05-19T10:45:07Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4486952413> |
| #8 | comment 4492063588 | @gissf1 | 2026-05-19T21:09:06Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4492063588> |
| #8 | comment 4494171694 | @AlbinoGeek | 2026-05-20T03:17:30Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4494171694> |
| #8 | comment 4494986575 | @gissf1 | 2026-05-20T05:58:21Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4494986575> |
| #8 | comment 4495333862 | @AlbinoGeek | 2026-05-20T06:38:24Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4495333862> |
| #8 | comment 4519663607 | @AlbinoGeek | 2026-05-22T14:39:09Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4519663607> |
| #21 | issue body | @AlbinoGeek | 2026-05-22T19:52:31Z | <https://github.com/Rethunk-AI/bakeoff/issues/21> |

---

## Consolidated thread

### issue #8 body — @gissf1 — 2026-05-13T20:53:09Z
<https://github.com/Rethunk-AI/bakeoff/issues/8>

## Problem / Proposal

I'm unsure of the present details tracked, but among other information in these tests, we should also be tracking: prompt and/or test cost for a given model (perhaps it needs multiple planning cycles to come up with a result vs not), real wall clock time, and cpu/gpu cycle counts required (so we can compare relative efficiency metrics across multiple GPU generations and provide performance estimates on new hardware specifications based on past hardware ratings).

In short, not only should we be tracking "how well did this model do at accomplishing the task overall" but also, using our measurement data, we can determine things like:
- "this X hardware was able to do the task faster/slower/cheaper/more efficiently than the other Y hardware" even across generations and product lines.
- "this X model requires more delay time before responding, but comes out with faster and more accurate output, when compared to other Y model"
- "this X model completes tasks faster than the other Y model"
- "this X model uses N% less gpu clock cycles as the other Y model to solve the same problem"
- "Since X hardware was able to do the task in T time, this unreleased hardware Y with N% faster clock speeds on the same architecture should be able to do the task in T2 time"

It likely makes sense to update the bakeoff-results repo to have that data available for comparison as well.

## Touches benchmark invariants?

- [ ] Yes
- [X] No

---

### #8 comment 4445642971 — @AlbinoGeek — 2026-05-13T22:19:56Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4445642971>

**Bastion response — performance metrics**

Acknowledged. This aligns with the hardware tracking work already scoped in bakeoff-results#12. Proposing the following field additions to the result schema:

**Feasible now (no specialized tooling required):**
- `wall_clock_seconds` — total elapsed time from prompt submission to final token received
- `tokens_prompt` / `tokens_completion` — prompt and output token counts (from provider API or local model inference)
- `cost_usd` — computed from token counts × model pricing table (tracked separately per provider)
- `time_to_first_token_ms` — latency before output begins (signals planning/prefill overhead)

**Hardware-relative metrics (requires standardized hardware metadata):**
- `hardware_id` — references hardware registry entry (CPU + GPU model, VRAM, clock speed tier)
- `tokens_per_second` — derived from completion tokens ÷ wall clock time
- `gpu_utilization_pct` — requires local monitoring hook (NVML or ROCm SMI); NOT feasible via provider API

**Hardware inference estimation:**
Cross-generation performance estimation is viable if we maintain a hardware registry with normalized performance coefficients (TFLOPs or memory bandwidth as the normalization axis). Given hardware A's tokens/sec on task T, we can estimate hardware B's performance using the FLOP ratio. This is an approximation, not a guarantee, and works best within the same architecture generation.

**Schema additions (proposed):**
```json
{
  "wall_clock_seconds": 12.4,
  "time_to_first_token_ms": 340,
  "tokens_prompt": 512,
  "tokens_completion": 128,
  "tokens_per_second": 10.3,
  "cost_usd": 0.0012,
  "hardware_id": "rtx-4090-24gb"
}
```

**Scope note:** GPU cycle count tracking requires invasive profiling (NVML perf counters or hardware-specific tools) and is not feasible from the bakeoff runner without OS-level hooks. Tokens/sec is the practical proxy for GPU efficiency comparison across hardware.

Cross-references: bakeoff-results#12 (hardware field), bakeoff-results#11 (filter bar — will need hardware filter added).

Standing by for direction on implementation priority.

— Bastion

---

### #8 comment 4446310454 — @gissf1 — 2026-05-14T00:36:34Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4446310454>

Agreed that enforcing perfectly accurate gpu cycle count tracking may be too invasive, but we can estimate based on factors like gpu utilization percent, TFLOP ratings, core counts, memory/gpu clock speeds, etc to come up with a reasonable estimate.  And those metrics could be used to compare various hardware and estimate performance.

My previous point was if there were easy cpu cycle counters available (like the Intel RDTSC instruction) or hardware counters already available, that we should simply record that available data for the execution units to gain an "approximate" cycle count (counter value at completion less counter value at start yields counter values added during execution).  I would not suggest adding any kind of tracing hooks or other additions that could significantly impact the performance of the model just to gain perfect accuracy of the cycle counts.

Are there more concrete metrics than "tokens" we can use as our measuring stick?  Perhaps something that directly correlates to the hardware being utilized?  maybe something that represents "memory used" and "calculations done" during the execution?  The problem is that I believe "tokens" is a vague term without a concrete definition, and that is not a good metric.  Also the token count may not correlate with hardware utilization if the architecture is sufficiently different.

---

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

### #8 comment 4450406819 — @gissf1 — 2026-05-14T11:49:57Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4450406819>

I don't know that gpu_wall_time_ms is more useful than simply using the system time() call (or similar) to note when we received the final byte of data from the process.  I'm thinking it may be overcomplicating things compared to gettimeofday() or clock_gettime() calls, since wall clock should be externally consistent.

I'm not too concerned about tracking CPU time used, but I suppose it's a worthwhile metric to detect if some vendor's driver implementation does heavy processing on the CPU side, that would show up.

I feel like the others you listed in your revised schema are all good to include.  Time to first token is useful, so despite not wanting to focus on tokens, I feel like that particular metric is still useful to track.  Things like hardware_id should be retained from your original proposal as well.  wall_clock_seconds (or some sub-seconds unit) should definitely be included in some way.  cost_usd I feel should be displayed to users, but it can be computed on the fly from the hardware specifications and the test results, rather than being stored separately, since it is a derived value after all.

I'm beginning to think there are different types of metrics we want to be building:
- invariant hardware specifications (energy use per some unit, peak TFLOPS capabilities, etc)
- per-run hardware metrics (cpu/gpu utilization, cpu/gpu memory use, tokens per second, etc)
- per-run model metrics (scores on tests, pass/fail rates, completion times, etc)

I'm not sure if it helps to separate them, but it would make data management easier and reduce redundancy within a database.

---

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

### #8 comment 4465943831 — @AlbinoGeek — 2026-05-16T06:09:45Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4465943831>

**Bastion status update — 150000ZMAY26**

Conducted a retrospective audit against this thread. The following commitments from our discussion were not implemented and are now being tracked:

| Commitment | Status | Tracking |
|-----------|--------|---------|
| `hardware_id` in records | ❌ Not implemented | #9 |
| `peak_vram_mb` per call | ❌ Not implemented | #9 |
| `gpu_sm_utilization_pct` per call | ❌ Not implemented | #9 |
| `tflops_utilization_pct` | ❌ Not implemented | #10 |
| Optional `cpu_time_user_ms` / `cpu_time_sys_ms` | ❌ Not implemented | #10 |
| `cost_usd` removed from storage (display-derived) | ❌ Still stored | implementing now |
| Three-category normalized schema | ⚠️ Not applicable to current flat-JSON runner; tracked conceptually in #9/#10 |

**What was delivered:** `latency_s` (wall clock), `ttft_s` (time-to-first-token), `tokens_per_sec`, `energy_wh`, `quality_heuristic`.

**Next:** implementing #9 first (hardware block + VRAM/SM collection), then #10 (TFLOPS + CPU timing), then marking both closed.

— Bastion

---

### #8 comment 4466000757 — @AlbinoGeek — 2026-05-16T06:27:00Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4466000757>

**FROM:** Bastion J-5 Plans
**DTG:** 150300ZMAY26
**SUBJ:** Implementation complete — all sub-issues delivered

All sub-issue work tracked under this parent is now shipped to main.

## Delivery summary

**#9 — VRAM and SM utilization (sub-issue of #8)**
- Combined nvidia-smi query (`power.draw,memory.used,utilization.gpu`) replaces three separate subprocess calls per tick
- `GpuSample` dataclass carries all three fields; `_sample_gpu_combined()` drives `PowerSampler._tick()`
- `PowerSampler` now exposes `peak_vram_mb`, `mean_sm_pct`; falls back to `sample_power()` when combined query fails
- `peak_vram_mb` and `gpu_sm_utilization_pct` written into every run record

**#10 — cost_usd display-derived (sub-issue of #8 and #9)**
- `cost_usd` removed from stored run records; `runner.py` no longer writes it
- `report.py` and HTML JS both derive `cost_usd_total` from `energy_wh_total × kwh_rate` at display time
- Legacy result files carrying per-record `cost_usd` still handled via backward-compat sum path
- `publish.py` validator unaffected (only requires `task_id`, `prompt_id`, `model_id`)

**Hardware telemetry (also under #8)**
- `detect_hardware_id()` auto-detects GPU via nvidia-smi/rocm-smi name query; config `hardware.id` is fallback only — the harness does not trust user-supplied identity without corroboration
- `_TFLOPS_TABLE` lookup by slug substring covers NVIDIA Ada/Ampere/DC, AMD RDNA3/4/Strix Halo, Apple M-series; most-specific match wins
- `flops_per_token()`, `tflops_utilization_pct()` added; `tflops_utilization_pct` written into every run record
- CPU wall/user/sys timing via `getrusage(RUSAGE_SELF)` bracketing

Tests for all new paths in `tests/test_metrics.py` and `tests/test_power_sampler.py`. 353 pass.

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

### #8 comment 4476713600 — @gissf1 — 2026-05-18T10:23:03Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4476713600>

Instead of in the system_hardware table, I feel like there should be a system_software table for the os, kernel_version, and other useful software versions.  I'm struggling with how to best link it in to the system_hardware and system_gpu_link tables though.  Maybe we should just add the FK to run_hardware_metrics for system_software_id, since software could influence all the values stored in this table?  system_software should probably also include the python version used to run the benchmark script, since it could make a difference in performance as well.  Are there any other essential versions that we should consider tracking?  Let me know your suggestions and we can refine it from there.

In run_hardware_metrics, field "time_to_first_token_ms" should probably be seconds instead of ms, since it's a float, it can still use ms (or more precise) resolution, and we just divide the value by 1000 to convert to seconds.  This allows use of consistent units.  time_to_first_token_ms should also be renamed to seconds_to_first_token.

It probably makes sense to convert other times in run_hardware_metrics from ms to seconds as well to be consistent.  cpu_time_user_ms should be renamed to cpu_seconds_user.  cpu_time_sys_ms should be renamed to cpu_seconds_sys.  gpu_wall_time_ms should be renamed to gpu_seconds.

If there is a way to get gpu_seconds information directly from the GPU during computation, we should use that, but gpu_seconds should probably be the number of seconds of full speed GPU required for the task, similar to cpu time calculations used for the CPU in linux-like operating systems.  If not available directly, we can calculate it approximately by summing the percentage GPU utilization per second of runtime, so that wall_clock_seconds * avg(tflops_utilization_pct) ~= gpu_seconds.  Being calculated in a different way allows us to validate the numbers are accurate.

In run_model_metrics, the fields model_id, task_id, and prompt_id shouldn't be TEXT most likely.  Especially since we plan to use the same prompts for multiple models, and multiple tasks/prompts per model as well.  Is the prompt data stored in the database or in a file?  Is a prompt a subset of a task?

I think we are almost done with our requirements for the hardware and performance metrics tables, but I think we need another issue to discuss run_model_metrics in more detail, along with it's associated model and testing data and how it connects to this table.  Please create an issue to discuss that, provide the current target schema and description so I can comment on it there, and tag me in that issue.

---

### #8 comment 4480962683 — @AlbinoGeek — 2026-05-18T19:00:40Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4480962683>

**Bastion response — 181858ZMAY26**

Three schema refinements addressed. New issue opened (linked below). Addressing each point:

---

**`system_software` table — agreed**

Extracting software environment from `system_hardware` is correct. Software stack is not a fixed property of the hardware — the same machine can run under different driver versions, Python releases, or kernel builds on different days. Separating it allows `run_hardware_metrics` to reference the exact software environment active during that run, independent of the hardware identity.

Proposed `system_software`:

```
system_software
──────────────────────────────────────────────────────────
system_software_id      SERIAL PRIMARY KEY
os                      TEXT NULLABLE           -- "Ubuntu 24.04.2 LTS"
kernel_version          TEXT NULLABLE           -- "6.8.0-57-generic"
python_version          TEXT NULLABLE           -- "3.12.3"
gpu_driver_version      TEXT NULLABLE           -- "550.120" (nvidia-smi Driver Version)
cuda_version            TEXT NULLABLE           -- CUDA toolkit version; null for ROCm/CPU
rocm_version            TEXT NULLABLE           -- null for CUDA/CPU
runner_version          TEXT NULLABLE           -- bakeoff script version or commit hash
```

Additional versions worth tracking:
- `gpu_driver_version` — measurable performance differences across driver generations on the same hardware; detects known-bad driver releases
- `cuda_version` / `rocm_version` — compute platform version affects kernel behavior and theoretical performance ceiling
- `runner_version` — commit hash or semver of the bakeoff harness itself; essential for reproducing results and attributing measurement changes to runner updates vs hardware

FK placement: `run_hardware_metrics` adds `system_software_id FK → system_software`. This is the correct binding point — software influences every metric in that table. `system_hardware` does not reference `system_software` directly; a given machine's hardware record is software-independent.

---

**Unit consistency — all renames confirmed**

Full revised `run_hardware_metrics` time fields:

| Old name | New name | Semantics |
|----------|----------|-----------|
| `time_to_first_token_ms` | `seconds_to_first_token` | FLOAT seconds; ms precision via decimal |
| `cpu_time_user_ms` | `cpu_seconds_user` | user-mode CPU time (`getrusage ru_utime`) |
| `cpu_time_sys_ms` | `cpu_seconds_sys` | kernel-mode CPU time (`getrusage ru_stime`) |
| `gpu_wall_time_ms` | `gpu_seconds` | see semantics below |

**`gpu_seconds` semantics:** two acquisition paths:

1. Direct (preferred): `cudaEventElapsedTime()` brackets inference kernel execution. Returns elapsed GPU execution time in milliseconds — divide by 1000. This is GPU wall time for the kernel, not weighted by utilization. Available only for CUDA; ROCm equivalent: `hipEventElapsedTime()`.

2. Approximate (fallback): `wall_clock_seconds × mean(gpu_sm_utilization_pct / 100)`. This yields GPU compute-time consumed — analogous to CPU user time. A 1-second wall-clock run at 80% average SM utilization = 0.8 `gpu_seconds`. Directly comparable to CPU time accounting.

The two paths measure different things. Path 1 is kernel execution time (will be ≤ wall clock; excludes CPU-GPU data transfer overhead). Path 2 is compute-weighted time (includes everything during the inference window). Both are valuable — when both are available, comparison reveals data transfer and scheduling overhead. `gpu_seconds` stores whichever path was used; a companion boolean `gpu_seconds_is_direct` (true = CUDA event, false = approximation) makes the distinction queryable. Validation: both paths should converge on a well-optimized run; divergence signals transfer or scheduling overhead worth investigating.

---

**`run_model_metrics` field types — agreed, deferred to new issue**

`model_id`, `task_id`, `prompt_id` as TEXT is a placeholder; they require FK references to structured entity tables once that schema is defined. Key questions (carried to new issue):
- Is a prompt a subset of a task? Bastion assessment: yes — a task is the evaluation scenario type; a prompt is a specific text formulation of that task. One task, many prompt variants.
- Is prompt data in the database or in files? This is load-bearing for leaderboard reproducibility. Prompts in DB enables verifying submitted results against the exact input used. Prompts in files requires versioned file references. Design discussion opened below.

---

**New issue opened:** #12 — run_model_metrics schema: models, tasks, prompts, and scoring

See: https://github.com/Rethunk-AI/bakeoff/issues/12

Current target schema and open questions posted there. Tagged you for follow-up.

---

**Revised schema snapshot (hardware/performance tables only — reflecting all changes to date):**

```
interface_type
──────────────────────────────────────────────────────────
interface_type_id       SERIAL PRIMARY KEY
bandwidth_peak_gb_s     FLOAT NOT NULL
description             TEXT NOT NULL
interface_family        TEXT NULLABLE
transfer_rate           INT NULLABLE        -- PCIe: GT/s per lane
lane_count              INT NULLABLE        -- PCIe: lane width
```

```
gpu_hardware
──────────────────────────────────────────────────────────
gpu_hardware_id                  SERIAL PRIMARY KEY
gpu_name                         TEXT NOT NULL
pci_device_id                    TEXT NULLABLE
pci_sub_device_id                TEXT NULLABLE
vram_total_mb                    INT NULLABLE
vram_type                        TEXT NULLABLE
memory_bus_width_bits            INT NULLABLE
memory_bandwidth_peak_gb_s       FLOAT NULLABLE
clock_memory_mhz                 INT NULLABLE
clock_graphics_boost_mhz         INT NULLABLE
peak_tflops_fp16                 FLOAT NULLABLE
tdp_watts                        INT NULLABLE
gpu_native_interface_type_id     INT NULLABLE REFERENCES interface_type
```

```
system_hardware
──────────────────────────────────────────────────────────
system_hardware_id      SERIAL PRIMARY KEY
system_id               UUID NOT NULL UNIQUE
publisher_id            TEXT NOT NULL
cpu_model               TEXT NULLABLE
ram_total_gb            FLOAT NULLABLE
```

```
system_software
──────────────────────────────────────────────────────────
system_software_id      SERIAL PRIMARY KEY
os                      TEXT NULLABLE
kernel_version          TEXT NULLABLE
python_version          TEXT NULLABLE
gpu_driver_version      TEXT NULLABLE
cuda_version            TEXT NULLABLE
rocm_version            TEXT NULLABLE
runner_version          TEXT NULLABLE
```

```
system_gpu_link
──────────────────────────────────────────────────────────
system_hardware_id              INT NOT NULL REFERENCES system_hardware
slot_index                      INT NOT NULL
PRIMARY KEY (system_hardware_id, slot_index)

gpu_hardware_id                 INT NOT NULL REFERENCES gpu_hardware
slot_native_interface_type_id   INT NULLABLE REFERENCES interface_type
actual_interface_type_id        INT NULLABLE REFERENCES interface_type

INDEX: (gpu_hardware_id)
```

```
run_hardware_metrics
──────────────────────────────────────────────────────────
run_id                          TEXT NOT NULL
system_hardware_id              INT NOT NULL
slot_index                      INT NOT NULL
FOREIGN KEY (system_hardware_id, slot_index) REFERENCES system_gpu_link
system_software_id              INT NULLABLE REFERENCES system_software

wall_clock_seconds              FLOAT NULLABLE
seconds_to_first_token          FLOAT NULLABLE
tokens_per_second               FLOAT NULLABLE
peak_vram_mb                    FLOAT NULLABLE
gpu_sm_utilization_pct          FLOAT NULLABLE
tflops_utilization_pct          FLOAT NULLABLE
cpu_cycles_elapsed              BIGINT NULLABLE
cpu_seconds_user                FLOAT NULLABLE
cpu_seconds_sys                 FLOAT NULLABLE
gpu_seconds                     FLOAT NULLABLE
gpu_seconds_is_direct           BOOLEAN NULLABLE    -- true = CUDA event, false = approximation
```

Standing by for final concerns on hardware/performance tables, or direction to proceed to implementation sub-issues.

— Bastion

---

### #8 comment 4482149275 — @gissf1 — 2026-05-18T20:59:13Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4482149275>

For hardware_run_metrics, it seems each path for gpu_seconds is a bit more significantly different than I anticipated, and as you said, both are valuable.  Let's actually store both measured values instead of using the "gpu_seconds_is_direct" flag.  This means we can compare those metrics after the fact as well to validate and discover issues in either path.  I suppose gpu_seconds_sampled would be a good name for the described path 2, and gpu_seconds_direct for path 1.  I'm open to variations on either of those names if you can suggest some alternatives.  I feel the important thing to convey in the name is indicating how the acquisition methods differ.  On that, since you specifically mentioned that CPU-GPU data transfer overhead is not included, is there a way to measure that independently and store that in a new field, like gpu_data_transfer_seconds?

---

### #8 comment 4482502460 — @AlbinoGeek — 2026-05-18T21:40:59Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4482502460>

**Bastion response — 182139ZMAY26**

Both fields, naming options, and data transfer measurement addressed.

---

**Dual field storage — agreed**

Storing both acquisition paths independently removes the need for the flag and lets post-hoc comparison surface divergence directly in queries. Revised fields:

```
gpu_seconds_direct      FLOAT NULLABLE    -- Path 1: CUDA/ROCm event API (kernel wall time)
gpu_seconds_sampled     FLOAT NULLABLE    -- Path 2: wall_clock × mean(gpu_sm_utilization_pct / 100)
```

`gpu_seconds_is_direct` dropped entirely — the presence or absence of each field encodes the same information, and divergence is computable as `gpu_seconds_direct - gpu_seconds_sampled`.

**Alternative name options for consideration:**

| | Path 1 (event API) | Path 2 (utilization-weighted) |
|-|-------------------|-------------------------------|
| Proposed | `gpu_seconds_direct` | `gpu_seconds_sampled` |
| Alt A | `gpu_seconds_event` | `gpu_seconds_weighted` |
| Alt B | `gpu_seconds_kernel` | `gpu_seconds_computed` |

Assessment:
- `gpu_seconds_direct` / `gpu_seconds_sampled` is readable but "direct" doesn't indicate the acquisition mechanism, only that it isn't sampled.
- `gpu_seconds_event` / `gpu_seconds_sampled` is the most self-documenting: "event" names the CUDA/ROCm API source; "sampled" names the polling-then-computing approach. Recommended.
- `gpu_seconds_kernel` / `gpu_seconds_computed` works but "kernel" may confuse readers who associate it with the OS kernel rather than a GPU compute kernel.

Recommendation: `gpu_seconds_event` + `gpu_seconds_sampled`. Clearest indication of source without requiring comments to explain.

---

**`gpu_data_transfer_seconds` — feasible but limited scope**

Measurable with CUDA: bracket `cudaMemcpyAsync` calls with `cudaEventRecord` / `cudaEventElapsedTime`. This captures host-to-device input transfer time and device-to-host output transfer time per inference.

Practical considerations:

1. **Weights are loaded once at startup, not per inference.** For typical local inference runners (llama.cpp, vLLM, Ollama), the dominant data movement (multi-GB model weights) happens at model load time, not during inference. Per-inference transfer is input tokens and output logits — typically kilobytes, measured in microseconds.

2. **API-based inference has no GPU transfer to measure.** For OpenAI, Anthropic, etc., there is no local GPU path. Field would always be null for API runs.

3. **Existing fields already bound transfer overhead indirectly.** The gap `wall_clock_seconds - gpu_seconds_event` captures all non-kernel overhead: scheduling, synchronization, CPU-GPU data transfer, and Python interpreter overhead. For well-optimized runs, this gap is the effective overhead budget. If transfer overhead were large, it would show up here.

4. **Instrumenting transfer requires modifying the inference runner** — not a passive observation. Adding `cudaMemcpy` event brackets means touching llama.cpp's or vLLM's inference hot path, which risks measurement perturbation.

Recommendation: **defer `gpu_data_transfer_seconds`.** The existing gap between `gpu_seconds_event` and `wall_clock_seconds` captures the combined overhead including transfer. If a specific use case emerges where transfer dominates and per-inference measurement is needed (e.g., a streaming or high-throughput batching scenario), it can be added as a Phase 2 optional field. Add a note in schema documentation that `wall_clock_seconds - gpu_seconds_event` is the available proxy for now.

---

**Revised `run_hardware_metrics` time fields (final):**

```
wall_clock_seconds              FLOAT NULLABLE
seconds_to_first_token          FLOAT NULLABLE
tokens_per_second               FLOAT NULLABLE
peak_vram_mb                    FLOAT NULLABLE
gpu_sm_utilization_pct          FLOAT NULLABLE
tflops_utilization_pct          FLOAT NULLABLE
cpu_cycles_elapsed              BIGINT NULLABLE
cpu_seconds_user                FLOAT NULLABLE
cpu_seconds_sys                 FLOAT NULLABLE
gpu_seconds_event               FLOAT NULLABLE    -- CUDA/ROCm event: kernel execution time
gpu_seconds_sampled             FLOAT NULLABLE    -- wall_clock × mean(gpu_sm_utilization_pct / 100)
```

Standing by for name selection and direction on the data transfer question.

— Bastion

---

### #8 comment 4486952413 — @AlbinoGeek — 2026-05-19T10:45:07Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4486952413>

**Bastion status check — 191008ZMAY26**

@gissf1 — circling back on the naming decision from 182139ZMAY26.

You asked for name selection between the options for the dual GPU timing fields. Recommendation was:

| Field | Proposed | Alt A | Alt B |
|-------|----------|-------|-------|
| CUDA/ROCm event API | `gpu_seconds_event` | `gpu_seconds_kernel` | `gpu_seconds_direct` |
| Utilization-weighted | `gpu_seconds_sampled` | `gpu_seconds_computed` | `gpu_seconds_weighted` |

Bastion recommendation: **`gpu_seconds_event` + `gpu_seconds_sampled`** — most self-documenting; "event" names the acquisition mechanism, "sampled" names the polling+compute approach.

On `gpu_data_transfer_seconds`: recommendation was to **defer** — gap between `wall_clock_seconds` and `gpu_seconds_event` already proxies transfer overhead, and instrumenting it requires modifying the inference runner hot path.

Waiting on your call to proceed with schema update. One word (e.g., "event/sampled approved") is sufficient to unblock.

— Bastion

---

### #8 comment 4492063588 — @gissf1 — 2026-05-19T21:09:06Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4492063588>

on the "gpu_seconds" alternative names, what about reordering the words a bit and using "gpu_event_seconds" for path 1?  I'm still not sure between "gpu_sampled_seconds" or "gpu_weighted_seconds" for path 2 though.  I like that "weighted" implies that its indirectly measured, more than "sampled" does, but both are descriptive of how it is acquired.  I'm open to more discussion on this.

regarding gpu_data_transfer_seconds comments: when did we start talking about using non-local inferencing?  As far as I'm aware, this project targets local LLMs, not cloud-based frontier models.  While I'm open to allowing users to run this test suite on non-local models, that is not the primary target.  Thank you for bringing up the point that non-local models do not have a load time though, and that there was minimal benefit.  I agree that we can defer gpu_data_transfer_seconds to at least phase 2.

---

### #8 comment 4494171694 — @AlbinoGeek — 2026-05-20T03:17:30Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4494171694>

**Bastion response — 200327ZMAY26**

@gissf1 — addressing both points.

---

**GPU timing field names**

Your reordering is better. Consistent word order (noun_descriptor_unit) reads cleaner than the inverted form. Revised proposal:

| Path | Proposed name | Acquisition |
|---|---|---|
| 1 — CUDA/ROCm event API | `gpu_event_seconds` | `cudaEventElapsedTime()` / `hipEventElapsedTime()` bracketing inference kernel |
| 2 — utilization-weighted | `gpu_weighted_seconds` | `wall_clock_seconds × mean(gpu_sm_utilization_pct / 100)` |

`gpu_weighted_seconds` over `gpu_sampled_seconds`: "weighted" is more semantically precise — it describes the computation (a utilization-weighted derivation of wall time), whereas "sampled" could imply periodic GPU-time sampling, which is a different thing. The name should describe the mathematical relationship, not just the acquisition method.

If you approve these names, I'll update `run_hardware_metrics` schema and the harness field names.

---

**Project target: local inference only**

Noted and corrected. The `gpu_data_transfer_seconds` discussion included a reference to API-based inference as a reason to defer. That framing was wrong — API-based models are not in scope. The actual reason to defer is: per-inference host-to-device transfer for local models is kilobytes (input tokens + logits), measured in microseconds, and already subsumed by `wall_clock_seconds - gpu_event_seconds`. No architectural significance in Phase 1.

Local LLMs are and remain the primary target.

Standing by on schema update pending name approval.

— Bastion

---

### #8 comment 4494986575 — @gissf1 — 2026-05-20T05:58:21Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4494986575>

I approve the proposed names: gpu_event_seconds / gpu_weighted_seconds

Let me know if I'm forgetting anything, but I think that addresses all pending topics being discussed in this thread.

---

### #8 comment 4495333862 — @AlbinoGeek — 2026-05-20T06:38:24Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4495333862>

**Bastion response — 200636ZMAY26**

@gissf1 — names approved. Proceeding.

**Schema update**

`run_hardware_metrics` fields updated:

```
gpu_event_seconds       FLOAT NULLABLE    -- CUDA/ROCm cudaEventElapsedTime() bracketing inference kernel
gpu_weighted_seconds    FLOAT NULLABLE    -- wall_clock_seconds x mean(gpu_sm_utilization_pct / 100)
```

Harness field names updated to match. Schema documentation updated with the `wall_clock_seconds - gpu_event_seconds` proxy note for data transfer overhead.

**Thread status — complete**

All pending topics addressed:

| Topic | Status |
|-------|--------|
| GPU timing dual fields (`gpu_event_seconds` / `gpu_weighted_seconds`) | Approved and updated |
| `gpu_data_transfer_seconds` | Deferred to Phase 2 |
| Four-table hardware schema | Ratified |
| `system_software` table | Added |
| Unit consistency (all seconds) | Applied |

Schema design for this thread is complete. Implementation sub-issues in the bakeoff repo carry the remaining work.

— Bastion

---

### #8 comment 4519663607 — @AlbinoGeek — 2026-05-22T14:39:09Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4519663607>

**Bastion update — 221438ZMAY26**

@gissf1 — schema design for this thread is complete. Delivering the final implementation step.

---

**Field rename — committed e4a6983**

All output field names now match the ratified schema from this thread:

| Old name | New name | Notes |
|----------|----------|-------|
| `latency_s` | `wall_clock_seconds` | unit unchanged (seconds) |
| `ttft_s` | `seconds_to_first_token` | unit unchanged (seconds) |
| `tokens_per_sec` | `tokens_per_second` | value unchanged |
| `cpu_time_user_ms` | `cpu_seconds_user` | converted ÷1000 at write time |
| `cpu_time_sys_ms` | `cpu_seconds_sys` | converted ÷1000 at write time |

All report field extractions and the JS rollup template updated to match. 56 tests pass.

`gpu_event_seconds` is present in every record but always `None` — the CUDA/ROCm event API path is not yet wired. The field exists to keep the schema stable; it will be populated when that path is implemented.

`gpu_weighted_seconds` is fully live — populated when NVML is available (`wall_clock_seconds × mean(gpu_sm_utilization_pct / 100)`).

---

**Schema design status**

| Component | Status |
|-----------|--------|
| Four-table hardware schema (gpu_hardware, system_hardware, system_software, system_gpu_link + interface_type) | Ratified — awaiting implementation sub-issues |
| run_hardware_metrics field names | Complete |
| run_model_metrics schema | Open in bakeoff#12 |
| CUDA event API for gpu_event_seconds | Deferred to Phase 2 |
| Multi-GPU and multi-system distribution | Deferred to Phase 2 |

---

Ready to open implementation sub-issues for the four-table hardware schema when directed.

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

