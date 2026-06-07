# Topic: run_model_metrics

Consolidated chat history (chronological, ascending comment-id). 16 entries. Verbatim quotes; attribution in each header. **#38 thread excluded from this variant.**

## Source entries (provenance TOC)

Entries used to build this topic and its summary. (Not migrated into `topic_summary/`; audit reference only.)

| ticket | entry | author | posted | url |
|---|---|---|---|---|
| #8 | issue body | @gissf1 | 2026-05-13T20:53:09Z | <https://github.com/Rethunk-AI/bakeoff/issues/8> |
| #8 | comment 4445642971 | @AlbinoGeek | 2026-05-13T22:19:56Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4445642971> |
| #8 | comment 4450406819 | @gissf1 | 2026-05-14T11:49:57Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4450406819> |
| #8 | comment 4462460688 | @AlbinoGeek | 2026-05-15T18:52:20Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4462460688> |
| #8 | comment 4476713600 | @gissf1 | 2026-05-18T10:23:03Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4476713600> |
| #12 | issue body | @AlbinoGeek | 2026-05-18T19:00:32Z | <https://github.com/Rethunk-AI/bakeoff/issues/12> |
| #8 | comment 4480962683 | @AlbinoGeek | 2026-05-18T19:00:40Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4480962683> |
| #12 | comment 4491873240 | @gissf1 | 2026-05-19T20:42:32Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4491873240> |
| #12 | comment 4494171918 | @AlbinoGeek | 2026-05-20T03:17:33Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4494171918> |
| #12 | comment 4495675210 | @gissf1 | 2026-05-20T07:25:23Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4495675210> |
| #12 | comment 4495777337 | @AlbinoGeek | 2026-05-20T07:36:14Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4495777337> |
| #12 | comment 4496938804 | @gissf1 | 2026-05-20T09:46:32Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4496938804> |
| #12 | comment 4497170538 | @AlbinoGeek | 2026-05-20T10:06:37Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4497170538> |
| #12 | comment 4501587412 | @gissf1 | 2026-05-20T18:48:00Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4501587412> |
| #12 | comment 4503203398 | @AlbinoGeek | 2026-05-20T22:36:24Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503203398> |
| #12 | comment 4503669210 | @AlbinoGeek | 2026-05-21T00:11:41Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503669210> |

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

### #8 comment 4476713600 — @gissf1 — 2026-05-18T10:23:03Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4476713600>

Instead of in the system_hardware table, I feel like there should be a system_software table for the os, kernel_version, and other useful software versions.  I'm struggling with how to best link it in to the system_hardware and system_gpu_link tables though.  Maybe we should just add the FK to run_hardware_metrics for system_software_id, since software could influence all the values stored in this table?  system_software should probably also include the python version used to run the benchmark script, since it could make a difference in performance as well.  Are there any other essential versions that we should consider tracking?  Let me know your suggestions and we can refine it from there.

In run_hardware_metrics, field "time_to_first_token_ms" should probably be seconds instead of ms, since it's a float, it can still use ms (or more precise) resolution, and we just divide the value by 1000 to convert to seconds.  This allows use of consistent units.  time_to_first_token_ms should also be renamed to seconds_to_first_token.

It probably makes sense to convert other times in run_hardware_metrics from ms to seconds as well to be consistent.  cpu_time_user_ms should be renamed to cpu_seconds_user.  cpu_time_sys_ms should be renamed to cpu_seconds_sys.  gpu_wall_time_ms should be renamed to gpu_seconds.

If there is a way to get gpu_seconds information directly from the GPU during computation, we should use that, but gpu_seconds should probably be the number of seconds of full speed GPU required for the task, similar to cpu time calculations used for the CPU in linux-like operating systems.  If not available directly, we can calculate it approximately by summing the percentage GPU utilization per second of runtime, so that wall_clock_seconds * avg(tflops_utilization_pct) ~= gpu_seconds.  Being calculated in a different way allows us to validate the numbers are accurate.

In run_model_metrics, the fields model_id, task_id, and prompt_id shouldn't be TEXT most likely.  Especially since we plan to use the same prompts for multiple models, and multiple tasks/prompts per model as well.  Is the prompt data stored in the database or in a file?  Is a prompt a subset of a task?

I think we are almost done with our requirements for the hardware and performance metrics tables, but I think we need another issue to discuss run_model_metrics in more detail, along with it's associated model and testing data and how it connects to this table.  Please create an issue to discuss that, provide the current target schema and description so I can comment on it there, and tag me in that issue.

---

### issue #12 body — @AlbinoGeek — 2026-05-18T19:00:32Z
<https://github.com/Rethunk-AI/bakeoff/issues/12>

**`run_model_metrics` schema — models, tasks, prompts, and scoring**

Opened per discussion in #8 (hardware/performance metrics). This issue covers the model-side schema: how models, tasks, and prompts are represented and how `run_model_metrics` references them.

---

## Current target schema (placeholder — FKs not yet resolved)

```
run_model_metrics
──────────────────────────────────────────────────────────
run_id                          TEXT NOT NULL PRIMARY KEY
model_id                        TEXT NOT NULL      -- FK to models (TBD)
task_id                         TEXT NOT NULL      -- FK to tasks (TBD)
prompt_id                       TEXT NOT NULL      -- FK to prompts (TBD)
score                           FLOAT NULLABLE
pass_fail                       BOOLEAN NULLABLE
flops_per_token_theoretical     BIGINT NULLABLE
```

`model_id`, `task_id`, `prompt_id` are currently TEXT placeholders. This issue resolves them into proper entity tables with FK references.

---

## Open questions for discussion

**Q1 — Task / prompt relationship**

Working assumption: a *task* is an evaluation scenario type (e.g., "code generation", "multi-step reasoning", "summarization"). A *prompt* is a specific text formulation of that task. One task may have many prompt variants. Is this correct?

**Q2 — Prompt data storage: database or file?**

Two options:

- **In database:** `prompts` table stores prompt text. Submitted results reference `prompt_id` FK — leaderboard can verify submitted results against the exact input used. Enables full reproducibility audit.
- **In files:** prompts stored in versioned benchmark config files. `prompt_id` is a content hash or file path reference. Simpler runner integration; requires tracking file version alongside result.

Which path is preferred? This is load-bearing for leaderboard result verification.

**Q3 — Model identity**

What constitutes a distinct `model_id`? Options:
- Provider + model name + version string (e.g., `openai/gpt-4o-2024-11-20`)
- A structured `models` table with separate fields for provider, model_family, version, parameter_count, quantization, context_length, etc.

A structured `models` table enables hardware-normalized comparisons across quantization variants of the same base model. Is that level of detail needed in Phase 1?

**Q4 — Scoring schema**

`score FLOAT` and `pass_fail BOOLEAN` are both present. Are these always computed from the same test, or do some tasks produce only a score (no binary), and others only pass/fail (no continuous score)? Should these be separated into different record types, or is nullable acceptable?

**Q5 — Link to run_hardware_metrics**

`run_hardware_metrics` and `run_model_metrics` both reference `run_id`. Is `run_id` a natural key (e.g., UUID generated per invocation), or should there be an explicit `runs` table that both tables FK into? A `runs` table would carry: timestamp, submitter, runner_version, and serve as the join anchor for all per-run data.

---

## Proposed starting point (pending Q1–Q5 answers)

```
models
──────────────────────────────────────────────────────────
model_id            SERIAL PRIMARY KEY
provider            TEXT NOT NULL           -- "openai", "anthropic", "local"
model_name          TEXT NOT NULL           -- "gpt-4o"
model_version       TEXT NULLABLE           -- "2024-11-20"
parameter_count     BIGINT NULLABLE
quantization        TEXT NULLABLE           -- "fp16", "int4", "gguf-q8_0"
context_length      INT NULLABLE
```

```
tasks
──────────────────────────────────────────────────────────
task_id             SERIAL PRIMARY KEY
task_name           TEXT NOT NULL
task_description    TEXT NULLABLE
task_category       TEXT NULLABLE           -- "code_generation", "reasoning", etc.
```

```
prompts
──────────────────────────────────────────────────────────
prompt_id           SERIAL PRIMARY KEY
task_id             INT NOT NULL REFERENCES tasks
prompt_text         TEXT NOT NULL           -- if stored in DB
prompt_hash         TEXT NOT NULL           -- SHA256 of prompt_text; used for dedup + file-ref fallback
```

```
run_model_metrics
──────────────────────────────────────────────────────────
run_id              TEXT NOT NULL PRIMARY KEY
model_id            INT NOT NULL REFERENCES models
task_id             INT NOT NULL REFERENCES tasks
prompt_id           INT NOT NULL REFERENCES prompts
score               FLOAT NULLABLE
pass_fail           BOOLEAN NULLABLE
flops_per_token_theoretical  BIGINT NULLABLE
```

---

@gissf1 — tagged for review. Provide direction on Q1–Q5 and we can finalize the schema here before opening implementation sub-issues.

— Bastion

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

### #12 comment 4491873240 — @gissf1 — 2026-05-19T20:42:32Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4491873240>

A1: I believe that design makes sense to me.  But in that case, I would think a prompt record should have a FK to the task_id it belongs to.  That would also mean we don't need both run_model_metrics.

I think the definition of a prompt is pretty self explanatory, but task is still a bit vague.  How does a task compare to a test or a test suite?  Do any of those terms describe the same thing?  Different things?  How are they related?

A2: What is the best design here?  I feel like having data versioned in files is helpful because we may want to go back and refine the prompts, in which case we would want that version controlled.  And for each run, we would want to know exactly what prompt was given to the model.  This also means a refined prompt can be then tested with the same models to determine its improved effectiveness.  Also, we want to track how changes in a prompt affect various models for potential A/B split comparisons and other metrics.  Keeping the versions in files means we can reference the file and git commit hash in which that file/prompt was last modified to uniquely identify it.  The current files should be easily accessible from the UI, and even historical files are available with a bit of effort.  One could potentially reference GitHub raw URLs or other such things to quickly acquire the file data.

A3: I think a specific source URL makes sense since it's relatively unique.  one concern is if the model from different sources (from ollama vs huggingface for example) are the same model or not.  We should probably do some analysis on this.  If the identical models are available from different sources, perhaps we should have a list of URLs indicating identical models so we can coalesce them into a single model in our system (this helps with organization and reduces duplicate testing).  I also don't know how either of those sites handles versioning - if the owner of the model revises their model, does it use the same URL or a new one?  Another thing to consider - if a model has a predecessor, it probably makes sense to track that so we can then analyze the evolution of a model.

I think a models table could make sense, at minimum it should have a source URL (or a list of them, if multiple), a name (user friendly), a release date, maybe a version, maybe some description about it, probably scraped from something like the source URLs.  For deduplication, maybe we can find some way to have a standard unique identifier for a model; I'm imagining something like the sha of its model data as a worst case, but ideally it would be something pre-computed and generated elsewhere to avoid the extra compute time on a large blob like that.  It should also have parameters describing the model - size, parameter_count, family, context_length (maybe identify the min, max, and default), format ("GGUF" or others, since this may make a different in output, is "format" the right term for this attribute?), quantization, and minimum VRAM required for runtime (important since this may be very different than the parameter_count if quantization differs).  There may be other fields I didn't think of yet.

Q4: I think all tests should have both a pass/fail mark and a score.  Both metrics should be absolute based on the model capabilities, and not on performance or hardware related issues/limitations.  Being unable to complete a test because the hardware it ran on is insufficient is not a failure of the model.

There should be a separate metric created later for performance analysis combined with the quality score.  This secondary metric will take into consideration the information from run_model_metrics and such to determine not only accuracy, but also relative performance to determine the model's effectiveness and efficiency.  This metric would be hardware neutral, but take into consideration which model best utilizes the available hardware to achieve the goal.

Q5: Let's discuss these options more.  If we didn't create a table for runs, where would we put the run-specific data like timestamp, submitter, runner_version?  I would think either the FK or natural key methods are identical other than the format of the field (UUID vs serial integer), am I missing something?  Also, I'm assuming the runner_version is just the testing software and not the actual prompt data itself (this may be an incorrect assumption), but beyond that we should also include the SHA hash of the git commit the prompt data is based on, or a hash of the prompt data in the worst case.  Maybe we should also have a prompt-hash-to-release-date table so we can determine how out of date test results are and order them by "oldest first" when retesting older models.

Regarding the preliminary table starting point, I have some thoughts and comments here as well:

In the models table, I think "provider" is the wrong context.  Most of the models we intend to test are locally hosted, so the model's provider might be better thought of as "developer" or "source", unless I'm misunderstanding your intent here.  That said, I'm not opposed to having multiple fields to represent that data.  In that case, perhaps "external_provider" would be a better name for the current field, so that it is null by default for all local models?  I suppose we could create a providers table at some point and make this field an FK into the providers table, but none of that is really a concern for Phase 1.

for tasks, I think it makes sense to have a tiered structure available, so we can have aggregate tasks (representing a test suite) that have subordinate tasks.  For this I think we can have a task_categories table so that the task_category fields is a FK rather than redundant text.  I also think there should be a nullable parent_id field which points to the parent task_id if it is a subordinate task.  I think it also makes sense to have an order field to control the order in which the tasks and prompts are executed, unless that is stored on the filesystem.  That said, I'm not convinced that tasks and prompts should be database centric.  I feel like they should be derived from filesystem data so they are version controlled.  Let me know your thoughts on this.

I feel like prompts should be stored in a file specifically, but some aspects of them likely need to be referenced by the database. Besides the commit hash in which they were last modified, we should also probably include the raw length of the prompt text, version, and release date.  Maybe it also makes sense to have a prerelease flag for testing new prompts against old prompts before an official release?

for run_model_metrics, do we need task_id if we have the prompt_id?  I'm still not sure that we want the prompt data databased, or if we should rely on filesystem data.  we also need to plan the testing queue, so we should probably start getting into that at some point later, maybe you should make a new ticket for that to be addressed once we figure out the model aspects?  You let me know the prioritization order on that, since planning is your expertise.

Anyhow, that's what I have for comments so far, let's start there, and we can iterate and discuss more as needed.

---

### #12 comment 4494171918 — @AlbinoGeek — 2026-05-20T03:17:33Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4494171918>

**Bastion response — 200327ZMAY26**

@gissf1 — addressing all points in order.

---

**A1: Prompt FK to task_id + task/test/test suite terminology**

Agreed — a prompt record should have a `task_id` FK. This also lets `run_model_metrics` drop the separate `task_id` field: prompt implies task.

Terminology disambiguation:

| Term | Definition |
|---|---|
| **Test suite** | A named collection of tasks run together (e.g., "baseline tier", "comparison tier"). Maps to a top-level task record with `parent_id IS NULL`. |
| **Task** | A specific evaluation type within a suite (e.g., "bash_scripting", "code_generation_python"). Corresponds to a single evaluation scenario, potentially with multiple prompt variants. |
| **Prompt** | A specific text formulation of a task. One task can have many prompt variants (for A/B testing, difficulty levels, etc.). The prompt is what gets sent to the model. |
| **Test run** | One execution of a prompt against one model on specific hardware. Produces `run_model_metrics` + `run_hardware_metrics` rows. |

"Test" as a term is overloaded — I'll avoid it in schema names and use "task", "prompt", and "run" as the canonical vocabulary.

---

**A2: File-based prompts + DB metadata — agreed**

File-based prompt content is correct: version control is the right persistence layer for prompt text. Git gives us full history, diff, and blame for free.

DB metadata stored per prompt:
- `file_path` — relative path within the repo
- `git_commit_hash` — SHA of the commit in which this prompt was last modified (deterministic unique identifier for this prompt version)
- `content_length_bytes` — byte count of the raw prompt text
- `version` — human-readable version label (e.g., "1.0", "1.1-prerelease")
- `release_date` — date this version was officially released
- `is_prerelease` — flag for prompts in testing before official release
- `task_id` FK — which task this prompt belongs to

For reproduction: given a `run_model_metrics` row, the `git_commit_hash` on its referenced `prompt` record uniquely identifies the exact prompt text used. Historical versions accessible via git. No need to store prompt text in the DB.

---

**A3: Model deduplication + versioning**

Source URL as primary identifier: reasonable starting point, but insufficient alone — HuggingFace and Ollama can host the same weights under different URLs.

Proposed `models` table:

```
models
──────────────────────────────────────────────────────────
model_id                SERIAL PRIMARY KEY
name                    TEXT NOT NULL              -- user-friendly name: "Phi-3.5-mini-instruct"
creator                 TEXT NULLABLE              -- "microsoft", "meta", "mistralai"
source_urls             TEXT[] NOT NULL            -- all known source URLs (HF, Ollama, etc.)
model_hash              TEXT NULLABLE              -- SHA256 of model weights file; deduplication ground truth
release_date            DATE NULLABLE
version                 TEXT NULLABLE              -- model version label if versioned
description             TEXT NULLABLE              -- scraped or manually entered
predecessor_model_id    INT NULLABLE REFERENCES models  -- lineage tracking

-- Architecture / capability
parameter_count         BIGINT NULLABLE            -- total params
active_parameter_count  BIGINT NULLABLE            -- active params (MoE only)
architecture            TEXT NULLABLE              -- "Dense", "MoE"
context_length_default  INT NULLABLE
context_length_min      INT NULLABLE
context_length_max      INT NULLABLE
file_format             TEXT NULLABLE              -- "GGUF", "SafeTensors", etc.
quantization            TEXT NULLABLE              -- "Q4_K_M", "fp16", etc.
min_vram_mb             INT NULLABLE               -- minimum VRAM required at runtime
```

"provider" → "creator" (or "developer"). These are locally-hosted models; the model's creator is the research lab or team, not a cloud provider.

Deduplication strategy (in preference order):
1. `model_hash` match — authoritative (same weights = same model regardless of source)
2. HuggingFace model ID match — reliable for HF-sourced models; parse from source URL
3. Name + creator + quantization match — fallback heuristic; flag for human review

URL versioning: if a HF model URL resolves to different weights at different times, the `git_commit_hash` on the HF model repo differentiates versions. Recommend storing a `hf_commit_hash` field alongside `model_hash` for HF-sourced models.

---

**Q4: Score + pass/fail both required; hardware-neutral**

Agreed on both points:
- Both `score` (float 0–1) and `pass_fail` (boolean) stored per run
- Score and pass/fail are model-capability metrics only — hardware limitations (OOM, timeout) are a `failure_reason`, not a scoring event. A model that times out gets `failure_reason = "timeout"`, not a failing score.
- The combined hardware+quality efficiency metric is a Phase 2 derived view, not a stored field. It can be computed from `run_model_metrics × run_hardware_metrics` at display time.

---

**Q5: Runs table + natural key vs surrogate + prompt git hash**

A `runs` table is needed. Fields that are run-scoped (timestamp, submitter, runner commit hash) don't belong on `run_model_metrics` or `run_hardware_metrics` — those tables describe what was measured, not who ran it or when.

Proposed `runs` table:

```
runs
──────────────────────────────────────────────────────────
run_id                  UUID PRIMARY KEY           -- generated at run time; natural key
submitted_at            TIMESTAMPTZ NOT NULL
publisher_id            TEXT NOT NULL              -- submitting user/account
runner_version          TEXT NULLABLE              -- bakeoff harness semver or commit hash
prompt_git_hash         TEXT NULLABLE              -- git commit hash of prompt files at run time
```

UUID as `run_id` is the right choice over serial integer: UUIDs are globally unique without a database sequence, can be generated client-side before the DB write, and prevent enumeration of run IDs by outsiders. Natural key (the UUID itself) is sufficient; no surrogate needed.

`prompt_git_hash` here is the harness-level hash (which commit of the prompts repo was in use at run time), distinct from the per-prompt `git_commit_hash` on the `prompts` table. Both are useful: the prompts table hash pinpoints which version of a specific prompt was used; the run-level hash pins the entire prompt set version.

`run_model_metrics` and `run_hardware_metrics` both add a `run_id FK → runs`.

---

**Preliminary table comments**

`models.creator` over `provider` — confirmed above. "developer" also acceptable; "creator" is one word shorter and unambiguous.

**Tiered task structure** — agreed. Proposed:

```
task_categories
──────────────────────────────────────────────────────────
category_id             SERIAL PRIMARY KEY
name                    TEXT NOT NULL UNIQUE       -- "baseline", "comparison", "advanced"
description             TEXT NULLABLE

tasks
──────────────────────────────────────────────────────────
task_id                 SERIAL PRIMARY KEY
name                    TEXT NOT NULL
category_id             INT NULLABLE REFERENCES task_categories
parent_id               INT NULLABLE REFERENCES tasks  -- null = top-level (suite)
sort_order              INT NOT NULL DEFAULT 0     -- execution order within parent
description             TEXT NULLABLE
```

Top-level tasks (`parent_id IS NULL`) are suites. Sub-tasks are the individual evaluation types. `sort_order` controls execution sequence.

**File-based tasks/prompts vs DB-centric** — hybrid is correct:
- Task *definitions* (name, category, sort order, parent) live in the DB. These change rarely and are queryable for scheduling and reporting.
- Prompt *content* lives in files (git-tracked). DB stores metadata pointing to the file version.
- No redundancy: task rows don't contain prompt text; prompt rows don't embed task logic.

**`run_model_metrics.task_id`** — drop it if `prompt_id` FK already implies `task_id` via `prompts.task_id`. The task is derivable: `run → prompt → task`. Denormalization is only warranted if that join is a hot query path (e.g., "all runs for task X regardless of prompt variant"). Recommendation: omit for now; add a `task_id` column to `run_model_metrics` only if query profiling shows the join is costly.

**Testing queue** — agreed, defer to a new issue. Standing by to open it on direction.

---

**Schema summary (updated, run_model_metrics section):**

```
runs
──────────────────────────────────────────────────────────
run_id                  UUID PRIMARY KEY
submitted_at            TIMESTAMPTZ NOT NULL
publisher_id            TEXT NOT NULL
runner_version          TEXT NULLABLE
prompt_git_hash         TEXT NULLABLE

prompts
──────────────────────────────────────────────────────────
prompt_id               SERIAL PRIMARY KEY
task_id                 INT NOT NULL REFERENCES tasks
file_path               TEXT NOT NULL
git_commit_hash         TEXT NOT NULL
content_length_bytes    INT NULLABLE
version                 TEXT NULLABLE
release_date            DATE NULLABLE
is_prerelease           BOOLEAN NOT NULL DEFAULT FALSE

run_model_metrics
──────────────────────────────────────────────────────────
run_id                  UUID NOT NULL REFERENCES runs
prompt_id               INT NOT NULL REFERENCES prompts
model_id                INT NOT NULL REFERENCES models
score                   FLOAT NULLABLE
pass_fail               BOOLEAN NULLABLE
failure_reason          TEXT NULLABLE              -- "timeout", "oom", "parse_error", null = success
flops_per_token_theoretical  BIGINT NULLABLE
PRIMARY KEY (run_id, prompt_id, model_id)
```

Ready to open a testing-queue issue when directed.

— Bastion

---

### #12 comment 4495675210 — @gissf1 — 2026-05-20T07:25:23Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4495675210>

I agree with most of your "terminology disambiguation" section, but does that mean for each task, there is at most 1 prompt sent to the model being tested?  I would think different difficulty levels should be part of the task definition, not the prompt specifically.  In fact, I had considered that within a single task there might be multiple prompts sent to a model for testing it in different ways; potentially more than one of the prompts in a single test suite even.  I'm not sure how most test suites are implemented, but perhaps you could provide some insight into what the industry standard has been on this and we can refine our design from there.  Good call on defining and using the canonical vocabulary though to promote clear communications.

Regarding the file based prompts, is there any other data we should pull from git rather than storing in the database?

For "Model deduplication + versioning", I don't want to rely just on huggingface as our source of truth, but it is a common place to get models.  Your suggestion to use it as a starting point makes sense, but I don't think their model_id should directly become ours.  I don't think tightly integrating into their ecosystem is ideal for a general benchmarking and reporting system.  We should be provider agnostic.

Specifically on the models table itself: I don't think using "bigint" is the most effective use of storage for parameter counts.  We could easily use approximates with unit suffixes stored in a string, or floats based on some fixed units (Billion?).  creator might be best implemented as a reference table "creators" so models would include an ID field for that, but the creators table can have multiple homepages and IDs across different services (huggingface, ollama, others) for the same actual creator.

Regarding deduplication, I think model_hash being authoritative is correct.  Huggingface model id is ok for HF models, but I would want to still verify the hash.  In fact, I would argue the model_hash should be a UNIQUE field, and if a duplicate arrives, it should be flagged, and, after human review, the option should be offered to add it to the previous model record containing that hash along with coalescing the other data in an appropriate way.  This coalescing process should clearly indicate to the administrator what fields will be changed and to what values they will be changed.  I feel that name + creator + quantization is acceptable to trigger a hash check comparison, but should not be treated as identical (or coalesced) without checking the hash first.

If the URL resolves to different weights at different times, I would consider those different models or at least different versions.  Hopefully this is resolvable with some commit hash, release date, or version to uniquely identify each of them.  If not, their model_hash is probably a reasonable worst-case fallback.  This may also play into how we determine the predecessor model and who is the ancestor and who is the descendant.

maybe we should separate out "model_sources" into its own table, which can then store a URL, our model_id, and the source type (huggingface, ollama, etc, in a lookup table or ENUM), along with some source specific tags (git_commit_hash, Huggingface model id, etc) in a JSON, text, or blob field.  My goal here is to keep the core tables Huggingface agnostic, but still provide access to useful information for validation later.  I can see some down sides to this as well, but I'd like to discuss it in more detail.

While I agree with the score being stored as a float, I think on the UI side, it should be presented as a percentage (so 0-100%, possibly with additional decimal precision).

run_id as UUID makes sense, as does the rest of your description about the runs table.

I feel like both tasks and prompts should be based on git- and filesystem- backed data.  The database would just be updated based on the file data, and used for quicker lookups by web users.  The task files should cover any kind of structure or conditions to the data.  Perhaps even referencing a script or other executable to handle logic and advanced functions required to plan and grade the prompt results.  I don't know for sure, and I'm open to discussing it, but I feel this is the most flexible approach.

Agreed on creating a testing queue ticket.

---

### #12 comment 4495777337 — @AlbinoGeek — 2026-05-20T07:36:14Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4495777337>

**Bastion response — 200735ZMAY26**

@gissf1 — addressing all points.

---

**Task/Prompt cardinality — one task, many prompts**

One task can have many prompts. Your instinct is correct. Industry standard:

- **HumanEval:** 164 individual code problems. Each problem is one prompt. "Code generation" is the task category; each problem is a prompt instance of that task.
- **IFEval:** 541 prompts testing instruction-following patterns. Prompts are organized by instruction type (e.g., "include keyword X", "respond in under N words"). The instruction type = task; each test instance = one prompt.
- **MMLU-Pro:** Questions organized by subject area (medicine, law, etc.). Subject area = task category; each question = one prompt.

Consistent pattern: **task = evaluation type or category; prompt = one specific test instance of that type.** One task routinely has many prompts (e.g., 10–50 bash scripting prompts to cover different patterns).

**Difficulty levels** — recommendation: store difficulty as a field on the `prompt` record (e.g., `difficulty_level ENUM('easy','medium','hard')` or INT 1–5), not as a separate sub-task hierarchy. This keeps the task tree flat for common queries and lets difficulty be filtered at query time without joining through an extra tree level. If a use case requires grouping by difficulty across all prompts regardless of task, a view handles it cleanly.

---

**Other git-derived data worth tracking**

Beyond content versioning (already covered by `git_commit_hash`), useful derivable fields:

| Field | Source | Notes |
|-------|--------|-------|
| `created_at` | First commit timestamp for the file | When this prompt was introduced |
| `modified_at` | Last commit timestamp for the file | When it was last changed |
| `author` | git blame (last modifier) | Who owns this prompt |

These should be **populated by the scanner** (the process that reads files and upserts to the DB), not manually entered. The DB values are cache — authoritative source is git.

One additional thing worth hashing: the prompt content itself (`content_sha256`) independent of the file path. This lets two prompts with identical text but different file paths be detected as duplicates. Different from `git_commit_hash` which is repo-wide.

---

**`model_sources` table — agreed**

Core `models` table stays provider-agnostic. Proposed:

```
source_types (lookup table)
──────────────────────────────────────────────────────────
source_type_id          SERIAL PRIMARY KEY
name                    TEXT NOT NULL UNIQUE       -- "huggingface", "ollama", "direct_url", "local_file"

model_sources
──────────────────────────────────────────────────────────
source_id               SERIAL PRIMARY KEY
model_id                INT NOT NULL REFERENCES models
source_type_id          INT NOT NULL REFERENCES source_types
url                     TEXT NOT NULL
source_metadata         JSONB NULLABLE             -- e.g. {"hf_commit_hash": "...", "hf_model_id": "...", "ollama_tag": "..."}
```

No HF-specific columns in `models`. All service-specific identifiers live in `source_metadata` JSONB on `model_sources`. Queryable via JSON operators when needed; invisible when not.

---

**Parameter count — float in billions**

Agreed. `BIGINT` is verbose and mismatches how humans discuss model size. Revised:

```
parameter_count_b       FLOAT NULLABLE             -- total params in billions (e.g. 46.7, 3.8)
active_parameter_count_b FLOAT NULLABLE            -- active params in billions (MoE only)
```

Precision: one decimal place is sufficient for all current model sizes. No information loss vs BIGINT for practical purposes.

---

**Creators table**

Agreed. Proposed:

```
creators
──────────────────────────────────────────────────────────
creator_id              SERIAL PRIMARY KEY
name                    TEXT NOT NULL UNIQUE       -- "meta", "microsoft", "mistralai"
display_name            TEXT NULLABLE              -- "Meta AI", "Microsoft Research"
homepage                TEXT NULLABLE
service_identifiers     JSONB NULLABLE             -- {"huggingface": "meta-llama", "ollama": "meta"}
```

`service_identifiers` JSONB keeps the lookup extensible without new columns per service. `models.creator_id INT NULLABLE REFERENCES creators` replaces the `creator TEXT` field.

---

**`model_hash` UNIQUE + deduplication workflow**

`model_hash` should be `UNIQUE NOT NULL` where known. Workflow:

1. New submission arrives with a `model_hash` that matches an existing row → **flag, do not insert**. Surface to administrator: "Possible duplicate of [existing model name]. Fields differ: [diff list]. Merge?"
2. Administrator reviews diff; accepts merge (coalesce) or rejects (keeps as distinct record, perhaps with a note).
3. Name+creator+quantization match without hash → trigger hash comparison as a verification hint. Do not auto-coalesce. Require hash confirmation before any merge.

No silent auto-merge in any code path. The merge confirmation UI should show exactly which fields change and to what values before the admin commits.

---

**File-first tasks and prompts**

Agreed. Design implications:

- **Filesystem = source of truth** for task definitions and prompt content.
- **DB = derived cache** — populated by a scanner process that reads files and upserts on each run or on a schedule.
- **Task hierarchy from filesystem structure** (recommended): directory layout encodes the task tree. Example:
  ```
  prompts/
    baseline/               ← suite (top-level task)
      bash_scripting/       ← task category
        001_create_dir.md   ← prompt instance
        002_parse_csv.md
      code_generation/
        ...
  ```
  Each directory level maps to a task row. Each `.md` (or `.yaml`) file maps to a prompt row. The scanner derives `parent_id` and `sort_order` from path and filename prefix.

- **Grader script reference** on the task definition file (YAML frontmatter or a separate `task.yaml` in each task directory). Example:
  ```yaml
  name: bash_scripting
  grader: graders/exact_match.py
  difficulty: medium
  ```
  `grader_script` field on `tasks` table; scanner populates it from the frontmatter.

- **Implication for prompt files:** each prompt file has frontmatter (difficulty, version, release_date, is_prerelease) + the prompt text body. Scanner extracts frontmatter to DB; content itself stays in the file.

---

**Testing queue issue**

Opening now as directed.

---

**Revised tables (updated from this discussion):**

```
creators
  creator_id, name, display_name, homepage, service_identifiers JSONB

source_types
  source_type_id, name

models
  model_id, name, creator_id FK, model_hash UNIQUE,
  parameter_count_b FLOAT, active_parameter_count_b FLOAT,
  architecture, context_length_default/min/max,
  file_format, quantization, min_vram_mb,
  release_date, version, description, predecessor_model_id FK

model_sources
  source_id, model_id FK, source_type_id FK, url, source_metadata JSONB

task_categories
  category_id, name UNIQUE, description

tasks
  task_id, name, category_id FK, parent_id FK (nullable),
  sort_order, description, grader_script

prompts
  prompt_id, task_id FK, file_path, git_commit_hash,
  content_sha256, content_length_bytes,
  version, release_date, is_prerelease, difficulty_level,
  created_at, modified_at, author

runs
  run_id UUID PK, submitted_at, publisher_id,
  runner_version, prompt_git_hash

run_model_metrics
  run_id FK, prompt_id FK, model_id FK,
  score FLOAT, pass_fail BOOLEAN, failure_reason TEXT,
  flops_per_token_theoretical BIGINT
  PK: (run_id, prompt_id, model_id)
```

Standing by for feedback or direction to proceed to implementation sub-issues.

— Bastion

---

### #12 comment 4496938804 — @gissf1 — 2026-05-20T09:46:32Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4496938804>

Difficulty level should be an integer, level 0 is basic tests, if the model achieve some sufficient level of success, then we can test the model with level 1 tests, etc.  This also means we can insert new levels on the top or bottom of the test hierarchy, allowing better testing of models of any capability level.  I think it should be stored in the prompt, as that is the most granular level, but perhaps the task level is easier to manage?  It may also make sense to have some kind of test dependency logic in the task or prompt files.  Let's discuss this more.

I think the modified_at field from git could be useful to track the prompt data and potentially the task structure as well.  the fields for author and created_at are probably not useful, unless you can explain why.

What is the benefit of detecting two prompts with identical text but different file paths as duplicates?  What are some situations in which that could be that useful?  I'm not opposed to it if there is a benefit, but I can't think of any way that would be useful.

on the model_hash discussion, there several outcomes of the administrator review: (1) accepting the merge (coalescing), as you suggested; (2) rejecting the coalesce/merge, and creating a new record, possibly with a note; and (3) rejecting the request entirely, possibly responding with a note saying how it was detected as a duplicate of another model, maybe specifying the model in that note and some information about it.  The details of option 3 and the response note can be discussed later when we are at that point in development.

Regarding the "Task hierarchy from filesystem structure", I don't know if that works perfectly.  I'm ok with filesystem structure to organize the data, but there needs to be some kind of yaml or other structured or programmatic data structure to store complex logic and generation rules for some tests.  I'm ok with the majority of the structure, but I feel there is also more potential variety than what was presented in the example.  What if we have a few levels of sub-tasks?  I agree with the idea of a yaml file in each task directory, but I feel like the order of the tests should be defined in the task.yaml file, or some sort mechanism or custom code should be identified in the task.yaml to sort the prompts into the needed order.  It should also include a glob to use to find appropriate prompts for the given test suite.  It may also make sense to allow a custom directory structure below the suite level available for test suites.  in whatever way task IDs are generated, it should be idempotent so the same task file always gets the same database IDs.  This prevents issues where a regeneration of some data causes FKs in the database to become invalid.

Grader script reference: as stated above, difficulty should be an integer internally, but perhaps it also makes sense to have relative options like "parent +1" or "another_test +1" or even "another_prompt_sha +1".  I don't know that the script data needs to be in the database, but perhaps a commit_sha of when the script file was last modified?

run_model_metrics.flops_per_token_theoretical is currently defined as bigint.  Is that necessary?  or could we simply use a larger unit like mflops or gflops (or others) with a float storage type?  Also, why is it "theoretical"?  I feel like there is a better adjective to use there.

---

### #12 comment 4497170538 — @AlbinoGeek — 2026-05-20T10:06:37Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4497170538>

**Bastion response — 201005ZMAY26**

@gissf1 — addressing all points.

---

**Difficulty level — integer, prompt-scoped**

Agreed: integer, 0-indexed, prompt-scoped. Storing at prompt level is correct — it is the most granular unit and allows mixing difficulty levels within a single task. The task level doesn't buy meaningful simplicity; difficulty is fundamentally a property of the specific test instance, not the evaluation type.

`difficulty_level INT NOT NULL DEFAULT 0` on `prompts`.

**Test dependency logic** — this is a meaningful addition. Recommendation: express dependencies in the `task.yaml` (or a `prompt.yaml` for prompt-level dependencies) as a structured list. Example:

```yaml
# prompts/baseline/bash_scripting/002_parse_csv.yaml
difficulty: 1
depends_on:
  min_pass_rate: 0.7        # at least 70% of level-0 prompts in this task must pass
  scope: task               # "task" | "suite" | specific prompt sha
```

The runner reads the dependency block before dispatching level-N prompts. If the prerequisite condition is not met, the runner skips level N (not a failure — a skip). This encoding keeps dependency logic in version-controlled files, not scattered across DB rows. Defer the full runner logic spec to the testing-queue issue.

---

**`author` and `created_at` — concede**

`modified_at` is the useful field; agree on that. Dropping `author` and `created_at` unless a concrete use case emerges. The scanner will still populate `modified_at` from the last-modifying commit timestamp.

---

**Content hash (`content_sha256`) for duplicate detection**

Primary use case: detecting unintentional prompt duplication across task directories during import or refactoring. If a prompt file is copied into two task directories and both are scanned, the DB gets two prompt records that will produce identical model outputs — wasting compute on repeat runs. The hash lets the scanner flag this at import time rather than at run time.

If your workflow keeps prompts strictly single-homed (one file, one task, no cross-linking), the duplicate detection provides no value. Recommend including `content_sha256` as a nullable indexed field but not adding a UNIQUE constraint on it — duplicates get flagged as warnings, not hard rejections, since identical prompt text for different tasks may be intentional in some edge cases.

---

**Model deduplication outcomes — three options confirmed**

Outcomes recorded:
1. Accept merge (coalesce): fields reconciled and combined into existing record.
2. Reject merge, create new record with note: treated as a distinct model, administrator annotates the reason.
3. Reject entirely with note referencing the existing duplicate record: submission discarded; administrator provides context.

Option 3 detail deferred. No further action needed here.

---

**Task hierarchy — idempotent IDs + flexible YAML**

Idempotent task IDs: use a deterministic natural key, not a SERIAL. Recommended approach: `task_id` = SHA256 of the task's canonical path relative to the prompts root (e.g., `baseline/bash_scripting`), truncated to a fixed prefix. SERIAL IDs are regeneration-unsafe as noted. The scanner upserts on the natural key hash, so FK references remain stable across re-scans.

`task.yaml` schema (proposed):

```yaml
name: bash_scripting
category: baseline
parent: null                   # relative path to parent task.yaml, or null for root
sort_order: 10                 # within-parent execution order
description: "..."
grader: graders/exact_match.py
grader_script_commit: null     # populated by scanner from git
prompt_glob: "*.md"            # glob relative to this task directory
prompt_sort: filename_prefix   # "filename_prefix" | "frontmatter_order" | custom
difficulty_gate:               # optional: skip if prerequisite not met
  min_pass_rate: 0.7
  scope: parent
```

Multi-level sub-tasks: directory nesting with a `task.yaml` at each level. Each level is a task row; scanner derives `parent_id` from the parent directory's task natural-key hash.

---

**Difficulty relative references**

"parent +1" and similar relative expressions: feasible but require a two-pass scan (build the difficulty map first, then resolve relative references). Recommend supporting both forms:
- Absolute integer: `difficulty: 2`
- Relative: `difficulty: parent+1` or `difficulty: task:bash_scripting/001+1`

Relative references resolve to their target's integer value + offset at scan time. The DB stores the resolved integer; the YAML stores the expression. Expressions with unresolvable targets fail the scan with a clear error.

This is scanner complexity, not schema complexity. The `prompts.difficulty_level INT` field stores the resolved value regardless of how it was expressed.

---

**`flops_per_token_theoretical` — rename and retype**

"Theoretical" means calculated from architecture parameters (parameter count × ops per forward pass), not empirically measured. Better term: `compute_per_token_estimated` — accurate without implying it is measured or precise.

Unit: switch to FLOAT storing GFLOPs. `BIGINT` for raw FLOP counts reaches the 64-bit limit at ~9 × 10^18 ops, which covers current models, but FLOAT in GFLOPs is more human-readable and sufficient precision.

Revised field: `compute_per_token_gflops FLOAT NULLABLE`.

---

**Updated schema deltas from this round:**

```
prompts
  difficulty_level     INT NOT NULL DEFAULT 0      -- was ENUM; now integer; 0=basic
  content_sha256       TEXT NULLABLE               -- nullable; duplicate detection hint only
  modified_at          TIMESTAMPTZ NULLABLE        -- git last-modified; drop created_at, author
  (drop: created_at, author)

tasks
  natural_key_hash     TEXT NOT NULL UNIQUE        -- SHA256 of canonical path; stable FK anchor
  grader_script_commit TEXT NULLABLE               -- git SHA of grader file last modification

run_model_metrics
  compute_per_token_gflops  FLOAT NULLABLE         -- was flops_per_token_theoretical BIGINT
```

Ready to open the testing-queue issue when directed.

— Bastion

---

### #12 comment 4501587412 — @gissf1 — 2026-05-20T18:48:00Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4501587412>

For difficulty level, I think "difficulty" is an acceptable field name; adding the "level" suffix seems redundant without another field using "difficulty" as a prefix, and even then, the meaning is rather clear without it.

Your suggestion on using content_sha256 to prevent running the same prompt multiple times makes sense.  I don't believe that will be a common scenario, but it should be identified at import time (or in CI) and if detected, it should attempt to avoid the wasted compute time.  If a task intends to duplicate another prompt, I think it makes more sense to have a prompt-level yaml file that indicates that it duplicates the original prompt file, along with the filename and such it is duplicating; then the content_hash for that prompt would likely be of the yaml file (and maybe including the original content_hash value), not the prompt data itself, maintaining uniqueness.  This means we can retain the UNIQUE constraint without concern to this as a special case.  I think it makes sense to also have a way to optionally define it in the task.yaml instead of only through a prompt-level yaml, in which case, maybe the content_hash of the prompt should be the relative file path concatenated with the original content_hash?

For "flops_per_token_theoretical", what about just "gflops_per_token" since "compute" is implied and being in the run_model_metrics table, it clearly describes values obtained through the run.  The UI can manage displaying the details that it's calculated and/or not empirically measured as needed.

---

### #12 comment 4503203398 — @AlbinoGeek — 2026-05-20T22:36:24Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503203398>

**Bastion response — 201926ZMAY26**

@gissf1 — addressing all three points.

---

**`difficulty` field name — agreed**

Drop `_level` suffix. `difficulty INT NOT NULL DEFAULT 0` on `prompts`.

---

**Duplicate prompt handling — YAML indirection approach**

Agreed. Revised design:

An intentionally duplicated prompt is represented by a YAML stub file (e.g., `002_duplicate.yaml`):

```yaml
type: alias
source: ../task_a/001_original.md
original_content_sha256: <sha of original prompt text>
```

Scanner computes `content_sha256` for an alias = SHA256(`relative_file_path` + `original_content_sha256`). Deterministic and unique — two aliases pointing to the same original from different paths produce different hashes.

`task.yaml` can declare aliases in bulk via an `aliases:` block, generating stub records without requiring one YAML file per alias.

UNIQUE constraint on `content_sha256` is preserved across all cases:
- Standard prompt → SHA256 of prompt text content
- Alias prompt → SHA256 of path + original hash
- Unintentional duplicate (non-alias with colliding content hash) → flagged at import, blocked from insert, surfaced to administrator

No special-case exemptions. The distinction between intentional and unintentional is explicit in the source files.

---

**`gflops_per_token` — agreed**

`gflops_per_token FLOAT NULLABLE`. Table context makes the meaning clear without the "compute" prefix.

---

**Updated schema deltas:**

```
prompts
  difficulty           INT NOT NULL DEFAULT 0    -- was difficulty_level
  content_sha256       TEXT UNIQUE               -- alias hashes = SHA256(path + original_hash)

run_model_metrics
  gflops_per_token     FLOAT NULLABLE            -- was compute_per_token_gflops
```

— Bastion

---

### #12 comment 4503669210 — @AlbinoGeek — 2026-05-21T00:11:41Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503669210>

**Bastion response — 210001ZMAY26**

@gissf1 — no outstanding requirements from you on this thread at this time.

All Q1–Q5 questions have been addressed across prior rounds. Schema deltas are captured and agreed:

```
creators        — creator_id, name, display_name, homepage, service_identifiers JSONB
source_types    — source_type_id, name
models          — model_id, name, creator_id FK, model_hash UNIQUE, parameter_count_b FLOAT,
                  active_parameter_count_b FLOAT, architecture, context_length_default/min/max,
                  file_format, quantization, min_vram_mb, release_date, version, description,
                  predecessor_model_id FK
model_sources   — source_id, model_id FK, source_type_id FK, url, source_metadata JSONB
task_categories — category_id, name UNIQUE, description
tasks           — task_id, name, category_id FK, parent_id FK (nullable), sort_order,
                  description, grader_script, grader_script_commit, natural_key_hash UNIQUE
prompts         — prompt_id, task_id FK, file_path, git_commit_hash, content_sha256 UNIQUE,
                  content_length_bytes, version, release_date, is_prerelease, difficulty INT,
                  modified_at
runs            — run_id UUID PK, submitted_at, publisher_id, runner_version, prompt_git_hash
run_model_metrics — run_id FK, prompt_id FK, model_id FK, score FLOAT, pass_fail BOOLEAN,
                    failure_reason TEXT, gflops_per_token FLOAT
                    PK: (run_id, prompt_id, model_id)
```

Pending forward actions (awaiting direction):
- Implementation sub-issues for each table/component
- Testing-queue issue (previously stated as "opening now" — confirm status)

Standing by for next direction.

— Bastion

---

