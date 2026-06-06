# Summary: run_hardware_metrics

## Final state

Per-run hardware telemetry table. Rows keyed on `run_id` + compound FK into `system_gpu_link`. All metric columns are nullable (runner-dependent availability).

```sql
run_hardware_metrics
────────────────────────────────────────────────────────────
run_id                   TEXT NOT NULL
system_hardware_id       INT  NOT NULL
slot_index               INT  NOT NULL
PRIMARY KEY              (run_id, system_hardware_id, slot_index)  -- implied; run_id alone may be PK if single-GPU
FOREIGN KEY (system_hardware_id, slot_index)
    REFERENCES system_gpu_link (system_hardware_id, slot_index)
system_software_id       INT NULLABLE REFERENCES system_software

-- Timing
wall_clock_seconds       FLOAT NULLABLE   -- CLOCK_MONOTONIC, prompt→final token
seconds_to_first_token   FLOAT NULLABLE   -- was time_to_first_token_ms; all times in seconds
tokens_per_second        FLOAT NULLABLE

-- GPU utilization / compute
peak_vram_mb             FLOAT NULLABLE   -- NVML nvmlDeviceGetMemoryInfo peak during run
gpu_sm_utilization_pct   FLOAT NULLABLE   -- NVML nvmlDeviceGetUtilizationRates mean 10 Hz
tflops_utilization_pct   FLOAT NULLABLE   -- tokens_completion / gpu_wall_time_s / model_flops_per_token / gpu_peak_tflops

-- Dual GPU timing fields (approved per 4494986575)
gpu_event_seconds        FLOAT NULLABLE   -- Path 1: cudaEventElapsedTime() / hipEventElapsedTime() kernel wall time
gpu_weighted_seconds     FLOAT NULLABLE   -- Path 2: wall_clock_seconds × mean(gpu_sm_utilization_pct / 100)

-- CPU timing (optional, getrusage RUSAGE_SELF)
cpu_cycles_elapsed       BIGINT NULLABLE  -- RDTSC/PMCCNTR_EL0 delta; x86/arm64 only
cpu_seconds_user         FLOAT NULLABLE   -- was cpu_time_user_ms; getrusage ru_utime ÷ 1000
cpu_seconds_sys          FLOAT NULLABLE   -- was cpu_time_sys_ms; getrusage ru_stime ÷ 1000
```

**Implemented fields (shipped in bakeoff runner as of 4466000757 + rename commit e4a6983):**
`wall_clock_seconds`, `seconds_to_first_token`, `tokens_per_second`, `peak_vram_mb`, `gpu_sm_utilization_pct`, `tflops_utilization_pct`, `cpu_seconds_user`, `cpu_seconds_sys`.
`gpu_event_seconds` present in every record but always `NULL` — CUDA event path not yet wired (per 4519663607).
`gpu_weighted_seconds` fully live when NVML available.

**As of issue #38 (4629938143):** @gissf1 flagged that #21 may not cover all end-of-#8 changes and that `run_hardware_metrics` should be verified against the final schema before the ticket is considered closed.

## Notable / unusual decisions

- **All time fields in seconds, not milliseconds** (per 4476713600): `time_to_first_token_ms` → `seconds_to_first_token`; `cpu_time_user_ms` → `cpu_seconds_user`; `cpu_time_sys_ms` → `cpu_seconds_sys`; `gpu_wall_time_ms` → `gpu_seconds`. Float precision preserves sub-millisecond resolution; consistent units simplify queries and comparisons.

- **Dual GPU timing fields instead of a flag** (per 4482149275 → 4482502460): originally proposed as a single `gpu_seconds` + `gpu_seconds_is_direct` boolean. @gissf1 directed storing both values independently so post-hoc divergence analysis is directly queryable. Gap `wall_clock_seconds - gpu_event_seconds` is the documented proxy for CPU-GPU data transfer + scheduling overhead.

- **`gpu_event_seconds` vs `gpu_weighted_seconds` naming** (per 4492063588 → 4494171694 → 4494986575): Bastion proposed `gpu_seconds_event` / `gpu_seconds_sampled`; @gissf1 preferred reordered form. Final approved names: `gpu_event_seconds` (CUDA/ROCm event API) + `gpu_weighted_seconds` (utilization-weighted derivation). "Weighted" chosen over "sampled" because it describes the mathematical relationship (utilization × wall time), not just the polling acquisition method.

- **`cost_usd` removed from storage** (per 4462460688, implemented 4466000757): agreed to be a derived display value. Computed at render time from `energy_wh × kwh_rate` (local) or provider pricing × token counts (API). Backward-compat sum path retained for legacy result files.

- **`system_software_id` FK on `run_hardware_metrics`, not on `system_hardware`** (per 4476713600 → 4480962683): software stack is not a fixed property of hardware — same machine can run different driver/kernel/Python on different days. FK lives on the per-run table so each run records the exact software environment active during that run.

- **Compound FK `(system_hardware_id, slot_index)` → `system_gpu_link`** (per issue #21): identifies which physical GPU in which PCIe slot on which host was active for the run. Forward-compatible with a future `run_gpu_usage` join table for multi-GPU runs without schema changes to other tables.

- **`tflops_utilization_pct` moved to `run_hardware_metrics`** (per 4462460688 three-category split): originally considered for `run_model_metrics`. Placed here because it depends on GPU hardware specs (`gpu_peak_tflops`) and is a hardware-utilization signal, not a model quality metric.

- **`gpu_data_transfer_seconds` deferred** (per 4492063588 → 4494171694): for local inference, per-inference host-to-device transfer is kilobytes (input tokens + output logits), measured in microseconds. Weights load at startup, not per inference. The gap `wall_clock_seconds - gpu_event_seconds` already proxies combined overhead. Instrumenting transfer directly would require modifying inference runner hot paths.

- **Tokens-per-second retained despite "tokens is vague" critique** (per 4446310454 → 4450406819): @gissf1 noted tokens are architecture-variable. Retained as a practical proxy only; `tflops_utilization_pct` (FLOPs-normalized) is the primary cross-hardware comparability metric.

## Open / unresolved

- **`run_id` PK / uniqueness semantics**: thread establishes `run_id TEXT NOT NULL` + compound FK but does not explicitly define whether `run_id` alone is the PK (single-GPU assumption) or whether the PK is `(run_id, system_hardware_id, slot_index)`. The multi-GPU deferral leaves this ambiguous. Trade-off: `run_id` as PK is simpler and matches current single-GPU runner; composite PK is required once multi-GPU `run_gpu_usage` join table is introduced.

- **Completeness of #21 vs end-of-#8** (raised in issue #38 body, 4629938143): @gissf1 explicitly flagged that #21 may not have captured all changes discussed at the end of #8 and directed a verification pass before closing. Verification has not yet been confirmed; ticket #8 / #21 closure status is therefore open.

- **`gpu_event_seconds` CUDA path not wired** (per 4519663607): field exists in runner output but is always `NULL`. Implementation deferred to Phase 2. No blocking decision required; just unimplemented.

## Cross-topic links

- **`system_gpu_link`** — compound FK `(system_hardware_id, slot_index)` ties each run to a specific GPU-in-slot-in-host. See `system_gpu_link` topic for `slot_native_interface_type_id` and `actual_interface_type_id` columns and PCIe limitation attribution logic.
- **`system_software`** — `system_software_id` FK records the software environment (OS, kernel, Python, GPU driver, CUDA/ROCm, runner version) active during the run. See `system_software` topic.
- **`run_model_metrics`** — sibling per-run table for model quality metrics (scores, pass/fail, `flops_per_token_theoretical`). Split from hardware metrics per 4462460688 three-category normalization. Schema details open in bakeoff#12.
- **`gpu_hardware`** — `peak_tflops_fp16` from this table is the denominator in `tflops_utilization_pct` calculation. See `gpu_hardware` topic for `_TFLOPS_TABLE` slug-match lookup.
- **`interface_type`** — referenced transitively via `system_gpu_link` for PCIe link degradation display. See `interface_type` topic for `lane_transfer_rate` rename (noted in #38 body) and PCIe gen/width per-axis comparison logic.
