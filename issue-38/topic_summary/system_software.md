# Summary: system_software

## Final state

`system_software` was proposed by @gissf1 (per 4476713600) and fully specced by @AlbinoGeek (per 4480962683), then ratified into implementation tickets #19 and #21.

### Resolved final table schema

```sql
system_software
────────────────────────────────────────────────
system_software_id  SERIAL PRIMARY KEY
os                  TEXT NULLABLE    -- e.g. "Ubuntu 24.04.2 LTS"
kernel_version      TEXT NULLABLE    -- e.g. "6.8.0-57-generic"
python_version      TEXT NULLABLE    -- e.g. "3.12.3"
gpu_driver_version  TEXT NULLABLE    -- nvidia-smi Driver Version field
cuda_version        TEXT NULLABLE    -- null for ROCm/CPU-only runners
rocm_version        TEXT NULLABLE    -- null for CUDA runners
runner_version      TEXT NULLABLE    -- bakeoff harness commit hash or semver
```

All columns nullable; no uniqueness constraint on the table (see deduplication decision below).

FK placement: `run_hardware_metrics.system_software_id INT NULLABLE REFERENCES system_software` (per #21 body). Nullable on initial migration to avoid breaking existing rows; intended to be set NOT NULL after clean backfill.

## Notable / unusual decisions

- **No deduplication on `system_software`** — the same software environment produces a new row per run. `system_hardware` deduplicates on `system_id`, but `system_software` intentionally does not; each run captures an independent software environment snapshot. Full-column-hash dedup was explicitly marked Phase 2 optional (per #19 body). Rationale: simplicity; snapshot semantics are more correct for an audit log.

- **`gpu_seconds_is_direct` companion boolean** — `run_hardware_metrics.gpu_seconds` can be measured via two paths: CUDA/ROCm event elapsed time (kernel execution time, preferred) or wall-clock × mean SM utilization (compute-weighted approximation). Rather than a separate column per method, a single `gpu_seconds` FLOAT plus `gpu_seconds_is_direct BOOLEAN` makes the measurement method queryable without schema branching (per 4480962683). This also enables validation: the two paths should converge on a well-optimized run; divergence reveals transfer/scheduling overhead.

- **`system_software` not linked to `system_hardware`** — @AlbinoGeek explicitly ruled against a FK from `system_hardware` → `system_software` (per 4480962683): "a given machine's hardware record is software-independent." The software reference lives only on `run_hardware_metrics`, binding the software environment to the run, not the machine.

- **Unit normalization: all time fields converted from ms to seconds** — `time_to_first_token_ms` → `seconds_to_first_token`, `cpu_time_user_ms` → `cpu_seconds_user`, `cpu_time_sys_ms` → `cpu_seconds_sys`, `gpu_wall_time_ms` → `gpu_seconds`. Proposed by @gissf1 (per 4476713600); confirmed by @AlbinoGeek (per 4480962683). FLOAT seconds retains sub-millisecond precision while keeping units uniform across all timing columns.

- **Multi-GPU forward compatibility** — #21 explicitly calls out that if a `run_gpu_usage` join table is ever added for multi-GPU runs, `run_hardware_metrics` would reference `run_gpu_usage.run_id` instead of carrying the compound FK directly. No other tables would change. This design choice was made proactively with no multi-GPU work scoped.

## Open / unresolved

- **`system_software` deduplication threshold** — Phase 2 dedup by full column hash was flagged as "optional optimization" (#19 body) but not scoped, designed, or accepted. No decision on: whether a `UNIQUE` constraint across all columns is viable (NULLs in unique constraints vary by DB), whether a separate hash column is preferred, or what triggers dedup (insert-time upsert vs. background job).
  - address: #19, <https://github.com/Rethunk-AI/bakeoff/issues/19>

- **`system_software_id` nullability in `run_hardware_metrics`** — the FK was added as NULLABLE with a note to set NOT NULL "once backfill is confirmed clean or if starting fresh" (#21 body). No thread content establishes when or how that backfill would be confirmed, or who is responsible for running it. The nullable-to-not-null transition is unscheduled and uncriteria'd.
  - address: #21, <https://github.com/Rethunk-AI/bakeoff/issues/21>

## Cross-topic links

- **system_hardware** (`system_hardware` table) — `system_software` has no direct FK to `system_hardware`; the binding is indirect via `run_hardware_metrics`.
- **system_gpu_link** — `run_hardware_metrics` carries both the compound FK `(system_hardware_id, slot_index) → system_gpu_link` and `system_software_id → system_software`; the full hardware context join requires all three tables (#21 body).
- **run_hardware_metrics** — primary consumer of `system_software_id`; all timing-field renames and the `gpu_seconds_is_direct` column also live here.
- **run_model_metrics / models / tasks / prompts** — deferred to #12 (referenced in 4480962683); out of scope for this topic.
