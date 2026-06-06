# Summary: system_software

## Final state

`system_software` is a resolved, ratified table (per #19). Separate from `system_hardware` by design — same machine can run under different driver versions or kernel builds on different days.

```sql
CREATE TABLE system_software (
    system_software_id  SERIAL PRIMARY KEY,
    os                  TEXT NULLABLE,          -- e.g. "Ubuntu 24.04.2 LTS"
    kernel_version      TEXT NULLABLE,          -- e.g. "6.8.0-57-generic"
    python_version      TEXT NULLABLE,          -- e.g. "3.12.3"
    gpu_driver_version  TEXT NULLABLE,          -- nvidia-smi Driver Version
    cuda_version        TEXT NULLABLE,          -- null for ROCm/CPU-only
    rocm_version        TEXT NULLABLE,          -- null for CUDA runners
    runner_version      TEXT NULLABLE           -- bakeoff harness commit hash or semver
);
```

`run_hardware_metrics` carries `system_software_id INT NULLABLE REFERENCES system_software`. `system_hardware` does NOT reference `system_software` — hardware identity is software-independent.

Deduplication policy: `system_software` does NOT deduplicate — each run inserts a new row as an environment snapshot. Deduplication by full column hash is an optional Phase 2 optimization.

Auto-detection sources (per #19):
- `os` — `platform.platform()` or `distro.name(pretty=True)`
- `kernel_version` — `platform.release()`
- `python_version` — `platform.python_version()`
- `gpu_driver_version` — `nvidia-smi --query-gpu=driver_version`
- `cuda_version` — `nvidia-smi` header or `torch.version.cuda`
- `rocm_version` — `rocm-smi --showversion`; null if unavailable
- `runner_version` — harness `__version__` or `git describe --tags --always`

## Notable / unusual decisions

- **Software separate from hardware** — software is a run-time snapshot, not a machine property. The FK lands on `run_hardware_metrics`, not `system_hardware`. Enables detecting measurement regressions caused by driver updates on the same physical hardware.
- **No deduplication on `system_software`** — identical environments across runs create separate rows. Avoids a race-prone upsert at run time; full-column-hash dedup deferred to Phase 2 as optimization only.
- **`runner_version` tracked** — commit hash or semver of the bakeoff harness. Attributes measurement deltas to runner changes vs. hardware/driver changes; essential for benchmark reproducibility.
- **`cuda_version` and `rocm_version` are mutually exclusive nulls** — design explicitly supports heterogeneous GPU ecosystems. Querying `WHERE cuda_version IS NOT NULL` or `WHERE rocm_version IS NOT NULL` partitions the corpus cleanly.
- **Implementation gap noted at issue #38** — @gissf1 flagged that `system_hardware` and `system_software` tables were missing or minimal in the live `schema/schema.sql` as of 2026-06-05 (per 4629938143). Tables not yet confirmed implemented; #38 is the remediation ticket.

## Open / unresolved

- **Implementation status** — #38 (per 4629938143) confirms the tables were absent or incomplete in `schema.sql` at audit time. Whether #19 acceptance criteria are actually met is unresolved pending @gissf1 sign-off on a consolidated schema patch.
- **`vram_type` FK resolution** — #38 raises whether `vram_type` on `gpu_hardware` is a free-text field or a FK; this is adjacent and affects the schema patch scope.

## Cross-topic links

- `run_hardware_metrics` → `system_software.system_software_id` (FK; software environment active during a run)
- `run_hardware_metrics` → `system_gpu_link(system_hardware_id, slot_index)` (compound FK; physical hardware context)
- `system_hardware` — hardware identity table; no direct reference to `system_software`
- `system_gpu_link` — join table binding `system_hardware` to `gpu_hardware`; referenced by `run_hardware_metrics` alongside `system_software`
