---
ticket: 19
type: issue-body
author: AlbinoGeek
posted: 2026-05-22T19:52:23Z
topic: uuids
url: https://github.com/Rethunk-AI/bakeoff/issues/19
title: schema: implement system_hardware and system_software tables
---

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
