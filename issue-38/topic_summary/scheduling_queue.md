# Summary: scheduling_queue

## Final state

### `run_queue` table (as shipped in `schema/schema.sql`, commit `dac26d5`)

| Column | Type | Notes |
|--------|------|-------|
| `queue_id` | UUID PRIMARY KEY DEFAULT gen_random_uuid() | |
| `run_id` | UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE | |
| `model_id` | INT NOT NULL REFERENCES models(model_id) | |
| `prompt_id` | INT NOT NULL REFERENCES prompts(prompt_id) | replaces earlier `prompt_set_id`; one entry per (model × prompt) |
| `source_file` | TEXT | path of originating job descriptor; populated at enqueue |
| `priority` | INT NOT NULL DEFAULT 100 | lower = claimed sooner; 0–9 critical, 10–49 high, 50–99 normal, 100+ low |
| `status` | TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING','CLAIMED','IN_PROGRESS','COMPLETE','FAILED','CANCELLED')) | TEXT+CHECK retained (ENUM deferred P2 per 4522610944) |
| `attempt_count` | INT NOT NULL DEFAULT 0 | |
| `max_attempts` | INT NOT NULL DEFAULT 5 | raised from 3 per gissf1 (per 4512182181) |
| `claimed_by` | TEXT | runner identity |
| `claimed_at` | TIMESTAMPTZ | |
| `started_at` | TIMESTAMPTZ | |
| `completed_at` | TIMESTAMPTZ | |
| `retry_after` | TIMESTAMPTZ | claim gate; NULL = no delay; replaces updated_at math gate |
| `error_detail` | TEXT | |
| `created_at` | TIMESTAMPTZ NOT NULL DEFAULT NOW() | |
| `updated_at` | TIMESTAMPTZ NOT NULL DEFAULT NOW() | audit trail; coexists with retry_after (per 4522610944) |

Index: `run_queue_claim_idx ON run_queue (hardware_tag, priority, created_at) WHERE status = 'PENDING'` — note: `hardware_tag` was dropped from the schema in favor of capability-based matching; index definition may need updating to remove `hardware_tag`.

### State machine

```
PENDING → CLAIMED → IN_PROGRESS → COMPLETE
                               → FAILED → PENDING (reaper resets when attempt_count < max_attempts AND retry_after <= NOW())
                                        → FAILED (terminal, attempt_count >= max_attempts; operator action required)
PENDING → CANCELLED (operator)
CLAIMED → PENDING (stale-claim reaper: claimed_at < NOW() - INTERVAL '10 minutes')
```

### Claim protocol

Atomic `FOR UPDATE SKIP LOCKED` claim query. Runner passes its `vram_gb`; query calculates VRAM needed inline via join to `quantization_methods.vram_multiplier`:

```sql
AND CEIL(m.active_parameter_count_b * qm.vram_multiplier * 1.15) <= $runner_vram_gb
AND (rq.retry_after IS NULL OR rq.retry_after <= NOW())
ORDER BY rq.priority ASC, rq.created_at ASC
LIMIT 1
FOR UPDATE SKIP LOCKED
```

No `hardware_tag` pre-assignment — capability filter is entirely at claim time.

### Retry policy

- `max_attempts = 5`; terminal failure stays `FAILED` (visible signal for operator investigation).
- On FAILED with `attempt_count < max_attempts`: set `retry_after = NOW() + base_interval`, bump `attempt_count`, bump `priority += 5 * attempt_count` (per 4513822952). Item stays FAILED until reaper flips it to PENDING once `retry_after <= NOW()`.
- Priority degradation table:

| Attempt | Increment | Running delta from base 100 |
|---------|-----------|------------------------------|
| 1 | +5 | 105 |
| 2 | +10 | 115 |
| 3 | +15 | 130 |
| 4 | +20 | 150 |
| 5 | +25 | 175 |

- `retry_after` base interval: fixed 5 minutes (left open; exponential deferred).
- Runner also implements independent exponential backoff to prevent tight-loop waste.

### `runners` table

| Column | Type | Notes |
|--------|------|-------|
| `runner_id` | TEXT PRIMARY KEY | stable agent ID; hostname for first process, hostname-2 etc. for subsequent |
| `hostname` | TEXT NOT NULL | |
| `process_id` | INT NOT NULL | PID; updated on restart via UPSERT |
| `effective_user` | TEXT NOT NULL | |
| `last_heartbeat` | TIMESTAMPTZ NOT NULL DEFAULT NOW() | updated every 60s |
| `status` | TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE','IDLE','DEAD')) | TEXT+CHECK; ENUM deferred P2 |
| `started_at` | TIMESTAMPTZ NOT NULL DEFAULT NOW() | |

`runs` table gains `runner_id TEXT REFERENCES runners(runner_id)` to record which runner executed each run (per 4513796211).

### Disk DR layer

Two-directory layout in filesystem: `queue/pending/` (job descriptors, source of truth for re-enqueue) and `queue/completed/` (moved on COMPLETE). No `in-db/` level — `source_file` column in `run_queue` tracks which file originated the row. File loader waits until `mtime < NOW() - 30s` (configurable via `config.yaml` / `BAKEOFF_CONFIG`) before reading, to avoid partial-write races on ext4 (default commit=5s; 30s gives margin for NFS and lazy-write scenarios, per 4519072195).

DR path: scan `queue/pending/`; for each file, check `run_model_metrics` for valid results — if present, move to `completed/` and skip; if absent, re-enqueue as PENDING.

### Operational modes

- **Standalone (default):** `runner --model model.json --prompts prompts.json`. No DB. Writes to `results/<model>/<timestamp>.json`.
- **Queue worker (opt-in):** `runner --queue`. Connects to DB, registers in `runners`, enters claim loop.
- Config: `config.yaml` in project root (not a separate `bakeoff.yaml`); database URL absent/empty = standalone.

### Reaper

Probabilistic: 10% chance after each job outcome, embedded in queue worker process (not a cron or separate script). Tasks: reset stale CLAIMED items, mark DEAD runners (`last_heartbeat < NOW() - 5 min`), reclaim CLAIMED items from dead runners. FAILED → PENDING promotion is implicit via claim gate + retry_after, not a reaper action.

Idle-loop prevention: when no PENDING or expired-retry items exist, worker sleeps until `MIN(retry_after)` across eligible FAILED items, or 60s poll interval if nothing pending.

### `quantization_methods` table

Introduced to replace the inline CASE expression in the claim query (per 4516038912). Also serves as the source for the bakeoff-results filter bar dropdown.

| Column | Type | Notes |
|--------|------|-------|
| `quantization_id` | SERIAL PRIMARY KEY | |
| `name` | TEXT NOT NULL UNIQUE | "fp32", "q4_k_m", etc. |
| `vram_multiplier` | DECIMAL NOT NULL | bytes per active parameter; no NULL/ELSE risk |
| `description` | TEXT | |

`models.quantization` (TEXT) becomes `models.quantization_id INT NOT NULL REFERENCES quantization_methods(quantization_id)`. Seed data ships as `seeds/quantization_methods.json`; runner reads directly in standalone mode (no live DB needed for VRAM calculation).

### Implementation note

`bench/queue.py` + `bench/store.py` shipped on `main` (merge `0d0288e`). Race-safety of rename-as-mutex claim mechanism stress-proven: 16 workers × 6,400 attempts, 0 double-claims (per 4570546649). 38 new tests; full suite 413 green.

---

## Notable / unusual decisions

- **`retry_after` as claim gate, not a state transition** — FAILED items stay FAILED while awaiting retry. This preserves high-attempt-count items as a visible operator signal ("something is wrong") rather than silently cycling back to PENDING. The claim query's `retry_after <= NOW()` filter is the only gate, avoiding a separate state and transition. (per 4512182181, adopted 4513822952)

- **VRAM is calculated at claim time, not stored** — `min_vram_gb` was rejected as a stored column. The formula `CEIL(active_parameter_count_b * vram_multiplier * 1.15)` runs per-row at claim time. 1.15 is a 15% overhead factor for KV cache + activations. This avoids staleness when quantization formats are added/revised. (per 4513796211, 4519072195)

- **`quantization_methods` lookup table over inline CASE expression** — a CASE expression in the claim query creates a single update point risk (new format = forgotten branch). The lookup table makes new quantization formats a data insert, not a code change. It also serves double duty as the dropdown data source for bakeoff-results UI. (per 4516038912)

- **Dependency ordering closed as non-issue** — initial design had a dependency mechanism to enforce baseline runs completing before comparison runs. gissf1 confirmed scores are absolute (not relative to a baseline), so every model run is fully independent. Priority-only ordering (baseline enqueued at lower numeric priority as a convention) is sufficient and no enforcement is needed. (per 4513796211, closed 4513822952)

- **Standalone default, queue-worker opt-in** — inverts initial assumption. The full DB-backed queue is a special case for the backend pipeline, not the primary user experience. Most external contributors run one model once; they should not need to deploy Postgres. (per 4516038912, adopted 4519072195)

- **Rename-as-mutex claim in file-based queue** — the disk queue implementation (`bench/queue.py`) uses filesystem rename as the atomic claim operation rather than Postgres `FOR UPDATE SKIP LOCKED`. This is the standalone-compatible variant of the same race-safety idiom, designed for the no-DB case. (per 4570546649)

---

## Open / unresolved

- **`retry_after` base interval** — fixed 5 minutes agreed as acceptable for now, but exponential base was discussed and not formally closed. Scaling capped at "runner not idle-looping for 20+ minutes" constraint. (per 4516038912)

- **`run_queue_claim_idx` with dropped `hardware_tag`** — the original index definition includes `hardware_tag` which was removed from the schema. Whether the shipped index definition was updated is not confirmed in the thread.

- **Quantization multiplier validation** — gissf1 asked Bastion to cross-check multipliers against public GGUF model file sizes. Bastion committed to a follow-up comment; no confirmation visible in the thread. Values are untested against live data. (per 4516038912, 4522610944)

- **`retry_after` vs. exponential backoff interaction** — final policy is flat 5-minute floor + priority-based deprioritization + runner-side exponential backoff. Whether the floor should itself scale exponentially was left open. (per 4516038912, 4522610944)

- **Prompt descriptor file format** — deferred to scoring architecture thread. Format expected to mirror model disk file pattern.

- **runners.status ENUM migration** — TEXT+CHECK retained for schema stability. Marked for P2 cleanup once schema stabilizes. (per 4522610944)

---

## Cross-topic links

- **`quantization_methods`** → `models.quantization_id` FK; also feeds bakeoff-results UI (filter bar dropdown). VRAM estimation in claim query depends on this table.
- **`runs`** → `run_queue.run_id` FK (ON DELETE CASCADE); `runs.runner_id` FK to `runners.runner_id` added here.
- **`models`** → `run_queue` joins via `runs` to reach `models.active_parameter_count_b` for VRAM calculation. `models.quantization_id` FK to `quantization_methods`.
- **`prompts`** → `run_queue.prompt_id` FK. Prompt descriptor file format deferred to scoring architecture topic.
- **`run_model_metrics`** → results written on COMPLETE; result envelope carries full duplicate of `run_model_metrics` fields for self-contained distribution. DR path checks this table to avoid re-running completed items.
- **hardware / `gpu_hardware`** (scheduling_queue → gpu_hardware topic) — hardware-assignment scope for capability matching lives in #22 (tflops_source / gpu_hardware topic). `runners` table intentionally stores no hardware properties; runner declares capability at claim time.
