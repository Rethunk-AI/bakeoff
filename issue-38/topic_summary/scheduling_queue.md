# Summary: scheduling_queue

## Final state

Design-complete per close-out comment 4570344944. Implementation shipped in `bench/queue.py` + `bench/store.py` (merge `0d0288e`); `run_queue` table + claim index in `schema/schema.sql` (commit `dac26d5`).

### `run_queue` table

| Column | SQL type | Notes |
|---|---|---|
| `queue_id` | `UUID PRIMARY KEY DEFAULT gen_random_uuid()` | |
| `run_id` | `UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE` | FK → runs |
| `model_id` | `INT NOT NULL REFERENCES models(model_id)` | FK → models |
| `prompt_id` | `INT NOT NULL REFERENCES prompts(prompt_id)` | one entry per (model, prompt) pair; `prompt_set_id` dropped (per 4512649799) |
| `source_file` | `TEXT` | path to originating job descriptor file in `queue/pending/`; DR anchor |
| `priority` | `INT NOT NULL DEFAULT 100` | lower = higher priority; caller-controlled |
| `status` | `TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING','CLAIMED','IN_PROGRESS','COMPLETE','FAILED','CANCELLED'))` | TEXT + CHECK (not ENUM; deferred cleanup per 4522610944) |
| `attempt_count` | `INT NOT NULL DEFAULT 0` | |
| `max_attempts` | `INT NOT NULL DEFAULT 5` | raised from 3 (per 4512182181) |
| `claimed_by` | `TEXT` | runner_id of claiming process |
| `claimed_at` | `TIMESTAMPTZ` | |
| `started_at` | `TIMESTAMPTZ` | |
| `completed_at` | `TIMESTAMPTZ` | |
| `retry_after` | `TIMESTAMPTZ` | claim gate; replaces `updated_at` math (per 4513822952/4513796211) |
| `error_detail` | `TEXT` | |
| `created_at` | `TIMESTAMPTZ NOT NULL DEFAULT NOW()` | |
| `updated_at` | `TIMESTAMPTZ NOT NULL DEFAULT NOW()` | audit trail; coexists with `retry_after`, distinct purposes (per 4522610944) |

Index: `run_queue_claim_idx ON run_queue (hardware_tag, priority, created_at) WHERE status = 'PENDING'` — note `hardware_tag` was dropped from the table; index likely revised to match capability-based filtering.

### `runners` table

| Column | SQL type | Notes |
|---|---|---|
| `runner_id` | `TEXT PRIMARY KEY` | stable agent ID, e.g. `"worker-01"` or `hostname` for first process |
| `hostname` | `TEXT NOT NULL` | |
| `process_id` | `INT NOT NULL` | |
| `effective_user` | `TEXT NOT NULL` | |
| `last_heartbeat` | `TIMESTAMPTZ NOT NULL DEFAULT NOW()` | |
| `status` | `TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE','IDLE','DEAD'))` | TEXT + CHECK; deferred ENUM migration |
| `started_at` | `TIMESTAMPTZ NOT NULL DEFAULT NOW()` | |

`runs` table gains `runner_id TEXT REFERENCES runners(runner_id)` (per 4513822952).

### `quantization_methods` table

| Column | SQL type | Notes |
|---|---|---|
| `quantization_id` | `SERIAL PRIMARY KEY` | |
| `name` | `TEXT NOT NULL UNIQUE` | e.g. `"fp32"`, `"q4_k_m"` |
| `vram_multiplier` | `DECIMAL NOT NULL` | bytes per active parameter; no ELSE fallback risk |
| `description` | `TEXT` | |

`models.quantization` TEXT column replaced by `models.quantization_id INT NOT NULL REFERENCES quantization_methods(quantization_id)` (per 4519072195). Seed data ships as `seeds/quantization_methods.json` — single source for DB seeding and runner standalone lookup.

### `models` table additions (settled in this thread)

`model_hash TEXT`, `model_source_mtime TIMESTAMPTZ`, `model_source_size BIGINT` added (per 4522610944/4524200643). `model_id` type: `UUID PRIMARY KEY` via UUID5 (replaces `SERIAL`) — deterministic, distributed-safe (per 4522610944).

### State machine (settled)

```
PENDING → CLAIMED → IN_PROGRESS → COMPLETE
                               → FAILED  → PENDING (if attempt_count < max_attempts, via reaper after retry_after expires)
                                         → FAILED  (terminal, attempt_count >= max_attempts)
PENDING → CANCELLED
CLAIMED → PENDING (stale claim: >10 min in CLAIMED; reaper clears claimed_by/claimed_at)
```

Retry: `retry_after = NOW() + 5m` on FAILED; priority bump `+ (5 × attempt_count)`. Reaper probabilistic (10%) embedded in queue worker after each job outcome.

### Claim protocol

`FOR UPDATE SKIP LOCKED` atomic claim; runner passes `$runner_vram_gb` at claim time; VRAM estimate calculated inline as `CEIL(m.active_parameter_count_b × qm.vram_multiplier × 1.15)` — no stored `min_vram_gb` column (per 4513822952).

### Operating modes

Standalone (default): no DB, reads model/prompt descriptor files, writes results to disk, exits. Queue worker (opt-in `--queue`): DB-connected, claims from `run_queue`, reaper embedded. Invariant recorded in `AGENTS.md` (per 4570546649).

### Disk-DR layout

`queue/pending/` + `queue/completed/` two-directory layout. File mtime pre-read delay: 30s default, configurable via `config.yaml` (`queue.file_mtime_min_age_seconds`). Configuration merged into existing `config.yaml` project root file — no separate `bakeoff.yaml` (per 4524304026).

---

## Notable / unusual decisions

- **No stored `min_vram_gb`; calculated at claim time** — `CEIL(active_parameter_count_b × vram_multiplier × 1.15)` computed per-row in the claim query's `WHERE` clause. Avoids staleness, single update point if overhead ratio changes. Unusual: running a JOIN + CEIL expression on every claim-query iteration. Downstream: requires `quantization_methods` to be seeded before any claim is attempted. (per 4512649799, 4513822952, 4516038912)

- **`retry_after` as the retry gate instead of `updated_at` math** — @gissf1 proposed a named column over timestamp arithmetic to allow non-failure scheduling (marketing/legal delayed runs) and to preserve FAILED visibility for operator triage. Items stay FAILED until reaper resets them — high `attempt_count` + no PENDING transition is an observable signal. (per 4513796211, 4513822952)

- **Priority increment = `5 × attempt_count`** — soft near-exponential degradation without a hard jump; 5 attempts from base 100 reaches 175. Retains hard ceiling at `max_attempts = 5`. (per 4513822952, 4516038912)

- **`hardware_tag` dropped; capability-based matching at claim time** — @gissf1 rejected manual hardware tags as a maintenance burden; model requirements are properties of the model, not of a queue entry. Runner passes its VRAM at claim time. Unusual: moves hardware routing logic out of the queue schema and into the claim query itself. (per 4512182181, 4512649799)

- **UUID5 for `model_id`** — deterministic hash-based UUID; primary key derived from `model_hash` (or `source_url|parameter_count_b|model_source_size` provisionally). Same model produces the same UUID on any host without a DB round-trip. (per 4522610944, 4524200643)

- **`prompt_set_id` → flat `prompt_id` per queue entry** — one entry per (model, prompt) pair. Grouping logic at submission layer, not in the queue. 100 prompts × 10 models = 1,000 independently trackable/retriable entries. (per 4512649799)

- **Standalone as default, queue worker as opt-in** — inverted from initial proposal; queue worker mode is the special case. Stress-tested: 16 workers × 6,400 claims, 0 double-claims via rename-as-mutex. (per 4516038912, 4519072195, 4570546649)

- **`quantization_methods` seeded from a JSON asset** — `seeds/quantization_methods.json` is the single source for DB seeding and runner standalone lookup; prevents divergence between what the DB knows and what the runner knows. (per 4524200643, 4524304026)

- **`updated_at` retained alongside `retry_after`** — @gissf1 explicitly flagged these as non-redundant: `retry_after` is a scheduling claim gate; `updated_at` is an audit trail updated on every state transition. (per 4522549514, 4522610944)

- **`runners.status` TEXT + CHECK, not ENUM** — deferred until schema stabilizes; comment left in `schema.sql` as `-- TODO(deferred P2): migrate TEXT+CHECK → ENUM once schema stabilizes (bakeoff#13)`. (per 4522549514, 4522610944)

---

## Open / unresolved

1. **Model capability ingestion path** — where does `active_parameter_count_b`, `quantization`, `model_source_size`, and `model_hash` originate? Model card at submission, CI artifact, or runner autodiscovery? Defined as a need ("needs a defined ingestion path before capability-matching is useful") but deferred without a resolution.
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4512649799>

2. **`quantization_methods` seed multiplier sign-off** — @gissf1 noted this is not their expertise and asked for cross-checking against reference model file sizes (within 1% tolerance). Bastion committed to verifying against llama.cpp MODELS.md + a known-size model download; no confirmation comment appears in this thread.
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524200643>

3. **Namespace UUID committed** — Bastion stated it would generate and commit `BAKEOFF_MODEL_NAMESPACE` to `bench/constants.py` without user action required. No confirmation that this commit is visible in the thread; unverified as complete here.
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524304026>

4. **`--strict-cache` default disambiguation** — policy was stated (size changed = INVALID; mtime changed, size same = recompute hash; mtime changed, content same = update mtime in disk file). Not confirmed as implemented; no test coverage reference in the close-out.
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524304026>

5. **Multi-runner per host** — hostname suffix scheme (`hostname`, `hostname-2`, etc.) proposed but explicitly marked not a priority. Collision risk on concurrent startup noted and unresolved.
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4516038912>

6. **Hardware-assignment scope** — close-out comment 4570344944 explicitly states "Hardware-assignment scope lives in #22. Closing." The relationship between runner VRAM capability matching (settled here) and whatever broader hardware-assignment mechanism #22 covers is undefined in this thread.
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4570344944>

---

## Cross-topic links

- **models** — FK `run_queue.model_id → models.model_id`; `models.quantization_id → quantization_methods.quantization_id`; `models.model_hash`, `model_source_size`, `model_source_mtime` settled here.
- **prompts** — FK `run_queue.prompt_id → prompts.prompt_id`; prompt descriptor file format deferred to scoring architecture thread.
- **runs** — FK `run_queue.run_id → runs.run_id`; `runs.runner_id → runners.runner_id` added here.
- **run_model_metrics** — results written to disk on COMPLETE; test result envelope duplicates `run_model_metrics` fields for self-contained distribution.
- **hardware / runners** (#22) — broader hardware-assignment scope explicitly tracked there; VRAM capability filtering (settled here) is a prerequisite input to that scope.
- **result signing** — referenced but deferred to a separate thread (#16 mentioned for signing stanza in `config.yaml`).
- **scoring / prompt descriptor** — deferred to scoring architecture thread.
