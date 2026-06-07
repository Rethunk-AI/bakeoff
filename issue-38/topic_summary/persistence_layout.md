# Summary: persistence_layout

## Final state

Schema as settled by the end of the thread (tickets #8, #12–15, #17–22).

### `creators`
```sql
creator_id          UUID PRIMARY KEY           -- UUID5, deterministic
name                TEXT NOT NULL UNIQUE
display_name        TEXT
homepage            TEXT
service_identifiers JSONB                      -- {"huggingface": "...", "ollama": "..."}
```

### `source_types`
```sql
source_type_id      SERIAL PRIMARY KEY
name                TEXT NOT NULL UNIQUE       -- "huggingface", "ollama", "direct_url", "local_file"
```

### `quantization_methods`
```sql
quantization_id     SERIAL PRIMARY KEY
name                TEXT NOT NULL UNIQUE       -- "fp32", "q4_k_m", etc.
vram_multiplier     DECIMAL NOT NULL           -- bytes per active parameter; no ELSE fallback
description         TEXT
```
*(27 seed rows; seeded from `seeds/quantization_methods.json`)*

### `model_architectures`
```sql
-- lookup; 4 rows (Dense / MoE / SSM / Hybrid); seeded from seeds/model_architectures.json
```

### `model_file_formats`
```sql
-- lookup; 6 rows (GGUF / SafeTensors / PyTorch / ONNX / ExLlamaV2 / MLX); seeded from seeds/model_file_formats.json
```

### `models`
```sql
model_id                UUID PRIMARY KEY       -- UUID5; primary: uuid5(namespace, model_hash); provisional otherwise
name                    TEXT NOT NULL
creator_id              UUID REFERENCES creators
model_hash              TEXT UNIQUE
parameter_count_b       FLOAT
active_parameter_count_b FLOAT
architecture            TEXT                   -- FK candidate to model_architectures
file_format             TEXT                   -- FK candidate to model_file_formats
quantization_id         INT NOT NULL REFERENCES quantization_methods
context_length_default  INT
context_length_min      INT
context_length_max      INT
release_date            DATE
version                 TEXT
description             TEXT
predecessor_model_id    UUID REFERENCES models
model_source_mtime      TIMESTAMPTZ
model_source_size       BIGINT
provisional             BOOLEAN                -- true until model_hash is computed from weights
```
`min_vram_mb` was removed; VRAM is calculated at claim time via `active_parameter_count_b × qm.vram_multiplier × 1.15` (per 4513822952).

### `model_sources`
```sql
source_id               SERIAL PRIMARY KEY
model_id                UUID REFERENCES models
source_type_id          INT REFERENCES source_types
url                     TEXT NOT NULL
source_metadata         JSONB                  -- hf_commit_hash, hf_model_id, ollama_tag, etc.
```

### `task_categories`
```sql
category_id             SERIAL PRIMARY KEY
name                    TEXT NOT NULL UNIQUE
description             TEXT
```

### `tasks`
```sql
task_id                 SERIAL PRIMARY KEY
name                    TEXT NOT NULL
category_id             INT REFERENCES task_categories
parent_id               INT REFERENCES tasks   -- null = top-level suite
sort_order              INT NOT NULL DEFAULT 0
description             TEXT
grader_script           TEXT
grader_script_commit    TEXT                   -- git SHA of grader file last modification
natural_key_hash        TEXT NOT NULL UNIQUE   -- SHA256 of canonical path; stable FK anchor
```

### `prompts`
```sql
prompt_id               SERIAL PRIMARY KEY
task_id                 INT NOT NULL REFERENCES tasks
file_path               TEXT NOT NULL
git_commit_hash         TEXT NOT NULL
content_sha256          TEXT UNIQUE            -- alias hash = SHA256(path + original_hash)
content_length_bytes    INT
version                 TEXT
release_date            DATE
is_prerelease           BOOLEAN NOT NULL DEFAULT FALSE
difficulty              INT NOT NULL DEFAULT 0 -- 0 = basic; integer scale
modified_at             TIMESTAMPTZ
```

### `runners`
```sql
runner_id               TEXT PRIMARY KEY       -- stable agent ID; e.g. "worker-01"
hostname                TEXT NOT NULL
process_id              INT NOT NULL
effective_user          TEXT NOT NULL
last_heartbeat          TIMESTAMPTZ NOT NULL DEFAULT NOW()
status                  TEXT NOT NULL DEFAULT 'ACTIVE'
                            CHECK (status IN ('ACTIVE','IDLE','DEAD'))
                            -- TODO(P2): migrate TEXT+CHECK → ENUM once schema stabilizes (#13)
started_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
```

### `runs`
```sql
run_id                  UUID PRIMARY KEY       -- client-generated
submitted_at            TIMESTAMPTZ NOT NULL
publisher_id            TEXT NOT NULL
runner_version          TEXT
prompt_git_hash         TEXT                   -- harness-level prompts repo commit at run time
runner_id               TEXT REFERENCES runners
```

### `run_queue`
```sql
queue_id                UUID PRIMARY KEY DEFAULT gen_random_uuid()
run_id                  UUID NOT NULL REFERENCES runs ON DELETE CASCADE
prompt_id               INT NOT NULL REFERENCES prompts
model_id                UUID NOT NULL REFERENCES models  -- via runs; claim query joins
priority                INT NOT NULL DEFAULT 100         -- lower = sooner; 0 = critical
status                  TEXT NOT NULL DEFAULT 'PENDING'
                            CHECK (status IN ('PENDING','CLAIMED','IN_PROGRESS','COMPLETE','FAILED','CANCELLED'))
attempt_count           INT NOT NULL DEFAULT 0
max_attempts            INT NOT NULL DEFAULT 5
claimed_by              TEXT
claimed_at              TIMESTAMPTZ
started_at              TIMESTAMPTZ
completed_at            TIMESTAMPTZ
retry_after             TIMESTAMPTZ                      -- claim gate; NULL = no delay constraint
error_detail            TEXT
source_file             TEXT                             -- DR artefact; path in queue/pending/
created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
```

### `run_model_metrics`
```sql
run_id                  UUID NOT NULL REFERENCES runs
prompt_id               INT NOT NULL REFERENCES prompts
model_id                UUID NOT NULL REFERENCES models
score                   FLOAT                           -- 0–1; model capability, not hardware-affected
pass_fail               BOOLEAN
failure_reason          TEXT                            -- "timeout", "oom", "parse_error"; null = success
gflops_per_token        FLOAT
PRIMARY KEY (run_id, prompt_id, model_id)
```

### Migration tables (design converged; implementation Phase 2)
```sql
schema_versions (schema_version_id SERIAL PK, description, schema_migration_script, record_migration_script, created_at)
schema_version_history (schema_version_id FK, applied_at, PK(schema_version_id))
schema_tables (table_id SERIAL PK, table_name UNIQUE, uuid_namespace UUID)  -- one fixed namespace per table
schema_hops (hop_id SERIAL PK, table_id FK, from_version FK, to_version FK, allow_migration BOOLEAN, UNIQUE(table_id, from_version, to_version))
```
`uuid_migrations` (with `old_uuid`/`new_uuid` columns) was abandoned in favor of `schema_hops`; FK resolution is computed from per-record `schema_version_id` + namespace re-derivation rather than looked up (per 4573134793).

### Hardware tables (#17–21)
`interface_type`, `system_hardware`, `system_software`, `gpu_hardware`, `system_gpu_link`, `run_hardware_metrics` — schemas locked per #17–21; disk persistence settled in #22: hardware block embedded inline in `runs/<run_id>.json`; `hardware-snapshot/<uuid>.json` dedup cache written at ingest; `seeds/interface_types.json` shipped (`3b0bce4`).

### Disk layout (`BAKEOFF_DATA_DIR`)
```
seeds/quantization_methods.json
seeds/model_architectures.json
seeds/model_file_formats.json
seeds/source_types.json
seeds/task_categories.json
seeds/interface_types.json
models/<uuid>.json          -- creator + sources embedded; model_source_mtime, model_source_size
tasks/<natural_key_hash>.json
prompts/<content_sha256>.json
runners/<runner_id>.json
runs/<run_id>.json          -- run_model_metrics + hardware block embedded
run_queue/ (ephemeral exception — job submission lifecycle only)
```
`schema_version: 1` (integer) present in all disk files from day one (per 4522798455, 4524243217).

---

## Notable / unusual decisions

- **VRAM not stored; computed at claim time** — `min_vram_mb` removed from `models`; VRAM estimated as `CEIL(active_parameter_count_b × qm.vram_multiplier × 1.15)` inline in the claim query, joining `quantization_methods`. Single update point; no staleness risk; claim query passes runner's available VRAM as a parameter (per 4513822952, 4519072195). Downstream: enables fully capability-based routing without any manual hardware tagging.

- **`quantization_methods` as a lookup table instead of a CASE expression** — gissf1 proposed this over Bastion's initial CASE-in-SQL approach (per 4516038912). Benefit: extends to new formats without schema or code changes; doubles as the source for the bakeoff-results UI filter dropdown.

- **UUID5 for `model_id` (deterministic, not SERIAL)** — `uuid5(BAKEOFF_MODEL_NAMESPACE, model_hash)` primary; provisional UUID from source URL + params when hash not yet computed; promoted once weights are pulled (per 4522610944, 4524304026). Enables distributed generation and cross-system consistency without a DB round-trip.

- **`prompt.difficulty` is an integer (0-indexed), not an ENUM** — supports inserting new difficulty levels at any tier without re-seeding; 0 = basic (per 4496938804, 4497170538, 4501587412, 4503203398). Field name shortened from `difficulty_level` to `difficulty` (per 4501587412).

- **`content_sha256` UNIQUE with alias hashing** — intentional prompt duplication handled via a YAML alias stub; alias `content_sha256 = SHA256(path + original_hash)`; unintentional collision is a hard import error. Prevents wasted compute on duplicate runs while permitting intentional re-use (per 4501587412, 4503203398).

- **`retry_after` replaces `updated_at` gate** — FAILED items stay FAILED (visible for operator review); a separate `retry_after TIMESTAMPTZ` controls claim eligibility. Cleaner claim query; also supports non-failure scheduling (per 4513822952). `updated_at` retained alongside it as an audit field (per 4522549514).

- **Standalone runner is the default mode; queue is opt-in** — inverted from Bastion's initial proposal at gissf1's direction (per 4516038912, 4519072195). Queue worker wraps the same inference core.

- **File-based prompt/task definitions; DB as derived cache** — filesystem is source of truth; scanner upserts DB on each run. `natural_key_hash = SHA256(canonical_path)` on `tasks` prevents FK invalidation across re-scans (per 4496938804, 4497170538).

- **`schema_hops` replaces `uuid_migrations`** — UUID columns dropped from the migration table; FK resolution computed from `schema_version_id` + per-table namespace re-derivation rather than stored as old→new UUID pairs. Avoids unbounded table growth at scale (per 4572098058, 4573134793).

- **Per-table UUID namespaces (not per-version)** — gissf1 identified that per-version namespaces force UUID recomputation for every record on every schema bump, exploding `uuid_migrations`. Corrected to one fixed namespace per table in `schema_tables` (per 4572098058, 4573134793).

- **Hardware block embedded inline in run disk file** — run file is self-contained; hardware deduplication happens at ingest, not at write time. Sparse/reference format deferred to Phase 2 (per 4533647917, 4568864985).

---

## Open / unresolved

1. **`schema_tables.uuid_namespace` storage: DB vs Go constants** — should the per-table UUID namespace be stored in `schema_tables` (DB + backing disk file, allowing tooling lookup without code changes) or defined only as Go constants (simpler, source of truth in code)? gissf1 stated "schema_tables table as you suggested" and "stored into backing files on disk" (per 4572098058), but no final implementation confirmation was given before the thread closed.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4572098058>

2. **`schema_hops` table name vs alternatives** — Bastion proposed `schema_hops`; gissf1 mentioned `table_migrations` or integrating into `schema_versions`. No explicit name ratification before #22 closed with migration framework punted to #25.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4573134793>

3. **BIOS UUID field set (hardware snapshot identity)** — Bastion proposed including `memory_speed_mhz`, `memory_channels`, `memory_interleave_profile` in UUID input, excluding `bios_notes` / `power_limit_w`. gissf1 did not explicitly confirm or modify the field list in Q2 follow-up (per 4533647917, 4568864985); the open confirmation "BIOS field list above — is this the right set?" was not answered before #22 closed.
   - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4533647917>

4. **`--strict-cache` default: warn vs abort on model source file drift** — Bastion proposed warn in standalone mode, abort in queue worker mode (per 4522610944). gissf1's reply described the three-case policy (size change → INVALID; mtime change + same size → recompute hash; mtime change + same content → update mtime) but did not explicitly confirm the warn/abort default split (per 4524200643).
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524200643>

5. **Prompt descriptor disk file format** — deferred to the scoring architecture thread (per 4519072195, 4524304026). No ticket number was assigned for that thread in this corpus.
   - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524304026>

---

## Cross-topic links

- **`run_model_metrics`** → `runs`, `prompts`, `models` — three-way composite PK; scoring and failure reason live here. Links to `evaluation_methodology` (score semantics, hardware-neutral pass/fail).
- **`run_hardware_metrics` / hardware tables** (#17–21) — `run_hardware_metrics` is 1:1 with `runs`; hardware block embedded in `runs/<run_id>.json`. Hardware schema locked outside this thread.
- **`run_queue`** — `prompt_id FK → prompts`; capability-match via `quantization_methods.vram_multiplier`; `retry_after` gate and `source_file` DR artefact. Links to runner/queue-worker design (terminology settled in #13).
- **`schema_versions` / `schema_hops`** — migration framework design punted to #25; `schema_tables.uuid_namespace` affects disk layout established in #15 and queue implementation in #13.
- **`quantization_methods` seed** → also serves as the source for the bakeoff-results UI filter bar (`evaluation_ui` topic, if present).
- **`tasks` / `prompts` disk files** → filesystem-backed; scanner derives DB rows. Links to any `evaluation_methodology` or `scoring` topic covering grader scripts and difficulty gating.
