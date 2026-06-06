# Summary: run_model_metrics

## Final state

Nine-table cluster covering the model-side of a benchmark run. Converged across #8, #12. Final schema (per 4503669210):

```sql
-- Model identity
CREATE TABLE creators (
    creator_id          SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    display_name        TEXT NULLABLE,
    homepage            TEXT NULLABLE,
    service_identifiers JSONB NULLABLE    -- {"huggingface": "meta-llama", "ollama": "meta"}
);

CREATE TABLE source_types (
    source_type_id      SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE    -- "huggingface", "ollama", "direct_url", "local_file"
);

CREATE TABLE models (
    model_id                    SERIAL PRIMARY KEY,
    name                        TEXT NOT NULL,
    creator_id                  INT NULLABLE REFERENCES creators,
    model_hash                  TEXT UNIQUE,             -- SHA256 of weights; deduplication ground truth
    parameter_count_b           FLOAT NULLABLE,          -- total params in billions
    active_parameter_count_b    FLOAT NULLABLE,          -- active params in billions (MoE only)
    architecture                TEXT NULLABLE,
    context_length_default      INT NULLABLE,
    context_length_min          INT NULLABLE,
    context_length_max          INT NULLABLE,
    file_format                 TEXT NULLABLE,           -- "GGUF", "SafeTensors", etc.
    quantization                TEXT NULLABLE,
    min_vram_mb                 INT NULLABLE,
    release_date                DATE NULLABLE,
    version                     TEXT NULLABLE,
    description                 TEXT NULLABLE,
    predecessor_model_id        INT NULLABLE REFERENCES models
);

CREATE TABLE model_sources (
    source_id           SERIAL PRIMARY KEY,
    model_id            INT NOT NULL REFERENCES models,
    source_type_id      INT NOT NULL REFERENCES source_types,
    url                 TEXT NOT NULL,
    source_metadata     JSONB NULLABLE    -- {"hf_commit_hash": "...", "hf_model_id": "...", "ollama_tag": "..."}
);

-- Task / prompt hierarchy
CREATE TABLE task_categories (
    category_id         SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    description         TEXT NULLABLE
);

CREATE TABLE tasks (
    task_id             SERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    category_id         INT NULLABLE REFERENCES task_categories,
    parent_id           INT NULLABLE REFERENCES tasks,    -- null = top-level suite
    sort_order          INT NOT NULL DEFAULT 0,
    description         TEXT NULLABLE,
    grader_script       TEXT NULLABLE,
    grader_script_commit TEXT NULLABLE,                   -- git SHA of grader file's last modification
    natural_key_hash    TEXT NOT NULL UNIQUE              -- SHA256 of canonical path; stable FK anchor
);

CREATE TABLE prompts (
    prompt_id               SERIAL PRIMARY KEY,
    task_id                 INT NOT NULL REFERENCES tasks,
    file_path               TEXT NOT NULL,
    git_commit_hash         TEXT NOT NULL,
    content_sha256          TEXT UNIQUE,                  -- alias hashes = SHA256(path + original_hash)
    content_length_bytes    INT NULLABLE,
    version                 TEXT NULLABLE,
    release_date            DATE NULLABLE,
    is_prerelease           BOOLEAN NOT NULL DEFAULT FALSE,
    difficulty              INT NOT NULL DEFAULT 0,
    modified_at             TIMESTAMPTZ NULLABLE
);

-- Run anchor
CREATE TABLE runs (
    run_id              UUID PRIMARY KEY,
    submitted_at        TIMESTAMPTZ NOT NULL,
    publisher_id        TEXT NOT NULL,
    runner_version      TEXT NULLABLE,
    prompt_git_hash     TEXT NULLABLE                     -- commit hash of prompt repo at run time
);

-- Per-run model measurements
CREATE TABLE run_model_metrics (
    run_id              UUID NOT NULL REFERENCES runs,
    prompt_id           INT NOT NULL REFERENCES prompts,
    model_id            INT NOT NULL REFERENCES models,
    score               FLOAT NULLABLE,
    pass_fail           BOOLEAN NULLABLE,
    failure_reason      TEXT NULLABLE,                    -- "timeout", "oom", "parse_error"; null = success
    gflops_per_token    FLOAT NULLABLE,
    PRIMARY KEY (run_id, prompt_id, model_id)
);
```

`run_model_metrics.task_id` is omitted — task is derivable via `prompt → task` join. Add only if join proves costly in profiling (per 4494171918).

## Notable / unusual decisions

- **`models.creator` renamed to `creator_id` FK → `creators` table** — most models are locally hosted; "provider" implies a cloud API vendor. "Creator" = the research lab or team. `creators` table holds `service_identifiers JSONB` to map one entity across HuggingFace, Ollama, and other registries without HF-coupling (per 4495675210, 4495777337).
- **`model_sources` split off `models`** — keeps the core `models` table provider-agnostic. All service-specific identifiers (HF commit hash, Ollama tag, etc.) go into `source_metadata JSONB` on `model_sources`. No HF columns in `models` (per 4495675210).
- **`parameter_count_b FLOAT` not `BIGINT`** — billions-unit float is human-readable and sufficient precision; BIGINT for raw counts hits 9×10¹⁸ limit but is verbose and mismatches how model sizes are discussed (per 4495777337).
- **`model_hash UNIQUE`, never auto-merged** — a hash collision surfaces to an administrator with a full field-diff for review. Three outcomes: accept coalesce, reject and create new record with note, or reject entirely with note (per 4495675210, 4497170538). Name+creator+quantization match triggers a hash check hint but does not auto-coalesce.
- **Prompt content is filesystem/git-backed; DB is a derived cache** — prompt text stays in version-controlled files; `prompts` table stores metadata (`file_path`, `git_commit_hash`, `content_sha256`) to make results reproducible. Scanner populates DB from files; git is the authoritative source (per 4491873240, 4495675210).
- **`content_sha256 UNIQUE` with alias indirection** — intentional prompt duplication uses a YAML stub file with `type: alias`; alias hash = SHA256(relative_path + original_content_sha256). Preserves UNIQUE constraint; unintentional duplicates are hard-rejected at import (per 4501587412, 4503203398).
- **`tasks.natural_key_hash` TEXT NOT NULL UNIQUE** — SERIAL PKs are regeneration-unsafe; re-scanning the filesystem would invalidate all FK references. Deterministic SHA256 of canonical path produces stable IDs across re-scans (per 4497170538).
- **`difficulty INT NOT NULL DEFAULT 0` on `prompts`, not on `tasks`** — difficulty is a property of the specific test instance; mixed difficulty levels within a single task are expected. Integer allows inserting new levels at either end without renumbering. Relative references (`parent+1`, `task:X/001+1`) resolve at scan time; DB stores resolved integer only (per 4496938804, 4497170538).
- **`gflops_per_token FLOAT`** — renamed from `flops_per_token_theoretical BIGINT`. "Theoretical" replaced with no qualifier; table context makes the estimation nature clear. GFLOPs unit is human-readable and precision-sufficient (per 4496938804, 4503203398).
- **`runs` table as join anchor** — run-scoped metadata (timestamp, submitter, runner version) does not belong on `run_model_metrics` or `run_hardware_metrics`. UUID PK prevents enumeration and can be generated client-side before DB write (per 4494171918). Both metrics tables FK into `runs`.
- **`failure_reason TEXT` separates hardware failures from model scores** — a model that hits OOM or timeout is not scored as a failure; `failure_reason` carries the cause and `score`/`pass_fail` stay null. The combined efficiency metric (quality × hardware performance) is a Phase 2 derived view, not stored (per 4491873240).
- **`prompt_git_hash` on `runs` vs `git_commit_hash` on `prompts`** — two distinct fields: `runs.prompt_git_hash` pins the entire prompt repo state at run time; `prompts.git_commit_hash` pins the last-modifying commit for a specific prompt file. Both are needed for full reproducibility auditing (per 4494171918).

## Open / unresolved

- **`run_model_metrics.task_id` denormalization** — recommended omission unless query profiling shows the `run → prompt → task` join is costly. No profiling data yet. Trade-off: denormalized FK adds one join level of convenience at the cost of potential inconsistency if `prompt.task_id` is updated (per 4494171918).
- **Difficulty dependency logic in `task.yaml`** — structured `depends_on` block (min pass rate, scope) was proposed for gating level-N prompts. Encoding is in version-controlled files, not DB rows. Full runner logic deferred to a testing-queue issue; no schema impact expected but not yet finalized (per 4497170538).
- **`tasks` filesystem structure depth** — multi-level sub-task nesting via directory `task.yaml` files was agreed in principle; `prompt_glob` and `prompt_sort` fields in `task.yaml` were proposed. Final `task.yaml` schema not ratified — details (custom sort mechanisms, glob scoping) were noted for further iteration (per 4497170538).
- **Testing queue issue** — @gissf1 agreed to open it; status as of final entry (4503669210) was "standing by for direction". No issue number recorded in this topic.
- **`model_hash` as `UNIQUE NOT NULL` vs `UNIQUE NULLABLE`** — spec text says `UNIQUE` but models may be submitted before the hash is computed (large file). Whether null is permitted was not explicitly resolved; the schema snapshot at 4503669210 omits `NOT NULL` from `model_hash`.

## Cross-topic links

- `runs.run_id` ← `run_hardware_metrics.run_id` — **run_hardware_metrics** topic (issue #21); both tables share the `runs` FK anchor
- `run_hardware_metrics.(system_hardware_id, slot_index)` → **system_gpu_link** (topic: system_gpu_link / issue #20) — hardware context for any run is a join through `system_gpu_link → system_hardware + gpu_hardware + interface_type`
