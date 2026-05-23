-- schema/schema.sql
-- Relational schema for the bakeoff benchmarking harness.
-- Design agreed upon in Rethunk-AI/bakeoff#12 and #14.
-- Do not hand-edit this file without a matching update to both issues.

-- ---------------------------------------------------------------------------
-- source_types
-- Enumeration of provider / delivery mechanisms for model weights.
-- ---------------------------------------------------------------------------
CREATE TABLE source_types (
    source_type_id SERIAL PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE   -- 'huggingface', 'ollama', 'direct_url', 'local_file'
);

-- Seed: canonical provider names agreed in #14
INSERT INTO source_types (name) VALUES
    ('huggingface'),
    ('ollama'),
    ('direct_url'),
    ('local_file');

-- ---------------------------------------------------------------------------
-- creators
-- Organisations or individuals that publish models.
-- ---------------------------------------------------------------------------
CREATE TABLE creators (
    creator_id          SERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    display_name        TEXT,
    homepage            TEXT,
    service_identifiers JSONB   -- e.g. {"huggingface": "microsoft", "ollama": "microsoft"}
);

-- ---------------------------------------------------------------------------
-- models
-- One row per distinct weights file / quantisation variant.
-- model_hash (SHA256 of weights) is the deduplication ground truth (#12).
-- parameter_count_b stored in billions as FLOAT to avoid waste on large values (#12).
-- predecessor_model_id tracks fine-tune / derivative lineage.
-- ---------------------------------------------------------------------------
CREATE TABLE models (
    model_id                  SERIAL PRIMARY KEY,
    name                      TEXT NOT NULL,
    creator_id                INT REFERENCES creators,
    model_hash                TEXT UNIQUE,     -- SHA256 of weights file; deduplication ground truth
    parameter_count_b         FLOAT,           -- total params in billions
    active_parameter_count_b  FLOAT,           -- active params (MoE only)
    architecture              TEXT,            -- 'Dense', 'MoE'
    context_length_default    INT,
    context_length_min        INT,
    context_length_max        INT,
    file_format               TEXT,            -- 'GGUF', 'SafeTensors', etc.
    quantization              TEXT,            -- 'Q4_K_M', 'fp16', etc.
    min_vram_mb               INT,
    release_date              DATE,
    version                   TEXT,
    description               TEXT,
    predecessor_model_id      INT REFERENCES models
);

-- ---------------------------------------------------------------------------
-- model_sources
-- Where a model can be fetched from (may have multiple rows per model).
-- updated = when all source_metadata fields (and model_hash) were sourced.
-- Multi-query consistency protocol: 1-minute window per round, retry on
-- value divergence, 5-minute hard cap (#14).
-- source_metadata is flat JSONB — no sub-objects (#14 signoff).
-- ---------------------------------------------------------------------------
CREATE TABLE model_sources (
    source_id       SERIAL PRIMARY KEY,
    model_id        INT NOT NULL REFERENCES models,
    source_type_id  INT NOT NULL REFERENCES source_types,
    url             TEXT NOT NULL,
    source_metadata JSONB,          -- flat: provider identity + stats co-located, no sub-objects
                                    -- e.g. {"ollama_tag": "llama3:8b-q4_K_M", "pulls": 3000}
                                    -- e.g. {"hf_commit": "abc123", "downloads": 50000, "likes": 1200}
    updated         TIMESTAMPTZ     -- when all source_metadata fields (and model_hash) were sourced
);

-- ---------------------------------------------------------------------------
-- task_categories
-- Broad groupings for task suites (baseline / comparison / advanced).
-- ---------------------------------------------------------------------------
CREATE TABLE task_categories (
    category_id SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,   -- 'baseline', 'comparison', 'advanced'
    description TEXT
);

-- ---------------------------------------------------------------------------
-- tasks
-- Tiered: parent_id IS NULL = top-level suite; non-null = sub-task.
-- natural_key_hash = SHA256 of canonical path relative to prompts root;
-- stable FK anchor across re-scans even after path renames (#12).
-- ---------------------------------------------------------------------------
CREATE TABLE tasks (
    task_id              SERIAL PRIMARY KEY,
    name                 TEXT NOT NULL,
    category_id          INT REFERENCES task_categories,
    parent_id            INT REFERENCES tasks,       -- null = top-level suite
    sort_order           INT NOT NULL DEFAULT 0,     -- execution order within parent
    description          TEXT,
    grader_script        TEXT,                       -- path to grader script
    grader_script_commit TEXT,                       -- git SHA of grader file last modification
    natural_key_hash     TEXT NOT NULL UNIQUE        -- SHA256 of canonical path; stable FK anchor
);

-- ---------------------------------------------------------------------------
-- prompts
-- Metadata for prompt files tracked in git.
-- content_sha256 UNIQUE enables deduplication; alias prompts use
-- SHA256(path + original_hash) (#12).
-- difficulty INT (not ENUM, not difficulty_level — agreed in #12).
-- is_prerelease supports A/B testing new prompts before full rollout.
-- ---------------------------------------------------------------------------
CREATE TABLE prompts (
    prompt_id            SERIAL PRIMARY KEY,
    task_id              INT NOT NULL REFERENCES tasks,
    file_path            TEXT NOT NULL,
    git_commit_hash      TEXT NOT NULL,     -- git SHA of commit that last modified this file
    content_sha256       TEXT UNIQUE,       -- SHA256 of prompt text (or alias hash); dedup anchor
    content_length_bytes INT,
    version              TEXT,              -- human-readable version label
    release_date         DATE,
    is_prerelease        BOOLEAN NOT NULL DEFAULT FALSE,
    difficulty           INT NOT NULL DEFAULT 0,   -- 0 = basic; higher = harder
    modified_at          TIMESTAMPTZ        -- git last-modified timestamp (populated by scanner)
);

-- ---------------------------------------------------------------------------
-- runs
-- A single benchmarking execution.
-- run_id is client-generated UUID — globally unique, prevents enumeration.
-- ---------------------------------------------------------------------------
CREATE TABLE runs (
    run_id          UUID PRIMARY KEY,
    submitted_at    TIMESTAMPTZ NOT NULL,
    publisher_id    TEXT NOT NULL,          -- submitting user/account
    runner_version  TEXT,                   -- bakeoff harness semver or commit hash
    prompt_git_hash TEXT                    -- git commit hash of prompt files at run time
);

-- ---------------------------------------------------------------------------
-- run_model_metrics
-- Per-(run, prompt, model) result row.
-- score NULL means the run failed for this cell.
-- gflops_per_token = theoretical GFLOPs per forward-pass token (agreed name, #12).
-- Composite PK: (run_id, prompt_id, model_id).
-- ---------------------------------------------------------------------------
CREATE TABLE run_model_metrics (
    run_id           UUID NOT NULL REFERENCES runs,
    prompt_id        INT  NOT NULL REFERENCES prompts,
    model_id         INT  NOT NULL REFERENCES models,
    score            FLOAT,          -- 0.0–1.0; null if run failed
    pass_fail        BOOLEAN,        -- null if run failed
    failure_reason   TEXT,           -- 'timeout', 'oom', 'parse_error'; null = success
    gflops_per_token FLOAT,          -- theoretical GFLOPs per forward pass token
    PRIMARY KEY (run_id, prompt_id, model_id)
);

-- ---------------------------------------------------------------------------
-- runners
-- One row per registered runner. public_key is base64-encoded Ed25519 public key.
-- ---------------------------------------------------------------------------
CREATE TABLE runners (
    runner_id     TEXT PRIMARY KEY,
    public_key    TEXT NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description   TEXT
);
