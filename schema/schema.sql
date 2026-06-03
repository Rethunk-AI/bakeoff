-- schema/schema.sql
-- Relational schema for the bakeoff benchmarking harness.
-- Design agreed in Rethunk-AI/bakeoff#12, #13, #14, #15.
-- Do not hand-edit without a matching update to the relevant issues.

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
-- quantization_methods (#13)
-- Lookup table for model quantization formats.
-- vram_multiplier = bytes per active parameter (e.g. 4.0 for fp32, 0.563 for q4_k_m).
-- Used in claim query: CEIL(active_parameter_count_b * vram_multiplier * 1.15) <= runner_vram_gb.
-- Seed data in seeds/quantization_methods.json.
-- ---------------------------------------------------------------------------
CREATE TABLE quantization_methods (
    quantization_id  SERIAL  PRIMARY KEY,
    name             TEXT    NOT NULL UNIQUE,
    vram_multiplier  DECIMAL NOT NULL,
    description      TEXT
);

INSERT INTO quantization_methods (name, vram_multiplier, description) VALUES
    -- Full precision
    ('fp32',    4.000, 'IEEE 754 single precision — 4 bytes/weight'),
    ('fp16',    2.000, 'IEEE 754 half precision — 2 bytes/weight'),
    ('bf16',    2.000, 'Brain float 16 — 2 bytes/weight'),
    -- 8-bit
    ('q8_0',    1.063, 'GGUF Q8_0 — 8 bits + 2-byte scale per 32-weight block'),
    -- 6-bit
    ('q6_k',    0.820, 'GGUF Q6_K — 6 bits/weight with K-quant super-blocks'),
    -- 5-bit
    ('q5_k_m',  0.684, 'GGUF Q5_K_M — 5-bit K-quant medium'),
    ('q5_k_s',  0.664, 'GGUF Q5_K_S — 5-bit K-quant small'),
    ('q5_0',    0.688, 'GGUF Q5_0 — 5 bits + 2-byte scale per 32-weight block'),
    ('q5_1',    0.750, 'GGUF Q5_1 — 5 bits + 4-byte scale+min per 32-weight block'),
    -- 4-bit
    ('q4_k_m',  0.563, 'GGUF Q4_K_M — 4-bit K-quant medium (recommended general use)'),
    ('q4_k_s',  0.545, 'GGUF Q4_K_S — 4-bit K-quant small'),
    ('q4_0',    0.563, 'GGUF Q4_0 — 4 bits + 2-byte scale per 32-weight block'),
    ('q4_1',    0.625, 'GGUF Q4_1 — 4 bits + 4-byte scale+min per 32-weight block'),
    -- 3-bit
    ('q3_k_l',  0.461, 'GGUF Q3_K_L — 3-bit K-quant large'),
    ('q3_k_m',  0.465, 'GGUF Q3_K_M — 3-bit K-quant medium'),
    ('q3_k_s',  0.410, 'GGUF Q3_K_S — 3-bit K-quant small'),
    -- 2-bit
    ('q2_k',    0.352, 'GGUF Q2_K — 2-bit K-quant'),
    -- imatrix quantization
    ('iq4_xs',  0.534, 'GGUF IQ4_XS — 4-bit imatrix extra-small'),
    ('iq4_nl',  0.563, 'GGUF IQ4_NL — 4-bit imatrix non-linear'),
    ('iq3_m',   0.441, 'GGUF IQ3_M — 3-bit imatrix medium'),
    ('iq3_s',   0.394, 'GGUF IQ3_S — 3-bit imatrix small'),
    ('iq3_xxs', 0.328, 'GGUF IQ3_XXS — 3-bit imatrix extra-extra-small'),
    ('iq2_m',   0.289, 'GGUF IQ2_M — 2-bit imatrix medium'),
    ('iq2_xs',  0.274, 'GGUF IQ2_XS — 2-bit imatrix extra-small'),
    ('iq2_xxs', 0.266, 'GGUF IQ2_XXS — 2-bit imatrix extra-extra-small'),
    ('iq1_m',   0.219, 'GGUF IQ1_M — 1-bit imatrix medium'),
    ('iq1_s',   0.188, 'GGUF IQ1_S — 1-bit imatrix small');

-- ---------------------------------------------------------------------------
-- model_architectures (#13/#15)
-- Lookup table for model architecture types.
-- Seed data in seeds/model_architectures.json.
-- New values require admin review — no automated insertion (#13).
-- ---------------------------------------------------------------------------
CREATE TABLE model_architectures (
    architecture_id  SERIAL  PRIMARY KEY,
    name             TEXT    NOT NULL UNIQUE,   -- 'Dense', 'MoE', 'SSM', 'Hybrid'
    description      TEXT
);

INSERT INTO model_architectures (name, description) VALUES
    ('Dense',  'Standard transformer with all parameters active per token'),
    ('MoE',    'Mixture of Experts — subset of parameters active per token'),
    ('SSM',    'State Space Model (e.g. Mamba)'),
    ('Hybrid', 'Mixed architecture combining MoE and Dense layers');

-- ---------------------------------------------------------------------------
-- model_file_formats (#13/#15)
-- Lookup table for model weight file formats.
-- Seed data in seeds/model_file_formats.json.
-- New values require admin review — no automated insertion (#13).
-- ---------------------------------------------------------------------------
CREATE TABLE model_file_formats (
    file_format_id  SERIAL  PRIMARY KEY,
    name            TEXT    NOT NULL UNIQUE,   -- 'GGUF', 'SafeTensors', etc.
    description     TEXT
);

INSERT INTO model_file_formats (name, description) VALUES
    ('GGUF',        'GGUF format — llama.cpp native, self-contained'),
    ('SafeTensors', 'HuggingFace SafeTensors format'),
    ('PyTorch',     'PyTorch .pt / .pth checkpoint'),
    ('ONNX',        'Open Neural Network Exchange format'),
    ('ExLlamaV2',   'ExLlamaV2 quantized format'),
    ('MLX',         'Apple MLX framework format');

-- ---------------------------------------------------------------------------
-- creators (#13/#15)
-- Organisations or individuals that publish models.
-- creator_id is deterministic UUID5(BAKEOFF_CREATOR_NAMESPACE, homepage).
-- Fallback: UUID5(BAKEOFF_CREATOR_NAMESPACE, display_name) with provisional=true.
-- ---------------------------------------------------------------------------
CREATE TABLE creators (
    creator_id          UUID PRIMARY KEY,
    name                TEXT NOT NULL,
    display_name        TEXT,
    homepage            TEXT,
    service_identifiers JSONB,   -- e.g. {"huggingface": "microsoft", "ollama": "microsoft"}
    provisional         BOOLEAN NOT NULL DEFAULT FALSE  -- true until homepage-based UUID confirmed
);

-- ---------------------------------------------------------------------------
-- models (#13/#15)
-- One row per distinct weights file / quantisation variant.
-- model_id is deterministic UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash) when hash known;
-- UUID5(BAKEOFF_MODEL_NAMESPACE, source_url|param_count_b|model_source_size) provisional.
-- Lookup FKs for architecture, file_format, quantization (#13).
-- min_vram is calculated (active_parameter_count_b * vram_multiplier * 1.15), not stored (#13).
-- ---------------------------------------------------------------------------
CREATE TABLE models (
    model_id                  UUID PRIMARY KEY,
    name                      TEXT NOT NULL,
    creator_id                UUID REFERENCES creators,
    model_hash                TEXT UNIQUE,               -- SHA256 of weights file; dedup ground truth (#12)
    parameter_count_b         FLOAT,                    -- total params in billions
    active_parameter_count_b  FLOAT,                    -- active params per forward pass (= total for Dense)
    architecture_id           INT REFERENCES model_architectures,
    context_length_default    INT,
    context_length_min        INT,
    context_length_max        INT,
    file_format_id            INT REFERENCES model_file_formats,
    quantization_id           INT REFERENCES quantization_methods,
    model_source_mtime        TIMESTAMPTZ,               -- mtime of cached weights file (#13)
    model_source_size         BIGINT,                   -- byte size of cached weights file (#13)
    release_date              DATE,
    version                   TEXT,
    description               TEXT,
    predecessor_model_id      UUID REFERENCES models,
    provisional               BOOLEAN NOT NULL DEFAULT FALSE  -- true until model_hash computed
);

-- ---------------------------------------------------------------------------
-- model_sources
-- Where a model can be fetched from (may have multiple rows per model).
-- ---------------------------------------------------------------------------
CREATE TABLE model_sources (
    source_id       SERIAL PRIMARY KEY,
    model_id        UUID NOT NULL REFERENCES models,
    source_type_id  INT NOT NULL REFERENCES source_types,
    url             TEXT NOT NULL,
    source_metadata JSONB,          -- flat: provider identity + stats, no sub-objects (#14)
                                    -- e.g. {"ollama_tag": "llama3:8b-q4_K_M", "pulls": 3000}
                                    -- e.g. {"hf_commit": "abc123", "downloads": 50000}
    updated         TIMESTAMPTZ     -- when source_metadata and model_hash were last sourced
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

-- Seed: dumb_model floor tier (Rethunk-AI/bakeoff#23)
INSERT INTO task_categories (name, description)
    VALUES ('dumb_model', 'Minimal-capability floor suite: deterministic scorers only.')
    ON CONFLICT (name) DO NOTHING;

-- ---------------------------------------------------------------------------
-- tasks
-- Tiered: parent_id IS NULL = top-level suite; non-null = sub-task.
-- natural_key_hash = SHA256 of canonical path relative to prompts root (#12).
-- ---------------------------------------------------------------------------
CREATE TABLE tasks (
    task_id              SERIAL PRIMARY KEY,
    name                 TEXT NOT NULL,
    category_id          INT REFERENCES task_categories,
    parent_id            INT REFERENCES tasks,
    sort_order           INT NOT NULL DEFAULT 0,
    description          TEXT,
    grader_script        TEXT,
    grader_script_commit TEXT,
    natural_key_hash     TEXT NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------------
-- prompts
-- Metadata for prompt files tracked in git (#12).
-- ---------------------------------------------------------------------------
CREATE TABLE prompts (
    prompt_id            SERIAL PRIMARY KEY,
    task_id              INT NOT NULL REFERENCES tasks,
    file_path            TEXT NOT NULL,
    git_commit_hash      TEXT NOT NULL,
    content_sha256       TEXT UNIQUE,
    content_length_bytes INT,
    version              TEXT,
    release_date         DATE,
    is_prerelease        BOOLEAN NOT NULL DEFAULT FALSE,
    difficulty           INT NOT NULL DEFAULT 0,
    modified_at          TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- runners (#13/#16)
-- One row per registered runner.
-- Merged: Ed25519 signing identity (#16) + queue worker tracking (#13).
-- public_key is base64-encoded Ed25519 public key.
-- hostname/process_id/effective_user/last_heartbeat populated by queue worker mode.
-- TODO(deferred P2 #13): migrate status TEXT+CHECK to ENUM once schema stabilises.
-- ---------------------------------------------------------------------------
CREATE TABLE runners (
    runner_id      TEXT        PRIMARY KEY,
    public_key     TEXT        NOT NULL,                     -- Ed25519 public key, base64 (#16)
    hostname       TEXT,                                     -- runner hostname (#13)
    process_id     INT,                                      -- runner PID (#13)
    effective_user TEXT,                                     -- OS user (#13)
    last_heartbeat TIMESTAMPTZ,                              -- updated every 60s by queue worker (#13)
    status         TEXT        NOT NULL DEFAULT 'ACTIVE'
                               CHECK (status IN ('ACTIVE', 'IDLE', 'DEAD')),
    registered_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at     TIMESTAMPTZ,                              -- when this process started (#13)
    description    TEXT
);

-- ---------------------------------------------------------------------------
-- runs
-- A single benchmarking execution.
-- run_id is client-generated UUID (#12).
-- runner_id FK records which runner executed this run (#13).
-- ---------------------------------------------------------------------------
CREATE TABLE runs (
    run_id          UUID PRIMARY KEY,
    submitted_at    TIMESTAMPTZ NOT NULL,
    publisher_id    TEXT NOT NULL,
    runner_version  TEXT,
    prompt_git_hash TEXT,
    runner_id       TEXT REFERENCES runners   -- null = standalone run, no queue registration (#13)
);

-- Extend runs: run-level completeness status (Rethunk-AI/bakeoff#23)
ALTER TABLE runs
    ADD COLUMN IF NOT EXISTS run_status TEXT
        CHECK (run_status IN ('complete', 'incomplete', 'failed'));

-- ---------------------------------------------------------------------------
-- run_queue (#13)
-- Operational queue for model test jobs. DB-authoritative; files in queue/
-- directory serve as bootstrap / disaster-recovery artefacts only.
-- Claim protocol: FOR UPDATE SKIP LOCKED; capability filter in claim query.
-- ---------------------------------------------------------------------------
CREATE TABLE run_queue (
    queue_id      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id        UUID        NOT NULL REFERENCES runs ON DELETE CASCADE,
    prompt_id     INT         NOT NULL REFERENCES prompts,
    priority      INT         NOT NULL DEFAULT 100,   -- lower = claimed sooner
    status        TEXT        NOT NULL DEFAULT 'PENDING'
                              CHECK (status IN ('PENDING', 'CLAIMED', 'IN_PROGRESS', 'COMPLETE', 'FAILED', 'CANCELLED')),
    attempt_count INT         NOT NULL DEFAULT 0,
    max_attempts  INT         NOT NULL DEFAULT 5,
    claimed_by    TEXT,                               -- runner_id of claiming runner
    claimed_at    TIMESTAMPTZ,
    started_at    TIMESTAMPTZ,
    completed_at  TIMESTAMPTZ,
    error_detail  TEXT,
    retry_after   TIMESTAMPTZ,                        -- claim gate: do not pick up before this time
    source_file   TEXT,                               -- path to queue/pending/<uuid>.json (DR artefact)
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for next-item-to-claim query (capability filter applied in query, not here)
CREATE INDEX run_queue_claim_idx
    ON run_queue (priority, created_at)
    WHERE status = 'PENDING';

-- ---------------------------------------------------------------------------
-- run_model_metrics
-- Per-(run, prompt, model) result row.
-- score NULL means the run failed for this cell.
-- gflops_per_token = theoretical GFLOPs per forward-pass token (#12).
-- ---------------------------------------------------------------------------
CREATE TABLE run_model_metrics (
    run_id           UUID NOT NULL REFERENCES runs,
    prompt_id        INT  NOT NULL REFERENCES prompts,
    model_id         UUID NOT NULL REFERENCES models,
    score            FLOAT,
    pass_fail        BOOLEAN,
    failure_reason   TEXT,
    gflops_per_token FLOAT,
    PRIMARY KEY (run_id, prompt_id, model_id)
);

-- Extend run_model_metrics: structured failure detail (Rethunk-AI/bakeoff#23)
ALTER TABLE run_model_metrics
    ADD COLUMN IF NOT EXISTS failure_detail TEXT;
-- (failure_reason already exists; its values now conform to the failure_code taxonomy)

-- ---------------------------------------------------------------------------
-- interface_type (#17)
-- GPU interface / interconnect bus types.
-- ---------------------------------------------------------------------------
CREATE TABLE interface_type (
    interface_type_id SERIAL PRIMARY KEY,
    name              TEXT NOT NULL UNIQUE
);

INSERT INTO interface_type (name) VALUES
    ('PCIe 3.0 x16'),
    ('PCIe 4.0 x16'),
    ('PCIe 5.0 x16'),
    ('NVLink 3'),
    ('NVLink 4'),
    ('Thunderbolt 3'),
    ('Thunderbolt 4'),
    ('USB4'),
    ('eGPU'),
    ('integrated');

-- ---------------------------------------------------------------------------
-- system_hardware (#19)
-- Host machine hardware snapshot.
-- ---------------------------------------------------------------------------
CREATE TABLE system_hardware (
    system_hardware_id SERIAL PRIMARY KEY,
    cpu_model          TEXT,
    cpu_cores          INT,
    cpu_threads        INT,
    ram_gb             FLOAT,
    motherboard        TEXT
);

-- ---------------------------------------------------------------------------
-- system_software (#19)
-- Host OS / driver snapshot.
-- ---------------------------------------------------------------------------
CREATE TABLE system_software (
    system_software_id SERIAL PRIMARY KEY,
    os_name            TEXT,
    os_version         TEXT,
    driver_version     TEXT,
    runtime_version    TEXT
);

-- ---------------------------------------------------------------------------
-- gpu_hardware (#18)
-- One row per physical GPU slot.
-- ---------------------------------------------------------------------------
CREATE TABLE gpu_hardware (
    gpu_hardware_id    SERIAL PRIMARY KEY,
    gpu_model          TEXT NOT NULL,
    vram_mb            INT,
    interface_type_id  INT REFERENCES interface_type,
    tflops_fp16        FLOAT,
    tflops_fp32        FLOAT,
    tflops_bf16        FLOAT
);

-- ---------------------------------------------------------------------------
-- system_gpu_link (#20)
-- Many-to-many: which GPUs are in which system snapshot.
-- ---------------------------------------------------------------------------
CREATE TABLE system_gpu_link (
    system_hardware_id INT NOT NULL REFERENCES system_hardware,
    gpu_hardware_id    INT NOT NULL REFERENCES gpu_hardware,
    slot_index         INT NOT NULL DEFAULT 0,
    PRIMARY KEY (system_hardware_id, gpu_hardware_id, slot_index)
);

-- ---------------------------------------------------------------------------
-- run_hardware_metrics (#21)
-- Hardware context for a run (one row per run).
-- ---------------------------------------------------------------------------
CREATE TABLE run_hardware_metrics (
    run_id             UUID PRIMARY KEY REFERENCES runs,
    system_hardware_id INT REFERENCES system_hardware,
    system_software_id INT REFERENCES system_software,
    gpu_hardware_id    INT REFERENCES gpu_hardware,
    peak_vram_mb       INT,
    power_limit_w      FLOAT,
    measured_tflops    FLOAT
);

-- ---------------------------------------------------------------------------
-- schema_versions (#25)
-- One row per schema generation. allow_migration gates data migration for
-- this version. schema_migration_script / record_migration_script are Go
-- template + sprig scripts executed by the migration runner.
-- ---------------------------------------------------------------------------
CREATE TABLE schema_versions (
    schema_version_id       INTEGER     PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    description             TEXT,
    allow_migration         BOOLEAN     NOT NULL DEFAULT FALSE,
    schema_migration_script TEXT,
    record_migration_script TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- schema_tables (#25)
-- One row per logical table / uuid_namespace generation.
-- deprecated_at non-null signals migration is pending; check schema_tables_join
-- for destination(s). Minor upgrades (no join row) update DDL in place and
-- bump current_version_id only.
-- ---------------------------------------------------------------------------
CREATE TABLE schema_tables (
    table_id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name         TEXT        NOT NULL,
    uuid_namespace     UUID        NOT NULL,
    initial_version_id INTEGER     NOT NULL REFERENCES schema_versions(schema_version_id),
    current_version_id INTEGER     NOT NULL REFERENCES schema_versions(schema_version_id),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    deprecated_at      TIMESTAMPTZ,
    UNIQUE (table_name, uuid_namespace)
);

-- ---------------------------------------------------------------------------
-- schema_tables_join (#25)
-- Migration graph edges for major migrations (splits, merges, UUID namespace
-- changes). One or more rows for a src_table = data must move.
-- No row = minor upgrade; DDL updated in place on the existing table.
-- INDEX on src_table alone is omitted: the composite PK (src_table, dst_table)
-- covers src-only lookups via leading-key index scan.
-- ---------------------------------------------------------------------------
CREATE TABLE schema_tables_join (
    src_table     UUID    NOT NULL REFERENCES schema_tables(table_id),
    dst_table     UUID    NOT NULL REFERENCES schema_tables(table_id),
    migrate_using INTEGER NOT NULL REFERENCES schema_versions(schema_version_id),
    PRIMARY KEY (src_table, dst_table),
    CHECK (src_table <> dst_table)
);
