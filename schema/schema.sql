-- schema/schema.sql
-- Relational schema for the bakeoff benchmarking harness.
-- Do not hand-edit without updating the migration runner and seed files in lockstep.

-- ---------------------------------------------------------------------------
-- source_types
-- Enumeration of provider / delivery mechanisms for model weights.
-- ---------------------------------------------------------------------------
CREATE TABLE source_types (
    source_type_id SERIAL PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE   -- 'huggingface', 'ollama', 'direct_url', 'local_file'
);

-- Seed: canonical provider names
INSERT INTO source_types (name) VALUES
    ('huggingface'),
    ('ollama'),
    ('direct_url'),
    ('local_file');

-- ---------------------------------------------------------------------------
-- quantization_methods
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
-- model_architectures
-- Lookup table for model architecture types.
-- Seed data in seeds/model_architectures.json.
-- New values require admin review — no automated insertion.
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
-- model_file_formats
-- Lookup table for model weight file formats.
-- Seed data in seeds/model_file_formats.json.
-- New values require admin review — no automated insertion.
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
-- creators
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
-- models
-- One row per distinct weights file / quantisation variant.
-- model_id is deterministic UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash) when hash known;
-- UUID5(BAKEOFF_MODEL_NAMESPACE, source_url|param_count_b|model_source_size) provisional.
-- Lookup FKs for architecture, file_format, quantization.
-- min_vram is calculated (active_parameter_count_b * vram_multiplier * 1.15), not stored.
-- ---------------------------------------------------------------------------
CREATE TABLE models (
    model_id                  UUID PRIMARY KEY,
    name                      TEXT NOT NULL,
    creator_id                UUID REFERENCES creators,
    model_hash                TEXT UNIQUE,               -- SHA256 of weights file; dedup ground truth
    parameter_count_b         FLOAT,                    -- total params in billions
    active_parameter_count_b  FLOAT,                    -- active params per forward pass (= total for Dense)
    architecture_id           INT REFERENCES model_architectures,
    context_length_default    INT,
    context_length_min        INT,
    context_length_max        INT,
    file_format_id            INT REFERENCES model_file_formats,
    quantization_id           INT REFERENCES quantization_methods,
    model_source_mtime        TIMESTAMPTZ,               -- mtime of cached weights file
    model_source_size         BIGINT,                   -- byte size of cached weights file
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
    source_metadata JSONB,          -- flat: provider identity + stats, no sub-objects
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

-- Seed: dumb_model floor tier
INSERT INTO task_categories (name, description)
    VALUES ('dumb_model', 'Minimal-capability floor suite: deterministic scorers only.')
    ON CONFLICT (name) DO NOTHING;

-- Seed: cyber_safety category
INSERT INTO task_categories (name, description)
VALUES (
  'cyber_safety',
  'Cyber capability proxy tests: injection resistance, refusal quality, dual-use code generation, agentic containment, exfiltration-via-reasoning, non-expert uplift.'
)
ON CONFLICT (name) DO NOTHING;

-- ---------------------------------------------------------------------------
-- tasks
-- Tiered: parent_id IS NULL = top-level suite; non-null = sub-task.
-- natural_key_hash = SHA256 of canonical path relative to prompts root.
-- uplift_baseline_task_id: reference task whose score forms the uplift baseline.
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

-- Extend tasks: uplift baseline reference
ALTER TABLE tasks
    ADD COLUMN IF NOT EXISTS uplift_baseline_task_id INT REFERENCES tasks(task_id);

-- ---------------------------------------------------------------------------
-- prompts
-- Metadata for prompt files tracked in git.
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
-- runners
-- One row per registered runner.
-- Merged: Ed25519 signing identity + queue worker tracking.
-- public_key is base64-encoded Ed25519 public key.
-- hostname/process_id/effective_user/last_heartbeat populated by queue worker mode.
-- TODO(deferred P2): migrate status TEXT+CHECK to ENUM once schema stabilises.
-- ---------------------------------------------------------------------------
CREATE TABLE runners (
    runner_id      TEXT        PRIMARY KEY,
    public_key     TEXT        NOT NULL,                     -- Ed25519 public key, base64
    hostname       TEXT,                                     -- runner hostname
    process_id     INT,                                      -- runner PID
    effective_user TEXT,                                     -- OS user
    last_heartbeat TIMESTAMPTZ,                              -- updated every 60s by queue worker
    status         TEXT        NOT NULL DEFAULT 'ACTIVE'
                               CHECK (status IN ('ACTIVE', 'IDLE', 'DEAD')),
    registered_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at     TIMESTAMPTZ,                              -- when this process started
    description    TEXT
);

-- ---------------------------------------------------------------------------
-- runs
-- A single benchmarking execution.
-- run_id is client-generated UUID.
-- runner_id FK records which runner executed this run.
-- ---------------------------------------------------------------------------
CREATE TABLE runs (
    run_id          UUID PRIMARY KEY,
    submitted_at    TIMESTAMPTZ NOT NULL,
    publisher_id    TEXT NOT NULL,
    runner_version  TEXT,
    prompt_git_hash TEXT,
    runner_id       TEXT REFERENCES runners   -- null = standalone run, no queue registration
);

-- Extend runs: run-level completeness status
ALTER TABLE runs
    ADD COLUMN IF NOT EXISTS run_status TEXT
        CHECK (run_status IN ('complete', 'incomplete', 'failed'));

-- ---------------------------------------------------------------------------
-- run_queue
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
-- gflops_per_token = theoretical GFLOPs per forward-pass token.
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

-- Extend run_model_metrics: structured failure detail
ALTER TABLE run_model_metrics
    ADD COLUMN IF NOT EXISTS failure_detail TEXT;
-- (failure_reason already exists; its values now conform to the failure_code taxonomy)

-- ---------------------------------------------------------------------------
-- agent_traces
-- Per-step trace for agentic evaluation runs (C4 cyber_safety, C6 agentic).
-- boundary_violated: true if the step crossed a containment / safety boundary.
-- cost_tokens: token cost for this action step; NULL for non-LLM steps.
-- UNIQUE (run_id, prompt_id, model_id, step_index) prevents duplicate steps.
-- ---------------------------------------------------------------------------
CREATE TABLE agent_traces (
    trace_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id            UUID        NOT NULL REFERENCES runs ON DELETE CASCADE,
    prompt_id         INT         NOT NULL REFERENCES prompts,
    model_id          UUID        NOT NULL REFERENCES models,
    step_index        INT         NOT NULL,
    action_type       TEXT        NOT NULL,
    action_payload    TEXT,
    observation       TEXT,
    boundary_violated BOOLEAN     NOT NULL DEFAULT FALSE,
    cost_tokens       INT,
    recorded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, prompt_id, model_id, step_index)
);

-- Extend run_model_metrics: link to agent_traces for agentic evaluation
ALTER TABLE run_model_metrics
    ADD COLUMN IF NOT EXISTS trace_id UUID REFERENCES agent_traces(trace_id);

-- ---------------------------------------------------------------------------
-- interface_type
-- GPU interface / interconnect types (#17). Seed data in seeds/interface_types.json.
-- bandwidth_peak_gb_s is the universal comparable field, bidirectional aggregate.
-- For PCIe it is lane_transfer_rate * lane_count * 2 / 8.
-- lane_transfer_rate (GT/s per lane) and lane_count are NULL for non-PCIe interfaces.
-- ---------------------------------------------------------------------------
CREATE TABLE interface_type (
    interface_type_id   SERIAL PRIMARY KEY,
    bandwidth_peak_gb_s FLOAT NOT NULL,
    description         TEXT  NOT NULL UNIQUE,   -- 'PCIe 4.0 x16', 'SXM5', 'Thunderbolt 4'
    interface_family    TEXT,                    -- 'PCIe', 'SXM', 'NVLink', 'Thunderbolt', 'USB', 'OCuLink'
    lane_transfer_rate  FLOAT,                   -- PCIe: GT/s per lane (16 for Gen 4, 2.5 for Gen 1)
    lane_count          INT                      -- PCIe: lane width (16 for x16)
);

INSERT INTO interface_type (description, interface_family, lane_transfer_rate, lane_count, bandwidth_peak_gb_s) VALUES
    ('PCIe 1.0 x1', 'PCIe', 2.5, 1, 0.625),
    ('PCIe 1.0 x4', 'PCIe', 2.5, 4, 2.5),
    ('PCIe 1.0 x8', 'PCIe', 2.5, 8, 5.0),
    ('PCIe 1.0 x16', 'PCIe', 2.5, 16, 10.0),
    ('PCIe 2.0 x1', 'PCIe', 5, 1, 1.25),
    ('PCIe 2.0 x4', 'PCIe', 5, 4, 5.0),
    ('PCIe 2.0 x8', 'PCIe', 5, 8, 10.0),
    ('PCIe 2.0 x16', 'PCIe', 5, 16, 20.0),
    ('PCIe 3.0 x1', 'PCIe', 8, 1, 2.0),
    ('PCIe 3.0 x4', 'PCIe', 8, 4, 8.0),
    ('PCIe 3.0 x8', 'PCIe', 8, 8, 16.0),
    ('PCIe 3.0 x16', 'PCIe', 8, 16, 32.0),
    ('PCIe 4.0 x1', 'PCIe', 16, 1, 4.0),
    ('PCIe 4.0 x4', 'PCIe', 16, 4, 16.0),
    ('PCIe 4.0 x8', 'PCIe', 16, 8, 32.0),
    ('PCIe 4.0 x16', 'PCIe', 16, 16, 64.0),
    ('PCIe 5.0 x1', 'PCIe', 32, 1, 8.0),
    ('PCIe 5.0 x4', 'PCIe', 32, 4, 32.0),
    ('PCIe 5.0 x8', 'PCIe', 32, 8, 64.0),
    ('PCIe 5.0 x16', 'PCIe', 32, 16, 128.0),
    ('SXM2', 'SXM', NULL, NULL, 300),
    ('SXM4', 'SXM', NULL, NULL, 600),
    ('SXM5', 'SXM', NULL, NULL, 900),
    ('NVLink 2.0', 'NVLink', NULL, NULL, 300),
    ('NVLink 3.0', 'NVLink', NULL, NULL, 600),
    ('NVLink 4.0', 'NVLink', NULL, NULL, 900),
    ('Thunderbolt 3', 'Thunderbolt', NULL, NULL, 10),
    ('Thunderbolt 4', 'Thunderbolt', NULL, NULL, 10),
    ('USB4', 'USB', NULL, NULL, 10),
    ('OCuLink 2.0', 'OCuLink', 16, 4, 16.0);

-- ---------------------------------------------------------------------------
-- vram_type
-- Video memory technology lookup (#38). Seed data in seeds/vram_types.json.
-- ---------------------------------------------------------------------------
CREATE TABLE vram_type (
    vram_type_id SERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,   -- 'GDDR6X', 'HBM2e'
    description  TEXT
);

INSERT INTO vram_type (name, description) VALUES
    ('GDDR5', 'Graphics DDR5'),
    ('GDDR5X', 'Graphics DDR5X'),
    ('GDDR6', 'Graphics DDR6'),
    ('GDDR6X', 'Graphics DDR6X (PAM4)'),
    ('GDDR7', 'Graphics DDR7'),
    ('HBM2', 'High Bandwidth Memory 2'),
    ('HBM2e', 'High Bandwidth Memory 2e'),
    ('HBM3', 'High Bandwidth Memory 3'),
    ('HBM3e', 'High Bandwidth Memory 3e'),
    ('LPDDR5', 'Low-power DDR5, shared with system memory'),
    ('LPDDR5X', 'Low-power DDR5X, shared with system memory');

-- ---------------------------------------------------------------------------
-- gpu_architecture
-- GPU microarchitecture lookup (#38). Seed data in seeds/gpu_architectures.json.
-- ---------------------------------------------------------------------------
CREATE TABLE gpu_architecture (
    gpu_architecture_id SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,   -- 'Ada Lovelace', 'RDNA 3.5'
    vendor              TEXT NOT NULL           -- 'NVIDIA', 'AMD', 'Intel'
);

INSERT INTO gpu_architecture (name, vendor) VALUES
    ('Pascal', 'NVIDIA'),
    ('Volta', 'NVIDIA'),
    ('Turing', 'NVIDIA'),
    ('Ampere', 'NVIDIA'),
    ('Ada Lovelace', 'NVIDIA'),
    ('Hopper', 'NVIDIA'),
    ('Blackwell', 'NVIDIA'),
    ('RDNA 2', 'AMD'),
    ('RDNA 3', 'AMD'),
    ('RDNA 3.5', 'AMD'),
    ('RDNA 4', 'AMD'),
    ('CDNA 2', 'AMD'),
    ('CDNA 3', 'AMD'),
    ('Alchemist', 'Intel'),
    ('Battlemage', 'Intel');

-- ---------------------------------------------------------------------------
-- tflops_source
-- Where a peak-TFLOPS figure came from (#38). Seed data in seeds/tflops_sources.json.
-- ---------------------------------------------------------------------------
CREATE TABLE tflops_source (
    tflops_source_id SERIAL PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,   -- 'vendor_spec', 'lookup_table', 'measured'
    description      TEXT
);

INSERT INTO tflops_source (name, description) VALUES
    ('vendor_spec', 'Peak figure from the vendor''s published specification'),
    ('lookup_table', 'Peak figure from the harness''s built-in model lookup table (bench/metrics.py)'),
    ('measured', 'Peak figure measured by the harness');

-- ---------------------------------------------------------------------------
-- system_hardware
-- The fixed physical host (#19). system_id is a stable per-host UUID generated once
-- at first run; rows upsert on it.
-- ---------------------------------------------------------------------------
CREATE TABLE system_hardware (
    system_hardware_id SERIAL PRIMARY KEY,
    system_id          UUID NOT NULL UNIQUE,
    publisher_id       TEXT NOT NULL,   -- submitting user / account
    cpu_model          TEXT,
    ram_total_gb       FLOAT
);

-- ---------------------------------------------------------------------------
-- system_software
-- Runtime environment snapshot for one run (#19). Not deduplicated: one row per run.
-- ---------------------------------------------------------------------------
CREATE TABLE system_software (
    system_software_id SERIAL PRIMARY KEY,
    os                 TEXT,   -- 'Ubuntu 24.04.2 LTS'
    kernel_version     TEXT,   -- '6.8.0-57-generic'
    python_version     TEXT,   -- '3.12.3'
    gpu_driver_version TEXT,   -- nvidia-smi Driver Version field
    cuda_version       TEXT,   -- NULL for ROCm / CPU-only runners
    rocm_version       TEXT,   -- NULL for CUDA runners
    runner_version     TEXT    -- bakeoff harness commit hash or semver
);

-- ---------------------------------------------------------------------------
-- gpu_hardware
-- Die-level GPU intrinsics (#18, #38): a model-level record shared by every system with
-- that GPU, deduplicated on (pci_device_id, pci_subsystem_device_id) else normalised gpu_name.
-- memory_bandwidth_peak_gb_s is stored, not derived.
-- ---------------------------------------------------------------------------
CREATE TABLE gpu_hardware (
    gpu_hardware_id              SERIAL PRIMARY KEY,
    gpu_name                     TEXT NOT NULL,
    pci_vendor_id                TEXT,   -- '0x10de'
    pci_device_id                TEXT,   -- '0x2684'
    pci_subsystem_vendor_id      TEXT,   -- board partner
    pci_subsystem_device_id      TEXT,   -- board partner variant
    gpu_architecture_id          INT REFERENCES gpu_architecture,
    vram_total_mb                INT,
    vram_type_id                 INT REFERENCES vram_type,
    memory_bus_width_bits        INT,
    memory_bandwidth_peak_gb_s   FLOAT,
    clock_memory_mhz             INT,
    clock_graphics_boost_mhz     INT,
    peak_tflops_fp16             FLOAT,
    tflops_source_id             INT REFERENCES tflops_source,
    tdp_w                        INT,
    gpu_native_interface_type_id INT REFERENCES interface_type   -- the card's rated spec
);

CREATE UNIQUE INDEX gpu_hardware_pci_identity_idx
    ON gpu_hardware (pci_device_id, pci_subsystem_device_id)
    WHERE pci_device_id IS NOT NULL AND pci_subsystem_device_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- system_gpu_link
-- A slot of a host and the GPU in it (#20). The slot is the fixed property, so the GPU
-- is data, not key. is_slot_limited is not stored: it is
-- slot_native_interface_type_id <> actual_interface_type_id.
-- ---------------------------------------------------------------------------
CREATE TABLE system_gpu_link (
    system_hardware_id            INT NOT NULL REFERENCES system_hardware,
    slot_index                    INT NOT NULL,
    gpu_hardware_id               INT NOT NULL REFERENCES gpu_hardware,
    slot_native_interface_type_id INT REFERENCES interface_type,   -- the slot's rated maximum
    actual_interface_type_id      INT REFERENCES interface_type,   -- the negotiated running state
    PRIMARY KEY (system_hardware_id, slot_index)
);

CREATE INDEX system_gpu_link_gpu_hardware_idx ON system_gpu_link (gpu_hardware_id);

-- ---------------------------------------------------------------------------
-- run_hardware_metrics
-- Hardware context for a run (one row per run, #21). (system_hardware_id, slot_index)
-- names the system_gpu_link row active for the run, so the GPU comes from the link.
-- ---------------------------------------------------------------------------
CREATE TABLE run_hardware_metrics (
    run_id             UUID PRIMARY KEY REFERENCES runs,
    system_hardware_id INT,
    slot_index         INT,
    system_software_id INT REFERENCES system_software,
    peak_vram_mb       INT,
    power_limit_w      FLOAT,
    measured_tflops    FLOAT,
    CONSTRAINT fk_run_hardware_metrics_system_gpu_link
        FOREIGN KEY (system_hardware_id, slot_index)
        REFERENCES system_gpu_link (system_hardware_id, slot_index)
);

-- ---------------------------------------------------------------------------
-- schema_versions
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
-- schema_tables
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
-- schema_tables_join
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

-- ---------------------------------------------------------------------------
-- schema_versions seed
-- Version 1: agent_traces table, uplift_baseline_task_id on tasks,
-- trace_id on run_model_metrics, cyber_safety category seed.
-- allow_migration=false until migration runner is implemented.
-- ---------------------------------------------------------------------------
INSERT INTO schema_versions (description, allow_migration)
VALUES (
    'C4+C6 schema: agent_traces table, tasks.uplift_baseline_task_id, run_model_metrics.trace_id, cyber_safety category',
    FALSE
);
