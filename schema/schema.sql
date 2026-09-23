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
-- GPU host / interconnect link types. Admin-controlled lookup; seed data in
-- seeds/interface_types.json.
-- bandwidth_peak_gb_s is bidirectional and the only field comparable across
-- families. lane_transfer_rate (GT/s per lane) and lane_count are set only for
-- PCIe-lane links so gen and width degradation can be attributed separately.
-- Degraded links get no rows: the display layer composes the description from
-- the native and actual rows referenced by system_gpu_link.
-- NVLink rows are reserved; no table references them yet.
-- ---------------------------------------------------------------------------
CREATE TABLE interface_type (
    interface_type_id   SERIAL PRIMARY KEY,
    bandwidth_peak_gb_s FLOAT  NOT NULL,
    description         TEXT   NOT NULL UNIQUE,   -- 'PCIe 4.0 x16', 'SXM5', 'Thunderbolt 4'
    interface_family    TEXT,                     -- 'PCIe', 'SXM', 'NVLink', 'Thunderbolt', 'OCuLink'
    lane_transfer_rate  FLOAT,                    -- FLOAT because PCIe 1.0 is 2.5 GT/s
    lane_count          INT
);

INSERT INTO interface_type (description, bandwidth_peak_gb_s, interface_family, lane_transfer_rate, lane_count) VALUES
    ('PCIe 1.0 x1',       0.5, 'PCIe',         2.5,  1),
    ('PCIe 1.0 x4',       2.0, 'PCIe',         2.5,  4),
    ('PCIe 1.0 x8',       4.0, 'PCIe',         2.5,  8),
    ('PCIe 1.0 x16',      8.0, 'PCIe',         2.5,  16),
    ('PCIe 2.0 x1',       1.0, 'PCIe',         5,    1),
    ('PCIe 2.0 x4',       4.0, 'PCIe',         5,    4),
    ('PCIe 2.0 x8',       8.0, 'PCIe',         5,    8),
    ('PCIe 2.0 x16',     16.0, 'PCIe',         5,    16),
    ('PCIe 3.0 x1',      1.97, 'PCIe',         8,    1),
    ('PCIe 3.0 x4',      7.88, 'PCIe',         8,    4),
    ('PCIe 3.0 x8',     15.75, 'PCIe',         8,    8),
    ('PCIe 3.0 x16',    31.51, 'PCIe',         8,    16),
    ('PCIe 4.0 x1',      3.94, 'PCIe',         16,   1),
    ('PCIe 4.0 x4',     15.75, 'PCIe',         16,   4),
    ('PCIe 4.0 x8',     31.51, 'PCIe',         16,   8),
    ('PCIe 4.0 x16',    63.02, 'PCIe',         16,   16),
    ('PCIe 5.0 x1',      7.88, 'PCIe',         32,   1),
    ('PCIe 5.0 x4',     31.51, 'PCIe',         32,   4),
    ('PCIe 5.0 x8',     63.02, 'PCIe',         32,   8),
    ('PCIe 5.0 x16',   126.03, 'PCIe',         32,   16),
    ('SXM2',            300.0, 'SXM',          NULL, NULL),
    ('SXM4',            600.0, 'SXM',          NULL, NULL),
    ('SXM5',            900.0, 'SXM',          NULL, NULL),
    ('NVLink 2.0',      300.0, 'NVLink',       NULL, NULL),
    ('NVLink 3.0',      600.0, 'NVLink',       NULL, NULL),
    ('NVLink 4.0',      900.0, 'NVLink',       NULL, NULL),
    ('Thunderbolt 3',    10.0, 'Thunderbolt',  NULL, NULL),
    ('Thunderbolt 4',    10.0, 'Thunderbolt',  NULL, NULL),
    ('OCuLink 2.0',     15.75, 'OCuLink',      16,   4);

-- ---------------------------------------------------------------------------
-- gpu_architectures
-- GPU micro-architecture lookup, used to group results by generation.
-- Not an identity input: it is implied by the PCI vendor + device IDs.
-- Seed data in seeds/gpu_architectures.json.
-- ---------------------------------------------------------------------------
CREATE TABLE gpu_architectures (
    gpu_architecture_id SERIAL PRIMARY KEY,
    name                TEXT   NOT NULL UNIQUE,
    description         TEXT
);

INSERT INTO gpu_architectures (name, description) VALUES
    ('Pascal',       'NVIDIA, 2016'),
    ('Volta',        'NVIDIA, 2017'),
    ('Turing',       'NVIDIA, 2018'),
    ('Ampere',       'NVIDIA, 2020'),
    ('Ada Lovelace', 'NVIDIA, 2022'),
    ('Hopper',       'NVIDIA, 2022'),
    ('Blackwell',    'NVIDIA, 2024'),
    ('RDNA 2',       'AMD, 2020'),
    ('RDNA 3',       'AMD, 2022'),
    ('RDNA 3.5',     'AMD, 2024 (Strix Point / Strix Halo iGPU)'),
    ('RDNA 4',       'AMD, 2025'),
    ('CDNA 2',       'AMD, 2021'),
    ('CDNA 3',       'AMD, 2023'),
    ('Xe-HPG',       'Intel, 2022 (Alchemist)'),
    ('Xe2',          'Intel, 2024 (Battlemage / Lunar Lake)');

-- ---------------------------------------------------------------------------
-- vram_types
-- GPU memory technology lookup. Seed data in seeds/vram_types.json.
-- ---------------------------------------------------------------------------
CREATE TABLE vram_types (
    vram_type_id SERIAL PRIMARY KEY,
    name         TEXT   NOT NULL UNIQUE,
    description  TEXT
);

INSERT INTO vram_types (name, description) VALUES
    ('GDDR5',   'Graphics DDR5'),
    ('GDDR5X',  'Graphics DDR5X'),
    ('GDDR6',   'Graphics DDR6'),
    ('GDDR6X',  'Graphics DDR6X (PAM4 signalling)'),
    ('GDDR7',   'Graphics DDR7 (PAM3 signalling)'),
    ('HBM2',    'High Bandwidth Memory 2'),
    ('HBM2e',   'High Bandwidth Memory 2e'),
    ('HBM3',    'High Bandwidth Memory 3'),
    ('HBM3e',   'High Bandwidth Memory 3e'),
    ('DDR5',    'System DDR5 shared with an integrated GPU'),
    ('LPDDR5',  'System LPDDR5 shared with an integrated GPU'),
    ('LPDDR5X', 'System LPDDR5X shared with an integrated GPU');

-- ---------------------------------------------------------------------------
-- compute_formats
-- Numeric precision formats that TFLOPS figures are quoted for. A new format
-- is a seed row, not a new column. Seed data in seeds/compute_formats.json.
-- ---------------------------------------------------------------------------
CREATE TABLE compute_formats (
    compute_format_id SERIAL PRIMARY KEY,
    name              TEXT   NOT NULL UNIQUE,
    description       TEXT
);

INSERT INTO compute_formats (name, description) VALUES
    ('fp64', 'IEEE 754 double precision'),
    ('fp32', 'IEEE 754 single precision'),
    ('tf32', 'TensorFloat-32 (tensor core)'),
    ('fp16', 'IEEE 754 half precision'),
    ('bf16', 'Brain float 16'),
    ('fp8',  '8-bit float (E4M3 / E5M2)'),
    ('fp4',  '4-bit float'),
    ('int8', '8-bit integer'),
    ('int4', '4-bit integer');

-- ---------------------------------------------------------------------------
-- tflops_sources
-- Provenance for gpu_tflops values. Manufacturer figures are kept but marked
-- as such so measured values can later be told apart and preferred.
-- contacts: JSONB array of {"type": ..., "value": ...}.
-- url_template: static URL or {pci_vendor_id} / {pci_device_id} / {gpu_name}
-- token substitution. url_script: Go template + sprig (the language used by
-- schema_versions scripts); takes precedence over url_template when non-null.
-- Seed data in seeds/tflops_sources.json.
-- ---------------------------------------------------------------------------
CREATE TABLE tflops_sources (
    tflops_source_id SERIAL PRIMARY KEY,
    name             TEXT   NOT NULL UNIQUE,
    contacts         JSONB,
    url_template     TEXT,
    url_script       TEXT
);

INSERT INTO tflops_sources (name) VALUES
    ('unknown/unverified'),
    ('Rethunk measured');

-- ---------------------------------------------------------------------------
-- gpu_hardware
-- Die/board-level GPU intrinsics: one row per GPU model, shared by every
-- system that has one. Per-slot placement lives in system_gpu_link.
-- The UNIQUE key dedups rows that carry full PCI identity; rows without it
-- fall back to gpu_name matching in the writer.
-- memory_bandwidth_peak_gb_s is stored, not derived, because the derivation
-- needs a per-vram_type data-rate factor that every reader would repeat.
-- gpu_native_interface_type_id is the card's rated link, independent of slot.
-- ---------------------------------------------------------------------------
CREATE TABLE gpu_hardware (
    gpu_hardware_id              SERIAL PRIMARY KEY,
    gpu_name                     TEXT   NOT NULL,
    pci_vendor_id                TEXT,             -- '0x10de'
    pci_device_id                TEXT,             -- '0x2684'
    pci_subsystem_vendor_id      TEXT,             -- board partner
    pci_subsystem_device_id      TEXT,             -- board variant
    gpu_architecture_id          INT    REFERENCES gpu_architectures,
    vram_total_mb                INT,
    vram_type_id                 INT    REFERENCES vram_types,
    memory_bus_width_bits        INT,
    memory_bandwidth_peak_gb_s   FLOAT,
    clock_memory_mhz             INT,
    clock_graphics_boost_mhz     INT,
    tdp_w                        INT,
    gpu_native_interface_type_id INT    REFERENCES interface_type,
    UNIQUE (pci_vendor_id, pci_device_id, pci_subsystem_vendor_id, pci_subsystem_device_id)
);

-- ---------------------------------------------------------------------------
-- gpu_tflops
-- Peak throughput per GPU model and compute format, with provenance.
-- ---------------------------------------------------------------------------
CREATE TABLE gpu_tflops (
    gpu_hardware_id   INT   NOT NULL REFERENCES gpu_hardware,
    compute_format_id INT   NOT NULL REFERENCES compute_formats,
    tflops_value      FLOAT NOT NULL,
    tflops_source_id  INT   NOT NULL REFERENCES tflops_sources,
    PRIMARY KEY (gpu_hardware_id, compute_format_id)
);

-- ---------------------------------------------------------------------------
-- system_hardware
-- Fixed physical host. system_id is a stable per-host UUID generated on first
-- run and persisted locally, so one machine is one row across runs and
-- publishers. cpu_threads is kept (SMT can be toggled in firmware); core count
-- is implied by cpu_model. Memory clock/channels/profile are the active
-- settings, not SPD ratings.
-- bios_notes: JSONB key/value firmware settings, e.g. {"bar_size_mb": 16384}.
-- ---------------------------------------------------------------------------
CREATE TABLE system_hardware (
    system_hardware_id        SERIAL PRIMARY KEY,
    system_id                 UUID   NOT NULL UNIQUE,
    publisher_id              TEXT   NOT NULL,
    cpu_model                 TEXT,
    cpu_threads               INT,
    cpu_base_clock_mhz        INT,
    cpu_peak_clock_mhz        INT,
    ram_total_gb              FLOAT,
    motherboard               TEXT,
    memory_speed_mhz          INT,
    memory_channels           INT,
    memory_interleave_profile TEXT,             -- 'XMP', 'EXPO', 'DOCP', 'manual'
    bios_notes                JSONB
);

-- ---------------------------------------------------------------------------
-- system_software
-- Software environment snapshot; one new row per run, never deduplicated.
-- cuda_version / rocm_version are null on the other vendor's stack.
-- ---------------------------------------------------------------------------
CREATE TABLE system_software (
    system_software_id SERIAL PRIMARY KEY,
    os                 TEXT,
    kernel_version     TEXT,
    python_version     TEXT,
    gpu_driver_version TEXT,
    cuda_version       TEXT,
    rocm_version       TEXT,
    runner_version     TEXT
);

-- ---------------------------------------------------------------------------
-- system_gpu_link
-- Which GPU model sits in which slot of a host. The slot is the identity; the
-- GPU in it is data, so two identical GPUs in one host are two rows.
-- Slot limitation is derived (slot_native_interface_type_id <>
-- actual_interface_type_id), not stored. "actual" is the negotiated link that
-- both slot and GPU agreed on.
-- ---------------------------------------------------------------------------
CREATE TABLE system_gpu_link (
    system_hardware_id            INT NOT NULL REFERENCES system_hardware,
    slot_index                    INT NOT NULL,
    gpu_hardware_id               INT NOT NULL REFERENCES gpu_hardware,
    slot_native_interface_type_id INT REFERENCES interface_type,
    actual_interface_type_id      INT REFERENCES interface_type,
    PRIMARY KEY (system_hardware_id, slot_index)
);

CREATE INDEX system_gpu_link_gpu_hardware_idx ON system_gpu_link (gpu_hardware_id);

-- ---------------------------------------------------------------------------
-- run_hardware_metrics
-- Hardware context for a run (one row per run). Hardware identity goes
-- through system_gpu_link so the run records which GPU in which slot of which
-- host it used.
-- ---------------------------------------------------------------------------
CREATE TABLE run_hardware_metrics (
    run_id             UUID PRIMARY KEY REFERENCES runs,
    system_hardware_id INT,
    slot_index         INT,
    system_software_id INT REFERENCES system_software,
    peak_vram_mb       INT,
    power_limit_w      FLOAT,
    measured_tflops    FLOAT,
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
