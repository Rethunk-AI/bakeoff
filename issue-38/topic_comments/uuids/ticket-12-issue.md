---
ticket: 12
type: issue-body
author: AlbinoGeek
posted: 2026-05-18T19:00:32Z
topic: uuids
url: https://github.com/Rethunk-AI/bakeoff/issues/12
title: run_model_metrics schema: models, tasks, prompts, and scoring
---

**`run_model_metrics` schema — models, tasks, prompts, and scoring**

Opened per discussion in #8 (hardware/performance metrics). This issue covers the model-side schema: how models, tasks, and prompts are represented and how `run_model_metrics` references them.

---

## Current target schema (placeholder — FKs not yet resolved)

```
run_model_metrics
──────────────────────────────────────────────────────────
run_id                          TEXT NOT NULL PRIMARY KEY
model_id                        TEXT NOT NULL      -- FK to models (TBD)
task_id                         TEXT NOT NULL      -- FK to tasks (TBD)
prompt_id                       TEXT NOT NULL      -- FK to prompts (TBD)
score                           FLOAT NULLABLE
pass_fail                       BOOLEAN NULLABLE
flops_per_token_theoretical     BIGINT NULLABLE
```

`model_id`, `task_id`, `prompt_id` are currently TEXT placeholders. This issue resolves them into proper entity tables with FK references.

---

## Open questions for discussion

**Q1 — Task / prompt relationship**

Working assumption: a *task* is an evaluation scenario type (e.g., "code generation", "multi-step reasoning", "summarization"). A *prompt* is a specific text formulation of that task. One task may have many prompt variants. Is this correct?

**Q2 — Prompt data storage: database or file?**

Two options:

- **In database:** `prompts` table stores prompt text. Submitted results reference `prompt_id` FK — leaderboard can verify submitted results against the exact input used. Enables full reproducibility audit.
- **In files:** prompts stored in versioned benchmark config files. `prompt_id` is a content hash or file path reference. Simpler runner integration; requires tracking file version alongside result.

Which path is preferred? This is load-bearing for leaderboard result verification.

**Q3 — Model identity**

What constitutes a distinct `model_id`? Options:
- Provider + model name + version string (e.g., `openai/gpt-4o-2024-11-20`)
- A structured `models` table with separate fields for provider, model_family, version, parameter_count, quantization, context_length, etc.

A structured `models` table enables hardware-normalized comparisons across quantization variants of the same base model. Is that level of detail needed in Phase 1?

**Q4 — Scoring schema**

`score FLOAT` and `pass_fail BOOLEAN` are both present. Are these always computed from the same test, or do some tasks produce only a score (no binary), and others only pass/fail (no continuous score)? Should these be separated into different record types, or is nullable acceptable?

**Q5 — Link to run_hardware_metrics**

`run_hardware_metrics` and `run_model_metrics` both reference `run_id`. Is `run_id` a natural key (e.g., UUID generated per invocation), or should there be an explicit `runs` table that both tables FK into? A `runs` table would carry: timestamp, submitter, runner_version, and serve as the join anchor for all per-run data.

---

## Proposed starting point (pending Q1–Q5 answers)

```
models
──────────────────────────────────────────────────────────
model_id            SERIAL PRIMARY KEY
provider            TEXT NOT NULL           -- "openai", "anthropic", "local"
model_name          TEXT NOT NULL           -- "gpt-4o"
model_version       TEXT NULLABLE           -- "2024-11-20"
parameter_count     BIGINT NULLABLE
quantization        TEXT NULLABLE           -- "fp16", "int4", "gguf-q8_0"
context_length      INT NULLABLE
```

```
tasks
──────────────────────────────────────────────────────────
task_id             SERIAL PRIMARY KEY
task_name           TEXT NOT NULL
task_description    TEXT NULLABLE
task_category       TEXT NULLABLE           -- "code_generation", "reasoning", etc.
```

```
prompts
──────────────────────────────────────────────────────────
prompt_id           SERIAL PRIMARY KEY
task_id             INT NOT NULL REFERENCES tasks
prompt_text         TEXT NOT NULL           -- if stored in DB
prompt_hash         TEXT NOT NULL           -- SHA256 of prompt_text; used for dedup + file-ref fallback
```

```
run_model_metrics
──────────────────────────────────────────────────────────
run_id              TEXT NOT NULL PRIMARY KEY
model_id            INT NOT NULL REFERENCES models
task_id             INT NOT NULL REFERENCES tasks
prompt_id           INT NOT NULL REFERENCES prompts
score               FLOAT NULLABLE
pass_fail           BOOLEAN NULLABLE
flops_per_token_theoretical  BIGINT NULLABLE
```

---

@gissf1 — tagged for review. Provide direction on Q1–Q5 and we can finalize the schema here before opening implementation sub-issues.

— Bastion
