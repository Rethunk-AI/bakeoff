# Summary: run_model_metrics

## Final state

Schema as settled at the close of #12 (per 4503669210). Tables below show column names, SQL types, and PK/FK as agreed.

**`runs`**
| Column | Type | Notes |
|---|---|---|
| `run_id` | UUID | PK — client-generated |
| `submitted_at` | TIMESTAMPTZ | NOT NULL |
| `publisher_id` | TEXT | NOT NULL |
| `runner_version` | TEXT | NULLABLE |
| `prompt_git_hash` | TEXT | NULLABLE — commit hash of prompt set at run time |

**`run_model_metrics`**
| Column | Type | Notes |
|---|---|---|
| `run_id` | UUID | FK → runs, PK component |
| `prompt_id` | INT | FK → prompts, PK component |
| `model_id` | INT | FK → models, PK component |
| `score` | FLOAT | NULLABLE |
| `pass_fail` | BOOLEAN | NULLABLE |
| `failure_reason` | TEXT | NULLABLE — "timeout", "oom", "parse_error"; null = success |
| `gflops_per_token` | FLOAT | NULLABLE |
PK: `(run_id, prompt_id, model_id)`

**`models`**
| Column | Type | Notes |
|---|---|---|
| `model_id` | SERIAL | PK |
| `name` | TEXT | NOT NULL |
| `creator_id` | INT | FK → creators |
| `model_hash` | TEXT | UNIQUE |
| `parameter_count_b` | FLOAT | NULLABLE — billions |
| `active_parameter_count_b` | FLOAT | NULLABLE — MoE only |
| `architecture` | TEXT | NULLABLE |
| `context_length_default` | INT | NULLABLE |
| `context_length_min` | INT | NULLABLE |
| `context_length_max` | INT | NULLABLE |
| `file_format` | TEXT | NULLABLE — "GGUF", "SafeTensors" |
| `quantization` | TEXT | NULLABLE |
| `min_vram_mb` | INT | NULLABLE |
| `release_date` | DATE | NULLABLE |
| `version` | TEXT | NULLABLE |
| `description` | TEXT | NULLABLE |
| `predecessor_model_id` | INT | FK → models, self-referential |

**`creators`**
| Column | Type | Notes |
|---|---|---|
| `creator_id` | SERIAL | PK |
| `name` | TEXT | NOT NULL UNIQUE |
| `display_name` | TEXT | NULLABLE |
| `homepage` | TEXT | NULLABLE |
| `service_identifiers` | JSONB | NULLABLE — {"huggingface": "...", "ollama": "..."} |

**`source_types`**
| Column | Type | Notes |
|---|---|---|
| `source_type_id` | SERIAL | PK |
| `name` | TEXT | NOT NULL UNIQUE — "huggingface", "ollama", "direct_url", "local_file" |

**`model_sources`**
| Column | Type | Notes |
|---|---|---|
| `source_id` | SERIAL | PK |
| `model_id` | INT | FK → models |
| `source_type_id` | INT | FK → source_types |
| `url` | TEXT | NOT NULL |
| `source_metadata` | JSONB | NULLABLE — HF commit hash, ollama tag, etc. |

**`task_categories`**
| Column | Type | Notes |
|---|---|---|
| `category_id` | SERIAL | PK |
| `name` | TEXT | NOT NULL UNIQUE |
| `description` | TEXT | NULLABLE |

**`tasks`**
| Column | Type | Notes |
|---|---|---|
| `task_id` | SERIAL | PK |
| `name` | TEXT | NOT NULL |
| `category_id` | INT | FK → task_categories, NULLABLE |
| `parent_id` | INT | FK → tasks, NULLABLE — null = top-level suite |
| `sort_order` | INT | NOT NULL DEFAULT 0 |
| `description` | TEXT | NULLABLE |
| `grader_script` | TEXT | NULLABLE |
| `grader_script_commit` | TEXT | NULLABLE — git SHA of grader last modification |
| `natural_key_hash` | TEXT | NOT NULL UNIQUE — SHA256 of canonical path; stable FK anchor |

**`prompts`**
| Column | Type | Notes |
|---|---|---|
| `prompt_id` | SERIAL | PK |
| `task_id` | INT | FK → tasks, NOT NULL |
| `file_path` | TEXT | NOT NULL |
| `git_commit_hash` | TEXT | NOT NULL |
| `content_sha256` | TEXT | UNIQUE — alias hashes = SHA256(path + original_hash) |
| `content_length_bytes` | INT | NULLABLE |
| `version` | TEXT | NULLABLE |
| `release_date` | DATE | NULLABLE |
| `is_prerelease` | BOOLEAN | NOT NULL DEFAULT FALSE |
| `difficulty` | INT | NOT NULL DEFAULT 0 — 0=basic; no `_level` suffix |
| `modified_at` | TIMESTAMPTZ | NULLABLE — from git last-modifying commit |

## Notable / unusual decisions

- **`task_id` dropped from `run_model_metrics`** (per 4494171918) — task is derivable via `run → prompt → task`; denormalization deferred until query profiling justifies it. Unusual choice that trades join cost for normalization.

- **UUID natural key for `run_id`** (per 4494171918) — client-side generation, prevents enumeration, no DB sequence dependency. Explicitly chosen over serial integer; UUID is both PK and natural key with no surrogate needed.

- **`cost_usd` excluded from storage** (per 4450406819, confirmed 4462460688) — treated as a derived display value computed from hardware energy specs × duration or provider pricing × token counts; never persisted. Avoids stale derived data in the table.

- **`task_id` stable via `natural_key_hash`** (per 4497170538) — SERIAL IDs on tasks are regeneration-unsafe; idempotent hash of canonical path used instead. FK references stay stable across re-scans, which is atypical for a SERIAL-primary-key design.

- **Duplicate prompt YAML indirection** (per 4503203398) — intentionally duplicated prompts use a YAML stub (`type: alias`) rather than a UNIQUE exemption. Alias `content_sha256` = SHA256(relative_path + original_hash), preserving the UNIQUE constraint with no special cases.

- **Filesystem-first for tasks and prompts** (per 4495675210, 4495777337) — DB is a derived cache populated by a scanner; git/filesystem is source of truth. Task hierarchy encoded in directory structure with `task.yaml` per level; scanner derives `parent_id` and `sort_order`. This inverts the typical DB-first design pattern.

- **`difficulty` as integer, prompt-scoped** (per 4496938804, 4497170538, 4501587412) — 0-indexed, stored on `prompts` not `tasks`; allows level-skipping gate logic in the runner without schema changes. Relative difficulty expressions (`parent+1`, `task:path+1`) resolve at scan time to integer; DB stores the resolved value only.

- **`gflops_per_token` over `flops_per_token_theoretical`** (per 4497170538, 4501587412, 4503203398) — renamed twice: first to `compute_per_token_gflops`, then to `gflops_per_token`; BIGINT replaced by FLOAT in GFLOPs. "Theoretical" dropped in favor of table-context clarity.

- **`failure_reason TEXT` as hardware-neutral failure gate** (per 4494171918) — OOM, timeout, and parse errors land in `failure_reason`, not in `score`/`pass_fail`. Model scoring is capability-only; hardware-caused failures are not scored against the model.

- **Score displayed as percentage, stored as float 0–1** (per 4495675210) — UI responsibility to render 0–100%; storage normalized. Decouples display convention from schema.

- **Three deduplication outcomes for `model_hash` conflicts** (per 4495675210, 4497170538) — (1) accept merge/coalesce; (2) reject merge, create new record with note; (3) reject entirely with note citing existing record. Option 3 detail deferred. No silent auto-merge in any path.

## Open / unresolved

- **Task/prompt cardinality within a single run** — it is settled that one task has many prompts, but the thread does not resolve whether a single run dispatches all prompts for a task or selects a subset, nor whether the runner skips remaining prompts in a task if an earlier one fails. The dependency/difficulty-gate logic (`difficulty_gate` in `task.yaml`) was proposed but deferred to the testing-queue issue rather than finalized here.
  - address: #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4496938804>

- **Prompt dependency relative references resolution pass** — the two-pass scan approach (build difficulty map, then resolve relative expressions like `parent+1`) was proposed but explicitly flagged as scanner complexity to be specced out, not settled here.
  - address: #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4496938804>

- **`task.yaml` alias bulk-declaration syntax** — the `aliases:` block in `task.yaml` was proposed (per 4503203398) as an alternative to per-prompt YAML stubs, but no schema or syntax was finalized; left as stated intent.
  - address: #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503203398>

- **Testing-queue issue** — opening was stated as "now" in 4495777337, confirmed as pending direction in 4503669210; no ticket number or link appears in this thread.
  - address: #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503669210>

- **Implementation sub-issues per table/component** — explicitly listed as a pending forward action awaiting direction at thread close (per 4503669210); no sub-issues opened within this thread.
  - address: #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503669210>

- **`model_hash` NULLABLE vs NOT NULL** — the thread says UNIQUE but does not resolve whether `model_hash` should be required. Early entries treat it as a known-good deduplication anchor; gissf1's comment implies it should be authoritative (per 4495675210), but the final schema snapshot shows it as UNIQUE without a NOT NULL constraint. No explicit resolution.
  - address: #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4495675210>

## Cross-topic links

- **`run_hardware_metrics`** — shares `run_id UUID FK → runs`; the `runs` table is the join anchor for all per-run data across both metric tables. Schema for `run_hardware_metrics` and its supporting tables (`system_hardware`, `system_gpu_link`, `system_software`, `interface_type`, `gpu_hardware`) was settled in #8 and is covered under the `run_hardware_metrics` topic.
- **`hardware_specs` / `system_hardware`** — `run_model_metrics` has no direct FK to hardware tables; the hardware context is reached via `run_id → runs` then to `run_hardware_metrics`. Separation was an explicit design goal.
- **Testing queue** — `prompts.difficulty` and the runner's level-gate logic are load-bearing inputs to the testing-queue scheduling design; topic dependency flagged in 4491873240 and 4495777337.
