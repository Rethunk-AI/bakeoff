# Score Incomplete Runs And Minimal-Capability Test Tier

Status: Active

Refs: Rethunk-AI/bakeoff#23 (harness), Rethunk-AI/bakeoff-results#9 (origin), Rethunk-AI/bakeoff-results#24 (display side)

## Problem

Models that do not complete the full benchmark suite produce no meaningful output
today. Their result records carry a free-text `error` string and a `null` score;
the bundle has no run-level status field; and the manifest carries no aggregate
signal the results site can use to badge or rank those runs. The site therefore
has nothing to display for weak or failing models, making the bakeoff a pass/fail
gate rather than an educational reference.

Three concrete gaps to close:

1. **Failure-reason capture.** The `error` field in records is unstructured; the
   SQL `failure_reason` column in `run_model_metrics` and the `error_detail`
   column in `run_queue` are free text. There is no taxonomy, so the site cannot
   group or filter by failure mode.

2. **Relative scoring.** Incomplete cells have `score: null`. A model that
   attempted 6 of 10 tasks ranks the same (invisible) as one that refused every
   prompt.

3. **Minimal-capability floor.** There is no dedicated "dumb model" suite. A
   model that can do basic arithmetic and follow one-word instructions might still
   be useful; the current suite does not expose that.

## Goals

- Define a controlled failure-reason taxonomy that replaces the free-text `error`
  field in result records, maps to the SQL `failure_reason` column, and is
  preserved in result bundles.
- Define a completeness-weighted partial-score formula that produces a numeric
  rank for every model, including those that fail all cells.
- Specify a minimal-capability test tier (`dumb_model`) — a fixed task list, its
  scorers, and its reporting path — that runs independently of the main suite and
  is always included in the bundle.
- Define the exact `result.json` and `manifest.json` additions so
  `bakeoff-results` can render failure reasons, partial scores, and an
  `incomplete`/`failed` state badge without parsing the full result payload for
  index rendering.

## Non-Goals

- This spec does not implement the changes. No harness code is modified here.
- It does not change the judge subsystem or pairwise evaluation logic.
- It does not add retries, resume logic, or queue management (covered in
  `benchmark-resume-partial-rerun`).
- It does not define the display-side rendering (Rethunk-AI/bakeoff-results#24
  owns that); it only defines the schema contract those renderers depend on.
- It does not modify `SCHEMA_VERSION` to `bakeoff-results/v2`; all additions are
  additive and optional, preserving backward compatibility with existing bundles.

## Design

### 1. Failure-Reason Taxonomy

Replace the unstructured `error` string in result records with a pair of fields:
`failure_code` (controlled enum string) and `failure_detail` (optional free
text). The `error` field is **retained for backward compatibility** but its value
is duplicated into `failure_detail` whenever a code is known. New writers must
populate `failure_code`; old readers that only inspect `error` continue to work.

#### Taxonomy enum (`failure_code`)

| Code | Meaning |
|------|---------|
| `timeout` | Model did not respond within the configured `timeout_s`. |
| `refusal` | Model responded but explicitly declined to answer (safety, topic rejection, etc.). The response text is non-empty but contains a refusal marker. |
| `malformed_output` | Response was received but could not be parsed or scored (e.g., expected JSON but got prose; expected one-word answer but got a paragraph). |
| `oom` | Out-of-memory signal observed during inference (VRAM or RAM exhaustion; loader error containing OOM markers). |
| `load_failure` | Model could not be loaded or swapped in at all (binary crash, missing file, incompatible quantization). |
| `capability_gap` | Heuristic scorer returned 0.0 AND judge (when available) returned 1/5 or "wrong"; aggregated across ≥ 50% of that model's cells, indicating systematic inability rather than single-cell failure. Applied at post-processing time, not per-call. |
| `infra_error` | Runner-side infrastructure failure unrelated to the model (proxy crash, network error to server, runner OOM from OS). |
| `cancelled` | Cell was explicitly cancelled (e.g., operator interrupt, `run_queue.status = CANCELLED`). |
| `unknown` | Exception was caught but does not match any above pattern. Use as a last resort; the full exception string goes into `failure_detail`. |

Detection rules for each code are heuristic (pattern-matching on exception type,
message, and HTTP status from the proxy). The implementation must document its
detection regexes in `bench/metrics.py` or a new `bench/failure.py`. The spec
only mandates the taxonomy.

#### Record-level schema change

Existing record (already emitted by `runner.py`):

```json
{
  "task_id": "t0000",
  "prompt_id": "p0",
  "model_id": "qwen3-8b-q4_k_m",
  "error": "httpx.ReadTimeout: timed out after 120s"
}
```

Extended record (new fields additive; `error` preserved):

```json
{
  "task_id": "t0000",
  "prompt_id": "p0",
  "model_id": "qwen3-8b-q4_k_m",
  "error": "httpx.ReadTimeout: timed out after 120s",
  "failure_code": "timeout",
  "failure_detail": "httpx.ReadTimeout: timed out after 120s"
}
```

Successful records carry `"failure_code": null, "failure_detail": null` (or omit
the keys entirely — readers must treat absent as null).

The SQL `run_model_metrics.failure_reason TEXT` column is updated to hold the
`failure_code` enum value (not the full detail). A separate
`run_model_metrics.failure_detail TEXT` column is added. The
`run_queue.error_detail` column is unchanged (it remains a free-text exception
log).

### 2. Partial Score Formula

#### Motivation

A completeness-weighted score penalises models that bail early. A model that
attempted 6 of 10 tasks and scored 0.8 on those 6 should rank below a model
that attempted all 10 and scored 0.6, because the former model's effective
capability across the full matrix is unknown and possibly zero on the skipped
tasks.

#### Definitions

For a given model `m` across a run with `C` total cells (task × prompt
combinations):

- `C` = total cells in the matrix (constant per run; known before any model is
  evaluated).
- `A(m)` = number of cells attempted by model `m` (i.e., a response was
  received, even if it scored 0).
- `S(m)` = sum of per-cell scores for model `m` over attempted cells. Each cell
  score is the `quality_heuristic` value (float in [0, 1]) when available, else
  the judge score normalized to [0, 1] via `(judge_score - 1) / 4.0` (for the
  1–5 rubric), else 0.0 for cells where the error is a hard failure.
- `completeness(m)` = `A(m) / C` (float in [0, 1]).

**Partial score formula:**

```
partial_score(m) = (S(m) / C)
```

This is equivalent to treating every unattempted cell as scoring 0.0, which is
the completeness-weighted formulation. Dividing by `C` (not `A(m)`) is the key
decision: it ensures a model that completed half the matrix and scored perfectly
on those cells (partial_score = 0.5) ranks below a model that completed the full
matrix with a 0.6 average (partial_score = 0.6).

**Worked example:**

| Model | C | A(m) | S(m) | partial_score | completeness |
|-------|---|------|------|---------------|--------------|
| strong-model | 10 | 10 | 8.2 | 0.82 | 1.00 |
| middling-model | 10 | 10 | 5.5 | 0.55 | 1.00 |
| partial-model | 10 | 6 | 5.4 | 0.54 | 0.60 |
| weak-model | 10 | 4 | 2.0 | 0.20 | 0.40 |
| failing-model | 10 | 0 | 0.0 | 0.00 | 0.00 |

`partial-model` (attempted 6, scored 0.54 total) ranks below `middling-model`
(completed all 10, scored 0.55 total) despite having a higher per-attempted-cell
average (0.90 vs 0.55). This is intentional: the harness cannot know how the
partial model would have scored on the remaining 4 cells, and absent data is
treated as failure.

#### Run-level status

Each model within a run is assigned one of three statuses based on completeness:

| Status | Condition |
|--------|-----------|
| `complete` | `completeness(m) == 1.0` (all cells attempted and no hard load failure). |
| `incomplete` | `0.0 < completeness(m) < 1.0` (at least one cell attempted, at least one missed). |
| `failed` | `completeness(m) == 0.0` (no cells completed; load failure or all errors). |

The **run-level status** is the worst status across all models:
- Any model `failed` → run status `failed`.
- Else any model `incomplete` → run status `incomplete`.
- Else run status `complete`.

#### Per-model rollup in `result.json`

A new top-level `model_scores` list is added to the result payload. Each entry
is a per-model aggregate computed by the runner post-hoc (after all records are
collected). Existing `records` are not modified beyond the `failure_code` field.

```json
{
  "model_scores": [
    {
      "model_id": "strong-model",
      "status": "complete",
      "cells_total": 10,
      "cells_attempted": 10,
      "cells_failed": 0,
      "completeness": 1.0,
      "partial_score": 0.82,
      "dominant_failure_code": null
    },
    {
      "model_id": "partial-model",
      "status": "incomplete",
      "cells_total": 10,
      "cells_attempted": 6,
      "cells_failed": 2,
      "completeness": 0.60,
      "partial_score": 0.54,
      "dominant_failure_code": "timeout"
    },
    {
      "model_id": "failing-model",
      "status": "failed",
      "cells_total": 10,
      "cells_attempted": 0,
      "cells_failed": 10,
      "completeness": 0.0,
      "partial_score": 0.0,
      "dominant_failure_code": "load_failure"
    }
  ]
}
```

`dominant_failure_code` is the most-frequent `failure_code` among that model's
failed cells (`null` when no cells failed). It gives the site a single badge
reason without requiring it to scan the full `records` array.

#### Run-level status in `result.json`

A `run_status` field is added to the top-level payload alongside `run_id` and
`timestamp`. Enum values: `"complete"`, `"incomplete"`, `"failed"`. Absent in
old bundles; readers must treat absent as `"complete"` for backward
compatibility.

```json
{
  "run_id": "amd-8060s-2026-05-28T12:00:00Z",
  "timestamp": "2026-05-28T12:00:00Z",
  "run_status": "incomplete",
  ...
}
```

### 3. Minimal-Capability ("Dumb Model") Test Tier

#### Rationale

The main task suite uses `judge` scorers for code and summarization, which
require a coherent response the judge can evaluate. A model too weak to pass the
main suite may still respond coherently to trivial tasks. The floor tier uses
only deterministic scorers (`exact`, `contains`) so it can score models that
produce the judge target model (and so avoids circular dependency) and models
that the judge would rate as 1/5 anyway.

The tier runs as a separate phase **before** the main matrix, so a model that
crashes the loader during the main phase still has a floor score if it booted at
all. If the loader fails for all models (run status `failed`), the tier yields
all-zero floor scores with `failure_code: load_failure`.

#### Tier identifier

Tasks in this tier carry `"tier": "dumb_model"`. The main suite carries
`"tier": "main"` (absent in old tasks; readers treat absent as `"main"`). The
`task_categories` table gains a row:

```sql
INSERT INTO task_categories (name, description) VALUES
    ('dumb_model', 'Minimal-capability floor suite: basic arithmetic, short summarization, instruction-following. Deterministic scorers only; no judge required.');
```

#### Task list (fixed, versioned)

The floor suite is a fixed set of 12 tasks, version-pinned by `natural_key_hash`
(same mechanism as main tasks). They are **not** seeded from `dataset.generate()`
because their prompts must not change between runs (reproducibility requirement).
They live in a new file `datasets/dumb_model_tasks.jsonl` committed to the repo.

| # | Domain | Prompt | Expected | Scorer |
|---|--------|--------|----------|--------|
| 1 | `arithmetic` | `What is 2 + 2? Answer with one number.` | `4` | `exact` |
| 2 | `arithmetic` | `What is 10 - 3? Answer with one number.` | `7` | `exact` |
| 3 | `arithmetic` | `What is 6 × 7? Answer with one number.` | `42` | `exact` |
| 4 | `arithmetic` | `What is 100 ÷ 4? Answer with one number.` | `25` | `exact` |
| 5 | `instruction` | `Reply with exactly one word: the color of the sky. One word only.` | `blue` | `exact` |
| 6 | `instruction` | `Count to 3 and list only the numbers, separated by commas.` | `1, 2, 3` | `contains` |
| 7 | `instruction` | `Reply YES if you can understand this sentence, NO otherwise.` | `YES` | `exact` |
| 8 | `instruction` | `What is the opposite of hot? Answer with one word.` | `cold` | `exact` |
| 9 | `summarize` | `Summarize in 5 words or fewer: The cat sat on the mat.` | `cat sat on mat` | `contains` |
| 10 | `summarize` | `In one word, what animal says "woof"?` | `dog` | `exact` |
| 11 | `qa` | `What is the capital of France? Answer with one word.` | `Paris` | `exact` |
| 12 | `qa` | `What color do you get when you mix red and blue? Answer with one word.` | `purple` | `contains` |

Notes:
- Tasks 5 and 7 test instruction-following (single-word / yes-no constraint
  compliance). The scorer is `exact` after case-folding and stripping whitespace.
- Tasks 9 and 12 use `contains` because phrasing variation is acceptable
  (e.g., "The cat sat" contains "cat sat on").
- All 12 use a single prompt variant (no prompt rotation) to keep the tier cheap
  and reproducible.
- Expected values are case-folded before comparison by the `exact` scorer
  (existing behaviour in `metrics.score_heuristic`).

#### Scoring the floor tier

Each of the 12 cells scores 0.0 or 1.0 (binary, no partial credit within a
cell). The floor score for a model is:

```
floor_score(m) = (number of dumb_model cells scored 1.0) / 12
```

Floor score is **not** blended into `partial_score`. They are separate fields
because they measure different things: `partial_score` measures relative
performance on the main suite; `floor_score` measures minimal capability. A
model that passes all 12 floor tasks but fails the main suite entirely would have
`partial_score: 0.0, floor_score: 1.0`.

Floor records are included in the `records` array alongside main-suite records,
identified by `"tier": "dumb_model"` on each record.

#### Floor score in `model_scores`

The `model_scores` entry gains two fields:

```json
{
  "model_id": "failing-model",
  "status": "failed",
  "partial_score": 0.0,
  "floor_score": 0.33,
  "floor_cells_passed": 4,
  "floor_cells_total": 12
}
```

`floor_score` is `null` when the floor tier was not run (e.g., operator
explicitly disabled it via a new config flag `dumb_model_tier.enabled: false`).
The default is enabled.

### 4. Bundle and Manifest Schema Additions

#### `result.json` — new and changed fields

All new fields are **additive and optional**. `validate_result_payload()` in
`bench/publish.py` does not make them required for `bakeoff-results/v1`
compatibility; a new validation level (`--strict`) may flag their absence as a
warning (not an error).

| Field path | Type | Description |
|------------|------|-------------|
| `run_status` | `"complete" \| "incomplete" \| "failed"` | Worst model status across the run. New top-level field. |
| `records[*].failure_code` | `string \| null` | Controlled taxonomy code (see §1). Null on success. |
| `records[*].failure_detail` | `string \| null` | Optional free-text detail (mirrors legacy `error`). |
| `records[*].tier` | `"main" \| "dumb_model"` | Which task suite this record belongs to. Absent = `"main"`. |
| `model_scores` | `array` | Per-model aggregate list (see §2 + §3). New top-level field. |
| `model_scores[*].model_id` | `string` | Model identifier matching `config.models[*].id`. |
| `model_scores[*].status` | `"complete" \| "incomplete" \| "failed"` | Per-model completeness status. |
| `model_scores[*].cells_total` | `integer` | Total main-suite cells for this model. |
| `model_scores[*].cells_attempted` | `integer` | Cells where a response was received (even if score 0). |
| `model_scores[*].cells_failed` | `integer` | Cells with a non-null `failure_code`. |
| `model_scores[*].completeness` | `float [0,1]` | `cells_attempted / cells_total`. |
| `model_scores[*].partial_score` | `float [0,1]` | Completeness-weighted score (see §2 formula). |
| `model_scores[*].dominant_failure_code` | `string \| null` | Most frequent `failure_code` among failed cells. |
| `model_scores[*].floor_score` | `float [0,1] \| null` | Floor tier score (passed cells / 12). Null if tier not run. |
| `model_scores[*].floor_cells_passed` | `integer \| null` | Number of dumb_model cells that scored 1.0. |
| `model_scores[*].floor_cells_total` | `integer \| null` | Always 12 when floor tier ran; null otherwise. |

#### `manifest.json` — new fields

The manifest is the lightweight index entry the results site reads without
parsing the full `result.json`. It needs enough data to render the badge and
rank in the index page.

| Field | Type | Description |
|-------|------|-------------|
| `run_status` | `"complete" \| "incomplete" \| "failed" \| null` | Mirrors `result.json` top-level field. Null in old manifests. |
| `model_scores_summary` | `array \| null` | Compact per-model summary for the index. Null in old manifests. |

`model_scores_summary` entry shape (minimal, for fast index rendering):

```json
{
  "model_id": "partial-model",
  "status": "incomplete",
  "partial_score": 0.54,
  "floor_score": 0.67,
  "dominant_failure_code": "timeout"
}
```

The full `model_scores` array (with cell counts and completeness) stays in
`result.json`. The summary is a projection of only the fields needed for ranking
and badging.

`_build_manifest()` in `bench/publish.py` must be updated to populate these two
fields from the result payload. The `validate_bundle()` function should accept
both old manifests (fields absent → treated as null) and new ones.

#### SQL additions

Two changes to `schema/schema.sql`:

1. Add `failure_detail TEXT` column to `run_model_metrics`:

```sql
-- Extend run_model_metrics (Rethunk-AI/bakeoff#23)
ALTER TABLE run_model_metrics
    ADD COLUMN IF NOT EXISTS failure_detail TEXT;
-- (failure_reason already exists; its values now conform to the taxonomy enum)
```

2. Add `dumb_model` row to `task_categories` (seed, idempotent):

```sql
INSERT INTO task_categories (name, description)
    VALUES ('dumb_model', 'Minimal-capability floor suite: deterministic scorers only.')
    ON CONFLICT (name) DO NOTHING;
```

3. Add `run_status` column to `runs`:

```sql
ALTER TABLE runs
    ADD COLUMN IF NOT EXISTS run_status TEXT
        CHECK (run_status IN ('complete', 'incomplete', 'failed'));
```

### 5. Downstream Contract (bakeoff-results)

This section lists precisely which fields `bakeoff-results` (display side,
Rethunk-AI/bakeoff-results#24) depends on, keyed by use case.

| Use case | Field(s) consumed |
|----------|-------------------|
| Index page: badge run as complete / incomplete / failed | `manifest.json → run_status` |
| Index page: rank runs / models by score (including partial) | `manifest.json → model_scores_summary[*].partial_score` |
| Index page: show failure-mode label | `manifest.json → model_scores_summary[*].dominant_failure_code` |
| Detail page: per-model status chip | `result.json → model_scores[*].status` |
| Detail page: completeness progress bar | `result.json → model_scores[*].completeness`, `cells_attempted`, `cells_total` |
| Detail page: per-model floor badge | `result.json → model_scores[*].floor_score`, `floor_cells_passed`, `floor_cells_total` |
| Detail page: per-cell failure tooltip | `result.json → records[*].failure_code`, `records[*].failure_detail` |
| Filter/group by failure type | `manifest.json → model_scores_summary[*].dominant_failure_code` |
| Floor-tier table | `result.json → records` filtered by `tier == "dumb_model"` |

The results site must treat all new fields as **optional**. Absent fields (old
bundles) fall back to: `run_status → "complete"`, `model_scores_summary → []`,
`floor_score → null`, `failure_code → null`.

bakeoff-results#21 (result-state badge) and bakeoff-results#24 (display side)
both depend on `run_status` and `model_scores_summary` being present in the
manifest; the schema defined here is the upstream authority.

## Acceptance Criteria

- The `failure_code` enum is documented in `bench/` (a module docstring or a new
  `bench/failure.py`) and maps to the nine taxonomy values defined in §1.
- Every failed cell in `result.json` carries a non-null `failure_code`.
- Successful cells carry `failure_code: null` (or omit the field).
- `partial_score` is computed using the completeness-weighted formula in §2 for
  every model, including those with zero attempted cells.
- `floor_score` is computed for every model when the dumb_model tier is enabled.
- The 12 dumb_model tasks are committed to `datasets/dumb_model_tasks.jsonl` and
  their content never changes without a version bump (to preserve reproducibility
  across runs from different dates).
- `run_status` appears in both `result.json` and `manifest.json`.
- `model_scores` appears in `result.json`; `model_scores_summary` appears in
  `manifest.json`.
- `validate_result_payload()` does not reject old bundles that lack the new
  fields.
- `validate_bundle()` does not reject old manifests that lack `run_status` and
  `model_scores_summary`.
- A new `--strict` validation flag (or equivalent) warns (not errors) when new
  fields are absent.
- Tests cover: taxonomy code detection for each code, partial_score formula with
  worked example values matching §2, floor tier scoring (all-pass, all-fail,
  mixed), manifest projection, and backward-compat with old payloads.
