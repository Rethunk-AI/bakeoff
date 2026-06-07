# Topic: scheduling_queue

Consolidated chat history (chronological, ascending comment-id). 15 entries. Verbatim quotes; attribution in each header. **#38 thread excluded from this variant.**

## Source entries (provenance TOC)

Entries used to build this topic and its summary. (Not migrated into `topic_summary/`; audit reference only.)

| ticket | entry | author | posted | url |
|---|---|---|---|---|
| #12 | comment 4491873240 | @gissf1 | 2026-05-19T20:42:32Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4491873240> |
| #13 | issue body | @AlbinoGeek | 2026-05-20T07:36:25Z | <https://github.com/Rethunk-AI/bakeoff/issues/13> |
| #13 | comment 4507791272 | @AlbinoGeek | 2026-05-21T11:35:01Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4507791272> |
| #13 | comment 4512182181 | @gissf1 | 2026-05-21T19:43:35Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4512182181> |
| #13 | comment 4512649799 | @AlbinoGeek | 2026-05-21T20:48:59Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4512649799> |
| #13 | comment 4513796211 | @gissf1 | 2026-05-22T00:08:29Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4513796211> |
| #13 | comment 4513822952 | @AlbinoGeek | 2026-05-22T00:14:43Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4513822952> |
| #13 | comment 4516038912 | @gissf1 | 2026-05-22T06:56:39Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4516038912> |
| #13 | comment 4519072195 | @AlbinoGeek | 2026-05-22T13:24:49Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4519072195> |
| #13 | comment 4522549514 | @gissf1 | 2026-05-22T20:37:38Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4522549514> |
| #13 | comment 4522610944 | @AlbinoGeek | 2026-05-22T20:49:49Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4522610944> |
| #13 | comment 4524200643 | @gissf1 | 2026-05-23T04:52:22Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524200643> |
| #13 | comment 4524304026 | @AlbinoGeek | 2026-05-23T05:40:34Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524304026> |
| #13 | comment 4570344944 | @AlbinoGeek | 2026-05-29T03:52:50Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4570344944> |
| #13 | comment 4570546649 | @AlbinoGeek | 2026-05-29T04:26:21Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4570546649> |

---

## Consolidated thread

### #12 comment 4491873240 — @gissf1 — 2026-05-19T20:42:32Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4491873240>

A1: I believe that design makes sense to me.  But in that case, I would think a prompt record should have a FK to the task_id it belongs to.  That would also mean we don't need both run_model_metrics.

I think the definition of a prompt is pretty self explanatory, but task is still a bit vague.  How does a task compare to a test or a test suite?  Do any of those terms describe the same thing?  Different things?  How are they related?

A2: What is the best design here?  I feel like having data versioned in files is helpful because we may want to go back and refine the prompts, in which case we would want that version controlled.  And for each run, we would want to know exactly what prompt was given to the model.  This also means a refined prompt can be then tested with the same models to determine its improved effectiveness.  Also, we want to track how changes in a prompt affect various models for potential A/B split comparisons and other metrics.  Keeping the versions in files means we can reference the file and git commit hash in which that file/prompt was last modified to uniquely identify it.  The current files should be easily accessible from the UI, and even historical files are available with a bit of effort.  One could potentially reference GitHub raw URLs or other such things to quickly acquire the file data.

A3: I think a specific source URL makes sense since it's relatively unique.  one concern is if the model from different sources (from ollama vs huggingface for example) are the same model or not.  We should probably do some analysis on this.  If the identical models are available from different sources, perhaps we should have a list of URLs indicating identical models so we can coalesce them into a single model in our system (this helps with organization and reduces duplicate testing).  I also don't know how either of those sites handles versioning - if the owner of the model revises their model, does it use the same URL or a new one?  Another thing to consider - if a model has a predecessor, it probably makes sense to track that so we can then analyze the evolution of a model.

I think a models table could make sense, at minimum it should have a source URL (or a list of them, if multiple), a name (user friendly), a release date, maybe a version, maybe some description about it, probably scraped from something like the source URLs.  For deduplication, maybe we can find some way to have a standard unique identifier for a model; I'm imagining something like the sha of its model data as a worst case, but ideally it would be something pre-computed and generated elsewhere to avoid the extra compute time on a large blob like that.  It should also have parameters describing the model - size, parameter_count, family, context_length (maybe identify the min, max, and default), format ("GGUF" or others, since this may make a different in output, is "format" the right term for this attribute?), quantization, and minimum VRAM required for runtime (important since this may be very different than the parameter_count if quantization differs).  There may be other fields I didn't think of yet.

Q4: I think all tests should have both a pass/fail mark and a score.  Both metrics should be absolute based on the model capabilities, and not on performance or hardware related issues/limitations.  Being unable to complete a test because the hardware it ran on is insufficient is not a failure of the model.

There should be a separate metric created later for performance analysis combined with the quality score.  This secondary metric will take into consideration the information from run_model_metrics and such to determine not only accuracy, but also relative performance to determine the model's effectiveness and efficiency.  This metric would be hardware neutral, but take into consideration which model best utilizes the available hardware to achieve the goal.

Q5: Let's discuss these options more.  If we didn't create a table for runs, where would we put the run-specific data like timestamp, submitter, runner_version?  I would think either the FK or natural key methods are identical other than the format of the field (UUID vs serial integer), am I missing something?  Also, I'm assuming the runner_version is just the testing software and not the actual prompt data itself (this may be an incorrect assumption), but beyond that we should also include the SHA hash of the git commit the prompt data is based on, or a hash of the prompt data in the worst case.  Maybe we should also have a prompt-hash-to-release-date table so we can determine how out of date test results are and order them by "oldest first" when retesting older models.

Regarding the preliminary table starting point, I have some thoughts and comments here as well:

In the models table, I think "provider" is the wrong context.  Most of the models we intend to test are locally hosted, so the model's provider might be better thought of as "developer" or "source", unless I'm misunderstanding your intent here.  That said, I'm not opposed to having multiple fields to represent that data.  In that case, perhaps "external_provider" would be a better name for the current field, so that it is null by default for all local models?  I suppose we could create a providers table at some point and make this field an FK into the providers table, but none of that is really a concern for Phase 1.

for tasks, I think it makes sense to have a tiered structure available, so we can have aggregate tasks (representing a test suite) that have subordinate tasks.  For this I think we can have a task_categories table so that the task_category fields is a FK rather than redundant text.  I also think there should be a nullable parent_id field which points to the parent task_id if it is a subordinate task.  I think it also makes sense to have an order field to control the order in which the tasks and prompts are executed, unless that is stored on the filesystem.  That said, I'm not convinced that tasks and prompts should be database centric.  I feel like they should be derived from filesystem data so they are version controlled.  Let me know your thoughts on this.

I feel like prompts should be stored in a file specifically, but some aspects of them likely need to be referenced by the database. Besides the commit hash in which they were last modified, we should also probably include the raw length of the prompt text, version, and release date.  Maybe it also makes sense to have a prerelease flag for testing new prompts against old prompts before an official release?

for run_model_metrics, do we need task_id if we have the prompt_id?  I'm still not sure that we want the prompt data databased, or if we should rely on filesystem data.  we also need to plan the testing queue, so we should probably start getting into that at some point later, maybe you should make a new ticket for that to be addressed once we figure out the model aspects?  You let me know the prioritization order on that, since planning is your expertise.

Anyhow, that's what I have for comments so far, let's start there, and we can iterate and discuss more as needed.

---

### issue #13 body — @AlbinoGeek — 2026-05-20T07:36:25Z
<https://github.com/Rethunk-AI/bakeoff/issues/13>

**Context:** Opened per discussion in #12. Once model, task, and prompt schemas are finalized, test runs need to be planned and queued before execution.

**Scope for discussion:**

- Queue structure: how runs are enqueued (which model × which prompt set × which hardware)
- Priority ordering: which runs execute first (e.g., oldest model, least tested, highest priority tier)
- Queue state machine: PENDING → IN_PROGRESS → COMPLETE / FAILED
- Retry policy: failed runs (timeout, OOM) — how many retries, backoff
- Dependency ordering: should baseline tier complete before comparison tier runs?
- Queue persistence: file-based (git-tracked) or DB-only?
- Runner interaction: how the bakeoff runner picks up and marks queue items
- Multi-runner support: can multiple runners claim queue items concurrently?

**Relation to other issues:**
- Blocked on: #12 (model/task/prompt schema) — queue items reference model_id, prompt_id
- Informs: run_hardware_metrics (which hardware a queued run is assigned to)

@gissf1 — tagging for input once #12 schema is closer to final.

— Bastion

---

### #13 comment 4507791272 — @AlbinoGeek — 2026-05-21T11:35:01Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4507791272>

**Bastion (J-5) — bakeoff#13 queue design proposal — 211132ZMAY26**

@gissf1 — schema.sql is committed (23b4286). `runs` and `run_model_metrics` tables provide the structural anchor for the queue. Proposing initial queue design below. Opening for @gissf1 input.

---

## Proposed: `run_queue` table

```sql
CREATE TABLE run_queue (
    queue_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    model_id        INT  NOT NULL REFERENCES models(model_id),
    prompt_set_id   INT  NOT NULL,              -- logical grouping; FK TBD when prompts table extended
    hardware_tag    TEXT NOT NULL,              -- "gpu-a100", "cpu-only", etc. — matches runner affinity
    priority        INT  NOT NULL DEFAULT 100,  -- lower = higher priority; 0 = critical
    status          TEXT NOT NULL DEFAULT 'PENDING'
                        CHECK (status IN ('PENDING','CLAIMED','IN_PROGRESS','COMPLETE','FAILED','CANCELLED')),
    attempt_count   INT  NOT NULL DEFAULT 0,
    max_attempts    INT  NOT NULL DEFAULT 3,
    claimed_by      TEXT,                       -- runner identity (hostname + PID)
    claimed_at      TIMESTAMPTZ,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_detail    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for next-item-to-claim query
CREATE INDEX run_queue_claim_idx
    ON run_queue (hardware_tag, priority, created_at)
    WHERE status = 'PENDING';
```

---

## State machine

```
PENDING → CLAIMED → IN_PROGRESS → COMPLETE
                  ↘             → FAILED  → PENDING (if attempt_count < max_attempts)
                                           → FAILED  (terminal, if attempt_count >= max_attempts)
PENDING → CANCELLED (operator cancel)
CLAIMED → PENDING  (claim timeout: >10 min in CLAIMED without IN_PROGRESS transition = stale claim)
```

**Claim protocol (atomic, race-safe):**
```sql
UPDATE run_queue
SET status = 'CLAIMED', claimed_by = $runner_id, claimed_at = NOW()
WHERE queue_id = (
    SELECT queue_id FROM run_queue
    WHERE status = 'PENDING'
      AND hardware_tag = $runner_hardware_tag
    ORDER BY priority ASC, created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
RETURNING *;
```

`FOR UPDATE SKIP LOCKED` is the standard Postgres queue-claim idiom. Multiple runners on the same hardware tag cannot claim the same item.

---

## Priority ordering

| Priority value | Meaning |
|---------------|---------|
| 0–9 | Critical (baseline tier, blocks comparisons) |
| 10–49 | High (new model, first submission) |
| 50–99 | Normal (scheduled refresh) |
| 100+ | Low (re-run for additional samples) |

**Dependency ordering:** baseline tier items enqueued at priority 0–9. Comparison-tier items enqueued at 50. Runner responsible for checking whether baseline is COMPLETE before marking comparison IN_PROGRESS (or queue can enforce via FK / trigger — open question for @gissf1).

---

## Retry policy

- `max_attempts = 3` default.
- Retry trigger: on FAILED with `attempt_count < max_attempts`, reset to PENDING, increment `attempt_count`, set `priority = priority + 10` (deprioritize relative to fresh items).
- No backoff delay in DB — runner responsible for exponential backoff before re-claiming.
- OOM / timeout failures: same retry path; `error_detail` records cause.
- Terminal failure (`attempt_count >= max_attempts`): stays FAILED. Manual operator action to reset or cancel.

---

## Persistence: DB-only (proposed)

Git-tracked file queue rejected. Rationale:
- `FOR UPDATE SKIP LOCKED` requires Postgres; file-based queues cannot provide race-safe multi-runner claims.
- Queue state is transient operational data, not result data. Git is append-only history; queue state is mutable — wrong abstraction.
- Schema.sql is already the substrate. Adding `run_queue` keeps all operational state in one plane.

If git-tracked queue is preferred for audit/portability reasons, @gissf1 to advise — Bastion will revise.

---

## Runner interaction

```
1. Runner starts → declares hardware_tag + runner_id
2. Claims next PENDING item (atomic UPDATE above)
3. Transitions to IN_PROGRESS (UPDATE status='IN_PROGRESS', started_at=NOW())
4. Executes run → writes results to run_model_metrics
5. Transitions to COMPLETE (UPDATE status='COMPLETE', completed_at=NOW())
   OR FAILED (UPDATE status='FAILED', error_detail=<cause>)
6. Retry logic: if FAILED and attempt_count < max_attempts → reset to PENDING
7. Loop back to step 2
```

Heartbeat: runner should UPDATE `updated_at` every 60s while IN_PROGRESS. Stale-claim reaper (cron or background goroutine) resets items where `status='CLAIMED'` and `claimed_at < NOW() - INTERVAL '10 minutes'`.

---

## Multi-runner support

Fully supported via `FOR UPDATE SKIP LOCKED`. Runners filter by `hardware_tag` — a GPU runner will not claim CPU-only items and vice versa. No coordinator process required.

---

## Open questions for @gissf1

1. **Dependency enforcement in DB vs application layer?** Should a DB trigger block CLAIMED→IN_PROGRESS on comparison items until baseline is COMPLETE, or does the runner check this at claim time?
2. **`hardware_tag` enumerated in DB or free-text?** Could add a `hardware_configs` reference table.
3. **`prompt_set_id` FK target?** Current `prompts` table has `prompt_id` per prompt. A prompt set could be a named group — suggest adding `prompt_sets` table, or using a JSON array in `run_queue`.
4. **Stale-claim reaper: runner-side or scheduled DB job?** Simple `pg_cron` expression vs application heartbeat loop.

— Bastion

---

### #13 comment 4512182181 — @gissf1 — 2026-05-21T19:43:35Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4512182181>

In the claim protocol and run_queue table, what if the model is simple enough not to need any specific hardware tag?  I think the whole "hardware tag" concept is actually not right, but perhaps you can convince me.  I think it should be as simple as doing math on what the model's minimum requirements are for a given runner's hardware, so perhaps running a 100B dense model on a 6GB GPU is not practical, but running a 21B parameter on 16GB GPU should be ok.  We shouldn't have to manually tag every model with what hardware we want it to run on.

I don't understand what question is being asked in the "Priority ordering" section.

For the retry policy, I think it also makes sense to modify the claim protocol query to ignore items that have been updated_at less than 5 minutes ago and have updated_at != created_at;  That handles all failure updates and limits, enforcing a minimum 5 minute retry delay at the claim level without stalling the queue for other models.  We should probably still have an exponential backoff in the runner, but this prevents wasting compute time in a loop when something is badly broken.  We should probably have a delay on the transition between FAILED and returning to PENDING, and maybe that eliminates the need for the delay on claim?  Let's discuss this all more.

I think max_attempts should be more like 5 by default, especially if we are decreasing priority each time it fails.

I think the queue should be database driven since it is "rapidly changing data", and requires atomic operations for execution, but it does need some persistence to disk.  At minimum, successfully completed test results should be written to disk, and in case of data corruption, we should be able to rebuild a functional queue with basic priorities and order restored.  The file-backed priorities should reflect the initial priorities given in the run_queue table, but we shouldn't be updating the files to keep them in sync with the database updates; the files would be there mostly for disaster recovery.  Once a run_queue item completes successfully it probably makes sense to move the initial file that populated the run_queue entry into some "completed" directory, so it doesn't get re-queued on restoration (once we get the results, we don't need to run a test again other than to update results later, so the file wouldn't need to be restored).

Can you please elaborate on your statement "Queue state is transient operational data, not result data. Git is append-only history; queue state is mutable — wrong abstraction."?

Regarding your questions:

1. Please elaborate, I don't understand your question.
2. As stated above, I don't think "hardware tag" is the right approach, but I'm open to being convinced.  If you are not sure this is the right approach, I can suggest alternatives and we can discuss.
3. perhaps this should be a task_id rather than a prompt?  I'm unsure of your motivation to making a single queue item cover more than a single prompt (which would then be just prompt_id).  Let's discuss this more.
4. Honestly, if the DB operation is atomic, I don't think it matters which is implemented first, or even if we have both.  My initial thought would be to avoid specialized scheduled DB jobs so that all our execution is handled within the runner and the DB is just a simple DB, but this is not a hard line, and I could imagine a dedicated runner being more efficient.  We could have a runner randomly choose to run this query after a failure (so maybe it does rand() with a 10% chance of running the reaper, otherwise resuming its normal operations).  We could adjust that % as needed to adjust the frequency.  The reaper process, wherever it runs from, should probably also handle checking on and converting FAILED runs back into PENDING after the transition delay expires.  I like the idea of having a runner application heartbeat loop in general because it allows us to centrally keep track of the runner processes and ensure they haven't died.  Let's discuss this question more as well.

---

### #13 comment 4512649799 — @AlbinoGeek — 2026-05-21T20:48:59Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4512649799>

**Bastion (J-5) — bakeoff#13 queue revision — 212014ZMAY26**

@gissf1 — substantive feedback received. Addressing each point in sequence.

---

## 1. Hardware tag → capability-based matching

Agreed. Manual `hardware_tag` is wrong — creates maintenance burden and doesn't generalize. Revised approach: store model requirements in `models`, store runner capabilities in a `runners` table, filter at claim time.

**`models` table additions:**
```sql
ALTER TABLE models ADD COLUMN min_vram_gb    INT     NOT NULL DEFAULT 0;
ALTER TABLE models ADD COLUMN param_count_b  DECIMAL;   -- e.g. 21.0 for 21B
ALTER TABLE models ADD COLUMN quantization   TEXT;      -- "fp16", "q4_k_m", etc.
```

**`runners` table (new):**
```sql
CREATE TABLE runners (
    runner_id      TEXT        PRIMARY KEY,   -- hostname:pid or stable agent ID
    vram_gb        INT         NOT NULL,
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status         TEXT        NOT NULL DEFAULT 'ACTIVE'
                               CHECK (status IN ('ACTIVE','IDLE','DEAD'))
);
```

**`run_queue`:** drop `hardware_tag`. Claim query filters by runner capability at claim time:

```sql
UPDATE run_queue
SET status = 'CLAIMED', claimed_by = $runner_id, claimed_at = NOW()
WHERE queue_id = (
    SELECT rq.queue_id
    FROM   run_queue rq
    JOIN   runs   r ON r.run_id   = rq.run_id
    JOIN   models m ON m.model_id = r.model_id
    WHERE  rq.status = 'PENDING'
      AND  m.min_vram_gb <= $runner_vram_gb
      AND  (rq.updated_at = rq.created_at
            OR rq.updated_at <= NOW() - INTERVAL '5 minutes')   -- retry gate; see §3
    ORDER BY rq.priority ASC, rq.created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
RETURNING *;
```

Runner passes its `vram_gb` at claim time — no pre-assignment required. Model requirements are stored once at ingestion (from model card or metadata). Additional capability axes (CPU cores, RAM) can be added later without touching the queue schema.

---

## 2. Priority ordering section — clarification

That section was unclear. The subsection "Dependency ordering" buried an implicit assumption I should have stated: in a comparative bakeoff, some runs are **baseline** (the reference model) and some are **comparison** runs. If scoring is relative (comparison scored against baseline output), baseline must complete first. The priority table was attempting to encode that ordering. I'm separating it into its own question — see §5 below.

---

## 3. Retry policy revision

Your `updated_at` claim gate is the right approach. Adopting with refinements:

**Claim-level retry gate** (your proposal, integrated into claim query above):
```sql
AND (rq.updated_at = rq.created_at
     OR rq.updated_at <= NOW() - INTERVAL '5 minutes')
```

On FAILED with `attempt_count < max_attempts`: reset immediately to PENDING, bump `updated_at`. Claim query won't pick it up for 5 minutes. No dedicated `retry_after` column needed — `updated_at` doubles as the delay marker. Clean.

A delay on the FAILED → PENDING transition itself would also work, but combined with the claim gate it becomes redundant. Recommend the claim gate only — fewer state transitions, simpler reaper logic.

**max_attempts = 5.** Agreed. With priority degradation each failure, terminal failures are deprioritized well before they hit the ceiling.

**Priority increment on retry:** `priority = priority + 20` per failure. With 5 attempts starting at normal (100), worst-case degradation reaches 180 — still processable, but well behind fresh items.

**Exponential backoff in runner** still recommended for per-runner loop protection. Even with the claim gate blocking re-claim, a tight loop burning compute against `FOR UPDATE SKIP LOCKED` wastes cycles. Runner should back off independently.

---

## 4. Disk persistence — DR layer

Agreed on the model. Revised design:

**DB is authoritative during operation.** All queue state mutations are DB-only. No live file sync.

**Files are bootstrap/DR artifacts.** Three directories:

```
queue/
  pending/    # job descriptor files — source of truth for re-enqueue on DR
  in-db/      # moved here after successful DB insert (optional but clearer)
  completed/  # moved here on run COMPLETE; prevents re-enqueue on DR
  failed/     # optional: terminal failures for operator review
```

**Normal flow:**
1. Job descriptor dropped into `queue/pending/` (by operator, CI, or API).
2. Enqueue process reads file → inserts `run_queue` row → moves file to `queue/in-db/`.
3. On COMPLETE: move file to `queue/completed/`. Results in `run_model_metrics` (DB).
4. DB and files intentionally diverge on queue state. Files only track: "was this job ever successfully completed?"

**DR path:**
1. DB lost or corrupted.
2. Scan `queue/pending/` and `queue/in-db/`: re-enqueue anything absent from `queue/completed/`. Initial priorities from file metadata.
3. CLAIMED/IN_PROGRESS/FAILED state is not restored — everything reverts to PENDING. Acceptable: worst case is re-running some jobs.

**On my "transient data" statement:** I conflated two use cases. Live queue state (CLAIMED/IN_PROGRESS transitions firing every few seconds) is wrong to sync continuously to git — generates meaningless commit noise. Your proposal is different: files as DR initialization artifacts, not live state. That's the right abstraction for recovery. My rejection of file-backed queues was too broad. Retract that; the revised design above is what I should have proposed initially.

---

## 5. Q1 — Dependency enforcement (elaborated)

Context: a bakeoff run may include a baseline model (the reference) and comparison models. If scoring is **relative** (comparison output scored against baseline output), the baseline must complete first.

Three options:

- **A. Priority-only** — enqueue baseline at priority 0, comparisons at priority 50. Statistical ordering; no enforcement. Simple. Risk: a slow or failed baseline item doesn't block comparisons at the DB level.
- **B. Dependency FK** — add `depends_on_queue_id UUID REFERENCES run_queue(queue_id)` to `run_queue`. Claim query filters: `AND (rq.depends_on_queue_id IS NULL OR (SELECT status FROM run_queue WHERE queue_id = rq.depends_on_queue_id) = 'COMPLETE')`. Strict, but adds schema complexity and a self-join on every claim.
- **C. Application-layer check** — runner checks dependency at IN_PROGRESS transition; re-queues if baseline not complete. Looser; keeps DB simple.

**Recommendation: A for now.** If scoring is a post-processing step (runner collects raw outputs; a separate scorer compares them later), the dependency problem dissolves — comparisons don't need baseline to complete before they run. Does your scoring model require online comparison at run time, or is scoring a separate offline pass? Answer determines whether B is worth the complexity.

---

## 6. Q3 — prompt_id vs prompt_set_id

Agreed: one queue entry per (model, prompt) pair. `prompt_id` as a direct FK to `prompts(prompt_id)` is correct. Grouping logic belongs at the submission layer, not the queue.

If a run covers 100 prompts × 10 models → 1,000 queue entries. Each is independently trackable, retriable, and scoreable. Flat queue is the right shape.

**Schema change:** replace `prompt_set_id INT` with `prompt_id INT NOT NULL REFERENCES prompts(prompt_id)`.

---

## 7. Q4 — reaper: runner-side, probabilistic

Your 10% random reaper proposal is sound. Formalized:

**Runner loop:**
1. Claim next compatible PENDING item (claim query in §1).
2. Transition to IN_PROGRESS → execute → COMPLETE or FAILED.
3. After any outcome: `if rand() < 0.10 { run_reaper() }`.
4. Heartbeat: UPDATE runners SET last_heartbeat = NOW(), status = 'ACTIVE' every 60s.

**Reaper tasks:**
- Stale CLAIMED reset: `status = 'CLAIMED' AND claimed_at < NOW() - INTERVAL '10 minutes'` → PENDING, clear `claimed_by`/`claimed_at`.
- Dead runner detection: `last_heartbeat < NOW() - INTERVAL '5 minutes'` → mark DEAD, reclaim their CLAIMED items.
- FAILED → PENDING via claim gate (handled implicitly by §3 — no explicit reaper step needed unless we add `retry_after`).

The `runners` heartbeat table (§1) gives dead-runner detection without a coordinator process. Reaper stays a simple opportunistic sweep.

Starting at 10% is reasonable. Tune the percentage as queue depth and runner count scale.

---

## Revised open questions

1. **Scoring model** — online (comparison vs baseline at run time) or offline post-processing pass? Determines if dependency enforcement (§5 option B) is needed.
2. **`queue/in-db/` or leave in `pending/`?** Moving on enqueue gives clear DR semantics; crashing mid-move leaves orphans. Simpler alternative: leave files in `pending/`, mark DB row with `source_file TEXT`, check `queue/completed/` on DR scan. Preference?
3. **Model capability ingestion** — where does `min_vram_gb` / `param_count_b` come from? Model card at submission time, CI artifact, or runner autodiscovery? Needs a defined ingestion path before the capability-matching claim query is useful.

— Bastion

---

### #13 comment 4513796211 — @gissf1 — 2026-05-22T00:08:29Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4513796211>

Starting with the proposed models table changes:

I don't think we need to add param_count_b to the models table since we already have parameter_count_b and active_parameter_count_b.  We can discuss using decimal vs float data types if that's significant for precision or accuracy, or some other reason.  I would think we can just use active_parameter_count_b, and for "Dense" models (in the architecture field), we just use the same value for active_parameter_count_b and parameter_count_b.

quantization field already exists in the models table as well.

As for min_vram_gb, do we really need to store that in the models table?  why not have a formula to calculate it based on the model's properties instead?  I don't think it needs to be stored directly if it's able to be calculated easily on the fly.

Regarding the runners table, I don't think we need to know the hardware details of a runner in the runners table.  The runner can just add its own specification limits when it makes the claim query to filter out run_queue models it's unable to test.  Also, if we're creating a table like that, it should probably include the host/ip, process ID, effective user, and other properties of the runner process and its state.

It might make sense to add a "runner_id" to the runs table to indicate which runner was used. 

Dependency ordering based on comparison vs baseline is pointless.  The resulting score should be absolute, in which case the comparison becomes a simple post-processing task of comparing the benchmark scores.  Every model test is independent.  A model's test score should not be biased relative to another model's test results.  If we notice the scores on a test are 100% passing across the board, then the test is pointless and likely broken in a way that it always passes.  If the test scores are 100% failing, either the test is broken, impossible, or the tested models are too dumb.  If we test it against frontier models, and they perfectly pass, that could be considered a test to determine if the test itself is at least sensible and valid.  If even the frontier models fail to pass a test, either it's extremely hard (beyond their capability even) or broken.  In this case, we need to do more analysis to determine which case it is.

We may want to have a retry_after field to simply schedule a test to begin at a certain time for marketing or legal purposes; it also allows us to avoid the math calculation (NOW() - INTERVAL '5 minutes') to compare each row against rq.updated_at on each iteration.  I'm not sure why you preferred the math route, but I'm open to hearing an explanation if you feel it has merit.  Also, leaving it in FAILED state for a bit can be a signal with high retry count that we need to look into it and analyze what is going on, as that would be a sign of problems to look into and resolve.

Priority increment on retry: Maybe we increment by something like (5 * attempt_count), which gets near to exponential backoff.  Agreed on retaining backoff in runner as well.

on the DB disk files, I think it makes sense that on initial load, it makes sense to load everything.  Any new files that appear after the initial load should probably have a check to ensure their modification time is at least 5 seconds in the past (or some larger time) to ensure the file has finished writing and is not partially written at the time it is read and loaded.  Other than that, I agreed on the normal flow.

On the DR path, I think it makes sense to refer back to the results and compare against in-db entries. If the run's results are present, then verify they are valid, and if so, we can ignore it (or move it to completed).  If not, we can move it back to pending and re-enqueue that item.

Q1: As mentioned above, the score should be absolute.  I think A is the only valid option from that perspective.

Q3: My only comment is that results from multiple prompts may be necessary for higher level tasks to compute a score.  Perhaps this is an average/aggregate score, or perhaps the task requires computing something in multiple ways, and then a final comparison at the end determines the high-level score.  This may also relate to the python code used for testing, so perhaps I'm getting ahead of myself.

Q4: I think an essential question is whether the runner is also dependent on the database?  if so, can a normal user also use the runner for a single model on their system without a long-running queue system like we intend to use?  If those are difficult to separate out, then perhaps a dedicated cron that periodically queries the DB to cleanup and reap dead processes makes sense.  My goal was to distribute that load across the runners.

Regarding the revised open questions:
1. Answered above.
2. I think marking the run_queue row with a field for source_file generally makes sense, regardless of the file move semantics.  Perhaps we don't need the "in-db" level if we're tracking this, as we can simply look in the DB or filesystem as needed to determine if it exists.  I'm open to leaving them in pending until they are done, at which point they move to completed.
3. This may need further discussion, possibly a new thread of itself at some point.

---

### #13 comment 4513822952 — @AlbinoGeek — 2026-05-22T00:14:43Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4513822952>

**Bastion (J-5) — bakeoff#13 schema refinements — 220013ZMAY26**

@gissf1 — corrections received and accepted. Addressing in sequence.

---

## 1. models table — existing columns, calculated VRAM

Confirmed: `parameter_count_b`, `active_parameter_count_b`, and `quantization` already exist. No new columns needed for those.

On `min_vram_gb` — agreed, do not store it. Calculate from existing fields:

```sql
-- Inline VRAM estimate for claim query
CEIL(
    m.active_parameter_count_b
    * CASE m.quantization
        WHEN 'fp32'    THEN 4.0
        WHEN 'fp16'    THEN 2.0
        WHEN 'bf16'    THEN 2.0
        WHEN 'int8'    THEN 1.0
        WHEN 'q8_0'    THEN 1.0
        WHEN 'q4_k_m'  THEN 0.45
        WHEN 'q4_0'    THEN 0.5
        WHEN 'q4_1'    THEN 0.5
        ELSE                 2.0   -- conservative default for unknown quant
      END
    * 1.15    -- 15% overhead for KV cache + activations
) AS estimated_vram_gb
```

Runner passes its available VRAM as a parameter; claim query filters `estimated_vram_gb <= $runner_vram_gb`. Calculation is per-row at claim time — no stored column, no staleness risk.

If quantization formats diverge from this list over time, the CASE expression is the single update point.

---

## 2. runners table — revised

Dropping hardware storage from runners. Runner declares its own capability at claim time. Table tracks identity, state, and heartbeat only:

```sql
CREATE TABLE runners (
    runner_id      TEXT        PRIMARY KEY,   -- stable agent ID (e.g. "worker-01")
    hostname       TEXT        NOT NULL,
    process_id     INT         NOT NULL,
    effective_user TEXT        NOT NULL,
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status         TEXT        NOT NULL DEFAULT 'ACTIVE'
                               CHECK (status IN ('ACTIVE','IDLE','DEAD')),
    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Runner registers on startup (INSERT or UPSERT). Heartbeats every 60s. Reaper marks DEAD when `last_heartbeat < NOW() - INTERVAL '5 minutes'` and reclaims their CLAIMED items.

`runner_id` in `runs` table: agreed — add `runner_id TEXT REFERENCES runners(runner_id)` to `runs`. Records which runner executed each run. Useful for diagnosing runner-specific failure patterns.

---

## 3. Claim query — revised

```sql
UPDATE run_queue
SET status = 'CLAIMED', claimed_by = $runner_id, claimed_at = NOW()
WHERE queue_id = (
    SELECT rq.queue_id
    FROM   run_queue rq
    JOIN   runs   r ON r.run_id   = rq.run_id
    JOIN   models m ON m.model_id = r.model_id
    WHERE  rq.status = 'PENDING'
      AND  (rq.retry_after IS NULL OR rq.retry_after <= NOW())
      AND  CEIL(
               m.active_parameter_count_b
               * CASE m.quantization
                   WHEN 'fp32'   THEN 4.0
                   WHEN 'fp16'   THEN 2.0
                   WHEN 'bf16'   THEN 2.0
                   WHEN 'int8'   THEN 1.0
                   WHEN 'q8_0'   THEN 1.0
                   WHEN 'q4_k_m' THEN 0.45
                   WHEN 'q4_0'   THEN 0.5
                   WHEN 'q4_1'   THEN 0.5
                   ELSE               2.0
                 END * 1.15
           ) <= $runner_vram_gb
    ORDER BY rq.priority ASC, rq.created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
RETURNING *;
```

`retry_after` replaces the `updated_at` gate entirely. Simpler: one boolean check per row vs. timestamp arithmetic. Agreed.

---

## 4. retry_after field — adopted

Accepting `retry_after TIMESTAMPTZ` over the `updated_at` math approach. Advantages you identified are correct:

- Simpler claim query (`retry_after <= NOW()` vs. `updated_at` delta math)
- Supports scheduling for non-failure reasons (marketing, legal, staged rollout)
- FAILED items stay FAILED — high `attempt_count` with no PENDING transition = visible signal for operator investigation

**Revised retry flow:**
1. Run → FAILED. Set `retry_after = NOW() + base_interval`, bump `attempt_count`, bump `priority`.
2. Item stays FAILED. Claim query ignores it (status ≠ 'PENDING').
3. Reaper (probabilistic, runner-side) finds: `status = 'FAILED' AND attempt_count < max_attempts AND retry_after <= NOW()` → reset to PENDING, clear `retry_after`.
4. Item re-enters claim pool.

`base_interval` for retry: 5 minutes is fine as a floor. Reaper/runner can use exponential: `5 * 2^(attempt_count - 1)` minutes, capped at e.g. 60 minutes.

---

## 5. Priority increment on retry

Adopting `priority + (5 * attempt_count)`:

| Attempt | Increment | Running delta (from base 100) |
|---------|-----------|-------------------------------|
| 1       | +5        | 105                           |
| 2       | +10       | 115                           |
| 3       | +15       | 130                           |
| 4       | +20       | 150                           |
| 5       | +25       | 175                           |

Soft exponential feel without a hard jump. Terminal at 5 attempts.

---

## 6. Dependency ordering — closed

Confirmed: scores are absolute. Every model run is independent. No baseline dependency needed. Priority-only (option A) sufficient — baseline items enqueued at lower priority numbers as a convention, not enforced by the queue. Closing this question.

---

## 7. Disk layer — revised

**`run_queue` schema addition:**
```sql
ALTER TABLE run_queue ADD COLUMN source_file TEXT;
```

Populated at enqueue time. No `in-db/` directory. Two-directory layout:

```
queue/
  pending/    # job descriptors (source of truth for enqueue)
  completed/  # moved here on COMPLETE
```

**New file detection:** loader checks `mtime < NOW() - 5s` before reading. Configurable. Prevents partial-write reads on rapid ingestion.

**Normal flow:**
1. File written to `queue/pending/`.
2. Enqueue: read file → insert `run_queue` (with `source_file = path`) → do not move file.
3. On COMPLETE: move file to `queue/completed/`. Results in `run_model_metrics`.

**DR path:**
1. Scan `queue/pending/`. For each file:
   a. Check `run_model_metrics` for results matching this job. If results present and valid → move file to `queue/completed/`, skip re-enqueue.
   b. If no valid results → insert `run_queue` row (PENDING). File stays in `pending/`.
2. No CLAIMED/IN_PROGRESS/FAILED state restored. All re-enqueued items start PENDING.

---

## 8. Q4 — runner/DB dependency (standalone vs queue mode)

Key question correctly identified: should a single user be able to run one model without the full DB-backed queue?

Proposed answer: **yes, via a `--standalone` flag**.

```
runner --standalone --model llama3-8b --prompts my-prompts.json
```

Standalone mode:
- Skips DB entirely. No `run_queue`, no `runners` table.
- Reads prompts from file, executes run, writes results to `results/<model>/<timestamp>.json`.
- No claim/heartbeat/reaper logic.
- Same inference code path as queue mode.

Queue mode (default):
- Connects to DB, registers in `runners`, enters claim loop.
- Reaper runs probabilistically (10%) after each job outcome.

If standalone mode is needed immediately, it's a clean separation: the inference core is shared; the queue integration is a thin wrapper around it. Single binary, two operating modes.

**Reaper as separate process:** if runner/DB coupling is tight and standalone users don't want a DB dependency, the reaper can be extracted as a lightweight daemon (`runner --reaper`) or a simple cron script that runs `psql` queries. Keeps the main runner lean. No strong preference — lean toward embedded probabilistic reaper for now; extract to dedicated daemon only if contention becomes a problem.

---

## 9. Q3 (multi-prompt scoring) — deferred

Agreed, leave open. Flat `prompt_id` per queue entry still works for aggregation — the scoring layer handles it. If aggregate scoring emerges as a requirement, it maps naturally onto grouping by `run_id` across queue entries. No schema change needed until the scoring architecture is clearer.

---

## Open questions remaining

1. **VRAM CASE expression completeness** — quantization formats in current `models` data? Need to enumerate actual values to avoid hitting the `ELSE 2.0` default. What does the `quantization` column contain today?
2. **runner_id stability** — stable agent ID (e.g. `"worker-01"`) or derived from `hostname:pid`? `hostname:pid` changes on restart; stable ID requires external assignment. Preference?
3. **retry_after base interval** — flat 5 min, or exponential from attempt 1? Ties into how aggressively we want to retry transient vs. persistent failures.

— Bastion

---

### #13 comment 4516038912 — @gissf1 — 2026-05-22T06:56:39Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4516038912>

Why not have a lookup table for quantization and then models.quantization can be a FK into that table with the size multiplier present in the quantization_methods table along with the name/description as needed.  This allows us to expand to new quantization methods without effort and simplifies lookup query syntax.  It also provides us with a source to look up quantization methods for the drop list in the bakeoff-results filter bar (either they can share the same table, or derive from the same source data).  It also means we don't have to worry about an "else" condition for the multiplier, as it should not be a null-able field.

On dependency ordering, what do you mean "baseline items enqueued at lower priority numbers as a convention, not enforced by the queue"?  any items enqueued can have a priority so we can control the order in which various items are tested.  Especially in the beginning, it makes sense to test easier tests to get some results before harder tests that might take a long time or not yet be ready.  We should have a priority queue for items surely, and some models should be a low priority.  I'm not sure what you meant by "as a convention" though, can you elaborate?  Also, I think the priority is implicitly enforced by the queue's claim query, unless I'm missing something.

on new file detection, do you think 5 seconds is enough delay before reading new files?  what is the filesystem commit time for most Linux filesystems?  I believe many have a commit time of 5 seconds, so maybe 15 seconds is a better delay to give some overlap?  what about with the "relatime", "lazytime", or other options?  I think it should be less than 10 minutes, but I'm open to a deeper discussion on that.

Regarding your key question of if should a single user be able to run one model without the full DB-backed queue.  I would argue that standalone should be the default mode of operation.  The queuing and this whole database schema is more of a special case for our backend model testing processes to better integrate with the bakeoff-results website.  Most users that would be submitting results would be doing so as a file with a one-off test on a certain model.  Obviously the file they are sending should contain all the necessary information for us to populate our DB tables, but also should be signed and verified however else we intended to do that.  The queuing system is more of a special case background process that can interact with the database used by the bakeoff-results website, and spins off standalone runners to do the actual testing process.

Perhaps I didn't clarify I feel that most of the database data should be backed into files for data recovery and idempotentcy.  For example, to specify a model for the runner to test, there should be a model file similar to our models table (and maybe the model_sources table as well), so the model data is accessible to the runner without requiring access to a live database.  That said, I don't feel the minutia of runner state and heartbeat updates need to be in a file at all.

I'm okay with embedding the reaper into whatever process handles the queuing, or not, as you feel is best.  Personally I would prefer it in the same code rather than as a separate script doing things in a completely different way using pure SQL and such, but that doesn't mean it needs to be in the queuing code if there's a better place.

Regarding your "Open questions remaining":

VRAM CASE expression completeness - currently there is no database, and therefore no quantization column, so it contains nothing.  I think you had the right approach to using the existing models data as a source to generate a list of quantization formats.  Between that list and some lookups online, it should be quite possible to determine reasonable scaling factors for each format.

runner_id stability - I think there should be a stable agent ID per host, at minimum, but as you mentioned the PID changes per restart of the process.  Maybe it makes sense to start with the stable host ID as the id for the first process as well.  If there are multiple processes at some point, we can use the runners table to track the actual processes while live.  Perhaps we can have some mangling process to convert from that host id in a predictable way for the subsequent processes if multiple are spawned.  Maybe use some kind of suffix to the host ID to indicate that this is the second (or other) live runner, or some fixed addition or bit mangling per incrementing process count.  I'm imagining something like adding 1 to the host ID for the second process, 2 for the third, and so on (adding n-1), but that's just a simple idea and may cause conflicts; I'm open to suggestions on this, but I don't think more than 1 runner process is essential to plan for in depth at the moment.

retry_after base interval - honestly I think it makes sense to do this exponential as well, but using a different scale since this is the minimum time between retries.  The priority retry scaling will make sure that other "more important" tests are done first, but if this is hit, it generally means we have nothing better to do.  I'm now thinking it makes sense to do something like 5 minutes minimum, but reaching the limit for retry #5 should not keep the runner waiting idle looping for 20 minutes or longer.  In fact, the queuing system should recognize that it doesn't have any work to do, and it should sleep until a few seconds before the next retry_after for a valid queue item that can be run on this node.  But back to the initial question, let's discuss more options and reasons, but I think it's acceptable to leave it fixed at 5 minutes.

I just realized I'm incorrectly interchanging queuing system and runner, perhaps we need to make a clearer distinction between them.  I would argue the runner should be the single process testing system, and the queuing system uses the runner to perform tests and manage the queue of models to test in use cases like ours.  That said, they may well be one-in-the-same process in the end.

---

### #13 comment 4519072195 — @AlbinoGeek — 2026-05-22T13:24:49Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4519072195>

**Bastion (J-5) — bakeoff#13 schema + design revision — 221322ZMAY26**

@gissf1 — full response below. Substantive revision on several points.

---

## 1. quantization_methods lookup table — ADOPTED

Correct call. CASE expression in the claim query is fragile and duplicated. Proposed table:

```sql
CREATE TABLE quantization_methods (
    quantization_id  SERIAL  PRIMARY KEY,
    name             TEXT    NOT NULL UNIQUE,   -- "fp32", "q4_k_m", etc.
    vram_multiplier  DECIMAL NOT NULL,           -- bytes per active parameter (e.g. 4.0 for fp32, 0.45 for q4_k_m)
    description      TEXT
);
```

`models.quantization` (currently TEXT) becomes `models.quantization_id INT NOT NULL REFERENCES quantization_methods(quantization_id)`. Not nullable — every model must declare a known quantization method.

Revised claim query VRAM filter:
```sql
AND CEIL(m.active_parameter_count_b * qm.vram_multiplier * 1.15) <= $runner_vram_gb
```
(with `JOIN quantization_methods qm ON qm.quantization_id = m.quantization_id`)

Benefits: extensible without schema change, no ELSE fallback risk, single source for the bakeoff-results filter bar dropdown. Initial seed data will be derived from GGUF/llama.cpp format specs. Will post seed SQL as a follow-up for review.

---

## 2. Priority ordering — clarified

Agree with your read. The phrase "as a convention, not enforced by the queue" was imprecise.

What I meant: the *semantic label* "this is a baseline" is human-assigned at enqueue time — nothing in the schema enforces that meaning. The numeric priority IS enforced by the claim query's `ORDER BY priority ASC`. Lower number = claimed sooner. If the caller assigns a baseline model priority 0, the queue guarantees it is processed before items at priority 50. The semantics of which items are "baseline" or "comparison" are up to the caller.

Since scores are absolute (Q1 closed), the baseline/comparison distinction is moot for ordering. Priority is a general-purpose field. Caller controls meaning.

---

## 3. File mtime delay — revised

5 seconds is likely insufficient. ext4 default `commit=5s` means a file written just before the commit interval could appear present to an mtime check but not be fully flushed. `relatime`/`lazytime` mount options complicate this further — atime updates are lazy, which doesn't directly affect write flush, but indicates an environment that deprioritizes filesystem precision.

Revised recommendation: default **30 seconds**, configurable. 30s provides comfortable margin over ext4 commit, NFS write-behind, and lazy-write scenarios. For most test submission workflows, a 30s pre-read delay is acceptable. High-throughput operators can reduce it. The configuration point matters more than the exact default.

---

## 4. Standalone as default — ADOPTED

Inverting the model. Standalone is default; queue worker mode is opt-in.

```
runner --model model.json --prompts prompts.json    # standalone, default
runner --queue                                       # queue worker mode, connects to DB
```

**Standalone:**
- No DB dependency.
- Reads model descriptor file + prompts file from disk.
- Executes run → writes results to `results/<model>/<timestamp>.json`.
- Signs/verifies results (TBD — separate thread when signing scheme is defined).
- Exits.

**Queue worker:**
- Connects to DB.
- Registers in `runners` table.
- Claims items from `run_queue`, invokes the same inference core as standalone.
- Reports results to `run_model_metrics`, moves `source_file` to `queue/completed/`.
- Runs probabilistic reaper after each outcome.

The inference core is shared. Queue integration is a thin wrapper around it. Single binary, mode determined by flags.

---

## 5. File-backed model data — ADOPTED

Runner must operate without a live DB. Propose a model descriptor file (JSON) that mirrors the relevant columns:

```json
{
  "model_id": 42,
  "name": "llama3-8b",
  "architecture": "Dense",
  "parameter_count_b": 8.0,
  "active_parameter_count_b": 8.0,
  "quantization": "q4_k_m",
  "sources": [
    { "type": "huggingface", "repo": "meta-llama/Meta-Llama-3-8B-Instruct" }
  ]
}
```

Runner reads this at startup. No DB lookup needed for model properties or VRAM estimate. The `quantization` field here is the name string — runner looks up `vram_multiplier` from a local bundled copy of `quantization_methods` seed data (or computes from a bundled table), without requiring a live DB connection.

DB `models` table is populated separately (by a loader or from the same descriptor file at submission time). Descriptor file is the canonical input; DB is derived.

Runner state, heartbeat, and queue mechanics are DB-only. No file backing needed for those. Agreed.

---

## 6. Reaper — embedded, same code

Embedded in the queue worker process. Not a separate SQL script or cron. Agreed. Probabilistic 10% after each job outcome. Extract to dedicated daemon only if contention warrants it later.

---

## 7. Open questions — responses

**VRAM CASE completeness:** no live DB yet — proceeding to build `quantization_methods` seed data from GGUF/safetensors standards + llama.cpp documentation. Will post as a follow-up comment with full seed SQL for review.

**runner_id stability:** hostname as stable base ID. Multiple processes on same host: `hostname`, `hostname-2`, `hostname-3`, etc. — simple suffix, collision-free for the foreseeable single-runner-per-host case. PID recorded separately in `runners.process_id` for process-level tracking. On restart, runner UPSERTs into `runners` with the same stable ID, updating `process_id` and `started_at`. Reaper handles reclaiming any CLAIMED items from the prior PID. More than one runner per host is not a priority to design for now; revisit when multi-runner on a single host is needed.

**retry_after interval:** fixed 5 minutes acceptable. On the idle-loop concern: queue worker should compute next wake time rather than busy-polling:

```sql
SELECT MIN(retry_after)
FROM run_queue
WHERE status = 'FAILED'
  AND attempt_count < max_attempts
  AND retry_after > NOW()
  AND <capability filter>
```

If no PENDING items and no FAILED items with expired `retry_after`, sleep until `MIN(retry_after)`. If result is NULL (nothing pending, nothing retrying within capability), sleep for a configurable poll interval (e.g. 60s) before re-checking. Prevents idle spin without a dedicated sleep-queue mechanism.

---

## 8. Runner vs. queuing system — terminology adopted

Adopting your distinction. Proposed terms going forward:

- **runner**: the inference process. Executes a single model test (standalone or queue-driven). No queue dependency.
- **queue worker**: manages `run_queue`. Claims items, invokes the runner's inference core, reports results, runs the reaper. DB-dependent.

Same binary, mode selected by flag. The inference core has no queue dependency; the queue worker wraps the inference core.

---

## Revised schema delta (from last revision)

```sql
-- New
CREATE TABLE quantization_methods (
    quantization_id  SERIAL  PRIMARY KEY,
    name             TEXT    NOT NULL UNIQUE,
    vram_multiplier  DECIMAL NOT NULL,
    description      TEXT
);

-- models: replace TEXT quantization with FK
ALTER TABLE models DROP COLUMN quantization;
ALTER TABLE models ADD COLUMN quantization_id INT NOT NULL REFERENCES quantization_methods(quantization_id);

-- runs: add runner_id
ALTER TABLE runs ADD COLUMN runner_id TEXT REFERENCES runners(runner_id);

-- runners: revised (no vram_gb; add hostname, process_id, effective_user, started_at)
CREATE TABLE runners (
    runner_id      TEXT        PRIMARY KEY,
    hostname       TEXT        NOT NULL,
    process_id     INT         NOT NULL,
    effective_user TEXT        NOT NULL,
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status         TEXT        NOT NULL DEFAULT 'ACTIVE'
                               CHECK (status IN ('ACTIVE','IDLE','DEAD')),
    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- run_queue: add source_file, retry_after (updated_at gate replaced)
ALTER TABLE run_queue ADD COLUMN source_file TEXT;
ALTER TABLE run_queue ADD COLUMN retry_after TIMESTAMPTZ;
```

---

## Open questions remaining

1. **Result signing/verification** — what scheme? Separate thread when ready.
2. **Model descriptor file format** — JSON confirmed above; versioned schema? Location on disk (beside the model weights, or a separate submission directory)?
3. **quantization_methods seed data** — Bastion to post initial rows as follow-up; @gissf1 to review multipliers.

— Bastion

---

### #13 comment 4522549514 — @gissf1 — 2026-05-22T20:37:38Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4522549514>

File mtime delay - I think 30 seconds is still quite reasonable.  How do you propose making it configurable?  Do we have a configuration file somewhere that we can add a configuration entry to?

File-backed model data - I agree there should be file backed data that mirrors the relevant DB columns.  I'm unsure whether JSON or YAML is a better choice though.  Perhaps JSON files are better so they can be easily read as necessary by the bakeoff-results web interface?  I will point out that the model file should contain the model_hash, and most other fields from the models table.

We need to define some terminology here to differentiate between these items:
- models table - this is the database table in the database, not referring to the disk files that back it up.
- model disk file - this is the file we are now talking about using JSON or YAML that stores a version of same data as we have in the database, and likely should mirror those fields for the most part.
- model source file - this is the model data downloaded from a model_source, likely in GGUF format or similar, containing the actual model's parameters and weights.  As far as I understand, this file is stored on disk, possibly cached, and loaded into the GPU for inferencing by the runner and whatever process it uses for testing.

It may also make sense to have "model_source_mtime" and "model_source_size" to indicate the exact mtime and size of the model source file on disk (if present) to detect if the downloaded model source file has changed since the model disk file was updated.  This should make it easier to detect if the model disk file matches the cached model source file or the model_hash needs to be updated.  If we are updating the model disk file, the update process should be ensure all data is in sync and updated as described before (verifying the acquisition timeframe and such).

I think it might make sense for many tables to have a file-backing so looking up data in the files can be analogous to the SQL versions.  

I think model_id should probably be a UUID (or generated based on name, parameters, and model_hash; maybe other fields too?) so it can be generated in a distributed manner without significant risk of ID collisions, and be consistent for the same model.

on runner_id stability: the runner should ensure that another process isn't alive using that suffix index before using it or else they will be competing to update the same record and have collisions.  Agreed though, not a priority issue for now.

Agreed on your retry_after sleep process.

For runners.status field, why not make that an ENUM field instead of text + check?

For run_queue, we still have an updated_at field, right?  I feel that may still be useful, but perhaps there's an analogue to that?  retry_after is not an a replacement for updated_at, as they would serve different purposes.

Responses to "Open questions remaining":
- Result signing/verification, I will address in the new thread.
- Model descriptor file format - I agree JSON makes sense, I think versioning might make sense too, so we can import older models and other DB data as needed.  I imagine this would be just a "version: 1.0" in the file, or similar.  Location on disk will likely depend on where we are being run from, but can you explain what locations you are asking about in more detail?  Perhaps providing a list of questions would be helpful.  I will reply in the new thread #15 with this response again to continue this conversation.
- I reviewed the quantization_methods seed data, but this is not my expertise, so I can't provide much additional confidence other than to say that none of the values seem obviously out of place.  I would suggest verifying it online against another source.  Perhaps you can see file sizes of some models without downloading the model weights, then do the math to ensure it matches the estimate for a model of that size within some tolerance for error (within 1% tolerance I think is reasonable).

---

### #13 comment 4522610944 — @AlbinoGeek — 2026-05-22T20:49:49Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4522610944>

**Bastion (J-5) — bakeoff#13 schema + design revision — 222048ZMAY26**

@gissf1 — addressing all points from latest message.

---

## 1. Configuration file — mtime delay and runner settings

No config file defined yet. Proposing one now.

Format: YAML (human-edited config; JSON reserved for data files — see §2). Default location: `bakeoff.yaml` in the working directory. Override via `BAKEOFF_CONFIG` env var or `--config` flag.

```yaml
queue:
  pending_dir: ./queue/pending
  completed_dir: ./queue/completed
  file_mtime_min_age_seconds: 30     # configurable; 30 is the default
  poll_interval_seconds: 60
  reaper_probability: 0.10

runner:
  standalone_results_dir: ./results
  log_level: info

database:
  url: ""   # required for queue worker mode; absent/empty = standalone
```

`file_mtime_min_age_seconds: 30` is the configurable knob you asked for. High-throughput operators can reduce it; conservative environments can raise it.

---

## 2. Model disk file format — JSON, with versioning

JSON over YAML for model disk files. Rationale: bakeoff-results web interface reads these directly — JavaScript `JSON.parse` is native. YAML requires a parser dependency on the frontend. YAML stays for human-edited config files (above); JSON is the data exchange format.

Versioning: `"schema_version": 1` at root. Loader rejects unsupported versions with a clear error — clean upgrade path without silent schema drift.

Revised model disk file (additions from prior revision in comments):

```json
{
  "schema_version": 1,
  "model_id": "<uuid-v5>",
  "name": "llama3-8b",
  "architecture": "Dense",
  "parameter_count_b": 8.0,
  "active_parameter_count_b": 8.0,
  "quantization": "q4_k_m",
  "model_hash": "<sha256-of-model-source-file>",
  "model_source_mtime": "2026-05-20T10:00:00Z",
  "model_source_size": 4924194816,
  "sources": [
    { "type": "huggingface", "repo": "meta-llama/Meta-Llama-3-8B-Instruct" }
  ]
}
```

`model_hash`, `model_source_mtime`, `model_source_size` — all added per your request. Runner startup (standalone mode): stat the model source file on disk; compare mtime and size against descriptor. Mismatch → warn (default) or abort (`--strict-cache` flag, queue worker default). Detects drift between cached weights and the descriptor without requiring a full re-hash.

Terminology locked as you defined it:
- **models table** — DB table.
- **model disk file** — JSON descriptor (schema above).
- **model source file** — weights on disk (GGUF or similar). Not a structured file this system writes; the runner reads it.

---

## 3. model_id — UUID v5 (deterministic, distributed-safe)

Your intuition is correct: deterministic generation is better than random for consistent cross-system IDs.

Recommendation: **UUID v5** (name-based SHA-1). One fixed namespace UUID committed to the codebase as a project constant. Name string concatenates the canonical fields that uniquely identify a model version:

```
name + ":" + architecture + ":" + quantization + ":" + str(active_parameter_count_b)
```

Same inputs → same UUID on any host, without a DB round-trip. The `model_id` in the disk file and in the `models` table are identical by construction. Distributed generation without collision risk (collisions require identical canonical strings — not a practical concern for distinct models).

`models.model_id` type: `UUID PRIMARY KEY` (replace `SERIAL`). DB insert uses the pre-generated UUID from the descriptor file — no autoincrement needed.

Namespace UUID: generate once and commit as a project constant before first model ingestion pipeline is written. Low risk to defer until that sprint.

---

## 4. runners.status — TEXT + CHECK retained (for now)

ENUM is cleaner when the value set is closed and stable. TEXT + CHECK is preferred while the schema is actively evolving.

Postgres ENUM: adding values requires `ALTER TYPE ... ADD VALUE`. In PG 12+ this is transactional, but removing values or changing order still requires a new type. TEXT + CHECK: changing the value set is one line in a migration. Simpler to iterate during design phase.

Recommendation: stay TEXT + CHECK now. Convert to ENUM when the status set is stable and Phase 1 schema work closes. Mark as a deferred cleanup item.

---

## 5. updated_at in run_queue — retained

Confirmed: `updated_at` is not replaced by `retry_after`. They serve distinct purposes:

- **`retry_after TIMESTAMPTZ`** — claim gate. "Do not pick this up before this time." Set on FAILED transitions. NULL = no delay constraint.
- **`updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`** — audit trail. "When was this row last modified." Always set; updated on every state transition via trigger.

Both columns coexist. No change needed to the current schema for `updated_at`.

---

## 6. File-backing scope

Affirm for model data; bound for operational state.

**File-backed (priority):**
- `models` — covered by model disk file (§2). Standalone runner does not need a live DB for model properties.
- `quantization_methods` — seed data ships bundled with the binary (embedded constant or JSON asset). One bundled lookup for the whole runner; no per-model file for quantization data.

**Not file-backed:**
- `runners` — live heartbeat/state. Regenerated on restart; no recovery value.
- `run_queue` live state (CLAIMED/IN_PROGRESS transitions) — not file-backed. DR path re-enqueues from `queue/pending/` descriptors.

For `prompts` and `run_model_metrics`: results written to disk on COMPLETE (already in the design). Prompt descriptor file format follows the same pattern as model disk files. Defer until scoring architecture is defined.

---

## Revised schema delta (incremental from 221322Z)

```sql
-- models: UUID primary key, add hash + source tracking
-- (Replace SERIAL model_id with UUID; exact migration depends on whether table is empty)
ALTER TABLE models ADD COLUMN model_hash TEXT;
ALTER TABLE models ADD COLUMN model_source_mtime TIMESTAMPTZ;
ALTER TABLE models ADD COLUMN model_source_size BIGINT;

-- run_queue: confirm updated_at trigger present (no schema change)
-- runners.status: TEXT + CHECK retained, no change
```

`model_id` type change (`SERIAL` → `UUID`) requires a migration that drops and re-adds the PK if the table is not empty. Clean now while the DB has no production data.

---

## Open questions remaining

1. **Namespace UUID** — generate and commit the project-constant namespace UUID before first model ingestion pipeline. No dependency; can be done independently.
2. **`--strict-cache` default** — warn (standalone) vs. abort (queue worker) on model source file drift. Preference?
3. **quantization_methods seed SQL sign-off** — Bastion will cross-check multipliers against public GGUF model file sizes (llama.cpp MODELS.md + a known-size model download). Follow-up comment when confirmed.
4. **Prompt descriptor file format** — defer to scoring architecture thread.

— Bastion

---

### #13 comment 4524200643 — @gissf1 — 2026-05-23T04:52:22Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524200643>

1. Configuration file - how can you say there is no configuration file defined?  there is a config.yaml in the bakeoff project root.  That is a good starting point I believe.  Where is that format defined?  We may need to add new configuration entries or refine what is there though.  I believe we can import all the suggested configuration options you would have put in bakeoff.yaml into config.yaml instead.

2. Model disk file format - agreed; JSON, with versioning.  Can you show the schema differences between the models table and the model disk file?  I want to be sure we aren't missing anything important and seeing them side-by-side would help me to verify that.

3. model_id - I think your suggestion is a good start, but we should also include the inactive parameter_count_b, model_source_size and possibly the model_hash among the components used to generate the UUID.

Namespace UUID: I'm not sure what you mean here.  Can you explain this a bit more?

4. runners.status - OK, since the usage is nearly identical, I'm open to TEXT+CHECK for now until our schema is more stable.  And yes, please ensure that you mark this is a deferred cleanup item to keep track of it.

6. File-backing scope:
   - quantization_methods: this should be a JSON asset so we can populate the database with the same data used by the runner when pulling from files.  We don't want to maintain duplicate data to avoid divergences.
   - I would like to comment that the test results file should include a duplicate of all data stored in run_model_metrics so it can be self contained and distributed to others as desired without any external dependencies other than potentially our web application for viewing results.  Other than that, I'm fine with deferring whatever is not yet needed.

Revised schema delta / models table: there should be no data in it currently other than the queued or test data we entered.  If the database exists (I don't believe it does yet) you can list any data in it and we can determine from there whether it is important or something we can just regenerate from existing sources.  If nothing important is in it, you can clear it now.  A related note, I don't think we absolutely need the model_source_mtime in the models table, but to keep the synchronization with the model disk file simpler, I'm open to retaining it there.

Answers to "Open questions remaining":
1. Namespace UUID: You can generate one, or do you want me to generate it?  If so, how do you propose I do so?  Do you just need random numbers or what?
2. cache should be strict by default.  if the file changed size, it's clearly not the same file.  If it changed mtime, but size is the same, we can calculate the current hash of the data file to verify if it changed or not.  If the mtime changed, but content did not, we can simply update the mtime in the model disk file to match the new mtime of the model weight source file.
3. quantization_methods seed SQL sign-off: yup, please notify me when you have answers confirmed, but I expect your results will be correct.
4. In general, it should be similar to the prompts table, but agreed to defer it to scoring thread.

---

### #13 comment 4524304026 — @AlbinoGeek — 2026-05-23T05:40:34Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4524304026>

**Bastion response — 230536ZMAY26**

@gissf1 — addressing each item in order.

---

**1. Configuration file**

`config.yaml` in the project root IS the config file. Format is defined by the YAML structure consumed in `bench/config.py` and documented through the runner's key usage. New entries merge into `config.yaml` — no separate `bakeoff.yaml`. The `signing` stanza (shipped in bakeoff#16) follows the existing pattern:

```yaml
signing:
  enabled: true
  key_path: "~/.bakeoff/runner_key.pem"
  runner_id: "amd-8060s"   # defaults to hardware.id if omitted
```

---

**2. Model disk file vs. `models` table — side by side**

| Field | `models` table | `models/<uuid>.json` |
|-------|---------------|----------------------|
| `schema_version` | — | integer (e.g. `1`) |
| `model_id` | `SERIAL PK` | `uuid` (UUID5, see §3) |
| `name` | `TEXT` | `name` (identical) |
| `creator_id` | `INT FK → creators` | `creator: {name, display_name, homepage, service_identifiers}` (embedded object) |
| `model_hash` | `TEXT UNIQUE` | `model_hash` (identical) |
| `parameter_count_b` | `FLOAT` | `parameter_count_b` (identical) |
| `active_parameter_count_b` | `FLOAT` | `active_parameter_count_b` (identical) |
| `architecture` | `TEXT` | `architecture` (identical) |
| `context_length_default/min/max` | 3 INT cols | `context_length: {default, min, max}` (nested object) |
| `file_format` | `TEXT` | `file_format` (identical) |
| `quantization` | `TEXT` | `quantization` (identical) |
| `min_vram_mb` | `INT` | `min_vram_mb` (identical) |
| `release_date` | `DATE` | ISO8601 date string (identical semantics) |
| `version` | `TEXT` | `version` (identical) |
| `description` | `TEXT` | `description` (identical) |
| `predecessor_model_id` | `INT FK → models` | `predecessor_uuid` (UUID of predecessor — no FK) |
| `sources` | — (in `model_sources`) | `sources: [{source_type, url, source_metadata, updated}]` (embedded array) |
| `created_at` | — | TIMESTAMPTZ (disk-only audit field) |
| `updated_at` | — | TIMESTAMPTZ (disk-only audit field) |

Divergences are minimal: FK columns collapse to embedded objects/UUID references; `context_length` is nested for readability; disk file gains `schema_version`, `created_at`, `updated_at`.

---

**3. model_id UUID composition**

Agreed on including `parameter_count_b`, `model_source_size`, and `model_hash`. Proposed two-tier strategy:

- **Primary (hash known):** `UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash)` — model_hash is the deduplication ground truth; same hash = same model regardless of source.
- **Provisional (hash not yet computed):** `UUID5(BAKEOFF_MODEL_NAMESPACE, source_url + "|" + str(parameter_count_b) + "|" + str(model_source_size))` — uses the three fields you specified. Provisional UUID is promoted to the hash-based UUID once weights are pulled and `model_hash` is computed.

`model_source_mtime` excluded from UUID components — not absolutely needed; adds fragility to stable identity.

---

**4. Namespace UUID — explanation**

UUID5 requires a fixed "namespace UUID" — a project constant that scopes the hash space so bakeoff UUIDs cannot collide with UUIDs from unrelated tools. For this project, one namespace is committed as a constant:

```python
# bench/constants.py
BAKEOFF_MODEL_NAMESPACE = UUID("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
```

This is a one-time randomly generated value that never changes. It is not user input. Bastion generates and commits it to `bench/constants.py` — no action required from you.

---

**5. runners.status — deferred cleanup tracked**

Will add a `-- TODO(deferred P2): migrate TEXT+CHECK → ENUM once schema stabilizes (bakeoff#13)` comment to the `runners` table in `schema.sql`.

---

**6. File-backing scope**

- **quantization_methods → JSON asset:** Agreed. File at `seeds/quantization_methods.json`. DB is seeded from this file; runner reads it directly in standalone operation. No duplicate maintenance.
- **Test results include full `run_model_metrics` duplicate:** Confirmed. The signed result envelope (`result.records`) carries `run_model_metrics` fields as first-class data, not nested or derivable-only. Self-contained for distribution — no external DB dependency to read output files.

---

**7. Schema delta — current DB state**

If the database has not been initialized with production data (most likely at this stage), the `model_id` type change (`SERIAL → UUID`) applies to a fresh schema with no migration needed. Flag here if you have data to preserve and Bastion will write a proper migration. Otherwise proceeding with a clean UUID-based schema.

---

**Answers to open questions**

1. **Namespace UUID:** Bastion generates and commits to `bench/constants.py`. No action required from you.
2. **Cache strict default:** Policy confirmed — (a) size changed → INVALID, re-ingest required; (b) mtime changed, size unchanged → recompute hash, compare; (c) mtime changed, content unchanged → update mtime in model disk file, no re-ingest. Strict by default.
3. **quantization_methods seed SQL:** Will cross-check multipliers against llama.cpp MODELS.md and a reference known-size model. Will post confirmation here.
4. **Prompt descriptor format:** Deferred to scoring architecture thread per prior agreement.

— Bastion

---

### #13 comment 4570344944 — @AlbinoGeek — 2026-05-29T03:52:50Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4570344944>

**Bastion (J-5) — bakeoff#13 close-out — 290347ZMAY26**

Design consensus reached and the structural deliverable has shipped. Closing as **design-complete**.

## Settled (per @gissf1)
- Capability-based runner matching (VRAM declared by runner, calculated not stored); `hardware_tag` dropped.
- State machine: `PENDING → CLAIMED → IN_PROGRESS → COMPLETE/FAILED/CANCELLED`; stale-claim reaper.
- Retry: `max_attempts = 5`; `retry_after` + priority increment `5 × attempt_count`.
- Absolute scores → no baseline-tier dependency.
- Two-dir disk-DR layout (`pending/` + `completed/`) with `source_file`.
- **Standalone runner is the default mode**; queue-worker is opt-in.

## Shipped
- `run_queue` table + `run_queue_claim_idx` (`FOR UPDATE SKIP LOCKED` claim idiom) landed in `schema/schema.sql` (commit `dac26d5`).

## Follow-on (tracked separately, not blocking close)
- Runner does not yet *consume* `run_queue`; the opt-in queue-worker loop in `runner.py` is an implementation task for the sprint that enables multi-runner operation. Standalone default behaviour is unchanged.

Hardware-assignment scope lives in #22. Closing.

— Bastion

---

### #13 comment 4570546649 — @AlbinoGeek — 2026-05-29T04:26:21Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4570546649>

**Bastion (J-5) — bakeoff#13 implementation landed — 290347ZMAY26**

The opt-in disk-backed queue designed here is now implemented on `main` (merge `0d0288e`):
- `bench/queue.py` — `pending/` + `completed/` two-dir DR layout; `claim()` is race-safe via rename-as-mutex with a post-rename PENDING guard (stress-proven: 16 workers × 6,400 attempts, 0 double-claims); retry with `5×attempt_count` backoff + priority bump; terminal `FAILED` after `max_attempts=5`; `reap_stale_claims()`.
- `bench/store.py` — atomic JSON record I/O under `BAKEOFF_DATA_DIR`, UUID5 helpers from `bench/constants.py`.
- Standalone runner default unchanged; queue is strictly opt-in (invariant recorded in AGENTS.md).
- Spec: `specs/disk-persistence-layer/`. 38 new tests, full suite 413 green.

— Bastion

---

