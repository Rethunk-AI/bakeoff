# Topic: process_meta

Consolidated chat history (chronological, ascending comment-id). 22 entries. Verbatim quotes; attribution in each header. **#38 thread excluded from this variant.**

## Source entries (provenance TOC)

Entries used to build this topic and its summary. (Not migrated into `topic_summary/`; audit reference only.)

| ticket | entry | author | posted | url |
|---|---|---|---|---|
| #8 | comment 4465943831 | @AlbinoGeek | 2026-05-16T06:09:45Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4465943831> |
| #8 | comment 4486952413 | @AlbinoGeek | 2026-05-19T10:45:07Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4486952413> |
| #8 | comment 4495333862 | @AlbinoGeek | 2026-05-20T06:38:24Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4495333862> |
| #12 | comment 4503469716 | @gissf1 | 2026-05-20T23:29:50Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503469716> |
| #12 | comment 4503669210 | @AlbinoGeek | 2026-05-21T00:11:41Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503669210> |
| #14 | comment 4506576614 | @gissf1 | 2026-05-21T09:11:19Z | <https://github.com/Rethunk-AI/bakeoff/issues/14#issuecomment-4506576614> |
| #14 | comment 4506717026 | @AlbinoGeek | 2026-05-21T09:26:14Z | <https://github.com/Rethunk-AI/bakeoff/issues/14#issuecomment-4506717026> |
| #12 | comment 4507779592 | @AlbinoGeek | 2026-05-21T11:33:18Z | <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4507779592> |
| #14 | comment 4507781031 | @AlbinoGeek | 2026-05-21T11:33:31Z | <https://github.com/Rethunk-AI/bakeoff/issues/14#issuecomment-4507781031> |
| #13 | comment 4519569754 | @AlbinoGeek | 2026-05-22T14:26:08Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4519569754> |
| #8 | comment 4519663607 | @AlbinoGeek | 2026-05-22T14:39:09Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4519663607> |
| #8 | comment 4522155645 | @gissf1 | 2026-05-22T19:33:07Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4522155645> |
| #8 | comment 4522303209 | @AlbinoGeek | 2026-05-22T19:53:06Z | <https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4522303209> |
| #15 | comment 4533235030 | @gissf1 | 2026-05-25T09:40:32Z | <https://github.com/Rethunk-AI/bakeoff/issues/15#issuecomment-4533235030> |
| #15 | comment 4533402406 | @gissf1 | 2026-05-25T10:08:22Z | <https://github.com/Rethunk-AI/bakeoff/issues/15#issuecomment-4533402406> |
| #15 | comment 4533410862 | @AlbinoGeek | 2026-05-25T10:09:50Z | <https://github.com/Rethunk-AI/bakeoff/issues/15#issuecomment-4533410862> |
| #13 | comment 4533665446 | @gissf1 | 2026-05-25T10:49:42Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4533665446> |
| #13 | comment 4533761289 | @AlbinoGeek | 2026-05-25T11:06:36Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4533761289> |
| #13 | comment 4570344944 | @AlbinoGeek | 2026-05-29T03:52:50Z | <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4570344944> |
| #22 | comment 4574254854 | @AlbinoGeek | 2026-05-29T11:12:10Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574254854> |
| #22 | comment 4574320981 | @gissf1 | 2026-05-29T11:20:32Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574320981> |
| #22 | comment 4574341430 | @AlbinoGeek | 2026-05-29T11:22:52Z | <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574341430> |

---

## Consolidated thread

### #8 comment 4465943831 — @AlbinoGeek — 2026-05-16T06:09:45Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4465943831>

**Bastion status update — 150000ZMAY26**

Conducted a retrospective audit against this thread. The following commitments from our discussion were not implemented and are now being tracked:

| Commitment | Status | Tracking |
|-----------|--------|---------|
| `hardware_id` in records | ❌ Not implemented | #9 |
| `peak_vram_mb` per call | ❌ Not implemented | #9 |
| `gpu_sm_utilization_pct` per call | ❌ Not implemented | #9 |
| `tflops_utilization_pct` | ❌ Not implemented | #10 |
| Optional `cpu_time_user_ms` / `cpu_time_sys_ms` | ❌ Not implemented | #10 |
| `cost_usd` removed from storage (display-derived) | ❌ Still stored | implementing now |
| Three-category normalized schema | ⚠️ Not applicable to current flat-JSON runner; tracked conceptually in #9/#10 |

**What was delivered:** `latency_s` (wall clock), `ttft_s` (time-to-first-token), `tokens_per_sec`, `energy_wh`, `quality_heuristic`.

**Next:** implementing #9 first (hardware block + VRAM/SM collection), then #10 (TFLOPS + CPU timing), then marking both closed.

— Bastion

---

### #8 comment 4486952413 — @AlbinoGeek — 2026-05-19T10:45:07Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4486952413>

**Bastion status check — 191008ZMAY26**

@gissf1 — circling back on the naming decision from 182139ZMAY26.

You asked for name selection between the options for the dual GPU timing fields. Recommendation was:

| Field | Proposed | Alt A | Alt B |
|-------|----------|-------|-------|
| CUDA/ROCm event API | `gpu_seconds_event` | `gpu_seconds_kernel` | `gpu_seconds_direct` |
| Utilization-weighted | `gpu_seconds_sampled` | `gpu_seconds_computed` | `gpu_seconds_weighted` |

Bastion recommendation: **`gpu_seconds_event` + `gpu_seconds_sampled`** — most self-documenting; "event" names the acquisition mechanism, "sampled" names the polling+compute approach.

On `gpu_data_transfer_seconds`: recommendation was to **defer** — gap between `wall_clock_seconds` and `gpu_seconds_event` already proxies transfer overhead, and instrumenting it requires modifying the inference runner hot path.

Waiting on your call to proceed with schema update. One word (e.g., "event/sampled approved") is sufficient to unblock.

— Bastion

---

### #8 comment 4495333862 — @AlbinoGeek — 2026-05-20T06:38:24Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4495333862>

**Bastion response — 200636ZMAY26**

@gissf1 — names approved. Proceeding.

**Schema update**

`run_hardware_metrics` fields updated:

```
gpu_event_seconds       FLOAT NULLABLE    -- CUDA/ROCm cudaEventElapsedTime() bracketing inference kernel
gpu_weighted_seconds    FLOAT NULLABLE    -- wall_clock_seconds x mean(gpu_sm_utilization_pct / 100)
```

Harness field names updated to match. Schema documentation updated with the `wall_clock_seconds - gpu_event_seconds` proxy note for data transfer overhead.

**Thread status — complete**

All pending topics addressed:

| Topic | Status |
|-------|--------|
| GPU timing dual fields (`gpu_event_seconds` / `gpu_weighted_seconds`) | Approved and updated |
| `gpu_data_transfer_seconds` | Deferred to Phase 2 |
| Four-table hardware schema | Ratified |
| `system_software` table | Added |
| Unit consistency (all seconds) | Applied |

Schema design for this thread is complete. Implementation sub-issues in the bakeoff repo carry the remaining work.

— Bastion

---

### #12 comment 4503469716 — @gissf1 — 2026-05-20T23:29:50Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503469716>

Are there any additional pending questions or requirements from me at this time on this thread?

---

### #12 comment 4503669210 — @AlbinoGeek — 2026-05-21T00:11:41Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503669210>

**Bastion response — 210001ZMAY26**

@gissf1 — no outstanding requirements from you on this thread at this time.

All Q1–Q5 questions have been addressed across prior rounds. Schema deltas are captured and agreed:

```
creators        — creator_id, name, display_name, homepage, service_identifiers JSONB
source_types    — source_type_id, name
models          — model_id, name, creator_id FK, model_hash UNIQUE, parameter_count_b FLOAT,
                  active_parameter_count_b FLOAT, architecture, context_length_default/min/max,
                  file_format, quantization, min_vram_mb, release_date, version, description,
                  predecessor_model_id FK
model_sources   — source_id, model_id FK, source_type_id FK, url, source_metadata JSONB
task_categories — category_id, name UNIQUE, description
tasks           — task_id, name, category_id FK, parent_id FK (nullable), sort_order,
                  description, grader_script, grader_script_commit, natural_key_hash UNIQUE
prompts         — prompt_id, task_id FK, file_path, git_commit_hash, content_sha256 UNIQUE,
                  content_length_bytes, version, release_date, is_prerelease, difficulty INT,
                  modified_at
runs            — run_id UUID PK, submitted_at, publisher_id, runner_version, prompt_git_hash
run_model_metrics — run_id FK, prompt_id FK, model_id FK, score FLOAT, pass_fail BOOLEAN,
                    failure_reason TEXT, gflops_per_token FLOAT
                    PK: (run_id, prompt_id, model_id)
```

Pending forward actions (awaiting direction):
- Implementation sub-issues for each table/component
- Testing-queue issue (previously stated as "opening now" — confirm status)

Standing by for next direction.

— Bastion

---

### #14 comment 4506576614 — @gissf1 — 2026-05-21T09:11:19Z
<https://github.com/Rethunk-AI/bakeoff/issues/14#issuecomment-4506576614>

correct, you may now proceed.

---

### #14 comment 4506717026 — @AlbinoGeek — 2026-05-21T09:26:14Z
<https://github.com/Rethunk-AI/bakeoff/issues/14#issuecomment-4506717026>

**Bastion response — 210925ZMAY26**

@gissf1 — acknowledged. Design locked:

- `ollama` added to `source_types`
- `source_metadata` JSONB — flat, provider identity + stats co-located, no sub-objects
- `updated` TIMESTAMPTZ — new native column, single acquisition timestamp for all data in the row (including `model_hash`)
- Multi-query consistency protocol: 1-minute window per round, earliest timestamp wins, retry on value divergence, 5-minute hard cap

Standing by for implementation sub-issues when directed.

— Bastion

---

### #12 comment 4507779592 — @AlbinoGeek — 2026-05-21T11:33:18Z
<https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4507779592>

Schema design complete. schema/schema.sql committed (Rethunk-AI/bakeoff@23b4286). All agreed columns, types, and FK relationships implemented per the design thread in this issue. Closing.

---

### #14 comment 4507781031 — @AlbinoGeek — 2026-05-21T11:33:31Z
<https://github.com/Rethunk-AI/bakeoff/issues/14#issuecomment-4507781031>

Design finalized and implemented. schema/schema.sql includes `updated TIMESTAMPTZ` column and flat `source_metadata JSONB` per final design (ollama added to source_types; no sub-objects; multi-query consistency protocol recorded in this thread). Closing.

---

### #13 comment 4519569754 — @AlbinoGeek — 2026-05-22T14:26:08Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4519569754>

**Bastion (J-5) — bakeoff#13 thread housekeeping — 221424ZMAY26**

Two open questions from this thread have been spun off into dedicated issues:

- **#15** — Model descriptor file format and capability ingestion (schema versioning, disk location, ingestion pipeline)
- **#16** — Result signing and verification scheme (Ed25519 recommendation, per-runner keys, sidecar format)

Remaining work in this thread: quantization_methods seed SQL (follow-up comment below).

— Bastion

---

### #8 comment 4519663607 — @AlbinoGeek — 2026-05-22T14:39:09Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4519663607>

**Bastion update — 221438ZMAY26**

@gissf1 — schema design for this thread is complete. Delivering the final implementation step.

---

**Field rename — committed e4a6983**

All output field names now match the ratified schema from this thread:

| Old name | New name | Notes |
|----------|----------|-------|
| `latency_s` | `wall_clock_seconds` | unit unchanged (seconds) |
| `ttft_s` | `seconds_to_first_token` | unit unchanged (seconds) |
| `tokens_per_sec` | `tokens_per_second` | value unchanged |
| `cpu_time_user_ms` | `cpu_seconds_user` | converted ÷1000 at write time |
| `cpu_time_sys_ms` | `cpu_seconds_sys` | converted ÷1000 at write time |

All report field extractions and the JS rollup template updated to match. 56 tests pass.

`gpu_event_seconds` is present in every record but always `None` — the CUDA/ROCm event API path is not yet wired. The field exists to keep the schema stable; it will be populated when that path is implemented.

`gpu_weighted_seconds` is fully live — populated when NVML is available (`wall_clock_seconds × mean(gpu_sm_utilization_pct / 100)`).

---

**Schema design status**

| Component | Status |
|-----------|--------|
| Four-table hardware schema (gpu_hardware, system_hardware, system_software, system_gpu_link + interface_type) | Ratified — awaiting implementation sub-issues |
| run_hardware_metrics field names | Complete |
| run_model_metrics schema | Open in bakeoff#12 |
| CUDA event API for gpu_event_seconds | Deferred to Phase 2 |
| Multi-GPU and multi-system distribution | Deferred to Phase 2 |

---

Ready to open implementation sub-issues for the four-table hardware schema when directed.

— Bastion

---

### #8 comment 4522155645 — @gissf1 — 2026-05-22T19:33:07Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4522155645>

That description sounds good.  You may proceed with opening sub-issues for this issue whenever you are ready and feel that prerequisites are met.

---

### #8 comment 4522303209 — @AlbinoGeek — 2026-05-22T19:53:06Z
<https://github.com/Rethunk-AI/bakeoff/issues/8#issuecomment-4522303209>

**Bastion response — 221947ZMAY26**

Sub-issues opened. Implementation dependency order:

| Issue | Title | Depends on |
|-------|-------|-----------|
| #17 | `interface_type` lookup table + seed data | — |
| #18 | `gpu_hardware` table + auto-detection | #17 |
| #19 | `system_hardware` + `system_software` tables | — |
| #20 | `system_gpu_link` join table + PCIe detection | #17, #18, #19 |
| #21 | `run_hardware_metrics` FK wiring | #20, #19 |

#17 and #19 have no dependencies and can proceed in parallel. #18 requires #17 first. #20 requires #17 + #18 + #19. #21 is the terminal step.

Standing by for implementation direction or priority assignment.

— Bastion

---

### #15 comment 4533235030 — @gissf1 — 2026-05-25T09:40:32Z
<https://github.com/Rethunk-AI/bakeoff/issues/15#issuecomment-4533235030>

Agreed; proceed and let me know when you are ready for my opinions, comments, and/or review.

---

### #15 comment 4533402406 — @gissf1 — 2026-05-25T10:08:22Z
<https://github.com/Rethunk-AI/bakeoff/issues/15#issuecomment-4533402406>

Acknowledged.

---

### #15 comment 4533410862 — @AlbinoGeek — 2026-05-25T10:09:50Z
<https://github.com/Rethunk-AI/bakeoff/issues/15#issuecomment-4533410862>

**Bastion — 251009ZMAY26**

@gissf1 — standing by. Hardware disk persistence thread is open at bakeoff#22 when you are ready.

— Bastion

---

### #13 comment 4533665446 — @gissf1 — 2026-05-25T10:49:42Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4533665446>

Acceptable and acknowledged.

---

### #13 comment 4533761289 — @AlbinoGeek — 2026-05-25T11:06:36Z
<https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4533761289>

**Bastion — 251103ZMAY26**

@gissf1 — acknowledged. Stub format and predecessor tracking policy settled. Standing by for ingest pipeline scoping when the queue design in this thread is ready to close.

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

### #22 comment 4574254854 — @AlbinoGeek — 2026-05-29T11:12:10Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574254854>

**Bastion response — 291107ZMAY26**

@gissf1 — confirming all points.

**table_migrations confirmed.** "schema_hops" rejected; "table_migrations" preferred. Noted and carried into the new thread.

**Merge direction noted.** table_migrations fields into schema_versions rather than a separate table; schema_version_history eliminated with its fields (schema_version_id FK + applied_at) moved onto schema_tables instead. The open question of exactly how the merge should be structured (one row per table per version? JSONB array? other?) is the first item in the new thread.

**UUID namespace strategy noted.** "UUID changes to any table can simply be migrated to a new table using a new UUID namespace" — flagged as an open design question in the new thread rather than assumed confirmed. Needs a clear answer before any implementation can proceed.

**No steps taken.** Per your direction: no table designs approved or implemented yet. The new thread is the venue for resolving the remaining open questions before anything is built.

**New issue:** https://github.com/Rethunk-AI/bakeoff/issues/25 — "Migration framework: schema_versions, table_migrations, idempotent DDL, Go template scripting." Shows the before/after table designs from this thread and the open questions to resolve next.

Ready to close bakeoff#22 if you confirm this thread's scope is now fully handed off. Or let me know if there are remaining items to resolve here first.

— Bastion

---

### #22 comment 4574320981 — @gissf1 — 2026-05-29T11:20:32Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574320981>

If you feel there are no more remaining questions on this thread, I'm ok with either holding this open until #25 is resolved or closing this as done; whatever you believe is best for managing the project.

---

### #22 comment 4574341430 — @AlbinoGeek — 2026-05-29T11:22:52Z
<https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574341430>

**Bastion — closing — 291122ZMAY26**

All Q1–Q5 items from this thread are resolved:

| Q | Status |
|---|--------|
| Q1 — interface_type seed | Shipped `3b0bce4` |
| Q2 — Hardware UUID field list | Confirmed |
| Q3 — GPU snapshot layout | Confirmed |
| Q4 — Run file embeds full hardware block | Confirmed |
| Q5 — Snapshot timing per session | Confirmed |

Migration framework design continues in #25. Closing this thread.

— Bastion

---

