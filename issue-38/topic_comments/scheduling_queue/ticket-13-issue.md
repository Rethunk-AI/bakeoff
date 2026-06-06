---
ticket: 13
type: issue-body
author: AlbinoGeek
posted: 2026-05-20T07:36:25Z
topic: scheduling_queue
url: https://github.com/Rethunk-AI/bakeoff/issues/13
title: Design and implement test run scheduling queue
---

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
