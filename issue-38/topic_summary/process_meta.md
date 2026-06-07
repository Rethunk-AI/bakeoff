# Summary: process_meta

## Final state

Thread covers process conventions and triage rulings across the schema-spec issue set (#8, #12–15, #17–22). Key settled conventions:

**Naming conventions locked:** All timing fields use full-word seconds suffixes (`wall_clock_seconds`, `seconds_to_first_token`, `tokens_per_second`, `cpu_seconds_user`, `cpu_seconds_sys`); CPU fields converted from ms at write time (per 4519663607). GPU timing dual-field naming settled as `gpu_event_seconds` (CUDA/ROCm event API) + `gpu_weighted_seconds` (utilization-weighted proxy) (per 4495333862).

**Design-then-implement sequencing:** No table is implemented until @gissf1 explicitly approves the design in-thread. "You may proceed" / "correct, you may now proceed" are the unlock phrases (per 4506576614, 4533235030). Bastion does not self-authorize implementation.

**Sub-issue dependency ordering as a first-class deliverable:** When an implementation spans multiple tables, Bastion opens a dependency-ordered sub-issue list before any implementation begins and presents it for direction (per 4522303209, #17→#18→#19→#20→#21 chain).

**Spin-off issues for scope that exceeds a thread:** Topics that grow beyond the parent issue's scope are split into dedicated child issues rather than resolved inline (per 4519569754: #15 for model descriptor format, #16 for result signing; #25 for migration framework).

**"table_migrations" over "schema_hops":** Naming vote settled in-thread before the new issue was opened (per 4574254854).

**Queue design settled (per 4570344944):** Standalone runner is the default mode; queue-worker (`run_queue`) is opt-in. State machine: `PENDING → CLAIMED → IN_PROGRESS → COMPLETE/FAILED/CANCELLED`. Retry: `max_attempts = 5`, priority increment `5 × attempt_count`. Capability-based runner matching via declared VRAM; `hardware_tag` dropped.

**Hardware sub-system factored into four tables:** `gpu_hardware`, `system_hardware`, `system_software`, `system_gpu_link` + `interface_type` lookup — ratified in #8 thread, then decomposed into five ordered sub-issues.

**Migration framework handed off to #25:** `schema_versions` / `table_migrations` merge direction established; no implementation authorized until #25 resolves open structural questions (per 4574254854, 4574341430).

## Notable / unusual decisions

- **`cost_usd` removed from storage (display-derived only):** Explicitly called out as a schema discipline — cost is never stored, always computed at display time. Audited and flagged as not yet implemented at the thread's outset (per 4465943831), implying it was a deliberate prior ruling being enforced, not a new decision.

- **`gpu_data_transfer_seconds` deferred on the grounds of proxy sufficiency:** The gap between `wall_clock_seconds` and `gpu_event_seconds` already proxies transfer overhead, so instrumenting it was explicitly rejected to avoid modifying the hot path (per 4486952413). Useful precedent: "existing delta is good enough" as a deferral rationale.

- **`gpu_event_seconds` present-but-null in schema from day one:** Field exists and is written as `None` in every record until the CUDA event API is wired, keeping schema stable across implementations (per 4519663607). Downstream consumers must treat null as "not yet collected," not "unsupported."

- **Multi-query consistency protocol for model source data:** 1-minute window per round, earliest timestamp wins, retry on value divergence, 5-minute hard cap (per 4506717026). Prevents stale/split-brain row writes during concurrent source queries.

- **Absolute scores, no baseline-tier dependency:** Score schema carries raw values only; any baseline comparison is a query-time concern, not stored (per 4570344944). Keeps the schema simpler and avoids coupling scores to a baseline versioning problem.

## Open / unresolved

- **Migration framework structural shape:** How `table_migrations` merges into `schema_versions` is unresolved — one row per table per version, JSONB array, or another form? (per 4574254854). No implementation may begin until this is answered in #25.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574254854>

- **UUID namespace migration strategy:** "UUID changes to any table can simply be migrated to a new table using a new UUID namespace" was flagged as an open design question requiring a clear answer before any implementation (per 4574254854). Resolution venue is #25.
  - address: #22, <https://github.com/Rethunk-AI/bakeoff/issues/22#issuecomment-4574254854>

- **Queue-worker implementation (runner consuming `run_queue`):** `run_queue` table is shipped; the opt-in queue-worker loop in `runner.py` is not yet implemented and was explicitly deferred as a sprint task for enabling multi-runner operation (per 4570344944).
  - address: #13, <https://github.com/Rethunk-AI/bakeoff/issues/13#issuecomment-4570344944>

- **Testing-queue issue status:** Bastion noted it was "previously stated as 'opening now'" and requested confirmation of status (per 4503669210); no confirmation appears in this thread.
  - address: #12, <https://github.com/Rethunk-AI/bakeoff/issues/12#issuecomment-4503669210>

## Cross-topic links

- **hardware** — four-table hardware schema (`gpu_hardware`, `system_hardware`, `system_software`, `system_gpu_link`, `interface_type`) finalized here; implementation sub-issues #17–#21 carry the build work.
- **run_queue** — queue design settled in #13 thread; `run_queue` table shipped; worker loop deferred.
- **model_sources / source_types** — `ollama` seeded, `source_metadata` JSONB flat-layout and `updated` TIMESTAMPTZ settled here (via #14 thread).
- **schema_versions / table_migrations** — merge direction established here (#22 thread); open structural questions handed off to #25 (migration framework topic).
- **model_descriptors** — scope split to #15; not resolved within this thread.
- **result_signing** — scope split to #16; not resolved within this thread.
