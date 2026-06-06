# Summary: process_meta

## Final state

The agreed pipeline for resolving the schema discrepancies documented in #38 is:

1. **Cache** every comment from the locked ticket set into `tickets/<ticket>/<comment_id>.md` (verbatim + metadata: post-time, author, ticket#, comment-id). Comments processed in ascending ID order (chronological).
2. **Split** each comment into per-topic files at `topic_comments/<topic>/<comment_id>.md`; a comment spanning multiple topics is duplicated into each relevant subfolder.
3. **Consolidate** per topic into `topics/<topic>.md` — full chat history for that topic, metadata included.
4. **Summarize** per topic into `topic_summary/<topic>.md`.

**Locked ticket set:** #8, #12, #13, #14, #15, #17, #18, #19, #20, #21, #22, #38 (per 4636100356).

**Seed topic taxonomy (after refinements):** `gpu_hardware` (absorbs `vram_type`), `interface_type`, `system_gpu_link`, `system_hardware`, `system_software`, `run_hardware_metrics`, `tflops_source`, `gpu_architecture`, `uuids`. Taxonomy refines further as comments are split.

**Venue:** disposable branch `issue-38` on the existing `Rethunk-AI/bakeoff` repo, files under `issue-38/`. Branch is never merged or PR'd; deleted when schema work is resolved and removal is approved by @gissf1 or NOMAD (per 4637426231 + 4637474046).

**Reusable tooling** (comment-aware issue store, incremental sync, the comment→topic→consolidate→summarize machinery as a generalized capability) split off to `Rethunk-Tech/bastion-ai-helpers#3` so #38 stays focused on schema substance (per 4635959813).

**Schema substance** routes to #38. Process / pipeline discussion routes to the corpus repo itself once stood up.

**Incremental rebuild:** when tickets are added later, only downstream files older than the new source are regenerated (keyed off mtime/commit-time); full re-ingestion is never needed.

## Notable / unusual decisions

- **Disposable branch over external host or new repo** — @gissf1 proposed committing files to a throwaway `issue-38` branch on the existing bakeoff repo (4637426231). Accepted (4637474046) as the cleanest option: no new infra, no external-access grant, no egress outside Rethunk-org, content never enters the issue-comment corpus so summaries cannot self-corrupt. Branch is deleted post-resolution, never merged.
- **vram_type folds into gpu_hardware** — dropped as a standalone topic per @gissf1 (4636026879); keeps the FK relationship visible inside `gpu_hardware` rather than fragmented.
- **FKs cross-referenced both ways** — every FK lands in both the holding topic and the referenced topic so neither side loses the relationship (4636026879 / 4636100356). Downstream use: prevents summaries from silently dropping FK context.
- **Overlong single-ticket threads identified as root cause** — @gissf1 explicitly called out that extremely long responses (either party) are the signal to split into a new ticket (4635710977). The comment→topic decomposition is the structural fix.
- **Tooling split to helpers#3** — Bastion proactively split the reusable pipeline machinery to `bastion-ai-helpers#3` without waiting for direction, citing thread-focus discipline (4635959813); @gissf1 agreed.
- **Schema substance stays in #38 only** — @gissf1 explicitly asked not to fragment schema discussion across new bakeoff tickets; per-topic split happens inside the corpus, not across GitHub issues (4636100356).

## Open / unresolved

- **Push authorization for the `issue-38` branch** — as of the last comment in this log (4637474046), the disposable-branch proposal had been surfaced to NOMAD but NOMAD had not yet cleared the push. The corpus build was running locally; access for @gissf1 was pending that clearance. This is a process gate, not a design question — it resolves the moment NOMAD approves.

## Cross-topic links

- `process_meta` feeds all schema topics: the pipeline output (`topic_summary/<topic>.md`) is the input used to verify and reconcile every table design.
- `gpu_hardware` — absorbs `vram_type` per this process; see `gpu_hardware` summary for FK detail.
- `uuids` — carved out as its own topic per @gissf1 (4636026879); UUID design decisions that appear in any ticket's comments are routed there.
- `Rethunk-Tech/bastion-ai-helpers#3` — holds the generalized tooling; not a schema topic but the process dependency for future incremental corpus rebuilds.
