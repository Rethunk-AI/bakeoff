# bakeoff#38 — schema-discussion corpus (#38-EXCLUDED variant)

Topic-decomposed corpus of the schema-design discussion, built to settle
[Rethunk-AI/bakeoff#38](https://github.com/Rethunk-AI/bakeoff/issues/38). This branch
(`issue-38-2`) is **disposable**: never merged, never PR'd, deleted when @gissf1 or NOMAD
says it has served its purpose.

## What makes this variant different

This is the companion to the `issue-38` branch, with two deliberate differences requested by
@gissf1 in [bastion-ai-helpers#3](https://github.com/Rethunk-Tech/bastion-ai-helpers/issues/3):

1. **Ticket #38's own thread is EXCLUDED.** The summaries here are built ONLY from the 11
   schema-spec tickets (#8, #12–15, #17–22) — the #38 audit meta-conversation is omitted. The
   purpose is to see the schema discussion's *own* settled state, without the #38 audit's
   renames/rulings overlaid. Where a decision was reached only in #38, it is correctly absent
   here. Compare against `issue-38` (the #38-inclusive baseline) to see exactly what #38 added.
2. **Three output improvements** (also @gissf1's, this thread):
   - **Open items carry their address.** Every entry in a summary's `## Open / unresolved`
     section ends with `address: #<ticket>, <comment-url>` — where the question should be
     answered, not just that it is open.
   - **Cross-cutting open-questions roll-up.** `topic_summary/open_questions.md` aggregates
     every unresolved decision across all topics into one addressed list — read it first to see
     what is still open and where.
   - **Per-topic provenance TOC.** Each `topics/<topic>.md` opens with a "Source entries"
     table listing the exact ticket# + comment-id + url used to build it. Audit aid only — it
     is **not** migrated into `topic_summary/`.

## Layout

```
issue-38/                       (same path as the issue-38 branch, so the two branches diff cleanly)
  tickets/<n>/<comment_id>.md   verbatim entry + metadata (author, post-time, ticket#, comment-id)
  tickets/<n>/issue.md          the issue body (first chronological entry of that ticket)
  tickets/_manifest.md          all 122 entries, chronological (ascending comment-id), #38 omitted
  topic_comments/<topic>/...    each entry copied under every topic it spans (multi-topic = duplicated)
  topics/<topic>.md             provenance TOC + consolidated chat history per topic, ascending comment-id
  topic_summary/<topic>.md      aggregate final-state + rationale + carried-open decisions (addresses on opens)
  topic_summary/open_questions.md   cross-cutting roll-up of every open decision, addressed
```

## Source ticket set (#38 excluded)

#8, #12, #13, #14, #15, #17, #18, #19, #20, #21, #22 — 11 tickets, 122 thread entries
(11 issue bodies + 111 comments). Classifications reuse the locked issue-38 run; only #38's
12 entries were filtered out before consolidation + re-summarization.

## Topic taxonomy

All 15 topics from the `issue-38` build survive the exclusion (none was #38-only):
`gpu_hardware`, `interface_type`, `system_gpu_link`, `system_hardware`, `system_software`,
`run_hardware_metrics`, `tflops_source`, `gpu_architecture`, `uuids`, `run_model_metrics`,
`scheduling_queue`, `model_descriptor`, `model_trust`, `persistence_layout`, `process_meta`.

## Reading order for review

1. `topic_summary/open_questions.md` — everything still open across the corpus, with addresses.
2. `topic_summary/<topic>.md` — per-topic final state + what is open. Summaries reflect the
   #38-excluded thread only.
3. `topics/<topic>.md` — provenance TOC + verbatim chat backing each summary.
4. `tickets/<n>/` — original per-ticket source.

## Build conventions / deviations

- **topic_comments are real copies, not symlinks** — GitHub-web browse on a point-in-time
  disposable branch renders symlinks as broken path-text; the freshness rationale for symlinks
  is moot on a static branch. Revisit if the corpus ever moves to a live-synced host.
- **Open decisions are carried, not collapsed** — where a thread did not converge, the summary
  states the open question with its trade-offs and its address, never an invented consensus.
- **Chronology by comment-id** — GitHub comment IDs are globally sequential, so ascending ID ==
  chronological; issue bodies sort first within their ticket.

Built by Bastion from the live thread state. Regenerable: re-hydrate the ticket set, filter
#38, re-run classify (reused) → consolidate → summarize → roll-up.
