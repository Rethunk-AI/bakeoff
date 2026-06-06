# bakeoff#38 — schema-discussion corpus

Topic-decomposed corpus of the schema-design discussion across 12 bakeoff tickets, built
to settle [Rethunk-AI/bakeoff#38](https://github.com/Rethunk-AI/bakeoff/issues/38). This
branch (`issue-38`) is **disposable**: never merged, never PR'd, deleted when @gissf1 or
NOMAD says it has served its purpose.

## Why this exists

The schema threads outgrew single-issue focus. Per @gissf1's decomposition, each comment is
split by topic, consolidated chronologically per topic, then summarized to final-state — so a
reviewer reads one topic at a time instead of re-threading 12 interleaved issues.

## Layout

```
issue-38/
  tickets/<n>/<comment_id>.md   verbatim entry + metadata (author, post-time, ticket#, comment-id)
  tickets/<n>/issue.md          the issue body (first chronological entry of that ticket)
  tickets/_manifest.md          all 134 entries, chronological (ascending comment-id)
  topic_comments/<topic>/...    each entry copied under every topic it spans (multi-topic = duplicated)
  topics/<topic>.md             consolidated chat history per topic, ascending comment-id = chronological
  topic_summary/<topic>.md      aggregate final-state + rationale + carried-open decisions (NOT play-by-play)
```

## Source ticket set (locked by @gissf1)

#8, #12, #13, #14, #15, #17, #18, #19, #20, #21, #22, #38 — 12 tickets, 134 thread entries
(12 issue bodies + 122 comments).

## Topic taxonomy

**Seed (locked):** `gpu_hardware` (vram_type folded in), `interface_type`, `system_gpu_link`,
`system_hardware`, `system_software`, `run_hardware_metrics`, `tflops_source`,
`gpu_architecture`, `uuids`.

**Minted during classification (taxonomy is open-ended):** `run_model_metrics`,
`scheduling_queue`, `model_descriptor`, `model_trust`, `persistence_layout`, `process_meta`
(`process_meta` = discussion about the pipeline/process itself, kept out of schema topics).

## Reading order for review

1. `topic_summary/<topic>.md` — final state + what's still open per topic. Start here.
2. `topics/<topic>.md` — the verbatim chat backing each summary, chronological.
3. `tickets/<n>/` — original per-ticket source if you need to trace an entry.

## Build conventions / deviations (flagged for review)

- **topic_comments are real copies, not symlinks.** The earlier symlink convention assumed a
  live local filesystem with upsert-freshness. The venue here is GitHub-web browse on a
  point-in-time disposable branch, where symlinks render as broken path-text and the freshness
  rationale is moot. Real copies render correctly for the reviewer. Revisit if the corpus ever
  moves to a live-synced host.
- **Open decisions are carried, not collapsed.** Where a thread did not converge, the summary
  states the open question with its trade-offs rather than inventing consensus. See the
  `## Open / unresolved` section in each summary.
- **Chronology by comment-id.** GitHub comment IDs are globally sequential, so ascending ID ==
  chronological; issue bodies sort first within their ticket.

Built by Bastion from the live thread state. Regenerable: re-hydrate the ticket set, re-run
classify → consolidate → summarize.
