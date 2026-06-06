---
ticket: 22
type: issue-body
author: AlbinoGeek
posted: 2026-05-25T09:53:53Z
topic: system_hardware
url: https://github.com/Rethunk-AI/bakeoff/issues/22
title: Hardware tables disk persistence — analysis and layout
---

## Context

Hardware schema locked in #17 (#18, #19, #20, #21). This thread analyses which hardware tables warrant disk files, what the layout should be, and how hardware records relate to run disk files.

Follows the disk persistence pattern established in #15 for models/tasks/prompts/runs: disk files carry `schema_version`, `created_at`, `updated_at`; FK columns collapse to embedded objects or UUID/ID references.

---

## Tables in scope

| Table | Issue | Purpose |
|-------|-------|---------|
| `interface_type` | #17 | Lookup — GPU bus type enum |
| `system_hardware` | #19 | Host CPU/RAM snapshot per run |
| `system_software` | #19 | Host OS/driver snapshot per run |
| `gpu_hardware` | #18 | Individual GPU slot spec |
| `system_gpu_link` | #20 | Many-to-many: GPUs in a system snapshot |
| `run_hardware_metrics` | #21 | Per-run hardware context (links run → hw snapshot) |

---

## Open questions

**Q1 — Lookup table disk files**

`interface_type` is a small admin-controlled enum (10 rows). Should it have a seed JSON file (like `seeds/quantization_methods.json`) or is it embedded in schema.sql only?

Proposed: seed file `seeds/interface_types.json`, parallel to existing seeds. Admin-review-gated like model_architectures/model_file_formats.

**Q2 — Hardware snapshot identity**

`system_hardware` and `system_software` rows are generated at run time by the runner probing the host machine. No UUID — they use SERIAL PKs.

Options for disk identity:
- A. Derive a deterministic UUID from the hardware fingerprint fields (cpu_model + cpu_cores + ram_gb + motherboard hash) — allows dedup across runs without DB
- B. Use DB-assigned SERIAL ID embedded in the run disk file — simpler, requires DB round-trip before file is self-contained
- C. No separate disk file for hardware — embed the hardware snapshot inline in the run disk file

**Q3 — gpu_hardware + system_gpu_link disk layout**

A system snapshot involves one or more GPU slots. Should the disk representation be:
- A. A single `hardware-snapshot/<uuid>.json` file containing both system_hardware fields and an embedded `gpus: [...]` array (inlined from system_gpu_link + gpu_hardware) — single file per snapshot
- B. Separate `gpu_hardware/<id>.json` files referenced by ID from the snapshot — allows GPU spec dedup across snapshots

**Q4 — run_hardware_metrics relationship to run disk file**

`run_hardware_metrics` is 1:1 with `runs` (one row per run). Should the hardware context be:
- A. Embedded inline in `runs/<run_id>.json` as a `hardware:` block — avoids a separate file, run file is self-contained
- B. A separate `run-hardware/<run_id>.json` file — parallel to the run, referenced by run_id

**Q5 — Runner hardware reporting timing**

Hardware snapshot is captured by the runner at claim time or job start. Should the runner:
- A. Create/update the hardware snapshot once per process start (reuse across runs in same session)
- B. Snapshot hardware fresh per run (detects GPU slot changes mid-session)

---

Tagging @gissf1 for design input per the pattern established in #15.

— Bastion
