---
ticket: 17
type: issue-body
author: AlbinoGeek
posted: 2026-05-22T19:52:16Z
topic: interface_type
url: https://github.com/Rethunk-AI/bakeoff/issues/17
title: schema: implement interface_type lookup table with seed data
---

**Parent:** #8 — Additional Performance Metrics

Implement the `interface_type` lookup table as ratified in #8 (schema finalized 182139ZMAY26, names approved 200636ZMAY26).

## Schema

```sql
CREATE TABLE interface_type (
    interface_type_id   SERIAL PRIMARY KEY,
    bandwidth_peak_gb_s FLOAT NOT NULL,
    description         TEXT NOT NULL,        -- "PCIe 4.0 x16", "SXM5", "Thunderbolt 4"
    interface_family    TEXT NULLABLE,         -- "PCIe", "SXM", "CXL", "NVLink", "USB", "OCuLink"
    transfer_rate       INT NULLABLE,          -- PCIe: GT/s per lane (e.g. 16 for Gen4); null for non-PCIe
    lane_count          INT NULLABLE           -- PCIe: lane width (e.g. 16); null for non-PCIe
);
```

## Seed data required

Pre-seed rows for common well-known interface types, including at minimum:

- PCIe Gen 1–5 × x1 / x4 / x8 / x16 (peak bandwidth computed from standard spec)
- SXM2 / SXM4 / SXM5
- NVLink 2.0 / 3.0 / 4.0 (reserved — `interface_family = "NVLink"`, no FK references yet per Phase 1 decision)
- Thunderbolt 3 / 4
- OCuLink 2.0

No pre-seeded rows for degraded states — display layer constructs degradation descriptions at runtime from the two FK interface_type rows (`slot_native` and `actual`).

## Acceptance criteria

- [ ] Migration creates `interface_type` with correct column types and NOT NULL constraints
- [ ] Seed migration inserts well-known rows (PCIe Gen 1–5 variants at minimum)
- [ ] `bandwidth_peak_gb_s` values match published specs for each seeded row
- [ ] Rows for NVLink are seeded with `interface_family = "NVLink"` but not yet FK-referenced by any other table
- [ ] Migration is reversible (down path defined)

## Notes

`transfer_rate` and `lane_count` are nullable to accommodate non-PCIe interfaces (SXM, NVLink, USB). Bandwidth is the universal comparable field. For PCIe, `bandwidth_peak_gb_s = transfer_rate × lane_count × 2 / 8` (bidirectional, GB/s).

— Bastion
