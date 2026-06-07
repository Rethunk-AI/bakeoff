---
ticket: 15
type: issue-body
author: AlbinoGeek
posted: 2026-05-22T14:25:32Z
title: bakeoff: model descriptor file format and capability ingestion
url: https://github.com/Rethunk-AI/bakeoff/issues/15
---

# bakeoff: model descriptor file format and capability ingestion

**Bastion (J-5) — bakeoff: model descriptor file format and capability ingestion — 221424ZMAY26**

Spun off from #13 (221322ZMAY26 reply). Two coupled open questions that belong in their own thread.

---

## Background

#13 established:
- Standalone runner reads a JSON model descriptor at startup — no DB required
- DB `models` table is populated from the same descriptor at submission time
- Descriptor is canonical input; DB is derived

Confirmed fields (#13 §5):

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

---

## Q1 — Versioned schema?

Should the descriptor include a `schema_version` field?

```json
{ "schema_version": 1, ... }
```

**With version field:** runner can reject or adapt descriptors from older/newer tooling. Migration path without re-scanning all descriptor files.

**Without:** simpler now; add when first breaking change arrives. Risk: unversioned files in the wild become ambiguous once schema evolves.

Recommendation: include `schema_version: 1` from day one. Cost is one field; benefit is a clean migration path before any files accumulate in the wild.

---

## Q2 — Disk location

**Option A — beside model weights:**
```
/models/llama3-8b-q4km/
  llama3-8b-q4km.gguf
  model.json             ← descriptor co-located with weights
```

**Option B — separate submission directory:**
```
/models/llama3-8b-q4km/llama3-8b-q4km.gguf
queue/pending/llama3-8b-q4km.json         ← descriptor as submission artifact
```

Option A: self-contained model bundle; runner infers descriptor path from model path.

Option B: descriptor is a submission artifact, not part of the model bundle. Supports submitting the same model weights under multiple configurations (different quantization labels, priority, prompt sets) without duplicating the weights directory.

Recommendation: Option B. Models are large and often read-only or shared. Submission configuration belongs in the queue layer, not the model bundle.

---

## Q3 — Capability ingestion pipeline

Where does model metadata (parameter counts, quantization, architecture) originate?

- **A. Manual** — submitter writes `model.json` by hand or from model card. Simple; error-prone for numeric fields.
- **B. CI artifact** — CI pipeline generates descriptor from HuggingFace metadata or model card at ingest time. Reliable; requires CI integration.
- **C. Runner autodiscovery** — runner parses GGUF metadata at startup, cross-checks against descriptor values. Catches descriptor drift; adds startup latency.

Recommendation: A for now (manual / tooling-assisted). Add C (GGUF metadata cross-check at startup) when runner is implemented — cheap to add, catches the common mistake of mismatched quantization label or wrong parameter count in the descriptor.

---

@gissf1 — opinions on Q1–Q3 above?

— Bastion

