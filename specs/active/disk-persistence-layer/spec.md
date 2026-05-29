# Disk Persistence Layer

Status: Active

## Problem

The bakeoff harness produces run results as ephemeral JSON files under `results/` with no structured persistence for model descriptors, queue state, or cross-run identity. There is no stable place to store model metadata (weights hash, creator, sources) or to coordinate work across multiple runners without a database.

## Goals

Add a disk-based persistence layer (directory-per-table, UUID-filename records) for model descriptors and run queue items, keeping the standalone runner's default matrix loop unchanged. All storage is file-system-only: no database, no daemon.

## Non-Goals

- No database (psycopg/sqlite) in `bench/`.
- URL-submit → upstream-metadata-pull pipeline deferred to P2.
- Embedded GGUF metadata cross-check deferred to P2.
- `bench/runner.py` default path is not changed; the queue is opt-in only.

## Design

### Disk layout

Base dir from env var `BAKEOFF_DATA_DIR` (default `~/.local/share/bakeoff`), resolved at call time:

```
<BAKEOFF_DATA_DIR>/
  models/<uuid>.json        persistent model descriptors
  tasks/<natural_key_hash>.json
  prompts/<content_sha256>.json
  runners/<runner_id>.json
  runs/<run_id>.json
  run_queue/                EPHEMERAL; see queue section
    pending/<queue_id>.json
    completed/<queue_id>.json
```

All persistent records carry `schema_version` (integer, currently 1), `created_at`, and `updated_at` (ISO-8601 UTC, disk record lifecycle). Writes are atomic (temp file + `os.replace`).

### `bench/store.py`

Shared foundation: `data_dir()`, `write_record`, `read_record`, `list_records`, `delete_record`, `model_uuid`, `provisional_model_uuid`, `creator_uuid`, `provisional_creator_uuid`.

### `bench/descriptor.py`

Model descriptor reader/validator/persister for hand-crafted seed JSON. `load_descriptor` raises `DescriptorError` immediately on missing or unsupported `schema_version`. `validate_descriptor` returns an aggregated list of issues. `descriptor_uuid` derives UUID from model_hash or provisional key. `save_descriptor` validates then persists.

### `bench/queue.py`

Opt-in disk queue. Claim is race-safe via rename-as-mutex: `os.rename(src, lock)` is the claim gate; concurrent runners that lose get `FileNotFoundError`. CLAIMED/IN_PROGRESS items stay in `pending/`; terminal items move to `completed/`. Backoff: `retry_after = now + 5*attempt_count minutes`, `priority += 5*attempt_count`.

## Acceptance Criteria

- `bench/store.py`, `bench/descriptor.py`, `bench/queue.py` pass ruff, mypy (py310), and pytest.
- `bench/runner.py` default path is unchanged.
- `BAKEOFF_DATA_DIR` is documented in `HUMANS.md`.
- Spec, plan, and tasks created under `specs/active/disk-persistence-layer/`.
