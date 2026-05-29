# Tasks

## P0

- [x] Implement `bench/store.py`: `data_dir()`, `write_record`, `read_record`, `list_records`, `delete_record`, audit stamping, UUID5 helpers.
- [x] Add `tests/test_store.py`: roundtrip, audit-field preservation, UUID determinism.
- [x] Implement `bench/descriptor.py`: `load_descriptor` (schema_version gate), `validate_descriptor` (issue list), `descriptor_uuid`, `save_descriptor`.
- [x] Add `tests/test_descriptor.py`: happy path, missing/bad schema_version, UUID hash vs provisional.
- [x] Implement `bench/queue.py`: `enqueue`, `claim` (rename-as-mutex, race-safe), `mark_in_progress`, `complete`, `fail` (backoff + terminal), `cancel`, `reap_stale_claims`.
- [x] Add `tests/test_queue.py`: end-to-end happy path, fail/retry/exhaustion, retry_after gate, stale-claim reaping, concurrent 8-thread race.
- [x] Update `AGENTS.md`: design invariant entry + layout block entries for store/descriptor/queue.
- [x] Update `HUMANS.md`: `BAKEOFF_DATA_DIR` env var documentation.
- [x] Add spec files and `specs/index.md` Active table row.

## P2 (deferred)

- [ ] URL-submit -> upstream-metadata-pull -> validate ingestion pipeline (`bench/descriptor.py`).
- [ ] Embedded GGUF metadata cross-check against descriptor fields.
- [ ] DB-sync: write store records to `schema/schema.sql` Postgres tables when a DB connection is available.
- [ ] Descriptor CLI entry point (`python -m bench.descriptor ingest <file>`).
- [ ] Runner registration (`bench/queue.py` runner heartbeat + `runners/` table).
