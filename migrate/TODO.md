# TODO

- Implement inline record migration path (schema fingerprint comparison to detect when no structural DDL change occurred, avoiding the shadow-table rebuild): `runner.go:174`, `runner.go:258`.
- Wire a `--rollback` flag to drop all orphan shadow tables found from a prior interrupted run, instead of always continuing automatically: `runner.go:619`.
