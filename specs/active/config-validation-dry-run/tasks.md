# Tasks

## P0

- [x] Create `bench.config` with reusable config loading and validation primitives.
- [x] Validate required top-level sections and user-facing path names.
- [x] Validate model IDs, duplicate IDs, GGUF path shape, and `mmproj-*` rejection.
- [x] Validate judge enablement, judge mode, judge ID collisions, and judge GGUF shape.
- [x] Validate dataset, prompt, server, cost, and output values used by the runner.
- [x] Wire validation into `bench.runner` before any side effects.
- [x] Preserve llama-swap generator invariant checks as defensive errors.
- [x] Add tests for valid minimal config and representative invalid configs.
- [x] Confirm `uv run python -m bench.runner --config config.yaml --dry-run` passes.

## P1

- [ ] Emit a concise dry-run summary listing validation, dataset, and proxy-config checks.
- [ ] Add a lightweight CI dry-run fixture for non-default config branches.
- [ ] Update `HUMANS.md` dry-run and troubleshooting text.

## P2

- [ ] Consider a `--validate-only` flag if dry-run output becomes too broad.
