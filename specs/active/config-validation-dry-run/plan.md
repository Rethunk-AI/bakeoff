# Plan

## Implementation Slices

- Add `bench.config` with loading, issue collection, and path-oriented validation.
- Replace `bench.runner.load_config` with the shared loader and validation call.
- Keep llama-swap generator checks as invariant guards, but avoid relying on them for normal user errors.
- Add dry-run output that confirms validation, dataset generation, proxy config generation, and YAML serialization.
- Add unit tests for validator behavior and one dry-run fixture test that exercises the CI path.
- Update `HUMANS.md` dry-run and troubleshooting notes.

## Test Plan

- `uv run pytest tests/test_config.py tests/test_llama_swap.py`
- `uv run python -m bench.runner --config config.yaml --dry-run`
- `uv run pre-commit run --all-files`
