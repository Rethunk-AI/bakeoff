## Summary

<!-- What changed and why. Link issues. -->

## Type

- [ ] feat (new feature)
- [ ] fix (bug fix)
- [ ] refactor (no functional change)
- [ ] docs (README / HUMANS / AGENTS)
- [ ] ci / chore / build
- [ ] bench (scorer / judge mode / runner phase)

## Touches benchmark invariants?

<!-- Check AGENTS.md "Design invariants" before ticking any of these. -->

- [ ] Serving / boot order (sequential, one model at a time)
- [ ] Judge phase (pairwise_all vs scored, order randomization)
- [ ] JSON record shape in `results/run-<ts>.json`
- [ ] Energy / cost axis

## Verification

- [ ] `ruff check bench/`
- [ ] `mypy bench/`
- [ ] `shellcheck bin/serve.sh run.sh`
- [ ] `python -m bench.runner --config config.yaml --dry-run`
- [ ] End-to-end run against real models (if serving or judge logic changed)

## Docs

- [ ] Updated the right tier (README / HUMANS / AGENTS) per `AGENTS.md` governance
