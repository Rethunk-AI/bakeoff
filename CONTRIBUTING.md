# Contributing

Thanks for opening this file — it means you're considering a patch. This is a
thin routing doc: setup lives in [`HUMANS.md`](HUMANS.md), design invariants
and editing conventions live in [`AGENTS.md`](AGENTS.md). Don't look for that
content here; we deliberately keep it in one place.

## Where to start

- **New to the project:** [`HUMANS.md`](HUMANS.md) — prerequisites, install,
  run, configure, troubleshoot.
- **Design constraints you must preserve:** [`AGENTS.md`](AGENTS.md) — one
  model in VRAM at a time, judge runs as its own phase, energy-as-cost, etc.
  Violating these silently will get a PR bounced.
- **The contract:** [`config.yaml`](config.yaml). Every new knob goes here
  first, then gets wired through `runner.py` / `llama-swap.sh`. Don't hard-code.

## Setup

Base environment (podman, `uv`, GGUFs): [`HUMANS.md` § Prerequisites](HUMANS.md#prerequisites). Don't re-derive that here.

Dev extras (ruff, mypy, pytest, types-PyYAML) on top of the base venv:

```sh
uv pip install -e ".[dev]"
```

## Before opening a PR

Run the same checks CI runs. The fastest path is
[`pre-commit`](https://pre-commit.com/):

```sh
pre-commit install        # one-time, installs the git hook
pre-commit run --all-files
```

`pre-commit` covers ruff, shellcheck, actionlint, and basic file hygiene. It
deliberately does **not** run mypy or pytest (they need the dev venv and are
slow for every commit). Run those once before pushing:

```sh
uv run mypy bench/
uv run python -m pytest
uv run python -m bench.runner --config config.yaml --dry-run
```

If you added a new config knob, the dry-run should parse it without error.

## Commits and PRs

- **Conventional Commits**: `type(scope): subject`. `feat`, `fix`, `docs`,
  `refactor`, `test`, `ci`, `build`, `style`, `chore`. Scope is the module
  (`runner`, `report`, `clients`, `metrics`, `llama-swap`) or a docs tier
  (`readme`, `agents`).
- **Body explains _why_**, not _what_. A diff already shows the what.
- **One logical unit per commit.** If a PR is a bundle of unrelated changes,
  reviewers will ask for a split.
- **PR description**: what broke / what you're improving, any design tradeoff
  you considered and rejected, how you tested. The
  [`pull_request_template.md`](.github/pull_request_template.md) prompts for
  this.

## What NOT to change without discussing first

- Design invariants in [`AGENTS.md`](AGENTS.md) (sequential phases,
  randomized pairwise order, energy-based cost, etc.).
- The JSON record shape emitted to `results/run-<ts>.json` — the HTML
  dashboard reads it verbatim, and so do downstream consumers.
- The three-tier documentation split. If you find yourself wanting a fourth
  top-level `*.md`, open an issue first explaining why
  README / HUMANS / AGENTS can't absorb it.

## Security

See [`SECURITY.md`](SECURITY.md) for private disclosure. Don't file public
issues for vulnerabilities.
