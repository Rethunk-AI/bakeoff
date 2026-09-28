#!/usr/bin/env bash
# Bootstrap venv (uv), install deps, run the benchmark harness, emit reports.
# Model serving is engined's job (must already be running); this script
# never starts, stops, or builds anything on its behalf.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# --inexact: don't strip a contributor's dev extras (pytest, ruff, ...) just
# because this run only needs the base runtime deps.
uv sync --quiet --inexact

# Subcommands. Default: run the benchmark with the default config.
case "${1:-}" in
  fetch)
    shift
    exec uv run python -m bench.download "$@"
    ;;
  *)
    exec uv run python -m bench.runner --config config.yaml "$@"
    ;;
esac
