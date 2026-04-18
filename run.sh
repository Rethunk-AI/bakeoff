#!/usr/bin/env bash
# Bootstrap venv (uv), install deps, run A/B benchmark, emit reports.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

[ -d .venv ] || uv venv .venv
uv pip install --quiet -r requirements.txt

uv run python -m bench.runner --config config.yaml "$@"
