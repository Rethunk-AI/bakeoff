#!/usr/bin/env bash
# llama-swap proxy launcher for the bakeoff harness.
#
# llama-swap is a Go proxy that sits in front of llama.cpp servers and
# swaps the active backend on demand. This script is the thin bash
# wrapper used by both humans and the benchmark runner.
#
# Subcommands:
#   up <config-path> [listen-addr]
#       Sweep any stale bench-llama-* containers from a prior run, then
#       exec the llama-swap binary. Default listen: 127.0.0.1:8080.
#   down
#       Best-effort SIGTERM the llama-swap process (pkill by binary
#       path), then sweep bench-llama-* containers.
#   sweep
#       Just remove every bench-llama-* podman container.
#   wait <listen-addr> [timeout_s]
#       Poll http://<listen-addr>/v1/models until 200 OK or timeout.
#
# Env:
#   LLAMA_SWAP_BIN    path to the llama-swap binary.
#                     Default: <repo>/.cache/llama-swap/llama-swap
#                     (populated by run.sh bootstrap.)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
: "${LLAMA_SWAP_BIN:=$root/.cache/llama-swap/llama-swap}"

die() { echo "llama-swap.sh: $*" >&2; exit 1; }

_sweep() {
  if ! command -v podman >/dev/null 2>&1; then
    return 0
  fi
  local removed=0
  local names
  names="$(podman ps -a --filter 'name=^bench-llama-' --format '{{.Names}}' 2>/dev/null || true)"
  if [[ -z "$names" ]]; then
    return 0
  fi
  while IFS= read -r n; do
    [[ -z "$n" ]] && continue
    podman rm -f "$n" >/dev/null 2>&1 || true
    echo "swept $n"
    removed=$((removed + 1))
  done <<< "$names"
  echo "swept $removed container(s)"
}

cmd_up() {
  local config="${1:?config-path required}"
  local listen="${2:-127.0.0.1:8080}"
  [[ -x "$LLAMA_SWAP_BIN" ]] || die "binary not found: $LLAMA_SWAP_BIN (run ./run.sh once to bootstrap)"
  [[ -f "$config" ]] || die "config not found: $config"
  command -v podman >/dev/null 2>&1 || die "podman not found on PATH"
  _sweep
  exec "$LLAMA_SWAP_BIN" --config "$config" --listen "$listen"
}

cmd_down() {
  if command -v pkill >/dev/null 2>&1; then
    pkill -TERM -f "$LLAMA_SWAP_BIN" 2>/dev/null || true
  fi
  _sweep
}

cmd_wait() {
  local listen="${1:?listen-addr required}"
  local timeout_s="${2:-120}"
  local url="http://${listen}/v1/models"
  local deadline=$(( $(date +%s) + timeout_s ))
  while (( $(date +%s) < deadline )); do
    if curl -sf -o /dev/null "$url"; then
      return 0
    fi
    sleep 0.5
  done
  echo "llama-swap.sh: timed out waiting for $url after ${timeout_s}s" >&2
  return 1
}

case "${1:-}" in
  up)    shift; cmd_up    "$@" ;;
  down)  shift; cmd_down  "$@" ;;
  sweep) shift; _sweep ;;
  wait)  shift; cmd_wait  "$@" ;;
  *)
    cat <<'EOF' >&2
Usage:
  llama-swap.sh up <config-path> [listen-addr]
  llama-swap.sh down
  llama-swap.sh sweep
  llama-swap.sh wait <listen-addr> [timeout_s]
EOF
    exit 1 ;;
esac
