#!/usr/bin/env bash
# Minimal llama.cpp server launcher for the benchmark harness.
# Runs one model at a time in a podman container. No router, no flock.
#
# Subcommands:
#   up <gguf-rel> <alias> <port> [ctx] [n_cpu_moe]
#       Start a container. <gguf-rel> is relative to $MODELS_DIR.
#   down [container-name]
#       Stop + remove. Default name: bench-llama-<port> (use 'down-all' to nuke).
#   down-all
#       Stop + remove every container matching bench-llama-*.
#   wait <port> [timeout_s]
#       Poll /health until OK or timeout. Exit 0 on ready, 1 on timeout.
#
# Env (override as needed):
#   MODELS_DIR   default: $HOME/.lmstudio/models
#   IMAGE        default: ghcr.io/ggml-org/llama.cpp:server-vulkan
#   NGL          default: 99  (offload all layers)
#   UBATCH       default: 512
#   CACHE_TYPE_K default: q8_0
#   CACHE_TYPE_V default: q8_0
#   FLASH_ATTN   default: 1   (set 0 to disable)
#   JINJA        default: 1
#   EXTRA_ARGS   appended verbatim to the llama-server command line

set -euo pipefail

: "${MODELS_DIR:=$HOME/.lmstudio/models}"
: "${IMAGE:=ghcr.io/ggml-org/llama.cpp:server-vulkan}"
: "${NGL:=99}"
: "${UBATCH:=512}"
: "${CACHE_TYPE_K:=q8_0}"
: "${CACHE_TYPE_V:=q8_0}"
: "${FLASH_ATTN:=1}"
: "${JINJA:=1}"
: "${EXTRA_ARGS:=}"

die() { echo "serve.sh: $*" >&2; exit 1; }

container_name_for_port() { echo "bench-llama-$1"; }

cmd_up() {
  local gguf_rel="${1:?gguf-rel required}"
  local alias="${2:?alias required}"
  local port="${3:?port required}"
  local ctx="${4:-4096}"
  local n_cpu_moe="${5:-}"

  command -v podman >/dev/null || die "podman not found"
  [[ -d "$MODELS_DIR" ]] || die "MODELS_DIR not found: $MODELS_DIR"
  [[ -f "$MODELS_DIR/$gguf_rel" ]] || die "gguf not found: $MODELS_DIR/$gguf_rel"

  local name; name="$(container_name_for_port "$port")"

  # Kill any stale instance on this port.
  podman rm -f "$name" >/dev/null 2>&1 || true

  # Pre-flight: is the port free?
  if ss -tlnp 2>/dev/null | grep -q ":${port} " || \
     netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
    die "port $port already in use"
  fi

  local fa_arg=()
  [[ "$FLASH_ATTN" == 1 ]] && fa_arg=(-fa on)
  local jinja_arg=()
  [[ "$JINJA" == 1 ]] && jinja_arg=(--jinja)
  local moe_arg=()
  [[ -n "$n_cpu_moe" ]] && moe_arg=(--n-cpu-moe "$n_cpu_moe")

  # shellcheck disable=SC2086
  podman run -d --name "$name" \
    --device /dev/dri \
    --security-opt label=disable \
    -p "127.0.0.1:${port}:8080" \
    -v "$MODELS_DIR:/m:ro" \
    "$IMAGE" \
    -m "/m/$gguf_rel" \
    -a "$alias" \
    --host 0.0.0.0 --port 8080 \
    -c "$ctx" -ub "$UBATCH" -ngl "$NGL" \
    -ctk "$CACHE_TYPE_K" -ctv "$CACHE_TYPE_V" \
    "${fa_arg[@]}" "${jinja_arg[@]}" "${moe_arg[@]}" \
    $EXTRA_ARGS >/dev/null
  echo "$name"
}

cmd_down() {
  local name="${1:-}"
  [[ -n "$name" ]] || die "container name required"
  podman rm -f "$name" >/dev/null 2>&1 || true
  echo "stopped $name"
}

cmd_down_all() {
  local removed=0
  while read -r n; do
    [[ -z "$n" ]] && continue
    podman rm -f "$n" >/dev/null 2>&1 || true
    echo "stopped $n"
    removed=$((removed + 1))
  done < <(podman ps -a --filter 'name=^bench-llama-' --format '{{.Names}}')
  echo "removed $removed container(s)"
}

cmd_wait() {
  local port="${1:?port required}"
  local timeout_s="${2:-120}"
  local deadline=$(( $(date +%s) + timeout_s ))
  while (( $(date +%s) < deadline )); do
    if curl -sf -o /dev/null "http://127.0.0.1:${port}/health"; then
      return 0
    fi
    sleep 0.5
  done
  echo "serve.sh: timed out waiting for :${port}/health after ${timeout_s}s" >&2
  return 1
}

case "${1:-}" in
  up)       shift; cmd_up       "$@" ;;
  down)     shift; cmd_down     "$@" ;;
  down-all) shift; cmd_down_all "$@" ;;
  wait)     shift; cmd_wait     "$@" ;;
  *)
    cat <<'EOF' >&2
Usage:
  serve.sh up <gguf-rel> <alias> <port> [ctx] [n_cpu_moe]
  serve.sh down <container-name>
  serve.sh down-all
  serve.sh wait <port> [timeout_s]
EOF
    exit 1 ;;
esac
