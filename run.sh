#!/usr/bin/env bash
# Bootstrap venv (uv), install deps, bootstrap the pinned llama-swap
# binary, run the benchmark harness, emit reports.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

# ---------------------------------------------------------------------------
# llama-swap binary pin. Bump LLAMA_SWAP_VERSION and the matching per-arch
# SHA256 together. Checksums come from the release's checksums.txt:
#   https://github.com/mostlygeek/llama-swap/releases/download/$VER/llama-swap_<NN>_checksums.txt
# ---------------------------------------------------------------------------
LLAMA_SWAP_VERSION="v203"
LLAMA_SWAP_VERSION_NUM="${LLAMA_SWAP_VERSION#v}"
LLAMA_SWAP_SHA256_linux_amd64="188a6608c42c1288903133e557d57a2720741c1b5838897d86b55124d3ce3304"
LLAMA_SWAP_SHA256_linux_arm64="c40a21143efcd5b84a56813edab38af00a34cca10c508b5b59fb73a9e41bb0a3"
LLAMA_SWAP_SHA256_darwin_amd64="c29970ae4e9eac17deca162f0409c7e6152c097c00d124ee8e0f553f7a9641ae"
LLAMA_SWAP_SHA256_darwin_arm64="7c588e2ba47691694b12194b4f678543c3d23066068a95f769b550027fe08697"

die() { echo "run.sh: $*" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    die "neither sha256sum nor shasum available on PATH"
  fi
}

detect_platform() {
  local os arch
  case "$(uname -s)" in
    Linux)  os=linux ;;
    Darwin) os=darwin ;;
    *) die "unsupported OS for pre-built llama-swap: $(uname -s)" ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64)  arch=amd64 ;;
    aarch64|arm64) arch=arm64 ;;
    *) die "unsupported arch for pre-built llama-swap: $(uname -m)" ;;
  esac
  echo "${os}_${arch}"
}

select_checksum() {
  case "$1" in
    linux_amd64)  echo "$LLAMA_SWAP_SHA256_linux_amd64" ;;
    linux_arm64)  echo "$LLAMA_SWAP_SHA256_linux_arm64" ;;
    darwin_amd64) echo "$LLAMA_SWAP_SHA256_darwin_amd64" ;;
    darwin_arm64) echo "$LLAMA_SWAP_SHA256_darwin_arm64" ;;
    *) return 1 ;;
  esac
}

bootstrap_llama_swap() {
  local plat checksum url stamp archive target_dir actual
  plat="$(detect_platform)"
  checksum="$(select_checksum "$plat")" \
    || die "no pinned checksum for platform: $plat"

  target_dir="$here/.cache/llama-swap"
  stamp="$target_dir/VERSION"
  if [[ -x "$target_dir/llama-swap" && -f "$stamp" ]] \
       && [[ "$(cat "$stamp")" == "$LLAMA_SWAP_VERSION $plat" ]]; then
    return 0
  fi

  mkdir -p "$target_dir"
  archive="$target_dir/llama-swap_${LLAMA_SWAP_VERSION_NUM}_${plat}.tar.gz"
  url="https://github.com/mostlygeek/llama-swap/releases/download/${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION_NUM}_${plat}.tar.gz"

  echo "run.sh: fetching llama-swap ${LLAMA_SWAP_VERSION} (${plat})..."
  curl -fsSL "$url" -o "$archive" || die "download failed: $url"

  actual="$(sha256_of "$archive")"
  if [[ "$actual" != "$checksum" ]]; then
    rm -f "$archive"
    die "SHA256 mismatch for $archive: got $actual, pinned $checksum"
  fi

  tar -xzf "$archive" -C "$target_dir"
  rm -f "$archive"
  [[ -x "$target_dir/llama-swap" ]] || die "extract did not yield executable llama-swap"
  echo "$LLAMA_SWAP_VERSION $plat" > "$stamp"
}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

[ -d .venv ] || uv venv .venv
uv pip install --quiet -r requirements.txt

# Subcommands. Default: run the benchmark with the default config.
case "${1:-}" in
  fetch)
    shift
    exec uv run python -m bench.download "$@"
    ;;
  *)
    bootstrap_llama_swap
    exec uv run python -m bench.runner --config config.yaml "$@"
    ;;
esac
