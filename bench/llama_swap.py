"""Generate a llama-swap config from the bakeoff config.yaml.

llama-swap is a Go proxy that sits in front of llama.cpp servers and swaps
the active backend on demand based on the `model` field of each OpenAI
chat request. We keep the existing podman + llama.cpp Vulkan image as the
inference engine and delegate the boot/teardown dance to llama-swap.

The generator is pure: `build(bakeoff_cfg, models_dir)` returns a dict
ready to `yaml.safe_dump`. Keeping it pure lets the tests lock the cmd
line structure without booting anything.

Design invariants preserved (see AGENTS.md):

  - One model in VRAM at a time. llama-swap's default swap behaviour
    unloads the current backend before starting the next; we do not use
    groups/profiles, so the default applies. Per-model `ttl: 0` stops
    an idle model from unloading mid-matrix and poisoning the next call
    with a silent re-boot.
  - `sendLoadingState: false` stops llama-swap from injecting a boot
    status string into `reasoning_content` during the swap. Without it,
    the client's `content`-then-`reasoning_content` fallback can pick up
    the loading text as the response body.
  - `mmproj-*` GGUFs are vision projectors, not standalone text models.
    The generator refuses them outright.

The `cmd` strings are rendered as single-line podman invocations so the
tests can pin structure by substring match.
"""

from __future__ import annotations

import re
from typing import Any

from bench.config import ConfigError, judge_id

HEALTH_ENDPOINT = "/health"
LLAMA_IMAGE_DEFAULT = "ghcr.io/ggml-org/llama.cpp:server-vulkan"
CONTAINER_NAME_PREFIX = "bench-llama-"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def container_name(model_id: str) -> str:
    """Stable container name for a model entry. Mirrored in `cmdStop`."""
    return f"{CONTAINER_NAME_PREFIX}{model_id}"


def _validate_id(model_id: str) -> None:
    if not _ID_RE.match(model_id):
        raise ConfigError(
            f"model id {model_id!r} must match {_ID_RE.pattern} "
            "(used as container name + llama-swap key)"
        )


def _validate_gguf_not_mmproj(gguf: str, model_id: str) -> None:
    basename = gguf.rsplit("/", 1)[-1]
    if basename.lower().startswith("mmproj"):
        raise ConfigError(
            f"model {model_id!r}: {basename!r} is an mmproj-* vision "
            "projector, not a standalone text model"
        )


def _render_cmd(
    entry: dict[str, Any],
    server_cfg: dict[str, Any],
    models_dir: str,
    image: str,
) -> str:
    mid = entry["id"]
    gguf = entry["gguf"]
    alias = entry.get("alias") or mid
    ctx = int(entry.get("ctx", server_cfg.get("ctx", 4096)))
    ngl = int(server_cfg.get("ngl", 99))
    ubatch = int(server_cfg.get("ubatch", 512))
    ctk = str(server_cfg.get("cache_type_k", "q8_0"))
    ctv = str(server_cfg.get("cache_type_v", "q8_0"))
    flash_attn = bool(server_cfg.get("flash_attn", True))
    jinja = bool(server_cfg.get("jinja", True))
    n_cpu_moe = entry.get("n_cpu_moe")

    parts: list[str] = [
        "podman",
        "run",
        "--rm",
        "--name",
        container_name(mid),
        "--device",
        "/dev/dri",
        "--security-opt",
        "label=disable",
        "-p",
        "127.0.0.1:${PORT}:8080",
        "-v",
        f"{models_dir}:/m:ro",
        image,
        "-m",
        f"/m/{gguf}",
        "-a",
        alias,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        str(ctx),
        "-ub",
        str(ubatch),
        "-ngl",
        str(ngl),
        "-ctk",
        ctk,
        "-ctv",
        ctv,
    ]
    if flash_attn:
        parts += ["-fa", "on"]
    if jinja:
        parts.append("--jinja")
    if n_cpu_moe is not None:
        parts += ["--n-cpu-moe", str(int(n_cpu_moe))]
    return " ".join(parts)


def _model_block(
    entry: dict[str, Any],
    server_cfg: dict[str, Any],
    models_dir: str,
    image: str,
) -> tuple[str, dict[str, Any]]:
    if "id" not in entry or "gguf" not in entry:
        raise ConfigError(f"entry missing id/gguf: {entry!r}")
    mid = str(entry["id"])
    _validate_id(mid)
    _validate_gguf_not_mmproj(str(entry["gguf"]), mid)
    block = {
        "cmd": _render_cmd(entry, server_cfg, models_dir, image),
        "proxy": "http://127.0.0.1:${PORT}",
        "checkEndpoint": HEALTH_ENDPOINT,
        "cmdStop": f"podman stop {container_name(mid)}",
        "ttl": 0,
    }
    return mid, block


def build(bakeoff_cfg: dict[str, Any], models_dir: str) -> dict[str, Any]:
    """Map a bakeoff config to a llama-swap config dict.

    `models_dir` is the **host** path to the GGUF root (resolved by the
    caller); it's mounted read-only into each backend container at `/m`.

    The judge entry, when present, becomes another llama-swap model keyed
    by `judge.id` (default `"judge"`). It is intentionally not deduped
    against the model pool: the judge should be a distinct key so runners
    can ask llama-swap for it by name without swap confusion.
    """
    server_cfg = dict(bakeoff_cfg.get("server") or {})
    image = str(server_cfg.get("image", LLAMA_IMAGE_DEFAULT))

    models_out: dict[str, Any] = {}
    for entry in bakeoff_cfg.get("models") or []:
        mid, block = _model_block(entry, server_cfg, models_dir, image)
        if mid in models_out:
            raise ConfigError(f"duplicate model id: {mid!r}")
        models_out[mid] = block

    judge = dict(bakeoff_cfg.get("judge") or {})
    if judge.get("enabled") and judge.get("gguf"):
        jid = judge_id(judge)
        judge_entry = {**judge, "id": jid}
        mid, block = _model_block(judge_entry, server_cfg, models_dir, image)
        if mid in models_out:
            raise ConfigError(
                f"judge id {mid!r} collides with a model id — set judge.id to a distinct value"
            )
        models_out[mid] = block

    return {
        "healthCheckTimeout": int(server_cfg.get("boot_timeout_s", 300)),
        "startPort": int(server_cfg.get("backend_start_port", 5800)),
        "logLevel": "info",
        "globalTTL": 0,
        "sendLoadingState": False,
        "models": models_out,
    }
