"""Unit tests for the llama-swap config generator.

The generator is pure — no subprocess, no filesystem, no network. These
tests lock the structure of the emitted config so a future llama-swap
schema change surfaces as a clear failure instead of a silent runtime
mismatch inside the proxy.
"""

from __future__ import annotations

import pytest

from bench.llama_swap import (
    CONTAINER_NAME_PREFIX,
    HEALTH_ENDPOINT,
    ConfigError,
    build,
    container_name,
)

MODELS_DIR = "/models"


def _cfg(**overrides):
    """Minimal valid bakeoff config. Override specific keys per test."""
    base = {
        "server": {
            "image": "ghcr.io/ggml-org/llama.cpp:server-vulkan",
            "ctx": 4096,
            "ngl": 99,
            "ubatch": 512,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "flash_attn": True,
            "jinja": True,
            "boot_timeout_s": 300,
            "backend_start_port": 5800,
        },
        "models": [
            {"id": "m_a", "alias": "alias-a", "gguf": "org/repo/a.gguf"},
            {"id": "m_b", "alias": "alias-b", "gguf": "org/repo/b.gguf"},
        ],
        "judge": {"enabled": False},
    }
    base.update(overrides)
    return base


# --- Globals -----------------------------------------------------------------


class TestGlobals:
    def test_health_check_timeout_wired_from_boot_timeout(self):
        cfg = _cfg()
        cfg["server"]["boot_timeout_s"] = 420
        out = build(cfg, MODELS_DIR)
        assert out["healthCheckTimeout"] == 420

    def test_start_port_wired_from_backend_start_port(self):
        cfg = _cfg()
        cfg["server"]["backend_start_port"] = 5900
        out = build(cfg, MODELS_DIR)
        assert out["startPort"] == 5900

    def test_global_ttl_is_zero(self):
        # Non-zero globalTTL would let an idle model unload mid-matrix; the
        # next call would silently re-boot inside the timed span and poison
        # per-call metrics.
        out = build(_cfg(), MODELS_DIR)
        assert out["globalTTL"] == 0

    def test_send_loading_state_disabled(self):
        # llama-swap injects a loading message into `reasoning_content` when
        # true. The ChatClient falls back to reasoning_content when content
        # is empty, so warmup could capture the loading text as the answer.
        out = build(_cfg(), MODELS_DIR)
        assert out["sendLoadingState"] is False

    def test_log_level_info(self):
        out = build(_cfg(), MODELS_DIR)
        assert out["logLevel"] == "info"

    def test_defaults_apply_when_server_block_missing(self):
        cfg = {"models": [{"id": "m", "gguf": "o/r/f.gguf"}]}
        out = build(cfg, MODELS_DIR)
        assert out["healthCheckTimeout"] == 300
        assert out["startPort"] == 5800


# --- Per-model block ---------------------------------------------------------


class TestModelBlock:
    def test_minimum_block_fields(self):
        out = build(_cfg(), MODELS_DIR)
        block = out["models"]["m_a"]
        assert block["proxy"] == "http://127.0.0.1:${PORT}"
        assert block["checkEndpoint"] == HEALTH_ENDPOINT
        assert block["cmdStop"] == f"podman stop {container_name('m_a')}"
        assert block["ttl"] == 0

    def test_cmd_contains_gguf_and_alias(self):
        out = build(_cfg(), MODELS_DIR)
        cmd = out["models"]["m_a"]["cmd"]
        assert "/m/org/repo/a.gguf" in cmd
        assert "-a alias-a" in cmd

    def test_cmd_uses_podman_run_with_rm(self):
        out = build(_cfg(), MODELS_DIR)
        cmd = out["models"]["m_a"]["cmd"]
        assert cmd.startswith("podman run --rm ")

    def test_cmd_binds_localhost_to_port_macro(self):
        out = build(_cfg(), MODELS_DIR)
        cmd = out["models"]["m_a"]["cmd"]
        # Container listens on 8080; llama-swap maps host ${PORT} -> 8080.
        assert "-p 127.0.0.1:${PORT}:8080" in cmd

    def test_cmd_mounts_models_dir_read_only(self):
        out = build(_cfg(), MODELS_DIR)
        cmd = out["models"]["m_a"]["cmd"]
        assert f"-v {MODELS_DIR}:/m:ro" in cmd

    def test_cmd_container_name_stable(self):
        out = build(_cfg(), MODELS_DIR)
        cmd = out["models"]["m_a"]["cmd"]
        assert "--name bench-llama-m_a" in cmd

    def test_cmd_has_device_dri_for_gpu(self):
        out = build(_cfg(), MODELS_DIR)
        cmd = out["models"]["m_a"]["cmd"]
        assert "--device /dev/dri" in cmd

    def test_cmd_server_flags(self):
        out = build(_cfg(), MODELS_DIR)
        cmd = out["models"]["m_a"]["cmd"]
        assert "-c 4096" in cmd
        assert "-ngl 99" in cmd
        assert "-ub 512" in cmd
        assert "-ctk q8_0" in cmd
        assert "-ctv q8_0" in cmd
        assert "-fa on" in cmd
        assert "--jinja" in cmd

    def test_flash_attn_disabled_omits_flag(self):
        cfg = _cfg()
        cfg["server"]["flash_attn"] = False
        out = build(cfg, MODELS_DIR)
        assert "-fa on" not in out["models"]["m_a"]["cmd"]

    def test_jinja_disabled_omits_flag(self):
        cfg = _cfg()
        cfg["server"]["jinja"] = False
        out = build(cfg, MODELS_DIR)
        assert "--jinja" not in out["models"]["m_a"]["cmd"]

    def test_alias_defaults_to_id_when_absent(self):
        cfg = _cfg()
        cfg["models"] = [{"id": "lonely", "gguf": "o/r/f.gguf"}]
        out = build(cfg, MODELS_DIR)
        assert "-a lonely" in out["models"]["lonely"]["cmd"]

    def test_per_model_ctx_override(self):
        cfg = _cfg()
        cfg["models"][0]["ctx"] = 8192
        out = build(cfg, MODELS_DIR)
        assert "-c 8192" in out["models"]["m_a"]["cmd"]
        # The other model still uses server.ctx.
        assert "-c 4096" in out["models"]["m_b"]["cmd"]

    def test_n_cpu_moe_surfaces_when_set(self):
        cfg = _cfg()
        cfg["models"][0]["n_cpu_moe"] = 999
        out = build(cfg, MODELS_DIR)
        assert "--n-cpu-moe 999" in out["models"]["m_a"]["cmd"]

    def test_n_cpu_moe_absent_when_not_set(self):
        out = build(_cfg(), MODELS_DIR)
        assert "--n-cpu-moe" not in out["models"]["m_a"]["cmd"]

    def test_image_override_flows_into_cmd(self):
        cfg = _cfg()
        cfg["server"]["image"] = "ghcr.io/example/custom:latest"
        out = build(cfg, MODELS_DIR)
        assert "ghcr.io/example/custom:latest" in out["models"]["m_a"]["cmd"]


# --- Judge -------------------------------------------------------------------


class TestJudge:
    def test_judge_appended_when_enabled(self):
        cfg = _cfg()
        cfg["judge"] = {
            "enabled": True,
            "alias": "judge",
            "gguf": "org/repo/j.gguf",
            "ctx": 8192,
        }
        out = build(cfg, MODELS_DIR)
        assert "judge" in out["models"]
        cmd = out["models"]["judge"]["cmd"]
        assert "/m/org/repo/j.gguf" in cmd
        assert "-c 8192" in cmd
        assert "-a judge" in cmd

    def test_judge_skipped_when_disabled(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": False, "gguf": "org/repo/j.gguf"}
        out = build(cfg, MODELS_DIR)
        assert "judge" not in out["models"]

    def test_judge_skipped_when_no_gguf(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": True}
        out = build(cfg, MODELS_DIR)
        assert "judge" not in out["models"]

    def test_judge_custom_id(self):
        cfg = _cfg()
        cfg["judge"] = {
            "enabled": True,
            "id": "arbiter",
            "alias": "arbiter",
            "gguf": "org/repo/j.gguf",
        }
        out = build(cfg, MODELS_DIR)
        assert "arbiter" in out["models"]
        assert "judge" not in out["models"]

    def test_judge_collision_with_model_id_raises(self):
        cfg = _cfg()
        cfg["models"].append({"id": "judge", "gguf": "org/repo/m.gguf"})
        cfg["judge"] = {"enabled": True, "gguf": "org/repo/j.gguf"}
        with pytest.raises(ConfigError, match="collides"):
            build(cfg, MODELS_DIR)


# --- Validation --------------------------------------------------------------


class TestValidation:
    def test_duplicate_model_id_raises(self):
        cfg = _cfg()
        cfg["models"].append({"id": "m_a", "gguf": "org/repo/dup.gguf"})
        with pytest.raises(ConfigError, match="duplicate"):
            build(cfg, MODELS_DIR)

    def test_missing_id_raises(self):
        cfg = _cfg()
        cfg["models"] = [{"gguf": "org/repo/f.gguf"}]
        with pytest.raises(ConfigError, match="missing"):
            build(cfg, MODELS_DIR)

    def test_missing_gguf_raises(self):
        cfg = _cfg()
        cfg["models"] = [{"id": "m"}]
        with pytest.raises(ConfigError, match="missing"):
            build(cfg, MODELS_DIR)

    @pytest.mark.parametrize(
        "bad_id",
        [
            "has space",
            "slash/in/id",
            "-leadingdash",
            "has$dollar",
            "",
        ],
    )
    def test_bad_id_characters_rejected(self, bad_id):
        cfg = _cfg()
        cfg["models"] = [{"id": bad_id, "gguf": "org/repo/f.gguf"}]
        with pytest.raises(ConfigError):
            build(cfg, MODELS_DIR)

    @pytest.mark.parametrize(
        "gguf",
        [
            "org/repo/mmproj-F16.gguf",
            "org/repo/MMPROJ.gguf",
            "org/repo/subdir/mmproj-Q8_0.gguf",
        ],
    )
    def test_mmproj_rejected(self, gguf):
        cfg = _cfg()
        cfg["models"] = [{"id": "bad", "gguf": gguf}]
        with pytest.raises(ConfigError, match="mmproj"):
            build(cfg, MODELS_DIR)

    def test_non_mmproj_with_mmproj_in_path_allowed(self):
        # Only the *basename* is checked; a repo named "mmproj-friend" that
        # ships a real text model called "weights.gguf" must still work.
        cfg = _cfg()
        cfg["models"] = [
            {"id": "ok", "gguf": "org/mmproj-friend/weights.gguf"},
        ]
        out = build(cfg, MODELS_DIR)
        assert "ok" in out["models"]


# --- Helpers -----------------------------------------------------------------


class TestContainerName:
    def test_prefix_applied(self):
        assert container_name("foo").startswith(CONTAINER_NAME_PREFIX)
        assert container_name("foo") == "bench-llama-foo"

    def test_round_trip_into_cmd_stop(self):
        out = build(_cfg(), MODELS_DIR)
        for mid, block in out["models"].items():
            assert block["cmdStop"] == f"podman stop {container_name(mid)}"
