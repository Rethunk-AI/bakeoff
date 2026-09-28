"""Unit tests for the engined fragment generator.

The generator is pure — no subprocess, no filesystem, no network. These
tests lock the fragment's structure with a real `tomllib` round-trip so a
future schema drift surfaces here instead of at a live `systemctl reload`.
"""

from __future__ import annotations

import tomllib

import pytest

from bench.config import ConfigError
from bench.engined import ENGINE_ID, render_fragment, route_ids

SPEC_DIR = "/usr/local/src/com.github/Rethunk-Tech/engined/engines/llama"
MODELS_DIR = "/home/user/.local/share/engined-models/llm-bench"


def _cfg(**overrides):
    base = {
        "engined": {"spec_dir": SPEC_DIR, "models_dir": MODELS_DIR},
        "server": {
            "ctx": 4096,
            "ngl": 99,
            "ubatch": 512,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "flash_attn": True,
            "jinja": True,
        },
        "models": [
            {"id": "m_a", "gguf": "org/repo/a.gguf"},
            {"id": "m_b", "gguf": "org/repo/b.gguf"},
        ],
        "judge": {"enabled": False},
    }
    base.update(overrides)
    return base


def _parse(cfg):
    return tomllib.loads(render_fragment(cfg))


# --- Engine block --------------------------------------------------------


class TestEngineBlock:
    def test_engine_declared_once(self):
        parsed = _parse(_cfg())
        engines = parsed["engine"]
        assert len(engines) == 1
        assert engines[0]["id"] == ENGINE_ID

    def test_spec_dir_and_models_dir_wired(self):
        parsed = _parse(_cfg())
        engine = parsed["engine"][0]
        assert engine["spec_dir"] == SPEC_DIR
        assert engine["models_dir"] == MODELS_DIR

    def test_models_max_defaults_to_one(self):
        parsed = _parse(_cfg())
        assert parsed["engine"][0]["models_max"] == 1

    def test_models_max_override(self):
        cfg = _cfg()
        cfg["engined"]["models_max"] = 2
        parsed = _parse(cfg)
        assert parsed["engine"][0]["models_max"] == 2

    def test_missing_spec_dir_raises(self):
        cfg = _cfg()
        del cfg["engined"]["spec_dir"]
        with pytest.raises(ConfigError, match="spec_dir"):
            render_fragment(cfg)

    def test_missing_models_dir_raises(self):
        cfg = _cfg()
        del cfg["engined"]["models_dir"]
        with pytest.raises(ConfigError, match="models_dir"):
            render_fragment(cfg)


# --- Routes ----------------------------------------------------------------


class TestRoutes:
    def test_one_route_per_model(self):
        parsed = _parse(_cfg())
        routes = parsed["route"]
        assert {r["model"] for r in routes} == {"m_a", "m_b"}

    def test_route_fields(self):
        parsed = _parse(_cfg())
        route = next(r for r in parsed["route"] if r["model"] == "m_a")
        assert route["engine"] == ENGINE_ID
        assert route["upstream"] == "local"
        assert route["filename"] == "org/repo/a.gguf"
        assert route["role"] == "chat"

    def test_route_args_carry_server_flags(self):
        parsed = _parse(_cfg())
        args = next(r for r in parsed["route"] if r["model"] == "m_a")["args"]
        assert args["parallel"] == 1
        assert args["ctx-size"] == 4096
        assert args["ubatch-size"] == 512
        assert args["n-gpu-layers"] == 99
        assert args["cache-type-k"] == "q8_0"
        assert args["cache-type-v"] == "q8_0"
        assert args["flash-attn"] == "on"
        assert args["jinja"] is True

    def test_per_model_ctx_override(self):
        cfg = _cfg()
        cfg["models"][0]["ctx"] = 8192
        parsed = _parse(cfg)
        args_a = next(r for r in parsed["route"] if r["model"] == "m_a")["args"]
        args_b = next(r for r in parsed["route"] if r["model"] == "m_b")["args"]
        assert args_a["ctx-size"] == 8192
        assert args_b["ctx-size"] == 4096

    def test_n_cpu_moe_surfaces_when_set(self):
        cfg = _cfg()
        cfg["models"][0]["n_cpu_moe"] = 999
        parsed = _parse(cfg)
        args = next(r for r in parsed["route"] if r["model"] == "m_a")["args"]
        assert args["n-cpu-moe"] == 999

    def test_n_cpu_moe_absent_when_not_set(self):
        parsed = _parse(_cfg())
        args = next(r for r in parsed["route"] if r["model"] == "m_a")["args"]
        assert "n-cpu-moe" not in args

    def test_flash_attn_disabled_omits_flag(self):
        cfg = _cfg()
        cfg["server"]["flash_attn"] = False
        parsed = _parse(cfg)
        args = next(r for r in parsed["route"] if r["model"] == "m_a")["args"]
        assert "flash-attn" not in args

    def test_jinja_disabled_omits_flag(self):
        cfg = _cfg()
        cfg["server"]["jinja"] = False
        parsed = _parse(cfg)
        args = next(r for r in parsed["route"] if r["model"] == "m_a")["args"]
        assert "jinja" not in args


# --- Judge -------------------------------------------------------------------


class TestJudge:
    def test_judge_appended_when_enabled(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": True, "gguf": "org/repo/j.gguf", "ctx": 8192}
        parsed = _parse(cfg)
        route = next(r for r in parsed["route"] if r["model"] == "judge")
        assert route["filename"] == "org/repo/j.gguf"
        assert route["args"]["ctx-size"] == 8192

    def test_judge_skipped_when_disabled(self):
        parsed = _parse(_cfg())
        assert "judge" not in {r["model"] for r in parsed["route"]}

    def test_judge_skipped_when_no_gguf(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": True}
        parsed = _parse(cfg)
        assert "judge" not in {r["model"] for r in parsed["route"]}

    def test_judge_custom_id(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": True, "id": "arbiter", "gguf": "org/repo/j.gguf"}
        parsed = _parse(cfg)
        ids = {r["model"] for r in parsed["route"]}
        assert "arbiter" in ids
        assert "judge" not in ids

    def test_judge_collision_with_model_id_raises(self):
        cfg = _cfg()
        cfg["models"].append({"id": "judge", "gguf": "org/repo/m.gguf"})
        cfg["judge"] = {"enabled": True, "gguf": "org/repo/j.gguf"}
        with pytest.raises(ConfigError, match="collides"):
            render_fragment(cfg)


# --- Validation --------------------------------------------------------------


class TestValidation:
    def test_duplicate_model_id_raises(self):
        cfg = _cfg()
        cfg["models"].append({"id": "m_a", "gguf": "org/repo/dup.gguf"})
        with pytest.raises(ConfigError, match="duplicate"):
            render_fragment(cfg)

    def test_missing_id_raises(self):
        cfg = _cfg()
        cfg["models"] = [{"gguf": "org/repo/f.gguf"}]
        with pytest.raises(ConfigError, match="missing"):
            render_fragment(cfg)

    def test_missing_gguf_raises(self):
        cfg = _cfg()
        cfg["models"] = [{"id": "m"}]
        with pytest.raises(ConfigError, match="missing"):
            render_fragment(cfg)

    @pytest.mark.parametrize(
        "bad_id",
        ["has space", "slash/in/id", "Has-Upper", "has$dollar", "", "under_score_ok_but_UP"],
    )
    def test_bad_id_characters_rejected(self, bad_id):
        cfg = _cfg()
        cfg["models"] = [{"id": bad_id, "gguf": "org/repo/f.gguf"}]
        with pytest.raises(ConfigError):
            render_fragment(cfg)

    def test_lowercase_dash_underscore_id_allowed(self):
        cfg = _cfg()
        cfg["models"] = [{"id": "qwen3_6-35b", "gguf": "org/repo/f.gguf"}]
        parsed = _parse(cfg)
        assert {r["model"] for r in parsed["route"]} == {"qwen3_6-35b"}

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
            render_fragment(cfg)

    def test_non_mmproj_with_mmproj_in_path_allowed(self):
        cfg = _cfg()
        cfg["models"] = [{"id": "ok", "gguf": "org/mmproj-friend/weights.gguf"}]
        parsed = _parse(cfg)
        assert {r["model"] for r in parsed["route"]} == {"ok"}


# --- route_ids ---------------------------------------------------------------


class TestRouteIds:
    def test_matches_models_plus_judge(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": True, "gguf": "org/repo/j.gguf"}
        assert route_ids(cfg) == ["m_a", "m_b", "judge"]

    def test_no_judge_when_disabled(self):
        assert route_ids(_cfg()) == ["m_a", "m_b"]
