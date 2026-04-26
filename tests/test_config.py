"""Unit tests for bench.config — load, validate, hash."""
from __future__ import annotations

import pytest

from bench.config import (
    ConfigError,
    ValidationIssue,
    config_hash,
    load_config,
    validate_config,
)


def _cfg(**overrides):
    """Minimal valid bakeoff config. Override or deep-merge keys per test."""
    base = {
        "server": {"ctx": 4096, "ngl": 99, "boot_timeout_s": 300},
        "models": [
            {"id": "m_a", "gguf": "org/repo/a.gguf"},
            {"id": "m_b", "gguf": "org/repo/b.gguf"},
        ],
        "prompts": [{"id": "p1", "text": "hello"}],
        "dataset": {"n": 10, "domains": ["qa"]},
    }
    base.update(overrides)
    return base


# --- load_config -------------------------------------------------------------


class TestLoadConfig:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "missing.yaml")

    def test_yaml_parse_error(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("key: [unclosed")
        with pytest.raises(ConfigError, match="YAML parse error"):
            load_config(f)

    def test_not_a_mapping(self, tmp_path):
        f = tmp_path / "list.yaml"
        f.write_text("- a\n- b\n")
        with pytest.raises(ConfigError, match="did not parse to a mapping"):
            load_config(f)

    def test_valid(self, tmp_path):
        import yaml
        f = tmp_path / "ok.yaml"
        f.write_text(yaml.dump(_cfg()))
        data = load_config(f)
        assert data["dataset"]["n"] == 10


# --- validate_config — happy path -------------------------------------------


class TestValidateConfigValid:
    def test_minimal_valid(self):
        assert validate_config(_cfg()) == []

    def test_judge_enabled_pairwise(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": True, "mode": "pairwise_all", "gguf": "org/repo/judge.gguf"}
        assert validate_config(cfg) == []

    def test_judge_enabled_scored(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": True, "mode": "scored", "gguf": "org/repo/judge.gguf"}
        assert validate_config(cfg) == []

    def test_cost_section(self):
        cfg = _cfg()
        cfg["cost"] = {"enabled": True, "kwh_rate_usd": 0.12, "sample_hz": 2.0}
        assert validate_config(cfg) == []

    def test_output_dir(self):
        cfg = _cfg()
        cfg["output"] = {"dir": "results"}
        assert validate_config(cfg) == []


# --- validate_config — required sections ------------------------------------


class TestRequiredSections:
    def test_missing_server(self):
        cfg = _cfg()
        del cfg["server"]
        issues = validate_config(cfg)
        paths = [i.path for i in issues]
        assert "server" in paths

    def test_missing_models(self):
        cfg = _cfg()
        del cfg["models"]
        issues = validate_config(cfg)
        assert any(i.path == "models" for i in issues)

    def test_empty_models(self):
        cfg = _cfg(models=[])
        issues = validate_config(cfg)
        assert any(i.path == "models" for i in issues)

    def test_missing_prompts(self):
        cfg = _cfg(prompts=[])
        issues = validate_config(cfg)
        assert any(i.path == "prompts" for i in issues)

    def test_missing_dataset(self):
        cfg = _cfg()
        del cfg["dataset"]
        issues = validate_config(cfg)
        assert any(i.path == "dataset" for i in issues)


# --- validate_config — dataset ----------------------------------------------


class TestDataset:
    def test_missing_n(self):
        cfg = _cfg()
        cfg["dataset"] = {"domains": ["qa"]}
        issues = validate_config(cfg)
        assert any(i.path == "dataset.n" for i in issues)

    def test_zero_n(self):
        cfg = _cfg()
        cfg["dataset"]["n"] = 0
        issues = validate_config(cfg)
        assert any(i.path == "dataset.n" for i in issues)

    def test_negative_n(self):
        cfg = _cfg()
        cfg["dataset"]["n"] = -1
        issues = validate_config(cfg)
        assert any(i.path == "dataset.n" for i in issues)

    def test_float_n(self):
        cfg = _cfg()
        cfg["dataset"]["n"] = 1.5
        issues = validate_config(cfg)
        assert any(i.path == "dataset.n" for i in issues)

    def test_missing_domains(self):
        cfg = _cfg()
        cfg["dataset"] = {"n": 10}
        issues = validate_config(cfg)
        assert any(i.path == "dataset.domains" for i in issues)

    def test_unknown_domain(self):
        cfg = _cfg()
        cfg["dataset"]["domains"] = ["qa", "bogus"]
        issues = validate_config(cfg)
        assert any("bogus" in i.message and i.path == "dataset.domains" for i in issues)

    def test_all_valid_domains(self):
        cfg = _cfg()
        cfg["dataset"]["domains"] = ["qa", "code", "summarize", "classify"]
        assert validate_config(cfg) == []


# --- validate_config — models -----------------------------------------------


class TestModels:
    def test_missing_model_id(self):
        cfg = _cfg(models=[{"gguf": "org/repo/a.gguf"}])
        issues = validate_config(cfg)
        assert any(".id" in i.path for i in issues)

    def test_duplicate_model_id(self):
        cfg = _cfg(models=[
            {"id": "dup", "gguf": "org/repo/a.gguf"},
            {"id": "dup", "gguf": "org/repo/b.gguf"},
        ])
        issues = validate_config(cfg)
        assert any("duplicate" in i.message for i in issues)

    def test_missing_gguf(self):
        cfg = _cfg(models=[{"id": "m_a"}])
        issues = validate_config(cfg)
        assert any(".gguf" in i.path for i in issues)

    def test_bad_gguf_shape_no_org(self):
        cfg = _cfg(models=[{"id": "m_a", "gguf": "just-a-file.gguf"}])
        issues = validate_config(cfg)
        assert any(".gguf" in i.path for i in issues)

    def test_bad_gguf_no_extension(self):
        cfg = _cfg(models=[{"id": "m_a", "gguf": "org/repo/file"}])
        issues = validate_config(cfg)
        assert any(".gguf" in i.path for i in issues)

    def test_mmproj_rejected(self):
        cfg = _cfg(models=[{"id": "m_a", "gguf": "org/repo/mmproj-clip-vit.gguf"}])
        issues = validate_config(cfg)
        assert any("mmproj" in i.message or "projector" in i.message for i in issues)


# --- validate_config — prompts ----------------------------------------------


class TestPrompts:
    def test_missing_prompt_id(self):
        cfg = _cfg(prompts=[{"text": "hello"}])
        issues = validate_config(cfg)
        assert any(".id" in i.path for i in issues)

    def test_duplicate_prompt_id(self):
        cfg = _cfg(prompts=[
            {"id": "p1", "text": "a"},
            {"id": "p1", "text": "b"},
        ])
        issues = validate_config(cfg)
        assert any("duplicate" in i.message for i in issues)


# --- validate_config — judge ------------------------------------------------


class TestJudge:
    def test_bad_judge_mode(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": True, "mode": "tournament", "gguf": "org/repo/j.gguf"}
        issues = validate_config(cfg)
        assert any(i.path == "judge.mode" for i in issues)

    def test_judge_id_collision(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": True, "id": "m_a", "gguf": "org/repo/j.gguf"}
        issues = validate_config(cfg)
        assert any(i.path == "judge.id" and "collides" in i.message for i in issues)

    def test_judge_mmproj_gguf(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": True, "gguf": "org/repo/mmproj-clip.gguf"}
        issues = validate_config(cfg)
        assert any(i.path == "judge.gguf" for i in issues)

    def test_judge_disabled_no_validation(self):
        cfg = _cfg()
        cfg["judge"] = {"enabled": False, "mode": "bogus"}
        assert validate_config(cfg) == []


# --- validate_config — server numeric fields --------------------------------


class TestServerNumerics:
    @pytest.mark.parametrize("field", ["ctx", "ngl", "ubatch", "boot_timeout_s", "swap_port", "backend_start_port"])
    def test_negative_server_field(self, field):
        cfg = _cfg()
        cfg["server"][field] = -1
        issues = validate_config(cfg)
        assert any(i.path == f"server.{field}" for i in issues)

    @pytest.mark.parametrize("field", ["ctx", "ngl", "ubatch", "boot_timeout_s", "swap_port", "backend_start_port"])
    def test_zero_server_field(self, field):
        cfg = _cfg()
        cfg["server"][field] = 0
        issues = validate_config(cfg)
        assert any(i.path == f"server.{field}" for i in issues)

    def test_optional_server_fields_absent_ok(self):
        cfg = _cfg()
        assert validate_config(cfg) == []


# --- validate_config — cost -------------------------------------------------


class TestCost:
    def test_negative_kwh_rate(self):
        cfg = _cfg()
        cfg["cost"] = {"enabled": True, "kwh_rate_usd": -0.1}
        issues = validate_config(cfg)
        assert any(i.path == "cost.kwh_rate_usd" for i in issues)

    def test_zero_kwh_rate_ok(self):
        cfg = _cfg()
        cfg["cost"] = {"enabled": True, "kwh_rate_usd": 0}
        assert validate_config(cfg) == []

    def test_zero_sample_hz(self):
        cfg = _cfg()
        cfg["cost"] = {"enabled": True, "sample_hz": 0}
        issues = validate_config(cfg)
        assert any(i.path == "cost.sample_hz" for i in issues)

    def test_cost_disabled_no_validation(self):
        cfg = _cfg()
        cfg["cost"] = {"enabled": False, "kwh_rate_usd": -99}
        assert validate_config(cfg) == []


# --- validate_config — output -----------------------------------------------


class TestOutput:
    def test_output_dir_not_string(self):
        cfg = _cfg()
        cfg["output"] = {"dir": 123}
        issues = validate_config(cfg)
        assert any(i.path == "output.dir" for i in issues)

    def test_output_dir_string_ok(self):
        cfg = _cfg()
        cfg["output"] = {"dir": "results/"}
        assert validate_config(cfg) == []


# --- config_hash ------------------------------------------------------------


class TestConfigHash:
    def test_returns_16_hex(self):
        h = config_hash(_cfg())
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_config_same_hash(self):
        assert config_hash(_cfg()) == config_hash(_cfg())

    def test_different_config_different_hash(self):
        a = _cfg()
        b = _cfg()
        b["dataset"]["n"] = 999
        assert config_hash(a) != config_hash(b)

    def test_key_order_invariant(self):
        a = {"z": 1, "a": 2}
        b = {"a": 2, "z": 1}
        assert config_hash(a) == config_hash(b)


# --- ValidationIssue str ----------------------------------------------------


class TestValidationIssue:
    def test_str(self):
        v = ValidationIssue(path="models[0].id", message="required")
        assert str(v) == "models[0].id: required"
