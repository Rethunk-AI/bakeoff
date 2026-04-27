"""Unit tests for bench.provenance."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bench.provenance import (
    _infer_quantization,
    build_model_metadata,
    collect,
    enrich_model_metadata,
)

# --- _infer_quantization ----------------------------------------------------


class TestInferQuantization:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("model-Q4_K_M.gguf", "Q4_K_M"),
            ("model-IQ3_XS.gguf", "IQ3_XS"),
            ("model-Q8_0.gguf", "Q8_0"),
            ("model-BF16.gguf", "BF16"),
            ("model-F16.gguf", "F16"),
            ("model-Q5_K_S.gguf", "Q5_K_S"),
            ("no-quant-info.gguf", None),
            ("", None),
        ],
    )
    def test_known_patterns(self, filename, expected):
        assert _infer_quantization(filename) == expected


# --- build_model_metadata ---------------------------------------------------


class TestBuildModelMetadata:
    def _cfg(self, models=None, server_ctx=4096):
        return {
            "server": {"ctx": server_ctx, "image": "ghcr.io/ggml-org/llama.cpp:server-vulkan"},
            "models": models
            or [
                {"id": "m_a", "alias": "alpha", "gguf": "org/repo/model-Q4_K_M.gguf"},
            ],
        }

    def test_basic_fields(self):
        meta = build_model_metadata(self._cfg())
        assert len(meta) == 1
        m = meta[0]
        assert m["id"] == "m_a"
        assert m["alias"] == "alpha"
        assert m["repo_id"] == "org/repo"
        assert m["filename"] == "model-Q4_K_M.gguf"
        assert m["quantization"] == "Q4_K_M"
        assert m["ctx"] == 4096

    def test_ctx_override(self):
        models = [{"id": "m_a", "gguf": "org/repo/a.gguf", "ctx": 2048}]
        meta = build_model_metadata(self._cfg(models=models))
        assert meta[0]["ctx"] == 2048

    def test_n_cpu_moe(self):
        models = [{"id": "m_a", "gguf": "org/repo/a.gguf", "n_cpu_moe": 999}]
        meta = build_model_metadata(self._cfg(models=models))
        assert meta[0]["n_cpu_moe"] == 999

    def test_empty_models(self):
        assert build_model_metadata({"server": {}, "models": []}) == []

    def test_gguf_path_parsing(self):
        models = [{"id": "x", "gguf": "myorg/myrepo/myfile-Q5_K_S.gguf"}]
        meta = build_model_metadata({"server": {}, "models": models})
        m = meta[0]
        assert m["repo_id"] == "myorg/myrepo"
        assert m["filename"] == "myfile-Q5_K_S.gguf"
        assert m["quantization"] == "Q5_K_S"

    def test_short_gguf_path(self):
        models = [{"id": "x", "gguf": "badpath.gguf"}]
        meta = build_model_metadata({"server": {}, "models": models})
        assert meta[0]["repo_id"] is None


# --- collect ----------------------------------------------------------------


class TestCollect:
    def _cfg(self):
        return {
            "server": {"ctx": 4096, "image": "ghcr.io/ggml-org/llama.cpp:server-vulkan"},
            "models": [{"id": "m_a", "gguf": "org/repo/a-Q4_K_M.gguf"}],
            "dataset": {"n": 10, "domains": ["qa"]},
            "prompts": [{"id": "p1"}],
        }

    def test_returns_required_keys(self, tmp_path):
        with (
            patch("bench.provenance._git_info") as git_mock,
            patch("bench.provenance._podman_version", return_value="4.9.0"),
            patch("bench.provenance._llama_swap_version", return_value="0.0.8"),
            patch("bench.provenance._package_versions", return_value={"httpx": "0.27.0"}),
        ):
            git_mock.return_value = {"sha": "abc1234", "branch": "main", "dirty": False}
            prov = collect(self._cfg(), seed=42, repo_root=tmp_path)

        for key in (
            "git",
            "config_hash",
            "seed",
            "python",
            "platform",
            "packages",
            "podman_version",
            "llama_swap_version",
            "server_image",
            "warnings",
        ):
            assert key in prov

    def test_git_sha_present(self, tmp_path):
        with (
            patch("bench.provenance._git_info") as git_mock,
            patch("bench.provenance._podman_version", return_value=None),
            patch("bench.provenance._llama_swap_version", return_value=None),
            patch("bench.provenance._package_versions", return_value={}),
        ):
            git_mock.return_value = {"sha": "deadbeef", "branch": "main", "dirty": True}
            prov = collect(self._cfg(), seed=99, repo_root=tmp_path)

        assert prov["git"]["sha"] == "deadbeef"
        assert prov["git"]["dirty"] is True
        assert prov["seed"] == 99

    def test_missing_git_adds_warning(self, tmp_path):
        with (
            patch("bench.provenance._git_info") as git_mock,
            patch("bench.provenance._podman_version", return_value=None),
            patch("bench.provenance._llama_swap_version", return_value=None),
            patch("bench.provenance._package_versions", return_value={}),
        ):
            git_mock.return_value = {"sha": None, "branch": None, "dirty": False}
            prov = collect(self._cfg(), seed=0, repo_root=tmp_path)

        assert any("git" in w for w in prov["warnings"])

    def test_missing_podman_adds_warning(self, tmp_path):
        with (
            patch("bench.provenance._git_info") as git_mock,
            patch("bench.provenance._podman_version", return_value=None),
            patch("bench.provenance._llama_swap_version", return_value="1.0"),
            patch("bench.provenance._package_versions", return_value={}),
        ):
            git_mock.return_value = {"sha": "abc", "branch": "main", "dirty": False}
            prov = collect(self._cfg(), seed=0, repo_root=tmp_path)

        assert any("podman" in w for w in prov["warnings"])

    def test_config_hash_stable(self, tmp_path):
        cfg = self._cfg()
        with (
            patch("bench.provenance._git_info") as git_mock,
            patch("bench.provenance._podman_version", return_value="4.0"),
            patch("bench.provenance._llama_swap_version", return_value="0.8"),
            patch("bench.provenance._package_versions", return_value={}),
        ):
            git_mock.return_value = {"sha": "abc", "branch": "main", "dirty": False}
            p1 = collect(cfg, seed=42, repo_root=tmp_path)
            p2 = collect(cfg, seed=42, repo_root=tmp_path)

        assert p1["config_hash"] == p2["config_hash"]
        assert len(p1["config_hash"]) == 16

    def test_server_image_captured(self, tmp_path):
        with (
            patch("bench.provenance._git_info") as git_mock,
            patch("bench.provenance._podman_version", return_value="4.0"),
            patch("bench.provenance._llama_swap_version", return_value="0.8"),
            patch("bench.provenance._package_versions", return_value={}),
        ):
            git_mock.return_value = {"sha": "abc", "branch": "main", "dirty": False}
            prov = collect(self._cfg(), seed=42, repo_root=tmp_path)

        assert prov["server_image"] == "ghcr.io/ggml-org/llama.cpp:server-vulkan"


# --- enrich_model_metadata --------------------------------------------------


def _fake_info(sha="abc123", tags=None, pipeline_tag="text-generation", private=False):
    return SimpleNamespace(
        sha=sha,
        tags=tags or ["gguf", "llama"],
        pipeline_tag=pipeline_tag,
        private=private,
    )


def _meta(repo_id="org/repo"):
    return [{"id": "m_a", "repo_id": repo_id, "filename": "model.gguf"}]


class TestEnrichModelMetadata:
    def test_off_mode_is_noop(self):
        result = enrich_model_metadata(_meta(), mode="off", warnings=[])
        assert "hf_sha" not in result[0]

    def test_best_effort_adds_hf_fields(self):
        fake = _fake_info()
        with patch(
            "bench.provenance._hf_model_info",
            return_value={
                "hf_sha": fake.sha,
                "hf_tags": list(fake.tags),
                "hf_pipeline_tag": fake.pipeline_tag,
                "hf_private": fake.private,
            },
        ):
            result = enrich_model_metadata(_meta(), mode="best-effort", warnings=[])
        assert result[0]["hf_sha"] == "abc123"
        assert "gguf" in result[0]["hf_tags"]
        assert result[0]["hf_pipeline_tag"] == "text-generation"

    def test_best_effort_failure_appends_warning(self):
        warnings: list[str] = []
        with patch("bench.provenance._hf_model_info", side_effect=Exception("404")):
            result = enrich_model_metadata(_meta(), mode="best-effort", warnings=warnings)
        assert any("HuggingFace" in w for w in warnings)
        assert "hf_sha" not in result[0]

    def test_strict_mode_raises_on_failure(self):
        with (
            patch("bench.provenance._hf_model_info", side_effect=Exception("timeout")),
            pytest.raises(RuntimeError, match="HuggingFace lookup failed"),
        ):
            enrich_model_metadata(_meta(), mode="strict", warnings=[])

    def test_strict_mode_succeeds_when_lookup_works(self):
        with patch(
            "bench.provenance._hf_model_info",
            return_value={
                "hf_sha": "xyz",
                "hf_tags": [],
                "hf_pipeline_tag": None,
                "hf_private": False,
            },
        ):
            result = enrich_model_metadata(_meta(), mode="strict", warnings=[])
        assert result[0]["hf_sha"] == "xyz"

    def test_no_repo_id_skips_enrichment(self):
        meta = [{"id": "m_a", "repo_id": None, "filename": "model.gguf"}]
        with patch("bench.provenance._hf_model_info") as mock_info:
            result = enrich_model_metadata(meta, mode="best-effort", warnings=[])
        mock_info.assert_not_called()
        assert "hf_sha" not in result[0]

    def test_preserves_existing_fields(self):
        with patch("bench.provenance._hf_model_info", return_value={"hf_sha": "q"}):
            result = enrich_model_metadata(_meta(), mode="best-effort", warnings=[])
        assert result[0]["id"] == "m_a"
        assert result[0]["filename"] == "model.gguf"

    def test_multiple_models_all_enriched(self):
        meta = [
            {"id": "m_a", "repo_id": "org/repo-a", "filename": "a.gguf"},
            {"id": "m_b", "repo_id": "org/repo-b", "filename": "b.gguf"},
        ]
        with patch("bench.provenance._hf_model_info", return_value={"hf_sha": "s"}):
            result = enrich_model_metadata(meta, mode="best-effort", warnings=[])
        assert all(r["hf_sha"] == "s" for r in result)
