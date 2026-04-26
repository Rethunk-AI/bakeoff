"""Unit tests for bench.provenance."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bench.provenance import _infer_quantization, build_model_metadata, collect


# --- _infer_quantization ----------------------------------------------------


class TestInferQuantization:
    @pytest.mark.parametrize("filename,expected", [
        ("model-Q4_K_M.gguf", "Q4_K_M"),
        ("model-IQ3_XS.gguf", "IQ3_XS"),
        ("model-Q8_0.gguf", "Q8_0"),
        ("model-BF16.gguf", "BF16"),
        ("model-F16.gguf", "F16"),
        ("model-Q5_K_S.gguf", "Q5_K_S"),
        ("no-quant-info.gguf", None),
        ("", None),
    ])
    def test_known_patterns(self, filename, expected):
        assert _infer_quantization(filename) == expected


# --- build_model_metadata ---------------------------------------------------


class TestBuildModelMetadata:
    def _cfg(self, models=None, server_ctx=4096):
        return {
            "server": {"ctx": server_ctx, "image": "ghcr.io/ggml-org/llama.cpp:server-vulkan"},
            "models": models or [
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
        with patch("bench.provenance._git_info") as git_mock, \
             patch("bench.provenance._podman_version", return_value="4.9.0"), \
             patch("bench.provenance._llama_swap_version", return_value="0.0.8"), \
             patch("bench.provenance._package_versions", return_value={"httpx": "0.27.0"}):
            git_mock.return_value = {"sha": "abc1234", "branch": "main", "dirty": False}
            prov = collect(self._cfg(), seed=42, repo_root=tmp_path)

        for key in ("git", "config_hash", "seed", "python", "platform",
                    "packages", "podman_version", "llama_swap_version",
                    "server_image", "warnings"):
            assert key in prov

    def test_git_sha_present(self, tmp_path):
        with patch("bench.provenance._git_info") as git_mock, \
             patch("bench.provenance._podman_version", return_value=None), \
             patch("bench.provenance._llama_swap_version", return_value=None), \
             patch("bench.provenance._package_versions", return_value={}):
            git_mock.return_value = {"sha": "deadbeef", "branch": "main", "dirty": True}
            prov = collect(self._cfg(), seed=99, repo_root=tmp_path)

        assert prov["git"]["sha"] == "deadbeef"
        assert prov["git"]["dirty"] is True
        assert prov["seed"] == 99

    def test_missing_git_adds_warning(self, tmp_path):
        with patch("bench.provenance._git_info") as git_mock, \
             patch("bench.provenance._podman_version", return_value=None), \
             patch("bench.provenance._llama_swap_version", return_value=None), \
             patch("bench.provenance._package_versions", return_value={}):
            git_mock.return_value = {"sha": None, "branch": None, "dirty": False}
            prov = collect(self._cfg(), seed=0, repo_root=tmp_path)

        assert any("git" in w for w in prov["warnings"])

    def test_missing_podman_adds_warning(self, tmp_path):
        with patch("bench.provenance._git_info") as git_mock, \
             patch("bench.provenance._podman_version", return_value=None), \
             patch("bench.provenance._llama_swap_version", return_value="1.0"), \
             patch("bench.provenance._package_versions", return_value={}):
            git_mock.return_value = {"sha": "abc", "branch": "main", "dirty": False}
            prov = collect(self._cfg(), seed=0, repo_root=tmp_path)

        assert any("podman" in w for w in prov["warnings"])

    def test_config_hash_stable(self, tmp_path):
        cfg = self._cfg()
        with patch("bench.provenance._git_info") as git_mock, \
             patch("bench.provenance._podman_version", return_value="4.0"), \
             patch("bench.provenance._llama_swap_version", return_value="0.8"), \
             patch("bench.provenance._package_versions", return_value={}):
            git_mock.return_value = {"sha": "abc", "branch": "main", "dirty": False}
            p1 = collect(cfg, seed=42, repo_root=tmp_path)
            p2 = collect(cfg, seed=42, repo_root=tmp_path)

        assert p1["config_hash"] == p2["config_hash"]
        assert len(p1["config_hash"]) == 16

    def test_server_image_captured(self, tmp_path):
        with patch("bench.provenance._git_info") as git_mock, \
             patch("bench.provenance._podman_version", return_value="4.0"), \
             patch("bench.provenance._llama_swap_version", return_value="0.8"), \
             patch("bench.provenance._package_versions", return_value={}):
            git_mock.return_value = {"sha": "abc", "branch": "main", "dirty": False}
            prov = collect(self._cfg(), seed=42, repo_root=tmp_path)

        assert prov["server_image"] == "ghcr.io/ggml-org/llama.cpp:server-vulkan"
