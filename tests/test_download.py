"""Unit tests for pure helpers in bench.download (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

import bench.download as bench_download
from bench.download import (
    collect_from_config,
    human_bytes,
    split_gguf_path,
    target_path,
)


class TestSplitGgufPath:
    def test_simple(self):
        assert split_gguf_path("org/repo/file.gguf") == ("org/repo", "file.gguf")

    def test_nested_filename(self):
        assert split_gguf_path("org/repo/sub/file.gguf") == ("org/repo", "sub/file.gguf")

    def test_deeply_nested(self):
        assert split_gguf_path("a/b/c/d/e.gguf") == ("a/b", "c/d/e.gguf")

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            split_gguf_path("bad/path")

    def test_single_segment_raises(self):
        with pytest.raises(ValueError):
            split_gguf_path("single.gguf")


class TestCollectFromConfig:
    def test_models_only(self):
        cfg = {
            "models": [
                {"id": "a", "gguf": "org1/repoA/fileA.gguf"},
                {"id": "b", "gguf": "org2/repoB/fileB.gguf"},
            ],
        }
        assert collect_from_config(cfg) == [
            ("org1/repoA", "fileA.gguf"),
            ("org2/repoB", "fileB.gguf"),
        ]

    def test_judge_included(self):
        cfg = {
            "models": [{"id": "a", "gguf": "o1/r1/f1.gguf"}],
            "judge": {"gguf": "o2/r2/f2.gguf"},
        }
        assert collect_from_config(cfg) == [
            ("o1/r1", "f1.gguf"),
            ("o2/r2", "f2.gguf"),
        ]

    def test_dedup_judge_matching_model(self):
        cfg = {
            "models": [{"id": "a", "gguf": "o/r/f.gguf"}],
            "judge": {"gguf": "o/r/f.gguf"},
        }
        assert collect_from_config(cfg) == [("o/r", "f.gguf")]

    def test_empty_config(self):
        assert collect_from_config({}) == []

    def test_models_missing_gguf_skipped(self):
        cfg = {"models": [{"id": "a"}, {"id": "b", "gguf": "o/r/f.gguf"}]}
        assert collect_from_config(cfg) == [("o/r", "f.gguf")]

    def test_judge_without_gguf_ignored(self):
        cfg = {"judge": {"enabled": False}}
        assert collect_from_config(cfg) == []


class TestHumanBytes:
    def test_none(self):
        assert human_bytes(None) == "?"

    def test_bytes(self):
        assert human_bytes(512) == "512 B"

    def test_kib(self):
        assert human_bytes(2048) == "2.0 KiB"

    def test_mib(self):
        assert human_bytes(5 * 1024 * 1024) == "5.0 MiB"

    def test_gib(self):
        assert human_bytes(3 * 1024**3) == "3.0 GiB"


class TestTargetPath:
    def test_joins_parts(self, tmp_path: Path):
        p = target_path(tmp_path, "org/repo", "file.gguf")
        assert p == tmp_path / "org" / "repo" / "file.gguf"

    def test_nested_filename(self, tmp_path: Path):
        p = target_path(tmp_path, "org/repo", "sub/file.gguf")
        assert p == tmp_path / "org" / "repo" / "sub" / "file.gguf"

    def test_rejects_filename_escaping_models_dir(self, tmp_path: Path):
        with pytest.raises(ValueError, match="escapes"):
            target_path(tmp_path, "org/repo", "../../../evil.gguf")

    def test_rejects_malformed_repo_id(self, tmp_path: Path):
        with pytest.raises(ValueError, match="invalid repo_id"):
            target_path(tmp_path, "repoid", "file.gguf")
        with pytest.raises(ValueError, match="invalid repo_id"):
            target_path(tmp_path, "org/repo/extra", "file.gguf")

    def test_download_one_creates_nothing_for_invalid_repo_id(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(bench_download, "hf_hub_download", lambda **kw: calls.append(kw))
        models_dir = tmp_path / "models"
        with pytest.raises(ValueError, match="invalid repo_id"):
            bench_download.download_one(models_dir, "org/repo/extra", "file.gguf", None)
        assert calls == []
        assert not models_dir.exists()
