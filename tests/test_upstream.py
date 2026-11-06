"""Tests for bench.upstream: real parsing on a recorded Hugging Face API response."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import httpx
import pytest

from bench import descriptor, store, upstream

FIXTURE = Path(__file__).parent / "fixtures" / "hf_qwen3_8b_gguf.json"
REPO_URL = "https://huggingface.co/Qwen/Qwen3-8B-GGUF"
FILE_URL = f"{REPO_URL}/resolve/main/Qwen3-8B-Q4_K_M.gguf"


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BAKEOFF_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def info() -> dict:
    return json.loads(FIXTURE.read_text())


def _client(payload: dict, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(status, json=payload))
    )


def test_file_url_to_stored_descriptor(info):
    desc = upstream.pull_descriptor(FILE_URL, _client(info))
    assert desc["name"] == "Qwen3-8B-Q4_K_M"
    assert desc["quantization"] == "q4_k_m"
    assert desc["file_format"] == "GGUF"
    assert desc["architecture"] == "qwen3"
    assert desc["model_hash"].startswith("sha256:d98cdcbd")
    assert desc["model_source_size"] == 5027783488
    assert desc["sources"][0]["source_metadata"]["license"] == "apache-2.0"
    assert descriptor.validate_descriptor(desc) == []
    saved = descriptor.save_descriptor(desc)
    assert store.read_record("models", saved)["name"] == "Qwen3-8B-Q4_K_M"


def test_repo_url_with_several_ggufs_is_ambiguous(info):
    with pytest.raises(upstream.UpstreamError, match="several GGUF files"):
        upstream.pull_descriptor(REPO_URL, _client(info))


def test_repo_url_with_one_gguf_resolves(info):
    info["siblings"] = [s for s in info["siblings"] if s["rfilename"] != "Qwen3-8B-Q8_0.gguf"]
    desc = upstream.pull_descriptor(REPO_URL, _client(info))
    assert desc["sources"][0]["source_metadata"]["hf_file"] == "Qwen3-8B-Q4_K_M.gguf"


def test_license_from_tag_when_card_lacks_it(info):
    info["cardData"].pop("license")
    desc = upstream.pull_descriptor(FILE_URL, _client(info))
    assert desc["sources"][0]["source_metadata"]["license"] == "apache-2.0"


@pytest.mark.parametrize(
    ("mutate", "expect"),
    [
        (lambda i: (i["cardData"].pop("license"), i.update(tags=["gguf"])), "license"),
        (lambda i: i.update(gated="auto"), "gated"),
        (lambda i: i.update(private=True), "private"),
        (
            lambda i: next(
                s for s in i["siblings"] if s["rfilename"].endswith("Q4_K_M.gguf")
            ).update(size=0),
            "model_source_size",
        ),
    ],
)
def test_upstream_validation_rejects(info, mutate, expect):
    bad = copy.deepcopy(info)
    mutate(bad)
    with pytest.raises(upstream.UpstreamError, match=expect):
        upstream.pull_descriptor(FILE_URL, _client(bad))


@pytest.mark.parametrize(
    "url",
    ["https://example.com/a/b", "http://huggingface.co/a/b", "https://huggingface.co/onlyone"],
)
def test_unsupported_urls(url):
    with pytest.raises(upstream.UpstreamError):
        upstream.parse_hf_url(url)


def test_missing_file_and_http_error(info):
    with pytest.raises(upstream.UpstreamError, match="not found"):
        upstream.pull_descriptor(f"{REPO_URL}/blob/main/nope.gguf", _client(info))
    with pytest.raises(upstream.UpstreamError, match="fetch failed"):
        upstream.pull_descriptor(FILE_URL, _client({}, status=404))


@pytest.mark.skipif(
    not os.environ.get("BAKEOFF_NETWORK_TESTS"),
    reason="live Hugging Face call; set BAKEOFF_NETWORK_TESTS=1",
)
def test_live_hugging_face():
    desc = upstream.pull_descriptor(FILE_URL)
    assert desc["file_format"] == "GGUF"
    assert desc["sources"][0]["source_metadata"]["license"]
