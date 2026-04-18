"""Unit tests for bench.clients SSE parsing.

The network paths (`_chat_stream` / `_chat_blocking`) are not exercised here —
those would need an httpx mock transport and are integration-flavored. These
tests cover the pure parsers only.
"""
from __future__ import annotations

from bench.clients import _extract_delta, _parse_sse_chunk


def test_parse_sse_chunk_plain():
    chunk = _parse_sse_chunk('data: {"choices":[{"delta":{"content":"hi"}}]}')
    assert chunk is not None
    assert chunk["choices"][0]["delta"]["content"] == "hi"


def test_parse_sse_chunk_done_sentinel():
    assert _parse_sse_chunk("data: [DONE]") is None


def test_parse_sse_chunk_empty_payload():
    assert _parse_sse_chunk("data: ") is None


def test_parse_sse_chunk_non_data_line():
    assert _parse_sse_chunk(": keep-alive") is None
    assert _parse_sse_chunk("event: ping") is None


def test_parse_sse_chunk_malformed_json():
    assert _parse_sse_chunk("data: not-json{") is None


def test_extract_delta_content_only():
    c, r = _extract_delta({"choices": [{"delta": {"content": "hello"}}]})
    assert c == "hello"
    assert r == ""


def test_extract_delta_reasoning_only():
    c, r = _extract_delta({"choices": [{"delta": {"reasoning_content": "think"}}]})
    assert c == ""
    assert r == "think"


def test_extract_delta_both():
    c, r = _extract_delta({
        "choices": [{"delta": {"content": "ans", "reasoning_content": "cot"}}],
    })
    assert c == "ans"
    assert r == "cot"


def test_extract_delta_no_choices():
    c, r = _extract_delta({})
    assert c == ""
    assert r == ""


def test_extract_delta_empty_delta():
    c, r = _extract_delta({"choices": [{"delta": {}}]})
    assert c == ""
    assert r == ""


def test_extract_delta_usage_chunk_has_no_delta():
    # Final usage chunk in llama.cpp SSE stream: choices is empty array
    # with stream_options.include_usage=true.
    c, r = _extract_delta({
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 7},
    })
    assert c == ""
    assert r == ""
