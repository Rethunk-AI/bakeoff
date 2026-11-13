"""Unit tests for bench.clients SSE parsing.

The network paths (`_chat_stream` / `_chat_blocking`) are not exercised here —
those would need an httpx mock transport and are integration-flavored. These
tests cover the pure parsers only.
"""

from __future__ import annotations

import httpx

from bench.clients import _engined_headers, _extract_delta, _parse_sse_chunk


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
    c, r = _extract_delta(
        {
            "choices": [{"delta": {"content": "ans", "reasoning_content": "cot"}}],
        }
    )
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


def test_engined_headers_captures_route_and_queue_ms():
    headers = httpx.Headers({"x-engined-route": "@/llama-bench/m_a", "x-engined-queue-ms": "12"})
    assert _engined_headers(headers) == {
        "x-engined-route": "@/llama-bench/m_a",
        "x-engined-queue-ms": "12",
    }


def test_engined_headers_absent_when_not_engined():
    assert _engined_headers(httpx.Headers({})) == {}


def test_engined_headers_route_only():
    headers = httpx.Headers({"x-engined-route": "@/llama-bench/judge"})
    assert _engined_headers(headers) == {"x-engined-route": "@/llama-bench/judge"}


def test_extract_delta_usage_chunk_has_no_delta():
    # Final usage chunk in llama.cpp SSE stream: choices is empty array
    # with stream_options.include_usage=true.
    c, r = _extract_delta(
        {
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 7},
        }
    )
    assert c == ""
    assert r == ""


def test_stream_error_chunk_fails_the_call(monkeypatch):
    import pytest

    from bench.clients import ChatClient

    body = 'data: {"error": {"message": "out of memory"}}\n\ndata: [DONE]\n\n'
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body))
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=transport, **kw))
    with pytest.raises(RuntimeError, match="out of memory"):
        ChatClient("http://x/v1", "m").chat([{"role": "user", "content": "hi"}])


def _stream_client(monkeypatch, body: str):
    from bench.clients import ChatClient

    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body))
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=transport, **kw))
    return ChatClient("http://x/v1", "m")


def test_decode_tps_excludes_prefill():
    from bench.clients import _decode_tps

    # 11 tokens, first at 5s, done at 6s: 10 decode tokens in 1s.
    assert _decode_tps(11, 6.0, 5.0) == 10.0
    assert _decode_tps(None, 6.0, 5.0) is None
    assert _decode_tps(0, 6.0, 5.0) is None


def test_stream_without_usage_counts_deltas_not_zero(monkeypatch):
    body = (
        'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"b"}}]}\n\ndata: [DONE]\n\n'
    )
    res = _stream_client(monkeypatch, body).chat([{"role": "user", "content": "hi"}])
    assert res.completion_tokens == 2
    assert res.prompt_tokens is None


def test_stream_without_usage_or_deltas_is_unknown(monkeypatch):
    res = _stream_client(monkeypatch, "data: [DONE]\n\n").chat([{"role": "user", "content": "hi"}])
    assert res.completion_tokens is None
    assert res.tokens_per_sec is None
