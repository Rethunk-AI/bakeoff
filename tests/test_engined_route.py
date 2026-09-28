"""The answering-route check and the per-record engined fields."""

from __future__ import annotations

from typing import Any

import pytest

from bench.clients import ChatResult
from bench.runner import call_one, engined_fields

ADDR = "@/llama-bench/m_a"


class _Answering:
    """A client whose one reply carries the given engined headers."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw

    def chat(self, _messages: list[dict[str, str]]) -> ChatResult:
        return ChatResult(
            text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            latency_s=0.01,
            ttft_s=None,
            tokens_per_sec=100.0,
            raw=self.raw,
        )


def _call(raw: dict[str, Any]) -> ChatResult:
    res, *_ = call_one(_Answering(raw), "", "hi", 0, False, 0.0, 1, expected_route=ADDR)
    return res


def test_matching_route_is_recorded_with_its_queue_wait() -> None:
    res = _call({"x-engined-route": ADDR, "x-engined-queue-ms": "7"})
    assert engined_fields(res) == {"engined_route": ADDR, "engined_queue_ms": 7}


@pytest.mark.parametrize("raw", [{}, {"x-engined-route": "@/llama-bench/m_b"}])
def test_a_missing_or_different_route_fails_the_row(raw: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="expected"):
        _call(raw)
