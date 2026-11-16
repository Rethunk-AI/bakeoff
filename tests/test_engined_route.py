"""The answering-route check and the per-record engined fields."""

from __future__ import annotations

import random
from typing import Any

import pytest

from bench.clients import ChatResult
from bench.dataset import Task
from bench.runner import _pairwise_all_phase, _scored_phase, call_one, engined_fields

ADDR = "@/llama-bench/m_a"
JUDGE_ADDR = "@/llama-bench/judge"


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


class _Judge:
    """A judge client that always answers, with the given engined headers."""

    def __init__(self, text: str, raw: dict[str, Any]) -> None:
        self.text = text
        self.raw = raw

    def chat(self, _messages: list[dict[str, str]]) -> ChatResult:
        return ChatResult(
            text=self.text,
            prompt_tokens=1,
            completion_tokens=1,
            latency_s=0.01,
            ttft_s=None,
            tokens_per_sec=100.0,
            raw=self.raw,
        )


def _task() -> Task:
    return Task(id="t1", domain="qa", user_prompt="what?")


def _record(model_id: str) -> dict[str, Any]:
    return {"task_id": "t1", "prompt_id": "p1", "model_id": model_id, "text": "answer"}


@pytest.mark.parametrize("route", [JUDGE_ADDR, "@/llama-bench/m_b", None])
def test_scored_judge_verdict_from_another_route_is_not_recorded(
    route: str | None,
) -> None:
    raw = {} if route is None else {"x-engined-route": route}
    out = _scored_phase(
        _Judge("4", raw),
        [{"id": "m_a"}],
        [_task()],
        [{"id": "p1"}],
        [_record("m_a")],
        expected_route=JUDGE_ADDR,
    )
    assert len(out) == 1
    if route == JUDGE_ADDR:
        assert out[0]["score"] == 4 and not out[0].get("error")
    else:
        assert out[0]["score"] is None
        assert "expected" in out[0]["error"]


@pytest.mark.parametrize("route", [JUDGE_ADDR, "@/llama-bench/m_b", None])
def test_pairwise_judge_verdict_from_another_route_is_not_recorded(
    route: str | None,
) -> None:
    raw = {} if route is None else {"x-engined-route": route}
    out = _pairwise_all_phase(
        _Judge("A", raw),
        [{"id": "m_a"}, {"id": "m_b"}],
        [_task()],
        [{"id": "p1"}],
        [_record("m_a"), _record("m_b")],
        random.Random(0),
        expected_route=JUDGE_ADDR,
    )
    assert len(out) == 1
    if route == JUDGE_ADDR:
        assert out[0]["winner"] in {"A", "B", "TIE"} and not out[0].get("error")
    else:
        assert out[0]["winner"] is None
        assert "expected" in out[0]["error"]
