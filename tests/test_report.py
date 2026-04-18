"""Unit tests for bench.report pure helpers (percentiles + rollup)."""
from __future__ import annotations

import math

import pytest

from bench.report import _pareto_frontier, _percentile, _rollup


# --- _percentile ----------------------------------------------------------

def test_percentile_empty_returns_none():
    assert _percentile([], 50) is None


def test_percentile_single_value():
    assert _percentile([5.0], 50) == 5.0
    assert _percentile([5.0], 99) == 5.0


def test_percentile_median_odd():
    assert _percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_percentile_median_even():
    # Linear-interpolation: midpoint between 2 and 3.
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_percentile_p95_five_values():
    # p95 of [1..5] with linear interp: k = 4 * 0.95 = 3.8 -> 4 + 0.8*(5-4) = 4.8
    assert math.isclose(_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95), 4.8)


def test_percentile_p99_clamps_to_max():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    # p99 ~ 4.96, bounded below the max.
    v = _percentile(xs, 99)
    assert v is not None
    assert v <= max(xs)
    assert v > _percentile(xs, 95)  # type: ignore[operator]


def test_percentile_unsorted_input_does_not_mutate():
    xs = [3.0, 1.0, 2.0]
    _percentile(xs, 50)
    assert xs == [3.0, 1.0, 2.0]


# --- _rollup percentile integration --------------------------------------

def _rec(model_id: str, latency: float, ttft: float | None = None):
    return {
        "task_id": "t0", "prompt_id": "p", "model_id": model_id,
        "text": "x", "latency_s": latency, "ttft_s": ttft,
        "tokens_per_sec": 10.0, "error": None,
    }


def test_rollup_emits_latency_percentiles():
    recs = [_rec("m1", v) for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
    out = _rollup(recs)
    assert out["m1"]["latency_p50_s"] == 3.0
    assert out["m1"]["latency_p95_s"] == pytest.approx(4.8)
    assert out["m1"]["latency_p99_s"] is not None


def test_rollup_ttft_none_when_all_missing():
    recs = [_rec("m1", 1.0, None), _rec("m1", 2.0, None)]
    out = _rollup(recs)
    assert out["m1"]["ttft_mean_s"] is None
    assert out["m1"]["ttft_p50_s"] is None


def test_rollup_ttft_populated_when_present():
    recs = [_rec("m1", 1.0, 0.1), _rec("m1", 2.0, 0.2), _rec("m1", 3.0, 0.3)]
    out = _rollup(recs)
    assert out["m1"]["ttft_mean_s"] == pytest.approx(0.2)
    assert out["m1"]["ttft_p50_s"] == pytest.approx(0.2)


def test_rollup_skips_errored_rows():
    recs = [
        _rec("m1", 1.0),
        {"model_id": "m1", "error": "boom"},
        _rec("m1", 3.0),
    ]
    out = _rollup(recs)
    assert out["m1"]["n"] == 2


# --- _pareto_frontier -----------------------------------------------------

def test_pareto_empty():
    assert _pareto_frontier([]) == []


def test_pareto_single_point_is_frontier():
    assert _pareto_frontier([("a", 0.8, 1.0)]) == ["a"]


def test_pareto_dominated_dropped():
    # b is dominated by a (same quality, more energy) -> a only.
    pts = [("a", 0.8, 1.0), ("b", 0.8, 2.0)]
    assert _pareto_frontier(pts) == ["a"]


def test_pareto_strict_dominance():
    # b has both worse quality and worse energy -> dropped.
    pts = [("a", 0.9, 1.0), ("b", 0.7, 2.0)]
    assert _pareto_frontier(pts) == ["a"]


def test_pareto_non_dominated_pair_sorted_by_energy():
    # Tradeoff pair: low-energy/low-quality + high-energy/high-quality.
    pts = [("cheap", 0.6, 0.5), ("strong", 0.95, 2.0)]
    assert _pareto_frontier(pts) == ["cheap", "strong"]


def test_pareto_three_models_mixed():
    # fast: low energy, low quality. balanced: medium. premium: best quality.
    # middling is dominated by balanced (same quality, higher energy).
    pts = [
        ("fast",      0.50, 0.30),
        ("balanced",  0.75, 1.00),
        ("middling",  0.75, 1.50),
        ("premium",   0.95, 3.00),
    ]
    assert _pareto_frontier(pts) == ["fast", "balanced", "premium"]


def test_pareto_ties_in_energy_best_quality_wins():
    # Two points at identical energy: the higher-quality one dominates.
    pts = [("low", 0.5, 1.0), ("high", 0.9, 1.0)]
    assert _pareto_frontier(pts) == ["high"]
