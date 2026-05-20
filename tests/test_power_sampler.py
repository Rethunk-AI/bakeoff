"""Unit tests for the high-frequency power sampler.

Covers the pure integration helper (`trapezoid_wh`) and the sampler
lifecycle (`__enter__` / `__exit__`) with `sample_power` monkey-patched
so no subprocess is spawned.
"""

from __future__ import annotations

import time

import pytest

from bench import metrics
from bench.metrics import PowerSampler, trapezoid_wh

# --- trapezoid_wh ---------------------------------------------------------


def test_trapezoid_empty_is_none():
    assert trapezoid_wh([]) is None


def test_trapezoid_single_sample_is_zero():
    # Single sample has no duration to integrate over; caller falls back
    # to rectangle rule in PowerSampler.energy_wh.
    assert trapezoid_wh([(0.0, 100.0)]) == 0.0


def test_trapezoid_constant_power():
    # 100W for 3600s = 100 Wh.
    samples = [(0.0, 100.0), (1800.0, 100.0), (3600.0, 100.0)]
    assert trapezoid_wh(samples) == pytest.approx(100.0)


def test_trapezoid_linear_ramp():
    # 0W -> 200W over 3600s is a triangle. Area = 0.5 * 200 * 3600 Ws
    # = 360_000 Ws = 100 Wh.
    samples = [(0.0, 0.0), (3600.0, 200.0)]
    assert trapezoid_wh(samples) == pytest.approx(100.0)


def test_trapezoid_step_function():
    # 50W for 1s, jump to 150W for 1s. Area = 50 + (50+150)/2 + 150 is wrong;
    # it's piecewise linear between samples.
    # Segments: (0,50)->(1,50): 50 Ws. (1,50)->(2,150): 100 Ws. Total 150 Ws.
    samples = [(0.0, 50.0), (1.0, 50.0), (2.0, 150.0)]
    ws = trapezoid_wh(samples)
    assert ws == pytest.approx(150.0 / 3600.0)


# --- PowerSampler lifecycle ----------------------------------------------


class _FakePower:
    def __init__(self, watts_seq: list[float]):
        self.seq = list(watts_seq)
        self.i = 0

    def __call__(self, gpu_index: int = 0) -> metrics.PowerSample:
        if self.i >= len(self.seq):
            # Hold last value — the sampler may call faster than our seq.
            w = self.seq[-1] if self.seq else 0.0
        else:
            w = self.seq[self.i]
            self.i += 1
        return metrics.PowerSample(w, True)


def test_sampler_collects_samples(monkeypatch):
    fake = _FakePower([80.0, 90.0, 100.0, 110.0])
    monkeypatch.setattr(metrics, "sample_power", fake)
    # Fast sampling; short sleep gives the background thread a few ticks.
    with PowerSampler(hz=200.0, gpu_index=0) as s:
        time.sleep(0.05)
    assert len(s.samples) >= 2
    assert s.energy_wh is not None
    assert s.energy_wh >= 0.0


def test_sampler_empty_when_sampler_fails(monkeypatch):
    def failing(_i: int = 0) -> metrics.PowerSample:
        return metrics.PowerSample(None, False)

    monkeypatch.setattr(metrics, "sample_power", failing)
    with PowerSampler(hz=100.0) as s:
        time.sleep(0.02)
    assert s.samples == []
    assert s.energy_wh is None
    assert s.mean_watts is None


def test_sampler_stop_is_quick(monkeypatch):
    # Event.wait must preempt, not leave a trailing sleep after the call.
    monkeypatch.setattr(metrics, "sample_power", lambda i=0: metrics.PowerSample(50.0, True))
    t0 = time.perf_counter()
    with PowerSampler(hz=1.0) as s:  # nominal interval = 1s
        pass  # exit immediately
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5  # must not block a full interval
    # Prime sample runs in __enter__, so there's at least one reading.
    assert len(s.samples) >= 1


def test_sampler_rectangle_fallback_on_single_sample(monkeypatch):
    # One successful prime sample, then sampler hits a failure path and
    # stops immediately. energy_wh should use rectangle rule (W * elapsed).
    calls = [metrics.PowerSample(120.0, True), metrics.PowerSample(None, False)]
    it = iter(calls)
    monkeypatch.setattr(
        metrics, "sample_power", lambda i=0: next(it, metrics.PowerSample(None, False))
    )
    with PowerSampler(hz=100.0) as s:
        time.sleep(0.02)
    # Could be 1 or 2 samples depending on thread timing.
    assert s.energy_wh is not None
    assert s.energy_wh >= 0.0


# --- combined GPU sampler (VRAM + SM utilization) -------------------------


def _fake_combined(watts: float, vram: float, sm: float):
    """Return a factory that always yields a successful GpuSample."""

    def _f(gpu_index: int = 0) -> metrics.GpuSample:
        return metrics.GpuSample(watts=watts, vram_mb=vram, sm_pct=sm, ok=True)

    return _f


def test_sampler_combined_populates_vram_and_sm(monkeypatch):
    monkeypatch.setattr(metrics, "_sample_gpu_combined", _fake_combined(80.0, 4096.0, 75.0))
    with PowerSampler(hz=200.0, gpu_index=0) as s:
        time.sleep(0.05)
    assert len(s.samples) >= 1
    assert len(s.vram_samples) >= 1
    assert len(s.sm_samples) >= 1
    assert s.peak_vram_mb is not None
    assert s.peak_vram_mb >= 4096.0
    assert s.mean_sm_pct is not None
    assert 70.0 <= s.mean_sm_pct <= 80.0


def test_sampler_combined_peak_vram_is_max(monkeypatch):
    # Simulate VRAM climbing during inference.
    readings = [
        metrics.GpuSample(80.0, 2000.0, 60.0, True),
        metrics.GpuSample(80.0, 5000.0, 70.0, True),
        metrics.GpuSample(80.0, 3000.0, 65.0, True),
    ]
    it = iter(readings)

    def _f(gpu_index: int = 0) -> metrics.GpuSample:
        return next(it, readings[-1])

    monkeypatch.setattr(metrics, "_sample_gpu_combined", _f)
    with PowerSampler(hz=1000.0, gpu_index=0) as s:
        time.sleep(0.02)
    assert s.peak_vram_mb is not None
    assert s.peak_vram_mb == pytest.approx(5000.0)


def test_sampler_combined_null_when_combined_and_power_both_fail(monkeypatch):
    monkeypatch.setattr(
        metrics, "_sample_gpu_combined", lambda i=0: metrics.GpuSample(None, None, None, False)
    )
    monkeypatch.setattr(metrics, "sample_power", lambda i=0: metrics.PowerSample(None, False))
    with PowerSampler(hz=100.0) as s:
        time.sleep(0.02)
    assert s.peak_vram_mb is None
    assert s.mean_sm_pct is None
    assert s.energy_wh is None


def test_sampler_combined_falls_back_to_power_for_watts(monkeypatch):
    # Combined fails (e.g. ROCm host), but sample_power succeeds for energy.
    monkeypatch.setattr(
        metrics, "_sample_gpu_combined", lambda i=0: metrics.GpuSample(None, None, None, False)
    )
    monkeypatch.setattr(metrics, "sample_power", lambda i=0: metrics.PowerSample(50.0, True))
    with PowerSampler(hz=200.0) as s:
        time.sleep(0.05)
    assert len(s.samples) >= 1
    assert s.energy_wh is not None
    assert s.peak_vram_mb is None
    assert s.mean_sm_pct is None
