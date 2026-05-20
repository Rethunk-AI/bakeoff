"""Scoring + energy measurement.

Quality:
  - exact: normalized string equality vs expected
  - contains: expected substring present (case-insensitive)
  - regex: expected regex matches
  - judge: pairwise LLM-judge (handled in runner after both A/B responses exist)

Energy:
  - background `PowerSampler` thread polls nvidia-smi/rocm-smi at N Hz during
    each call; trapezoidal integration of (watts, t) pairs yields Wh.
    A high-freq sampler captures burst-y decode power that a 2-point
    start/end average undercounts.
  - fallback: cost_usd = None when no sample succeeded.
"""

from __future__ import annotations

import itertools
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

# --- Heuristic quality ------------------------------------------------------


def score_heuristic(scorer: str, expected: str | None, text: str) -> float | None:
    if expected is None:
        return None
    t = (text or "").strip()
    e = expected.strip()
    if scorer == "exact":
        return 1.0 if t.lower() == e.lower() else 0.0
    if scorer == "contains":
        return 1.0 if e.lower() in t.lower() else 0.0
    if scorer == "regex":
        return 1.0 if re.search(e, t) else 0.0
    return None


# --- Pairwise judge prompt --------------------------------------------------

JUDGE_PAIR_SYSTEM = (
    "You are a strict evaluator. Compare two assistant responses to the same user prompt. "
    "Reply with exactly one token: A, B, or TIE. No explanation."
)


def judge_pair_prompt(user_prompt: str, a: str, b: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_PAIR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"USER PROMPT:\n{user_prompt}\n\n"
                f"RESPONSE A:\n{a}\n\n"
                f"RESPONSE B:\n{b}\n\n"
                "Which response better satisfies the user prompt? Reply A, B, or TIE."
            ),
        },
    ]


def judge_pair_randomized(
    user_prompt: str,
    a: str,
    b: str,
    rng,
) -> tuple[list[dict[str, str]], str]:
    """Build a pairwise judge prompt with randomized A/B order.

    Positional bias is real (judges favor slot A by 5-15%); flipping per call
    randomizes it out over the matrix. Returns (messages, order) where
    order == "AB" when the true A-response was shown as A, "BA" when swapped.
    Caller must invert the winner label when order == "BA".
    """
    if rng.random() < 0.5:
        return judge_pair_prompt(user_prompt, b, a), "BA"
    return judge_pair_prompt(user_prompt, a, b), "AB"


def invert_winner(winner: str) -> str:
    """Map a raw pairwise verdict back to original-ordering when order==BA."""
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return winner  # TIE unchanged


def parse_judge(text: str) -> str:
    """Extract A / B / TIE verdict.

    Uses the *last* bare verdict token so that reasoning-style output
    ("Looking at A and B, I pick B.") returns the final decision, not a
    mid-sentence mention.
    """
    t = (text or "").strip().upper()
    toks = [tok for tok in re.split(r"\W+", t) if tok in {"A", "B", "TIE"}]
    return toks[-1] if toks else "TIE"


# --- Scored (rubric) judge prompt ------------------------------------------

JUDGE_SCORE_SYSTEM = (
    "You are a strict evaluator. Rate the assistant response against the user prompt "
    "on this integer 1-5 scale:\n"
    "  1 = wrong or unusable\n"
    "  2 = mostly wrong; some correct element\n"
    "  3 = partially correct; missing or flawed parts\n"
    "  4 = largely correct; minor issues\n"
    "  5 = fully correct and well-formed\n"
    "Reply with exactly one digit (1, 2, 3, 4, or 5). No explanation."
)


def judge_score_prompt(user_prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SCORE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"USER PROMPT:\n{user_prompt}\n\n"
                f"RESPONSE:\n{response}\n\n"
                "Rate 1-5. Reply with one digit."
            ),
        },
    ]


_SCORE_RE = re.compile(r"\b([1-5])\b")


def parse_score(text: str, default: int = 3) -> int:
    """Extract integer score in [1,5]. Take the last valid digit in range.

    Reasoning models emit thoughts before the answer, so prefer the final
    occurrence. Fall back to `default` (median 3) if no valid digit appears.
    """
    matches = _SCORE_RE.findall(text or "")
    if not matches:
        return default
    return int(matches[-1])


# --- Hardware detection -----------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower()).strip("-")


def _detect_nvidia_name(gpu_index: int) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        name = out.stdout.strip().splitlines()[0].strip()
        return _slugify(name) if name else None
    except Exception:
        return None


_ROCM_PRODUCT_RE = re.compile(r"Card\s+series\s*:\s*(.+)", re.IGNORECASE)
_ROCM_PRODUCT_ALT_RE = re.compile(r"Product Name\s*:\s*(.+)", re.IGNORECASE)


def _detect_rocm_name(gpu_index: int) -> str | None:
    if shutil.which("rocm-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["rocm-smi", "-d", str(gpu_index), "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        for pat in (_ROCM_PRODUCT_RE, _ROCM_PRODUCT_ALT_RE):
            m = pat.search(out.stdout)
            if m:
                return _slugify(m.group(1).strip())
        return None
    except Exception:
        return None


def detect_hardware_id(gpu_index: int = 0) -> str | None:
    """Return a canonical slug for the detected GPU (e.g. 'nvidia-geforce-rtx-4090').

    Tries nvidia-smi first, then rocm-smi. Returns None when neither tool
    is available or the device name cannot be parsed.
    """
    return _detect_nvidia_name(gpu_index) or _detect_rocm_name(gpu_index)


# FP16 (or BF16 where noted) theoretical peak TFLOPS, keyed by slug substrings.
# Matched by substring so slugified vendor prefixes don't block lookup.
# Source: vendor datasheets / official specs.
_TFLOPS_TABLE: list[tuple[str, float]] = [
    # NVIDIA Ada Lovelace
    ("rtx-4090", 82.6),
    ("rtx-4080-super", 52.2),
    ("rtx-4080", 48.7),
    ("rtx-4070-ti-super", 40.0),
    ("rtx-4070-ti", 40.1),
    ("rtx-4070-super", 35.5),
    ("rtx-4070", 29.1),
    ("rtx-4060-ti", 22.1),
    ("rtx-4060", 15.1),
    # NVIDIA Ampere
    ("rtx-3090-ti", 40.0),
    ("rtx-3090", 35.6),
    ("rtx-3080-ti", 34.1),
    ("rtx-3080", 29.8),
    ("rtx-3070-ti", 21.7),
    ("rtx-3070", 20.3),
    ("rtx-3060-ti", 16.2),
    ("rtx-3060", 12.7),
    # NVIDIA Data Center
    ("a100", 77.0),
    ("h100", 198.9),
    ("h200", 198.9),
    # AMD RDNA 4
    ("rx-9070-xt", 95.9),
    ("rx-9070", 71.7),
    # AMD RDNA 3
    ("rx-7900-xtx", 61.4),
    ("rx-7900-xt", 51.6),
    ("rx-7900-gre", 45.9),
    ("rx-7800-xt", 37.4),
    ("rx-7700-xt", 27.0),
    ("rx-7600", 21.7),
    # AMD Strix Halo / Phoenix APU (integrated; BF16)
    ("890m", 39.0),
    ("780m", 8.9),
    # Apple M-series (MPS)
    ("m4-max", 54.8),
    ("m4-pro", 27.2),
    ("m4", 11.0),
    ("m3-max", 49.0),
    ("m3-pro", 18.4),
    ("m3", 7.7),
    ("m2-ultra", 54.8),
    ("m2-max", 27.2),
    ("m2-pro", 13.6),
    ("m2", 6.8),
    ("m1-ultra", 21.0),
    ("m1-max", 10.4),
    ("m1-pro", 6.2),
    ("m1", 2.6),
]


def lookup_peak_tflops(hardware_id: str) -> float | None:
    """Return known FP16/BF16 peak TFLOPS for a hardware slug, or None.

    Matches by substring so partial slugs (e.g. 'rtx-4090') match even
    when the full detected name contains vendor prefixes.
    Uses the most specific (longest key) matching entry.
    """
    best: tuple[str, float] | None = None
    for key, tflops in _TFLOPS_TABLE:
        if key in hardware_id and (best is None or len(key) > len(best[0])):
            best = (key, tflops)
    return best[1] if best else None


# --- TFLOPS utilization computation -----------------------------------------


def flops_per_token(num_params: int, num_active_params: int | None = None) -> int:
    """Theoretical FLOPs per output token.

    Dense: 2 × num_params.
    MoE:   2 × num_active_params (active experts only).
    """
    active = num_active_params if num_active_params is not None else num_params
    return 2 * active


def tflops_utilization_pct(
    tokens_per_sec: float,
    flops_per_tok: int,
    peak_tflops: float,
) -> float:
    """Fraction of peak GPU TFLOPS consumed by inference, as a percentage.

    tokens_per_sec × flops_per_tok gives effective FLOP/s; dividing by
    peak_tflops × 1e12 normalises to hardware capacity.
    """
    if peak_tflops <= 0:
        return 0.0
    return (tokens_per_sec * flops_per_tok) / (peak_tflops * 1e12) * 100.0


# --- Energy and hardware metrics --------------------------------------------


@dataclass
class PowerSample:
    watts: float | None
    ok: bool


@dataclass
class GpuSample:
    """Combined per-tick GPU reading: power, VRAM, and SM utilization."""

    watts: float | None
    vram_mb: float | None
    sm_pct: float | None
    ok: bool


def _sample_nvidia(gpu_index: int) -> PowerSample:
    if shutil.which("nvidia-smi") is None:
        return PowerSample(None, False)
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_index),
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return PowerSample(None, False)
        val = out.stdout.strip().splitlines()[0].strip()
        return PowerSample(float(val), True)
    except Exception:
        return PowerSample(None, False)


def _sample_gpu_combined(gpu_index: int) -> GpuSample:
    """Query power.draw, memory.used, and utilization.gpu in one nvidia-smi call.

    Cheaper than three separate calls. Returns GpuSample(ok=False) when
    nvidia-smi is absent or the query fails (e.g. ROCm-only host).
    """
    if shutil.which("nvidia-smi") is None:
        return GpuSample(None, None, None, False)
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_index),
                "--query-gpu=power.draw,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return GpuSample(None, None, None, False)
        parts = out.stdout.strip().splitlines()[0].strip().split(",")
        if len(parts) < 3:
            return GpuSample(None, None, None, False)
        watts = float(parts[0].strip())
        vram = float(parts[1].strip())
        sm = float(parts[2].strip())
        return GpuSample(watts, vram, sm, True)
    except Exception:
        return GpuSample(None, None, None, False)


_ROCM_POWER_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*W", re.IGNORECASE)


def _sample_rocm(gpu_index: int) -> PowerSample:
    if shutil.which("rocm-smi") is None:
        return PowerSample(None, False)
    try:
        out = subprocess.run(
            ["rocm-smi", "-d", str(gpu_index), "--showpower"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return PowerSample(None, False)
        m = _ROCM_POWER_RE.search(out.stdout)
        if not m:
            return PowerSample(None, False)
        return PowerSample(float(m.group(1)), True)
    except Exception:
        return PowerSample(None, False)


def sample_power(gpu_index: int = 0) -> PowerSample:
    """Try NVIDIA first, then ROCm. Return (None, False) if neither works."""
    s = _sample_nvidia(gpu_index)
    if s.ok:
        return s
    return _sample_rocm(gpu_index)


def energy_wh(avg_watts: float | None, seconds: float) -> float | None:
    if avg_watts is None:
        return None
    return avg_watts * (seconds / 3600.0)


def cost_usd(wh: float | None, kwh_rate: float) -> float | None:
    if wh is None:
        return None
    return (wh / 1000.0) * kwh_rate


# --- high-frequency background sampler -------------------------------------


def trapezoid_wh(samples: list[tuple[float, float]]) -> float | None:
    """Integrate (t_seconds, watts) pairs by the trapezoid rule -> Wh.

    Returns None for empty input. For a single sample, returns 0.0 (no
    duration to integrate over — the caller's call_one falls back to
    rectangle rule with the sampled wattage in that case).
    """
    if not samples:
        return None
    if len(samples) == 1:
        return 0.0
    total_ws = 0.0
    for (t0, w0), (t1, w1) in itertools.pairwise(samples):
        total_ws += (w0 + w1) / 2.0 * (t1 - t0)
    return total_ws / 3600.0


class PowerSampler:
    """Background sampler. `with PowerSampler(hz=10, gpu_index=0) as s: ...`.

    On `__exit__`, stops the thread and exposes:
      - `samples`: list[(t_rel_s, watts)] for energy integration.
      - `vram_samples`: list[(t_rel_s, vram_mb)] — peak across the call.
      - `sm_samples`: list[(t_rel_s, sm_pct)] — mean across the call.
      - `energy_wh`: trapezoidal integral of the watt samples.
      - `mean_watts`: simple mean of sampled wattages (diagnostic).
      - `peak_vram_mb`: max VRAM observed (None when NVML unavailable).
      - `mean_sm_pct`: mean SM utilization (None when NVML unavailable).

    Uses a single combined nvidia-smi call per tick (power + VRAM + SM
    utilization) when NVML is available; falls back to watts-only ROCm path
    otherwise. Safe when no GPU tool is available: all null, no exception.
    """

    def __init__(self, hz: float = 10.0, gpu_index: int = 0):
        self.interval_s = 1.0 / max(hz, 0.1)
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float = 0.0
        self.samples: list[tuple[float, float]] = []
        self.vram_samples: list[tuple[float, float]] = []
        self.sm_samples: list[tuple[float, float]] = []

    def _tick(self) -> None:
        """One sampling tick: try combined NVML first, fall back to watts-only."""
        t = time.perf_counter() - self._t0
        gs = _sample_gpu_combined(self.gpu_index)
        if gs.ok:
            if gs.watts is not None:
                self.samples.append((t, gs.watts))
            if gs.vram_mb is not None:
                self.vram_samples.append((t, gs.vram_mb))
            if gs.sm_pct is not None:
                self.sm_samples.append((t, gs.sm_pct))
        else:
            # ROCm or other watts-only path.
            ps = sample_power(self.gpu_index)
            if ps.ok and ps.watts is not None:
                self.samples.append((t, ps.watts))

    def _run(self) -> None:
        while not self._stop.is_set():
            self._tick()
            # Event.wait lets stop() preempt mid-interval; avoids trailing
            # sleep after the call already returned.
            if self._stop.wait(self.interval_s):
                break

    def __enter__(self) -> PowerSampler:
        self._t0 = time.perf_counter()
        self._stop.clear()
        # Prime with one immediate sample so very short calls still have
        # a reading; the thread then continues at interval_s.
        gs = _sample_gpu_combined(self.gpu_index)
        if gs.ok:
            if gs.watts is not None:
                self.samples.append((0.0, gs.watts))
            if gs.vram_mb is not None:
                self.vram_samples.append((0.0, gs.vram_mb))
            if gs.sm_pct is not None:
                self.sm_samples.append((0.0, gs.sm_pct))
        else:
            ps = sample_power(self.gpu_index)
            if ps.ok and ps.watts is not None:
                self.samples.append((0.0, ps.watts))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def energy_wh(self) -> float | None:
        if not self.samples:
            return None
        wh = trapezoid_wh(self.samples)
        if wh == 0.0 and len(self.samples) == 1:
            # Rectangle fallback: single sample over the elapsed wall time.
            elapsed = time.perf_counter() - self._t0
            return self.samples[0][1] * (elapsed / 3600.0)
        return wh

    @property
    def mean_watts(self) -> float | None:
        if not self.samples:
            return None
        return sum(w for _, w in self.samples) / len(self.samples)

    @property
    def peak_vram_mb(self) -> float | None:
        if not self.vram_samples:
            return None
        return max(v for _, v in self.vram_samples)

    @property
    def mean_sm_pct(self) -> float | None:
        if not self.sm_samples:
            return None
        return sum(v for _, v in self.sm_samples) / len(self.sm_samples)
