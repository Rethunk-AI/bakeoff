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


# --- Energy -----------------------------------------------------------------


@dataclass
class PowerSample:
    watts: float | None
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
      - `samples`: list[(t_rel_s, watts)] taken while active.
      - `energy_wh`: trapezoidal integral of the samples.
      - `mean_watts`: simple mean of sampled wattages (diagnostic).

    Safe to use when no power source is available: `sampler.samples` will
    be empty and `energy_wh` None. Never raises from the thread.
    """

    def __init__(self, hz: float = 10.0, gpu_index: int = 0):
        self.interval_s = 1.0 / max(hz, 0.1)
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float = 0.0
        self.samples: list[tuple[float, float]] = []

    def _run(self) -> None:
        while not self._stop.is_set():
            ps = sample_power(self.gpu_index)
            if ps.ok and ps.watts is not None:
                self.samples.append((time.perf_counter() - self._t0, ps.watts))
            # Event.wait lets stop() preempt mid-interval; avoids trailing
            # sleep after the call already returned.
            if self._stop.wait(self.interval_s):
                break

    def __enter__(self) -> PowerSampler:
        self._t0 = time.perf_counter()
        self._stop.clear()
        # Prime with one immediate sample so very short calls still have
        # a reading; the thread then continues at interval_s.
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
