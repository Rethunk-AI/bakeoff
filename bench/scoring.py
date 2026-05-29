"""Completeness-weighted partial score and floor score rollup.

Implements the per-cell scoring, per-model rollup, and run-level status
aggregation defined in specs/score-incomplete-and-dumb-model-tier/spec.md §2
and §3.

Public API (consumed by runner.py):
    cell_score(record)              -> float in [0, 1]
    model_rollup(...)               -> dict (model_scores entry)
    run_status_from_scores(...)     -> str  ('complete'|'incomplete'|'failed')
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Import dominant_failure_code from bench.failure (written by parallel agent).
# Fall back to a minimal local implementation so tests pass standalone.
# ---------------------------------------------------------------------------

try:
    from bench.failure import dominant_failure_code as _dominant_failure_code
except ImportError:  # pragma: no cover — real import used in production
    from collections import Counter

    def _dominant_failure_code(codes: list[str]) -> str | None:
        """Minimal fallback: most-common code; None for empty input."""
        if not codes:
            return None
        return Counter(codes).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Per-cell scoring (spec §2 S(m) definition)
# ---------------------------------------------------------------------------


def cell_score(record: dict) -> float:
    """Return the per-cell score for *record* in [0, 1].

    Priority order (spec §2):
    1. Non-null ``error`` or ``failure_code`` → 0.0 (hard failure).
    2. ``quality_heuristic`` is a float in [0, 1] → that value.
    3. ``judge_score`` (1-5 int/float) present → ``(judge_score - 1) / 4.0``
       clamped to [0, 1].
    4. Fallthrough → 0.0.
    """
    # Rule 1: hard failure
    if record.get("failure_code") is not None or record.get("error") is not None:
        return 0.0

    # Rule 2: quality heuristic
    qh = record.get("quality_heuristic")
    if isinstance(qh, (int, float)) and 0.0 <= float(qh) <= 1.0:
        return float(qh)

    # Rule 3: judge score (1-5 rubric)
    js = record.get("judge_score")
    if js is not None:
        normalised = (float(js) - 1.0) / 4.0
        return max(0.0, min(1.0, normalised))

    # Rule 4: default
    return 0.0


# ---------------------------------------------------------------------------
# Per-model rollup (spec §2 / §3)
# ---------------------------------------------------------------------------


def model_rollup(
    model_id: str,
    main_records: list[dict],
    floor_records: list[dict],
    cells_total: int,
) -> dict:
    """Return the ``model_scores`` entry for *model_id*.

    Parameters
    ----------
    model_id:
        Identifier string for the model.
    main_records:
        All result records for this model from the main task suite
        (``tier == "main"`` or tier absent).
    floor_records:
        All result records for this model from the ``dumb_model`` tier.
        Pass an empty list when the floor tier was not run.
    cells_total:
        Total number of cells in the main matrix (constant per run; used as
        the denominator C in the partial-score formula).

    Returns
    -------
    dict with keys matching ``model_scores[*]`` in spec §2 / §3.
    """
    # --- cells_attempted: response received; failure_code and error both null
    def _is_attempted(r: dict) -> bool:
        return r.get("failure_code") is None and not r.get("error")

    def _failure_code(r: dict) -> str | None:
        fc = r.get("failure_code")
        if fc is not None:
            return str(fc)
        # fall back to truthy error string
        if r.get("error"):
            return "unknown"
        return None

    attempted = [r for r in main_records if _is_attempted(r)]
    failed = [r for r in main_records if _failure_code(r) is not None]

    cells_attempted = len(attempted)
    cells_failed = len(failed)

    # --- completeness
    completeness = 0.0 if cells_total == 0 else cells_attempted / cells_total

    # --- partial_score: S(m) / C  (divide by C, not by attempted)
    score_sum = sum(cell_score(r) for r in attempted)
    partial_score = 0.0 if cells_total == 0 else max(0.0, min(1.0, score_sum / cells_total))

    # --- status
    has_load_failure = any(
        _failure_code(r) == "load_failure" for r in main_records
    )
    if completeness == 1.0 and not has_load_failure:
        status = "complete"
    elif completeness == 0.0:
        status = "failed"
    else:
        status = "incomplete"

    # --- dominant_failure_code
    failure_codes = [c for c in (_failure_code(r) for r in failed) if c is not None]
    dominant = _dominant_failure_code(failure_codes) if failure_codes else None

    # --- floor score
    if not floor_records:
        floor_score = None
        floor_cells_passed = None
        floor_cells_total = None
    else:
        floor_cells_total = len(floor_records)
        floor_cells_passed = sum(1 for r in floor_records if cell_score(r) == 1.0)
        floor_score = floor_cells_passed / floor_cells_total

    return {
        "model_id": model_id,
        "status": status,
        "cells_total": cells_total,
        "cells_attempted": cells_attempted,
        "cells_failed": cells_failed,
        "completeness": completeness,
        "partial_score": partial_score,
        "dominant_failure_code": dominant,
        "floor_score": floor_score,
        "floor_cells_passed": floor_cells_passed,
        "floor_cells_total": floor_cells_total,
    }


# ---------------------------------------------------------------------------
# Run-level status (spec §2)
# ---------------------------------------------------------------------------


def run_status_from_scores(model_scores: list[dict]) -> str:
    """Return the worst status across all model score entries.

    Rules (spec §2):
    - Any model 'failed'     → 'failed'
    - Else any 'incomplete'  → 'incomplete'
    - Else                   → 'complete'
    - Empty list             → 'complete'
    """
    statuses = {entry.get("status") for entry in model_scores}
    if "failed" in statuses:
        return "failed"
    if "incomplete" in statuses:
        return "incomplete"
    return "complete"
