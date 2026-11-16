"""Completeness-weighted partial score and floor score rollup.

Implements per-cell scoring, the per-model rollup, and run-level status
aggregation.

``partial_score = S / D``. S is the sum of per-cell scores over judged-or-
deterministic cells. D is matrix size C minus unjudged cells: attempted cells
with neither a ``quality_heuristic`` nor a ``judge_score``. Those cells are
omitted from S as well, so a skipped judge does not dilute the score.
Unattempted cells still count as 0.0 against D. Dividing by attempted A was
rejected: it rewards stopping early, and the harness cannot know how the
missing cells would have scored. ``cells_unjudged`` is the coverage gap.
``completeness = A / C``. The worked examples live in tests/test_scoring.py.

Public API (consumed by runner.py):
    cell_score(record)              -> float in [0, 1]
    apply_scored_judgements(...)    -> None (copies scored marks onto records)
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


def apply_scored_judgements(records: list[dict], judgements: list[dict]) -> None:
    """Copy scored-mode rubric marks onto matching records as ``judge_score``.

    Pairwise judgements have no 1-5 mark; they are left alone so a null
    heuristic is not overwritten with a fake 0.
    """
    by_key = {(r.get("task_id"), r.get("prompt_id"), r.get("model_id")): r for r in records}
    for judgement in judgements:
        if judgement.get("mode") != "scored" or judgement.get("score") is None:
            continue
        rec = by_key.get(
            (judgement.get("task_id"), judgement.get("prompt_id"), judgement.get("model_id"))
        )
        if rec is not None:
            rec["judge_score"] = judgement["score"]


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
    dict with keys matching a ``model_scores[*]`` entry in result.json.
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

    def _is_unjudged(r: dict) -> bool:
        # Needs a judge mark: attempted, no usable heuristic, no judge_score.
        qh = r.get("quality_heuristic")
        if isinstance(qh, (int, float)) and 0.0 <= float(qh) <= 1.0:
            return False
        return r.get("judge_score") is None

    attempted = [r for r in main_records if _is_attempted(r)]
    failed = [r for r in main_records if _failure_code(r) is not None]
    unjudged = [r for r in attempted if _is_unjudged(r)]
    scored = [r for r in attempted if not _is_unjudged(r)]

    cells_attempted = len(attempted)
    cells_failed = len(failed)
    cells_unjudged = len(unjudged)

    # --- completeness
    completeness = 0.0 if cells_total == 0 else cells_attempted / cells_total

    # --- partial_score: S / (C - unjudged); unattempted still count as 0
    score_sum = sum(cell_score(r) for r in scored)
    scored_denom = cells_total - cells_unjudged
    partial_score = 0.0 if scored_denom == 0 else max(0.0, min(1.0, score_sum / scored_denom))

    # --- status
    has_load_failure = any(_failure_code(r) == "load_failure" for r in main_records)
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
        "cells_unjudged": cells_unjudged,
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
