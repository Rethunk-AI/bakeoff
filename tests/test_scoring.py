"""Unit tests for bench.scoring — partial score and floor rollup.

Coverage strategy: spec §2 worked-example table (exact values), status
assignment, floor scoring variants, and run_status_from_scores worst-case.
"""

from __future__ import annotations

import unittest

from bench.scoring import cell_score, model_rollup, run_status_from_scores

# ---------------------------------------------------------------------------
# Helpers to build synthetic records
# ---------------------------------------------------------------------------

def _ok(quality_heuristic: float) -> dict:
    """Successful record scored by heuristic."""
    return {"quality_heuristic": quality_heuristic, "failure_code": None, "error": None}


def _judge(judge_score: int | float) -> dict:
    """Successful record scored by judge (1-5)."""
    return {"judge_score": judge_score, "failure_code": None, "error": None}


def _fail(failure_code: str = "timeout") -> dict:
    """Failed record with a failure_code."""
    return {"failure_code": failure_code, "error": None}


def _error(msg: str = "some error") -> dict:
    """Legacy-style error record (no failure_code)."""
    return {"error": msg}


# ---------------------------------------------------------------------------
# cell_score
# ---------------------------------------------------------------------------

class TestCellScore(unittest.TestCase):

    def test_failure_code_non_null_returns_zero(self):
        self.assertEqual(cell_score({"failure_code": "timeout", "error": None}), 0.0)

    def test_error_non_null_returns_zero(self):
        self.assertEqual(cell_score({"failure_code": None, "error": "something went wrong"}), 0.0)

    def test_quality_heuristic_used_when_present(self):
        self.assertAlmostEqual(cell_score(_ok(0.75)), 0.75)

    def test_quality_heuristic_zero(self):
        self.assertEqual(cell_score(_ok(0.0)), 0.0)

    def test_quality_heuristic_one(self):
        self.assertEqual(cell_score(_ok(1.0)), 1.0)

    def test_judge_score_5_maps_to_one(self):
        self.assertAlmostEqual(cell_score(_judge(5)), 1.0)

    def test_judge_score_1_maps_to_zero(self):
        self.assertAlmostEqual(cell_score(_judge(1)), 0.0)

    def test_judge_score_3_maps_to_half(self):
        self.assertAlmostEqual(cell_score(_judge(3)), 0.5)

    def test_fallthrough_returns_zero(self):
        self.assertEqual(cell_score({}), 0.0)

    def test_quality_heuristic_takes_priority_over_judge(self):
        # Both present: quality_heuristic wins
        rec = {"quality_heuristic": 0.9, "judge_score": 1, "failure_code": None, "error": None}
        self.assertAlmostEqual(cell_score(rec), 0.9)

    def test_failure_code_beats_quality_heuristic(self):
        rec = {"quality_heuristic": 0.9, "failure_code": "refusal", "error": None}
        self.assertEqual(cell_score(rec), 0.0)


# ---------------------------------------------------------------------------
# Spec §2 worked example (exact numbers)
# ---------------------------------------------------------------------------

class TestWorkedExample(unittest.TestCase):
    """Verify every row in the spec §2 worked-example table."""

    def _make_main(self, quality_scores: list[float], failures: int = 0) -> list[dict]:
        """Build main_records: `quality_scores` attempted cells + `failures` failed cells."""
        records = [_ok(s) for s in quality_scores]
        records += [_fail() for _ in range(failures)]
        return records

    def _rollup(self, main_records, cells_total=10):
        return model_rollup("m", main_records, [], cells_total)

    # strong-model: C=10, A=10, S=8.2, partial_score=0.82, completeness=1.00
    def test_strong_model_partial_score(self):
        # 10 cells attempted; scores sum to 8.2 (e.g. eight 1.0s and two 0.1s)
        records = [_ok(1.0)] * 8 + [_ok(0.1)] * 2
        r = self._rollup(records)
        self.assertEqual(r["cells_attempted"], 10)
        self.assertAlmostEqual(r["partial_score"], 0.82, places=10)
        self.assertAlmostEqual(r["completeness"], 1.0, places=10)
        self.assertEqual(r["status"], "complete")

    # middling-model: C=10, A=10, S=5.5, partial_score=0.55, completeness=1.00
    def test_middling_model_partial_score(self):
        # 10 cells; 5x1.0 + 5x0.1 = 5.5
        records = [_ok(1.0)] * 5 + [_ok(0.1)] * 5
        r = self._rollup(records)
        self.assertEqual(r["cells_attempted"], 10)
        self.assertAlmostEqual(r["partial_score"], 0.55, places=10)
        self.assertAlmostEqual(r["completeness"], 1.0, places=10)
        self.assertEqual(r["status"], "complete")

    # partial-model: C=10, A=6, S=5.4, partial_score=0.54, completeness=0.60
    def test_partial_model_partial_score(self):
        # 6 attempted (scores sum 5.4) + 4 not in main_records (just absent)
        # Use: 5x1.0 + 1x0.4 = 5.4, but only 6 records supplied; C=10
        records = [_ok(1.0)] * 5 + [_ok(0.4)]
        r = self._rollup(records)
        self.assertEqual(r["cells_attempted"], 6)
        self.assertAlmostEqual(r["partial_score"], 0.54, places=10)
        self.assertAlmostEqual(r["completeness"], 0.6, places=10)
        self.assertEqual(r["status"], "incomplete")

    # weak-model: C=10, A=4, S=2.0, partial_score=0.20, completeness=0.40
    def test_weak_model_partial_score(self):
        # 4 attempted (scores sum 2.0: four 0.5s), rest absent
        records = [_ok(0.5)] * 4
        r = self._rollup(records)
        self.assertEqual(r["cells_attempted"], 4)
        self.assertAlmostEqual(r["partial_score"], 0.20, places=10)
        self.assertAlmostEqual(r["completeness"], 0.4, places=10)
        self.assertEqual(r["status"], "incomplete")

    # failing-model: C=10, A=0, S=0.0, partial_score=0.00, completeness=0.00
    def test_failing_model_partial_score(self):
        # 10 failed records (no attempts)
        records = [_fail("load_failure")] * 10
        r = self._rollup(records)
        self.assertEqual(r["cells_attempted"], 0)
        self.assertAlmostEqual(r["partial_score"], 0.00, places=10)
        self.assertAlmostEqual(r["completeness"], 0.00, places=10)
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["dominant_failure_code"], "load_failure")


# ---------------------------------------------------------------------------
# Status assignment
# ---------------------------------------------------------------------------

class TestStatusAssignment(unittest.TestCase):

    def test_complete_when_all_cells_attempted_no_load_failure(self):
        records = [_ok(0.8)] * 5
        r = model_rollup("m", records, [], 5)
        self.assertEqual(r["status"], "complete")

    def test_incomplete_when_partial_attempt(self):
        records = [_ok(0.5)] * 3  # only 3 of 5
        r = model_rollup("m", records, [], 5)
        self.assertEqual(r["status"], "incomplete")

    def test_failed_when_no_cells_attempted(self):
        records = [_fail("timeout")] * 5
        r = model_rollup("m", records, [], 5)
        self.assertEqual(r["status"], "failed")

    def test_incomplete_not_failed_when_zero_completeness_with_some_records_but_all_fail(self):
        # cells_total=10, main_records has 5 failed, 5 absent → completeness=0 → failed
        records = [_fail("refusal")] * 5
        r = model_rollup("m", records, [], 10)
        # cells_attempted=0, completeness=0.0 → status=failed
        self.assertEqual(r["status"], "failed")

    def test_cells_total_zero_edge_case(self):
        r = model_rollup("m", [], [], 0)
        self.assertEqual(r["partial_score"], 0.0)
        self.assertEqual(r["completeness"], 0.0)

    def test_load_failure_drives_incomplete_not_complete(self):
        # completeness == 1.0 but a load_failure is present → incomplete
        records = [_ok(0.9)] * 4 + [{"failure_code": "load_failure", "error": None}]
        # 4 attempted + 1 load_failure; cells_total=5 → completeness=0.8 → incomplete
        r = model_rollup("m", records, [], 5)
        self.assertEqual(r["status"], "incomplete")

    def test_load_failure_with_all_cells_present_but_one_failed(self):
        # All 5 cells present: 4 ok + 1 load_failure → completeness=4/5=0.8 → incomplete
        records = [_ok(1.0)] * 4 + [{"failure_code": "load_failure", "error": None}]
        r = model_rollup("m", records, [], 5)
        self.assertEqual(r["status"], "incomplete")


# ---------------------------------------------------------------------------
# Floor scoring (spec §3)
# ---------------------------------------------------------------------------

class TestFloorScoring(unittest.TestCase):

    def test_floor_not_run_yields_none_fields(self):
        r = model_rollup("m", [_ok(0.5)] * 5, [], 5)
        self.assertIsNone(r["floor_score"])
        self.assertIsNone(r["floor_cells_passed"])
        self.assertIsNone(r["floor_cells_total"])

    def test_floor_all_pass(self):
        floor = [_ok(1.0)] * 12
        r = model_rollup("m", [], floor, 10)
        self.assertAlmostEqual(r["floor_score"], 1.0)
        self.assertEqual(r["floor_cells_passed"], 12)
        self.assertEqual(r["floor_cells_total"], 12)

    def test_floor_all_fail(self):
        floor = [_fail()] * 12
        r = model_rollup("m", [], floor, 10)
        self.assertAlmostEqual(r["floor_score"], 0.0)
        self.assertEqual(r["floor_cells_passed"], 0)
        self.assertEqual(r["floor_cells_total"], 12)

    def test_floor_mixed_4_of_12(self):
        floor = [_ok(1.0)] * 4 + [_fail()] * 8
        r = model_rollup("m", [], floor, 10)
        self.assertAlmostEqual(r["floor_score"], 4 / 12)
        self.assertEqual(r["floor_cells_passed"], 4)
        self.assertEqual(r["floor_cells_total"], 12)

    def test_floor_partial_quality_does_not_count_as_pass(self):
        # floor_cells_passed counts cell_score == 1.0 only; 0.8 should NOT count
        floor = [_ok(0.8)] * 6 + [_ok(1.0)] * 6
        r = model_rollup("m", [], floor, 10)
        self.assertEqual(r["floor_cells_passed"], 6)
        self.assertAlmostEqual(r["floor_score"], 6 / 12)


# ---------------------------------------------------------------------------
# run_status_from_scores
# ---------------------------------------------------------------------------

class TestRunStatusFromScores(unittest.TestCase):

    def test_empty_list_is_complete(self):
        self.assertEqual(run_status_from_scores([]), "complete")

    def test_all_complete(self):
        scores = [{"status": "complete"}, {"status": "complete"}]
        self.assertEqual(run_status_from_scores(scores), "complete")

    def test_any_failed_dominates(self):
        scores = [
            {"status": "complete"},
            {"status": "incomplete"},
            {"status": "failed"},
        ]
        self.assertEqual(run_status_from_scores(scores), "failed")

    def test_incomplete_without_failed(self):
        scores = [{"status": "complete"}, {"status": "incomplete"}]
        self.assertEqual(run_status_from_scores(scores), "incomplete")

    def test_single_failed(self):
        self.assertEqual(run_status_from_scores([{"status": "failed"}]), "failed")

    def test_single_incomplete(self):
        self.assertEqual(run_status_from_scores([{"status": "incomplete"}]), "incomplete")


if __name__ == "__main__":
    unittest.main()
