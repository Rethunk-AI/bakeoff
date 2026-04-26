"""Unit tests for bench.resume and runner phase-order invariants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from bench.resume import (
    ResumeError,
    build_pending,
    build_pending_judge_pairs,
    build_pending_judge_scores,
    check_compat,
    load_prior,
    pairwise_key,
    row_key,
    scored_key,
    tag_fresh,
    tag_reused,
)

# --- TestRowKeys -------------------------------------------------------------


class TestRowKeys:
    def test_row_key(self):
        r = {"task_id": "t1", "prompt_id": "p1", "model_id": "m_a"}
        assert row_key(r) == ("t1", "p1", "m_a")

    def test_row_key_coerces_to_str(self):
        r = {"task_id": 1, "prompt_id": 2, "model_id": 3}
        assert row_key(r) == ("1", "2", "3")

    def test_pairwise_key_basic(self):
        j = {"task_id": "t1", "prompt_id": "p1", "a_model": "m_a", "b_model": "m_b"}
        assert pairwise_key(j) == ("t1", "p1", frozenset({"m_a", "m_b"}))

    def test_pairwise_key_symmetric(self):
        j_ab = {"task_id": "t1", "prompt_id": "p1", "a_model": "m_a", "b_model": "m_b"}
        j_ba = {"task_id": "t1", "prompt_id": "p1", "a_model": "m_b", "b_model": "m_a"}
        assert pairwise_key(j_ab) == pairwise_key(j_ba)

    def test_pairwise_key_coerces_to_str(self):
        j = {"task_id": 1, "prompt_id": 2, "a_model": "x", "b_model": "y"}
        tid, pid, _fs = pairwise_key(j)
        assert tid == "1"
        assert pid == "2"

    def test_scored_key(self):
        j = {"task_id": "t1", "prompt_id": "p1", "model_id": "m_a"}
        assert scored_key(j) == ("t1", "p1", "m_a")

    def test_scored_key_coerces_to_str(self):
        j = {"task_id": 9, "prompt_id": 8, "model_id": 7}
        assert scored_key(j) == ("9", "8", "7")


# --- TestLoadPrior -----------------------------------------------------------


class TestLoadPrior:
    def _write(self, tmp_path: Path, data: Any) -> Path:
        p = tmp_path / "result.json"
        p.write_text(json.dumps(data))
        return p

    def test_valid_payload_returned(self, tmp_path):
        p = self._write(tmp_path, {"tasks": [], "records": []})
        result = load_prior(p)
        assert result == {"tasks": [], "records": []}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ResumeError, match="not found"):
            load_prior(tmp_path / "no_such_file.json")

    def test_malformed_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json}")
        with pytest.raises(ResumeError, match="malformed JSON"):
            load_prior(bad)

    def test_non_dict_raises(self, tmp_path):
        p = self._write(tmp_path, [1, 2, 3])
        with pytest.raises(ResumeError, match="not a JSON object"):
            load_prior(p)

    def test_missing_tasks_raises(self, tmp_path):
        p = self._write(tmp_path, {"records": []})
        with pytest.raises(ResumeError, match="tasks"):
            load_prior(p)

    def test_missing_records_raises(self, tmp_path):
        p = self._write(tmp_path, {"tasks": []})
        with pytest.raises(ResumeError, match="records"):
            load_prior(p)

    def test_extra_fields_allowed(self, tmp_path):
        p = self._write(tmp_path, {"tasks": [], "records": [], "run_id": "r1"})
        result = load_prior(p)
        assert result["run_id"] == "r1"


# --- TestCheckCompat ---------------------------------------------------------


def _cfg(models=None, prompts=None):
    return {
        "models": models or [{"id": "m_a"}, {"id": "m_b"}],
        "prompts": prompts or [{"id": "p1"}],
    }


def _prior(tasks=None, models=None, prompts=None, seed=42):
    return {
        "tasks": tasks if tasks is not None else [{"id": "t1"}],
        "records": [],
        "config": {
            "models": models or [{"id": "m_a"}, {"id": "m_b"}],
            "prompts": prompts or [{"id": "p1"}],
        },
        "provenance": {"seed": seed},
    }


class TestCheckCompat:
    def test_compatible_returns_empty(self):
        errors = check_compat(_cfg(), seed=42, task_ids=["t1"], prior=_prior())
        assert errors == []

    def test_seed_mismatch_reports_error(self):
        errors = check_compat(_cfg(), seed=99, task_ids=["t1"], prior=_prior(seed=42))
        assert any("seed" in e for e in errors)

    def test_seed_none_in_prior_skips_check(self):
        prior = _prior()
        prior["provenance"]["seed"] = None
        errors = check_compat(_cfg(), seed=42, task_ids=["t1"], prior=prior)
        assert not any("seed" in e for e in errors)

    def test_task_set_mismatch_reports_error(self):
        prior = _prior(tasks=[{"id": "t1"}, {"id": "t2"}])
        errors = check_compat(_cfg(), seed=42, task_ids=["t1"], prior=prior)
        assert any("task" in e for e in errors)

    def test_prompt_mismatch_reports_error(self):
        prior = _prior(prompts=[{"id": "p2"}])
        errors = check_compat(_cfg(), seed=42, task_ids=["t1"], prior=prior)
        assert any("prompt" in e for e in errors)

    def test_model_mismatch_reports_error(self):
        prior = _prior(models=[{"id": "m_a"}, {"id": "m_c"}])
        errors = check_compat(_cfg(), seed=42, task_ids=["t1"], prior=prior)
        assert any("model" in e for e in errors)

    def test_model_superset_ok(self):
        # Current run has same model set — set equality check
        cfg = _cfg(models=[{"id": "m_a"}, {"id": "m_b"}])
        prior = _prior(models=[{"id": "m_b"}, {"id": "m_a"}])
        errors = check_compat(cfg, seed=42, task_ids=["t1"], prior=prior)
        assert not any("model" in e for e in errors)

    def test_multiple_mismatches_all_reported(self):
        prior = _prior(tasks=[{"id": "t2"}], prompts=[{"id": "p2"}])
        errors = check_compat(_cfg(), seed=99, task_ids=["t1"], prior=prior)
        assert len(errors) >= 3  # seed, tasks, prompts


# --- TestBuildPending --------------------------------------------------------


def _record(model_id, task_id="t1", prompt_id="p1", error=None):
    return {
        "model_id": model_id,
        "task_id": task_id,
        "prompt_id": prompt_id,
        "error": error,
    }


class TestBuildPending:
    def _models(self, ids=("m_a", "m_b")):
        return [{"id": i} for i in ids]

    def test_all_complete_returns_empty_sets(self):
        prior = [_record("m_a"), _record("m_b")]
        pending = build_pending(self._models(), ["t1"], ["p1"], prior)
        assert pending == {"m_a": set(), "m_b": set()}

    def test_missing_row_is_pending(self):
        prior = [_record("m_a")]
        pending = build_pending(self._models(), ["t1"], ["p1"], prior)
        assert ("t1", "p1") not in pending["m_a"]
        assert ("t1", "p1") in pending["m_b"]

    def test_errored_row_is_pending(self):
        prior = [_record("m_a", error="timeout")]
        pending = build_pending(self._models(["m_a"]), ["t1"], ["p1"], prior)
        assert ("t1", "p1") in pending["m_a"]

    def test_all_missing_gives_full_set(self):
        pending = build_pending(self._models(), ["t1", "t2"], ["p1", "p2"], [])
        for mid in ("m_a", "m_b"):
            assert len(pending[mid]) == 4  # 2 tasks x 2 prompts

    def test_mixed_complete_and_pending(self):
        prior = [
            _record("m_a", task_id="t1", prompt_id="p1"),
            _record("m_a", task_id="t1", prompt_id="p2", error="fail"),
        ]
        pending = build_pending(self._models(["m_a"]), ["t1"], ["p1", "p2"], prior)
        assert ("t1", "p1") not in pending["m_a"]
        assert ("t1", "p2") in pending["m_a"]

    def test_returns_entry_for_every_model(self):
        pending = build_pending(self._models(["m_a", "m_b", "m_c"]), ["t1"], ["p1"], [])
        assert set(pending.keys()) == {"m_a", "m_b", "m_c"}

    def test_rerun_errors_false_skips_errored(self):
        prior = [_record("m_a", error="timeout")]
        pending = build_pending(
            self._models(["m_a"]), ["t1"], ["p1"], prior, rerun_errors=False
        )
        assert ("t1", "p1") not in pending["m_a"]

    def test_rerun_missing_false_skips_missing(self):
        prior = []  # m_a has no rows at all — missing
        pending = build_pending(
            self._models(["m_a"]), ["t1"], ["p1"], prior, rerun_missing=False
        )
        assert ("t1", "p1") not in pending["m_a"]

    def test_filter_models_excludes_unselected(self):
        pending = build_pending(
            self._models(), ["t1"], ["p1"], [],
            filter_models={"m_a"},
        )
        assert ("t1", "p1") in pending["m_a"]
        assert pending["m_b"] == set()

    def test_filter_tasks_excludes_unselected(self):
        pending = build_pending(
            self._models(["m_a"]), ["t1", "t2"], ["p1"], [],
            filter_tasks={"t1"},
        )
        assert ("t1", "p1") in pending["m_a"]
        assert ("t2", "p1") not in pending["m_a"]

    def test_filter_prompts_excludes_unselected(self):
        pending = build_pending(
            self._models(["m_a"]), ["t1"], ["p1", "p2"], [],
            filter_prompts={"p1"},
        )
        assert ("t1", "p1") in pending["m_a"]
        assert ("t1", "p2") not in pending["m_a"]


# --- TestJudgePending --------------------------------------------------------


def _pairwise_j(a, b, task_id="t1", prompt_id="p1", winner="m_a", error=None):
    return {
        "mode": "pairwise", "task_id": task_id, "prompt_id": prompt_id,
        "a_model": a, "b_model": b, "winner": winner, "error": error,
    }


def _scored_j(model_id, score, task_id="t1", prompt_id="p1", error=None):
    return {
        "mode": "scored", "task_id": task_id, "prompt_id": prompt_id,
        "model_id": model_id, "score": score, "error": error,
    }


class TestBuildPendingJudgePairs:
    def _models(self):
        return [{"id": "m_a"}, {"id": "m_b"}]

    def test_all_judged_returns_empty(self):
        j = [_pairwise_j("m_a", "m_b")]
        pending = build_pending_judge_pairs(j, self._models(), ["t1"], ["p1"])
        assert len(pending) == 0

    def test_missing_pair_is_pending(self):
        pending = build_pending_judge_pairs([], self._models(), ["t1"], ["p1"])
        assert len(pending) == 1

    def test_errored_pair_is_pending(self):
        j = [_pairwise_j("m_a", "m_b", winner=None, error="timeout")]
        pending = build_pending_judge_pairs(j, self._models(), ["t1"], ["p1"])
        assert len(pending) == 1

    def test_symmetric_key_recognized(self):
        j = [_pairwise_j("m_b", "m_a")]  # swapped order
        pending = build_pending_judge_pairs(j, self._models(), ["t1"], ["p1"])
        assert len(pending) == 0

    def test_multiple_tasks_prompts(self):
        # 2 tasks x 2 prompts = 4 pairs total; 1 already judged
        j = [_pairwise_j("m_a", "m_b", task_id="t1", prompt_id="p1")]
        pending = build_pending_judge_pairs(j, self._models(), ["t1", "t2"], ["p1", "p2"])
        assert len(pending) == 3


class TestBuildPendingJudgeScores:
    def _models(self):
        return [{"id": "m_a"}, {"id": "m_b"}]

    def test_all_scored_returns_empty(self):
        j = [_scored_j("m_a", 4), _scored_j("m_b", 3)]
        pending = build_pending_judge_scores(j, self._models(), ["t1"], ["p1"])
        assert len(pending) == 0

    def test_missing_score_is_pending(self):
        j = [_scored_j("m_a", 4)]
        pending = build_pending_judge_scores(j, self._models(), ["t1"], ["p1"])
        assert ("t1", "p1", "m_b") in pending

    def test_errored_score_is_pending(self):
        j = [_scored_j("m_a", None, error="timeout"), _scored_j("m_b", 3)]
        pending = build_pending_judge_scores(j, self._models(), ["t1"], ["p1"])
        assert ("t1", "p1", "m_a") in pending
        assert ("t1", "p1", "m_b") not in pending

    def test_multiple_tasks_prompts(self):
        j = [_scored_j("m_a", 4, task_id="t1", prompt_id="p1")]
        pending = build_pending_judge_scores(j, self._models(), ["t1", "t2"], ["p1", "p2"])
        assert len(pending) == 7  # 2 models x 2 tasks x 2 prompts - 1 done


# --- TestTagging -------------------------------------------------------------


class TestTagging:
    def _records(self):
        return [
            {"model_id": "m_a", "task_id": "t1", "prompt_id": "p1"},
            {"model_id": "m_b", "task_id": "t1", "prompt_id": "p1"},
        ]

    def test_tag_reused_adds_field(self):
        tagged = tag_reused(self._records(), "prior-run-1")
        assert all(r["resumed_from"] == "prior-run-1" for r in tagged)

    def test_tag_reused_does_not_mutate_originals(self):
        originals = self._records()
        tag_reused(originals, "x")
        assert "resumed_from" not in originals[0]

    def test_tag_fresh_adds_field(self):
        tagged = tag_fresh(self._records(), "prior-run-1")
        assert all(r["source_run_id"] == "prior-run-1" for r in tagged)

    def test_tag_fresh_does_not_mutate_originals(self):
        originals = self._records()
        tag_fresh(originals, "x")
        assert "source_run_id" not in originals[0]

    def test_tag_reused_preserves_other_fields(self):
        tagged = tag_reused(self._records(), "r1")
        assert tagged[0]["model_id"] == "m_a"

    def test_tag_fresh_preserves_other_fields(self):
        tagged = tag_fresh(self._records(), "r1")
        assert tagged[0]["model_id"] == "m_a"

    def test_empty_list_handled(self):
        assert tag_reused([], "r1") == []
        assert tag_fresh([], "r1") == []


# --- TestPhaseOrder ----------------------------------------------------------


def _make_task(tid: str):
    from bench.dataset import Task
    return Task(id=tid, domain="qa", user_prompt="q", scorer="exact", expected="a")


def _stub_recs(mid, task_id="t1", prompt_id="p1"):
    return [{"model_id": mid, "task_id": task_id, "prompt_id": prompt_id, "error": None}]


class TestPhaseOrder:
    """Verify _run_model_phases calls run_model_phase in config order."""

    def _call(self, models, pending_by_model, prior_run_id=None, side_effects=None):
        from bench.runner import _run_model_phases
        tasks = [_make_task("t1")]
        prompts = [{"id": "p1", "system": "sys"}]
        if side_effects is None:
            side_effects = [_stub_recs(m["id"]) for m in models
                            if pending_by_model is None or pending_by_model.get(m["id"])]

        with patch("bench.runner.run_model_phase", side_effect=side_effects) as mock_phase:
            result = _run_model_phases(
                models, tasks, prompts,
                base_url="http://localhost/v1",
                cost_cfg={},
                timeout_s=10.0,
                warmup=False,
                pending_by_model=pending_by_model,
                prior_run_id=prior_run_id,
            )
        return mock_phase, result

    def test_normal_run_calls_each_model_once(self):
        models = [{"id": "m_a"}, {"id": "m_b"}]
        mock, _ = self._call(models, pending_by_model=None,
                             side_effects=[_stub_recs("m_a"), _stub_recs("m_b")])
        assert mock.call_count == 2
        assert mock.call_args_list[0][0][0] == {"id": "m_a"}
        assert mock.call_args_list[1][0][0] == {"id": "m_b"}

    def test_normal_run_preserves_config_order(self):
        models = [{"id": "m_b"}, {"id": "m_a"}]
        mock, _ = self._call(models, pending_by_model=None,
                             side_effects=[_stub_recs("m_b"), _stub_recs("m_a")])
        called_ids = [c[0][0]["id"] for c in mock.call_args_list]
        assert called_ids == ["m_b", "m_a"]

    def test_normal_run_passes_no_pending(self):
        models = [{"id": "m_a"}]
        mock, _ = self._call(models, pending_by_model=None,
                             side_effects=[_stub_recs("m_a")])
        _, kwargs = mock.call_args
        assert kwargs.get("pending") is None

    def test_resumed_skips_complete_model(self):
        models = [{"id": "m_a"}, {"id": "m_b"}]
        # m_a has no pending cells; m_b has one
        pending = {"m_a": set(), "m_b": {("t1", "p1")}}
        mock, _ = self._call(models, pending_by_model=pending,
                             side_effects=[_stub_recs("m_b")])
        assert mock.call_count == 1
        assert mock.call_args_list[0][0][0] == {"id": "m_b"}

    def test_resumed_passes_pending_set(self):
        models = [{"id": "m_a"}]
        pending_set = {("t1", "p1")}
        pending = {"m_a": pending_set}
        mock, _ = self._call(models, pending_by_model=pending,
                             side_effects=[_stub_recs("m_a")])
        _, kwargs = mock.call_args
        assert kwargs.get("pending") == pending_set

    def test_resumed_tags_fresh_records(self):
        models = [{"id": "m_a"}]
        pending = {"m_a": {("t1", "p1")}}
        _, result = self._call(models, pending_by_model=pending,
                               prior_run_id="run-prior",
                               side_effects=[_stub_recs("m_a")])
        assert all(r.get("source_run_id") == "run-prior" for r in result)

    def test_normal_run_no_tagging(self):
        models = [{"id": "m_a"}]
        _, result = self._call(models, pending_by_model=None,
                               prior_run_id=None,
                               side_effects=[_stub_recs("m_a")])
        assert "source_run_id" not in result[0]
