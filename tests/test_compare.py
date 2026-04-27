"""Unit tests for bench.compare."""

from __future__ import annotations

import json

from bench.compare import _compat_warnings, _delta, compare, compare_markdown, load_result

# --- Fixture helpers ---------------------------------------------------------


def _record(
    model_id,
    task_id="t1",
    prompt_id="p1",
    latency_s=1.0,
    tokens_per_sec=100.0,
    quality_heuristic=0.8,
    energy_wh=0.01,
    cost_usd=0.001,
    error=None,
):
    return {
        "model_id": model_id,
        "task_id": task_id,
        "prompt_id": prompt_id,
        "latency_s": latency_s,
        "ttft_s": None,
        "tokens_per_sec": tokens_per_sec,
        "quality_heuristic": quality_heuristic,
        "energy_wh": energy_wh,
        "cost_usd": cost_usd,
        "error": error,
    }


def _pairwise_judgement(winner, a="m_a", b="m_b", task_id="t1", prompt_id="p1"):
    return {
        "mode": "pairwise",
        "task_id": task_id,
        "prompt_id": prompt_id,
        "a_model": a,
        "b_model": b,
        "winner": winner,
        "order": "AB",
    }


def _scored_judgement(model_id, score, task_id="t1", prompt_id="p1"):
    return {
        "mode": "scored",
        "task_id": task_id,
        "prompt_id": prompt_id,
        "model_id": model_id,
        "score": score,
    }


def _payload(
    models=("m_a", "m_b"),
    records=None,
    judgements=None,
    run_id="run-1",
    seed=42,
    config_hash="abc123",
):
    cfg = {
        "models": [{"id": m} for m in models],
        "prompts": [{"id": "p1"}],
    }
    tasks = [{"id": "t1", "domain": "qa", "user_prompt": "q"}]
    prov = {"seed": seed, "config_hash": config_hash}
    return {
        "run_id": run_id,
        "timestamp": "20260101-120000",
        "config": cfg,
        "provenance": prov,
        "tasks": tasks,
        "records": records or [_record("m_a"), _record("m_b")],
        "judgements": judgements or [],
    }


# --- _compat_warnings --------------------------------------------------------


class TestCompatWarnings:
    def test_identical_payloads_no_warnings(self):
        p = _payload()
        assert _compat_warnings(p, p) == []

    def test_different_seeds_warns(self):
        base = _payload(seed=42)
        cand = _payload(seed=99)
        warns = _compat_warnings(base, cand)
        assert any("seed" in w for w in warns)

    def test_different_config_hash_warns(self):
        base = _payload(config_hash="aaa")
        cand = _payload(config_hash="bbb")
        warns = _compat_warnings(base, cand)
        assert any("config hash" in w for w in warns)

    def test_different_model_ids_warns(self):
        base = _payload(models=("m_a", "m_b"))
        cand = _payload(models=("m_a", "m_c"))
        warns = _compat_warnings(base, cand)
        assert any("model" in w.lower() for w in warns)

    def test_different_task_count_warns(self):
        base = _payload()
        cand = _payload()
        cand["tasks"].append({"id": "t2"})
        warns = _compat_warnings(base, cand)
        assert any("task" in w for w in warns)

    def test_different_judge_mode_warns(self):
        base = _payload(judgements=[_pairwise_judgement("m_a")])
        cand = _payload(judgements=[_scored_judgement("m_a", 4)])
        warns = _compat_warnings(base, cand)
        assert any("judge mode" in w for w in warns)


# --- _delta ------------------------------------------------------------------


class TestDelta:
    def test_positive_delta(self):
        assert _delta(1.0, 1.5) == "+0.500"

    def test_negative_delta(self):
        assert _delta(1.5, 1.0) == "-0.500"

    def test_none_base(self):
        assert _delta(None, 1.0) == "—"

    def test_none_cand(self):
        assert _delta(1.0, None) == "—"

    def test_both_none(self):
        assert _delta(None, None) == "—"


# --- compare_markdown --------------------------------------------------------


class TestCompareMarkdown:
    def _base(self):
        return _payload(
            records=[_record("m_a", latency_s=1.0), _record("m_b", latency_s=2.0)],
        )

    def _cand(self):
        return _payload(
            run_id="run-2",
            records=[_record("m_a", latency_s=0.8), _record("m_b", latency_s=2.2)],
        )

    def test_header_present(self):
        md = compare_markdown(self._base(), self._cand())
        assert "# Comparison" in md

    def test_model_rows_present(self):
        md = compare_markdown(self._base(), self._cand())
        assert "m_a" in md
        assert "m_b" in md

    def test_latency_delta_present(self):
        md = compare_markdown(self._base(), self._cand())
        assert "-0.200" in md

    def test_warnings_rendered(self):
        md = compare_markdown(self._base(), self._cand(), warnings=["seeds differ"])
        assert "seeds differ" in md

    def test_no_warnings_no_block(self):
        md = compare_markdown(self._base(), self._cand(), warnings=[])
        assert "Warning" not in md

    def test_scored_judge_section(self):
        base = _payload(
            judgements=[
                _scored_judgement("m_a", 4),
                _scored_judgement("m_b", 3),
            ]
        )
        cand = _payload(
            judgements=[
                _scored_judgement("m_a", 5),
                _scored_judgement("m_b", 2),
            ]
        )
        md = compare_markdown(base, cand)
        assert "scored mode" in md
        assert "+1.00" in md

    def test_pairwise_judge_section(self):
        base = _payload(
            judgements=[
                _pairwise_judgement("m_a"),
                _pairwise_judgement("m_a"),
            ]
        )
        cand = _payload(
            judgements=[
                _pairwise_judgement("m_b"),
                _pairwise_judgement("m_b"),
            ]
        )
        md = compare_markdown(base, cand)
        assert "pairwise mode" in md

    def test_mixed_judge_modes_no_judge_section(self):
        base = _payload(judgements=[_pairwise_judgement("m_a")])
        cand = _payload(judgements=[_scored_judgement("m_a", 4)])
        md = compare_markdown(base, cand)
        assert "Judge scores" not in md
        assert "Judge win rates" not in md

    def test_missing_model_in_candidate(self):
        base = _payload(
            models=("m_a", "m_b"),
            records=[_record("m_a"), _record("m_b")],
        )
        cand = _payload(
            models=("m_a",),
            records=[_record("m_a")],
        )
        md = compare_markdown(base, cand)
        assert "m_a" in md
        assert "m_b" in md

    def test_null_energy_handled(self):
        base = _payload(
            records=[
                _record("m_a", energy_wh=None, cost_usd=None),
            ]
        )
        cand = _payload(
            records=[
                _record("m_a", energy_wh=None, cost_usd=None),
            ]
        )
        md = compare_markdown(base, cand)
        assert "—" in md


# --- load_result and compare() -----------------------------------------------


class TestLoadAndCompare:
    def test_load_result(self, tmp_path):
        p = tmp_path / "run.json"
        payload = _payload()
        p.write_text(json.dumps(payload))
        loaded = load_result(p)
        assert loaded["run_id"] == "run-1"

    def test_compare_stdout(self, tmp_path, capsys):
        base_p = tmp_path / "base.json"
        cand_p = tmp_path / "cand.json"
        base_p.write_text(json.dumps(_payload()))
        cand_p.write_text(json.dumps(_payload(run_id="run-2")))
        rc = compare(base_p, cand_p)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Comparison" in out

    def test_compare_output_file(self, tmp_path):
        base_p = tmp_path / "base.json"
        cand_p = tmp_path / "cand.json"
        out_p = tmp_path / "report.md"
        base_p.write_text(json.dumps(_payload()))
        cand_p.write_text(json.dumps(_payload(run_id="run-2")))
        rc = compare(base_p, cand_p, output=out_p)
        assert rc == 0
        assert out_p.exists()
        assert "Comparison" in out_p.read_text()

    def test_compare_strict_fails_on_warning(self, tmp_path):
        base_p = tmp_path / "base.json"
        cand_p = tmp_path / "cand.json"
        base_p.write_text(json.dumps(_payload(seed=42)))
        cand_p.write_text(json.dumps(_payload(seed=99)))
        rc = compare(base_p, cand_p, strict=True)
        assert rc != 0

    def test_compare_non_strict_succeeds_with_warning(self, tmp_path, capsys):
        base_p = tmp_path / "base.json"
        cand_p = tmp_path / "cand.json"
        base_p.write_text(json.dumps(_payload(seed=42)))
        cand_p.write_text(json.dumps(_payload(seed=99)))
        rc = compare(base_p, cand_p, strict=False)
        assert rc == 0
