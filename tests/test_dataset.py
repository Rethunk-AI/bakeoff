"""Unit tests for bench.dataset generator and JSONL writer."""
from __future__ import annotations

import json

from bench.dataset import Task, generate, write_jsonl

ALL_DOMAINS = ["qa", "code", "summarize", "classify"]


class TestGenerate:
    def test_returns_requested_count(self):
        tasks = generate(n=20, domains=ALL_DOMAINS, seed=42)
        assert len(tasks) == 20

    def test_zero_returns_empty(self):
        assert generate(n=0, domains=ALL_DOMAINS) == []

    def test_ids_are_unique_and_ordered(self):
        tasks = generate(n=25, domains=ALL_DOMAINS, seed=7)
        ids = [t.id for t in tasks]
        assert len(set(ids)) == len(ids)
        # ids are t0000, t0001, ... assigned by loop index, so monotone.
        assert ids == sorted(ids)

    def test_same_seed_is_deterministic(self):
        a = generate(n=30, domains=ALL_DOMAINS, seed=123)
        b = generate(n=30, domains=ALL_DOMAINS, seed=123)
        assert a == b

    def test_different_seed_differs(self):
        a = generate(n=30, domains=ALL_DOMAINS, seed=1)
        b = generate(n=30, domains=ALL_DOMAINS, seed=2)
        # Not strictly guaranteed, but with 30 picks across 4 domains a
        # full collision would be astronomically unlikely.
        assert a != b

    def test_domain_filter_single(self):
        tasks = generate(n=10, domains=["qa"], seed=42)
        assert all(t.domain == "qa" for t in tasks)
        # QA tasks are scored by substring match against the expected capital.
        assert all(t.scorer == "contains" for t in tasks)
        assert all(t.expected for t in tasks)

    def test_code_uses_judge_scorer(self):
        tasks = generate(n=10, domains=["code"], seed=42)
        assert all(t.domain == "code" for t in tasks)
        assert all(t.scorer == "judge" for t in tasks)
        assert all(t.expected is None for t in tasks)

    def test_summarize_uses_judge_scorer(self):
        tasks = generate(n=10, domains=["summarize"], seed=42)
        assert all(t.domain == "summarize" and t.scorer == "judge" for t in tasks)

    def test_classify_uses_exact_scorer(self):
        tasks = generate(n=10, domains=["classify"], seed=42)
        assert all(t.domain == "classify" for t in tasks)
        assert all(t.scorer == "exact" for t in tasks)
        assert all(t.expected in {"POSITIVE", "NEGATIVE"} for t in tasks)

    def test_all_domains_represented_over_large_sample(self):
        # With 100 draws and 4 uniform choices, each domain is virtually
        # certain to appear (prob of any missing < 4 * 0.75^100 ≈ 1.3e-12).
        tasks = generate(n=100, domains=ALL_DOMAINS, seed=42)
        assert {t.domain for t in tasks} == set(ALL_DOMAINS)

    def test_unknown_domain_mixed_with_known_still_completes(self):
        tasks = generate(n=15, domains=["qa", "nope"], seed=42)
        # Only qa tasks materialise, but we still get 15 of them.
        assert len(tasks) == 15
        assert all(t.domain == "qa" for t in tasks)


class TestWriteJsonl:
    def test_round_trip(self, tmp_path):
        tasks = generate(n=5, domains=ALL_DOMAINS, seed=42)
        path = tmp_path / "sub" / "tasks.jsonl"
        write_jsonl(tasks, path)

        assert path.exists()
        lines = path.read_text().splitlines()
        assert len(lines) == 5

        rebuilt = [Task(**json.loads(line)) for line in lines]
        assert rebuilt == tasks

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "tasks.jsonl"
        write_jsonl([], path)
        assert path.exists()
        assert path.read_text() == ""
