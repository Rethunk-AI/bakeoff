"""Regression: OOM classification must not match the substring ``oom``."""

from bench.failure import classify


def test_oom_ignores_substring_false_positives() -> None:
    for message in ("no room left", "zoom error", "request doomed", "boom"):
        assert classify(message=message) != "oom"


def test_oom_matches_word_bounded_phrases() -> None:
    for message in ("out of memory", "OOM", "oom-killed"):
        assert classify(message=message) == "oom"
