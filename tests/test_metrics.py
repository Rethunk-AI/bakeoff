"""Unit tests for bench.metrics pure functions.

Nothing here boots a model or shells out; the energy-sampling helpers are
excluded because they are the only impure functions in the module.
"""

from __future__ import annotations

import random

import pytest

from bench.metrics import (
    JUDGE_PAIR_SYSTEM,
    JUDGE_SCORE_SYSTEM,
    cost_usd,
    energy_wh,
    flops_per_token,
    gpu_weighted_seconds,
    invert_winner,
    judge_pair_prompt,
    judge_pair_randomized,
    judge_score_prompt,
    lookup_peak_tflops,
    parse_judge,
    parse_score,
    score_heuristic,
    tflops_utilization_pct,
)

# --- score_heuristic --------------------------------------------------------


class TestScoreHeuristic:
    def test_expected_none_returns_none(self):
        assert score_heuristic("exact", None, "anything") is None

    def test_unknown_scorer_returns_none(self):
        assert score_heuristic("bogus", "x", "x") is None

    @pytest.mark.parametrize("scorer", ["exact", "contains", "regex"])
    def test_empty_text_is_zero(self, scorer):
        assert score_heuristic(scorer, "hello", "") == 0.0

    # exact
    def test_exact_match_case_insensitive(self):
        assert score_heuristic("exact", "Paris", "paris") == 1.0

    def test_exact_strips_whitespace(self):
        assert score_heuristic("exact", "Paris", "  Paris  ") == 1.0

    def test_exact_substring_is_not_exact(self):
        assert score_heuristic("exact", "Paris", "Paris, France") == 0.0

    # contains
    def test_contains_substring_present(self):
        assert score_heuristic("contains", "Paris", "The capital is Paris.") == 1.0

    def test_contains_is_case_insensitive(self):
        assert score_heuristic("contains", "PARIS", "paris, france") == 1.0

    def test_contains_missing(self):
        assert score_heuristic("contains", "Paris", "London") == 0.0

    # regex
    def test_regex_matches(self):
        assert score_heuristic("regex", r"^\d+$", "12345") == 1.0

    def test_regex_no_match(self):
        assert score_heuristic("regex", r"^\d+$", "abc") == 0.0

    def test_regex_uses_search_not_fullmatch(self):
        # `re.search`, so partial hits count.
        assert score_heuristic("regex", r"\d+", "order 42 please") == 1.0


# --- parse_judge ------------------------------------------------------------


class TestParseJudge:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("A", "A"),
            ("B", "B"),
            ("TIE", "TIE"),
            ("a", "A"),  # lowercased -> upper
            (" b ", "B"),
        ],
    )
    def test_bare_verdicts(self, text, expected):
        assert parse_judge(text) == expected

    def test_empty_or_none_defaults_tie(self):
        assert parse_judge("") == "TIE"
        assert parse_judge(None) == "TIE"  # type: ignore[arg-type]

    def test_picks_last_verdict_token(self):
        # Reasoning-model output: earlier tokens are deliberation; final is the call.
        assert parse_judge("Between A and B, I pick B.") == "B"

    def test_tie_on_no_verdict_token(self):
        assert parse_judge("I cannot decide between them") == "TIE"

    def test_handles_punctuation_noise(self):
        assert parse_judge("verdict: (A).") == "A"


# --- invert_winner ----------------------------------------------------------


class TestInvertWinner:
    def test_a_inverts_to_b(self):
        assert invert_winner("A") == "B"

    def test_b_inverts_to_a(self):
        assert invert_winner("B") == "A"

    def test_tie_unchanged(self):
        assert invert_winner("TIE") == "TIE"

    def test_unknown_unchanged(self):
        # Defensive: anything else passes through so callers can see the junk.
        assert invert_winner("WAT") == "WAT"


# --- parse_score ------------------------------------------------------------


class TestParseScore:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1", 1),
            ("2", 2),
            ("3", 3),
            ("4", 4),
            ("5", 5),
        ],
    )
    def test_single_digit(self, raw, expected):
        assert parse_score(raw) == expected

    def test_default_when_no_digit(self):
        assert parse_score("no number here") == 3
        assert parse_score("no number here", default=1) == 1

    def test_empty_or_none(self):
        assert parse_score("") == 3
        assert parse_score(None) == 3  # type: ignore[arg-type]

    def test_ignores_out_of_range_digits(self):
        # 0, 6, 7, 8, 9 must not count.
        assert parse_score("score 9, rating 4") == 4

    def test_prefers_last_valid_digit(self):
        # Reasoning output: intermediate mentions shouldn't beat the final answer.
        assert parse_score("Considering 2 or 3, final answer: 5") == 5

    def test_word_boundary_only(self):
        # "A1" should not match because \b requires a word boundary around the digit.
        assert parse_score("A1") == 3  # no valid isolated digit -> default


# --- judge prompt builders --------------------------------------------------


class TestJudgePromptBuilders:
    def test_pair_prompt_structure(self):
        msgs = judge_pair_prompt("Q?", "resp-A", "resp-B")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": JUDGE_PAIR_SYSTEM}
        assert msgs[1]["role"] == "user"
        body = msgs[1]["content"]
        assert "Q?" in body
        assert "resp-A" in body
        assert "resp-B" in body
        # Slot order in the body must match the call arguments (A then B here).
        assert body.index("resp-A") < body.index("resp-B")

    def test_score_prompt_structure(self):
        msgs = judge_score_prompt("Q?", "the response")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": JUDGE_SCORE_SYSTEM}
        assert msgs[1]["role"] == "user"
        assert "Q?" in msgs[1]["content"]
        assert "the response" in msgs[1]["content"]


# --- judge_pair_randomized --------------------------------------------------


class TestJudgePairRandomized:
    def test_order_ab_keeps_original_slots(self):
        # rng.random() >= 0.5 -> AB branch
        class StubRNG:
            def random(self):
                return 0.9

        msgs, order = judge_pair_randomized("Q?", "A-text", "B-text", StubRNG())
        assert order == "AB"
        body = msgs[1]["content"]
        assert body.index("A-text") < body.index("B-text")

    def test_order_ba_swaps_slots(self):
        # rng.random() < 0.5 -> BA branch: original-B is shown first.
        class StubRNG:
            def random(self):
                return 0.1

        msgs, order = judge_pair_randomized("Q?", "A-text", "B-text", StubRNG())
        assert order == "BA"
        body = msgs[1]["content"]
        assert body.index("B-text") < body.index("A-text")

    def test_seeded_rng_is_deterministic(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        orders1 = [judge_pair_randomized("Q", "a", "b", rng1)[1] for _ in range(50)]
        orders2 = [judge_pair_randomized("Q", "a", "b", rng2)[1] for _ in range(50)]
        assert orders1 == orders2

    def test_both_orders_are_reachable_over_many_calls(self):
        rng = random.Random(0)
        orders = {judge_pair_randomized("Q", "a", "b", rng)[1] for _ in range(200)}
        assert orders == {"AB", "BA"}

    def test_invert_winner_undoes_slot_swap(self):
        """End-to-end invariant: if order == 'BA', inverting the raw verdict
        returns the label against the original A/B identities.
        """
        rng = random.Random(1)
        # Pretend the judge always picks slot-A (the first shown response).
        wins_original = []
        for _ in range(20):
            _, order = judge_pair_randomized("Q", "a", "b", rng)
            raw = "A"  # judge always picks whoever is shown first
            canonical = invert_winner(raw) if order == "BA" else raw
            wins_original.append(canonical)
        # When order == "AB" the true A wins; when "BA" inversion flips to B.
        # The mix must contain both outcomes because orders vary.
        assert set(wins_original) == {"A", "B"}


# --- energy_wh / cost_usd ---------------------------------------------------


class TestEnergyAndCost:
    def test_energy_none_watts_returns_none(self):
        assert energy_wh(None, 3600) is None

    def test_energy_basic_unit_conversion(self):
        # 100 W for one hour == 100 Wh
        assert energy_wh(100.0, 3600.0) == pytest.approx(100.0)

    def test_energy_subsecond(self):
        # 200 W for 0.5 s == 200 * (0.5/3600) Wh
        assert energy_wh(200.0, 0.5) == pytest.approx(200.0 * 0.5 / 3600.0)

    def test_energy_zero_seconds_is_zero(self):
        assert energy_wh(500.0, 0.0) == 0.0

    def test_cost_none_wh_returns_none(self):
        assert cost_usd(None, 0.15) is None

    def test_cost_basic(self):
        # 1000 Wh == 1 kWh -> 1 kWh * $0.15 = $0.15
        assert cost_usd(1000.0, 0.15) == pytest.approx(0.15)

    def test_cost_zero_rate(self):
        assert cost_usd(500.0, 0.0) == 0.0


# --- TFLOPS utilization -----------------------------------------------------


class TestFlopsPerToken:
    def test_dense_model(self):
        # 2 × 7B params
        assert flops_per_token(7_000_000_000) == 14_000_000_000

    def test_moe_uses_active_params(self):
        # MoE with 3B active out of 35B total
        assert flops_per_token(35_000_000_000, 3_000_000_000) == 6_000_000_000

    def test_no_active_falls_back_to_total(self):
        assert flops_per_token(9_000_000_000, None) == 18_000_000_000


class TestTflopsUtilization:
    def test_basic(self):
        # 40 tok/s × 14B flops/tok = 560 GFLOPs; / 82.6 TFLOPS ≈ 0.678%
        result = tflops_utilization_pct(40.0, 14_000_000_000, 82.6)
        assert result == pytest.approx((40.0 * 14e9) / (82.6e12) * 100, rel=1e-4)

    def test_zero_peak_tflops_returns_zero(self):
        assert tflops_utilization_pct(100.0, 14_000_000_000, 0.0) == 0.0

    def test_full_utilization(self):
        # tokens_per_sec = peak_tflops × 1e12 / flops_per_token → 100%
        peak = 10.0
        fpt = 1_000_000_000
        tokens = (peak * 1e12) / fpt
        assert tflops_utilization_pct(tokens, fpt, peak) == pytest.approx(100.0, rel=1e-6)


class TestLookupPeakTflops:
    def test_rtx_4090(self):
        assert lookup_peak_tflops("nvidia-geforce-rtx-4090") == pytest.approx(82.6)

    def test_strix_halo(self):
        # Strix Halo slug contains "890m"
        assert lookup_peak_tflops("amd-radeon-graphics-890m") == pytest.approx(39.0)

    def test_unknown_returns_none(self):
        assert lookup_peak_tflops("unknown-widget-9000") is None

    def test_most_specific_match_wins(self):
        # "rtx-4080-super" is longer than "rtx-4080" and should win
        result = lookup_peak_tflops("nvidia-geforce-rtx-4080-super")
        assert result == pytest.approx(52.2)


# --- gpu_weighted_seconds ----------------------------------------------------


class TestGpuWeightedSeconds:
    def test_none_sm_pct_returns_none(self):
        # Path 2 is unavailable when NVML is not present; must propagate None.
        assert gpu_weighted_seconds(10.0, None) is None

    def test_full_utilization_equals_wall_seconds(self):
        # 100% SM utilization: weighted time equals wall clock time.
        assert gpu_weighted_seconds(5.0, 100.0) == pytest.approx(5.0)

    def test_zero_utilization_returns_zero(self):
        # 0% SM: GPU did nothing, weighted time is zero.
        assert gpu_weighted_seconds(5.0, 0.0) == pytest.approx(0.0)

    def test_half_utilization(self):
        # 50% SM over 10 s wall → 5.0 s weighted.
        assert gpu_weighted_seconds(10.0, 50.0) == pytest.approx(5.0)

    def test_fractional_utilization(self):
        # 75% SM over 8 s wall → 6.0 s weighted.
        assert gpu_weighted_seconds(8.0, 75.0) == pytest.approx(6.0)
