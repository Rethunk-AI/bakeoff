"""Unit tests for bench.failure — failure-reason taxonomy classifier.

Exercises classify(), is_refusal(), and dominant_failure_code() without
any network or model access.  One focused test per taxonomy code plus
edge cases for dominant_failure_code().

Run with:
    uv run python -m unittest tests.test_failure -v
"""

from __future__ import annotations

import unittest

from bench.failure import FAILURE_CODES, classify, dominant_failure_code, is_refusal


class TestClassifyTimeout(unittest.TestCase):
    def test_message_timed_out(self):
        self.assertEqual(classify(message="Request timed out after 120s"), "timeout")

    def test_message_timeout_keyword(self):
        self.assertEqual(classify(message="timeout waiting for response"), "timeout")

    def test_httpx_read_timeout(self):
        try:
            import httpx

            exc = httpx.ReadTimeout("timed out")
            self.assertEqual(classify(exc), "timeout")
        except ImportError:
            self.skipTest("httpx not installed")

    def test_message_time_out_with_space(self):
        # "timed out" (with space) must also match.
        self.assertEqual(classify(message="timed out after 60 seconds"), "timeout")


class TestClassifyOom(unittest.TestCase):
    def test_out_of_memory(self):
        self.assertEqual(classify(message="out of memory error"), "oom")

    def test_cuda_memory(self):
        self.assertEqual(classify(message="CUDA out of memory: cuda memory exhausted"), "oom")

    def test_vram(self):
        self.assertEqual(classify(message="VRAM limit exceeded"), "oom")

    def test_alloc_fail(self):
        self.assertEqual(classify(message="alloc failed: no contiguous block"), "oom")


class TestClassifyLoadFailure(unittest.TestCase):
    def test_failed_to_load(self):
        self.assertEqual(classify(message="failed to load model weights"), "load_failure")

    def test_no_such_file(self):
        self.assertEqual(classify(message="no such file: model.gguf"), "load_failure")

    def test_cannot_load(self):
        self.assertEqual(classify(message="cannot load incompatible quantization"), "load_failure")

    def test_swap_fail(self):
        self.assertEqual(classify(message="swap failed during model boot"), "load_failure")


class TestClassifyInfraError(unittest.TestCase):
    def test_connection_refused(self):
        self.assertEqual(classify(message="Connection refused on port 8080"), "infra_error")

    def test_econnrefused(self):
        self.assertEqual(classify(message="ECONNREFUSED 127.0.0.1:8080"), "infra_error")

    def test_proxy_keyword(self):
        self.assertEqual(classify(message="proxy error: upstream unreachable"), "infra_error")

    def test_httpx_connect_error(self):
        try:
            import httpx

            exc = httpx.ConnectError("connection refused")
            self.assertEqual(classify(exc), "infra_error")
        except ImportError:
            self.skipTest("httpx not installed")


class TestClassifyCancelled(unittest.TestCase):
    def test_cancelled_keyword(self):
        self.assertEqual(classify(message="task cancelled by operator"), "cancelled")

    def test_canceled_us_spelling(self):
        self.assertEqual(classify(message="run was canceled"), "cancelled")

    def test_interrupt_keyword(self):
        self.assertEqual(classify(message="interrupted by signal"), "cancelled")

    def test_keyboard_interrupt_exc(self):
        exc = KeyboardInterrupt()
        self.assertEqual(classify(exc), "cancelled")


class TestClassifyRefusal(unittest.TestCase):
    def test_i_cannot(self):
        self.assertEqual(
            classify(response_text="I cannot help with that request."), "refusal"
        )

    def test_i_cant(self):
        self.assertEqual(
            classify(response_text="I can't assist with this."), "refusal"
        )

    def test_im_sorry(self):
        self.assertEqual(
            classify(response_text="I'm sorry, but that is not something I can do."),
            "refusal",
        )

    def test_as_an_ai(self):
        self.assertEqual(
            classify(response_text="As an AI, I don't have personal opinions."), "refusal"
        )

    def test_refusal_only_when_response_text_set(self):
        # Without response_text, a refusal phrase in message does NOT produce refusal.
        result = classify(message="I cannot do this")
        # Should fall through to unknown (no other pattern matches).
        self.assertEqual(result, "unknown")

    def test_empty_response_text_not_refusal(self):
        result = classify(response_text="")
        self.assertEqual(result, "unknown")


class TestClassifyMalformed(unittest.TestCase):
    def test_malformed_message(self):
        self.assertEqual(classify(message="malformed response from model"), "malformed_output")

    def test_could_not_parse(self):
        self.assertEqual(classify(message="could not parse JSON from response"), "malformed_output")

    def test_unparseable(self):
        self.assertEqual(classify(message="unparseable output"), "malformed_output")

    def test_invalid_json(self):
        self.assertEqual(classify(message="invalid json: unexpected token"), "malformed_output")


class TestClassifyUnknown(unittest.TestCase):
    def test_unrecognized_exception(self):
        exc = ValueError("something weird happened")
        self.assertEqual(classify(exc), "unknown")

    def test_unrecognized_message(self):
        self.assertEqual(classify(message="something completely unexpected"), "unknown")

    def test_no_args_at_all(self):
        # Even with no information provided, returns unknown (graceful fallback).
        self.assertEqual(classify(), "unknown")


class TestClassifyPrecedence(unittest.TestCase):
    def test_cancelled_beats_timeout(self):
        # Message matches both cancelled and timeout; cancelled wins.
        self.assertEqual(
            classify(message="cancelled after timed out"), "cancelled"
        )

    def test_timeout_beats_infra(self):
        # Message matches timeout and network; timeout wins.
        self.assertEqual(
            classify(message="network timeout"), "timeout"
        )

    def test_oom_beats_load_failure(self):
        # "failed to load" + "out of memory": oom checked before load_failure.
        self.assertEqual(
            classify(message="failed to load: out of memory"), "oom"
        )

    def test_infra_beats_refusal(self):
        # infra_error checked before refusal.
        self.assertEqual(
            classify(
                message="connection refused",
                response_text="I cannot help with that.",
            ),
            "infra_error",
        )


class TestIsRefusal(unittest.TestCase):
    def test_none_returns_false(self):
        self.assertFalse(is_refusal(None))

    def test_empty_returns_false(self):
        self.assertFalse(is_refusal(""))

    def test_i_wont(self):
        self.assertTrue(is_refusal("I won't do that."))

    def test_am_unable(self):
        self.assertTrue(is_refusal("I am unable to comply."))

    def test_i_am_not_able(self):
        self.assertTrue(is_refusal("I am not able to assist."))

    def test_do_not_feel_comfortable(self):
        self.assertTrue(is_refusal("I do not feel comfortable answering."))

    def test_normal_text_is_not_refusal(self):
        self.assertFalse(is_refusal("The capital of France is Paris."))


class TestDominantFailureCode(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(dominant_failure_code([]))

    def test_clear_majority(self):
        codes = ["timeout", "timeout", "timeout", "refusal", "unknown"]
        self.assertEqual(dominant_failure_code(codes), "timeout")

    def test_single_item(self):
        self.assertEqual(dominant_failure_code(["oom"]), "oom")

    def test_all_same(self):
        self.assertEqual(dominant_failure_code(["infra_error"] * 5), "infra_error")

    def test_tie_broken_by_failure_codes_order(self):
        # timeout (index 0) vs refusal (index 1): timeout wins the tie.
        codes = ["refusal", "timeout"]
        self.assertEqual(dominant_failure_code(codes), "timeout")

    def test_tie_second_position_wins_over_third(self):
        # refusal (index 1) vs malformed_output (index 2): refusal wins.
        codes = ["malformed_output", "refusal"]
        self.assertEqual(dominant_failure_code(codes), "refusal")

    def test_failure_codes_tuple_order_preserved(self):
        # Verify FAILURE_CODES itself has the 9 expected members.
        self.assertEqual(len(FAILURE_CODES), 9)
        self.assertIn("timeout", FAILURE_CODES)
        self.assertIn("capability_gap", FAILURE_CODES)
        self.assertIn("unknown", FAILURE_CODES)


if __name__ == "__main__":
    unittest.main()
