"""Sprint 2 / PR 2.2 — ``ClassifiedRecoveryStrategy`` per-reason behaviour.

These tests are *targeted unit tests* that pin the dispatch-by-reason
table inside ``ClassifiedRecoveryStrategy._dispatch``.  Compared with
``test_fallback_chain.py`` (end-to-end engine flows), each test here
asserts a single reason → decision mapping in isolation.

The total surface is small: each ``FailoverReason`` should map to a
deterministic combination of (retry, activate_fallback, compress_context,
strip_thinking) plus a sensible ``wait_seconds``.  The dispatch table
in the strategy docstring is the source of truth; these tests make
sure the table stays accurate.
"""

from __future__ import annotations

import unittest

from aether.runtime.contracts import TurnContext
from aether.runtime.error_classifier import FailoverReason
from aether.runtime.provider_errors import (
    ProviderInvocationError,
    ResponseInvalidError,
    StreamStallError,
)
from aether.runtime.recovery import (
    AttemptState,
    ClassifiedRecoveryStrategy,
    RecoveryDecision,
)


def _ctx() -> TurnContext:
    return TurnContext(session_id="t", iteration=0, metadata={})


def _decide(
    error: ProviderInvocationError,
    *,
    attempt: int = 1,
    strategy: ClassifiedRecoveryStrategy | None = None,
) -> RecoveryDecision:
    strategy = strategy or ClassifiedRecoveryStrategy(
        max_attempts=3,
        base_wait_seconds=0.0,
        rate_limit_fallback_threshold_seconds=30.0,
    )
    return strategy.decide(error, AttemptState(attempt=attempt), _ctx())


class RateLimitDispatchTests(unittest.TestCase):
    def test_short_retry_after_does_not_request_fallback(self) -> None:
        decision = _decide(
            ProviderInvocationError(status_code=429, retry_after_seconds=5.0),
            strategy=ClassifiedRecoveryStrategy(
                base_wait_seconds=0.0,
                rate_limit_fallback_threshold_seconds=30.0,
            ),
        )
        self.assertTrue(decision.retry)
        self.assertEqual(decision.wait_seconds, 5.0)
        self.assertFalse(decision.activate_fallback)
        self.assertEqual(decision.classified_reason, "rate_limit")
        self.assertEqual(decision.reason, "rate-limit:wait")

    def test_long_retry_after_triggers_fallback(self) -> None:
        decision = _decide(
            ProviderInvocationError(status_code=429, retry_after_seconds=120.0),
            strategy=ClassifiedRecoveryStrategy(
                base_wait_seconds=0.0,
                rate_limit_fallback_threshold_seconds=30.0,
                max_wait_seconds=180.0,
            ),
        )
        self.assertTrue(decision.retry)
        self.assertGreater(decision.wait_seconds, 30.0)
        self.assertTrue(decision.activate_fallback)
        self.assertEqual(decision.reason, "rate-limit:fallback")


class ResponseInvalidDispatchTests(unittest.TestCase):
    def test_response_invalid_eagerly_falls_back_with_zero_wait(self) -> None:
        decision = _decide(
            ResponseInvalidError(validation_errors=["raw.error"]),
        )
        self.assertTrue(decision.retry)
        self.assertEqual(decision.wait_seconds, 0.0)
        self.assertTrue(decision.activate_fallback)
        self.assertEqual(decision.classified_reason, "response_invalid")


class StreamStallDispatchTests(unittest.TestCase):
    def test_stream_stalled_retries_in_place(self) -> None:
        decision = _decide(StreamStallError(stalled_after_seconds=90.0))
        self.assertTrue(decision.retry)
        self.assertEqual(decision.wait_seconds, 0.0)
        # The provider has already self-disabled streaming on the way out;
        # rotating to another provider would lose that state.
        self.assertFalse(decision.activate_fallback)


class ContextOverflowDispatchTests(unittest.TestCase):
    def test_400_context_overflow_signals_compression(self) -> None:
        decision = _decide(
            ProviderInvocationError(
                status_code=400,
                body_summary='{"error":{"message":"context length exceeded"}}',
            )
        )
        self.assertTrue(decision.retry)
        self.assertTrue(decision.compress_context)
        self.assertEqual(decision.classified_reason, "context_overflow")

    def test_413_payload_too_large_signals_compression(self) -> None:
        decision = _decide(
            ProviderInvocationError(
                status_code=413,
                body_summary='{"error":{"message":"payload too large"}}',
            )
        )
        self.assertTrue(decision.retry)
        self.assertTrue(decision.compress_context)
        self.assertEqual(decision.classified_reason, "payload_too_large")


class ThinkingSignatureDispatchTests(unittest.TestCase):
    def test_thinking_signature_400_strips_thinking(self) -> None:
        decision = _decide(
            ProviderInvocationError(
                status_code=400,
                body_summary='{"error":{"message":"Invalid signature on thinking block"}}',
            )
        )
        self.assertTrue(decision.retry)
        self.assertTrue(decision.strip_thinking)
        self.assertFalse(decision.compress_context)
        self.assertEqual(decision.classified_reason, "thinking_signature")


class HardStopDispatchTests(unittest.TestCase):
    def test_billing_402_gives_up_and_requests_fallback(self) -> None:
        decision = _decide(
            ProviderInvocationError(
                status_code=402,
                body_summary='{"error":{"message":"Insufficient credits"}}',
            )
        )
        self.assertFalse(decision.retry)
        self.assertTrue(decision.activate_fallback)
        self.assertEqual(decision.classified_reason, "billing")

    def test_auth_401_gives_up_and_requests_fallback(self) -> None:
        decision = _decide(
            ProviderInvocationError(status_code=401, body_summary="invalid api key"),
        )
        self.assertFalse(decision.retry)
        self.assertTrue(decision.activate_fallback)
        self.assertEqual(decision.classified_reason, "auth")

    def test_format_error_400_gives_up_with_fallback(self) -> None:
        decision = _decide(
            ProviderInvocationError(
                status_code=400,
                body_summary="invalid temperature value",
            )
        )
        self.assertFalse(decision.retry)
        self.assertTrue(decision.activate_fallback)
        self.assertEqual(decision.classified_reason, "format_error")

    def test_model_not_found_404_gives_up_with_fallback(self) -> None:
        decision = _decide(
            ProviderInvocationError(
                status_code=404,
                body_summary='{"error":{"message":"model not found"}}',
            )
        )
        self.assertFalse(decision.retry)
        self.assertTrue(decision.activate_fallback)
        self.assertEqual(decision.classified_reason, "model_not_found")


class BackoffDispatchTests(unittest.TestCase):
    def test_overloaded_503_uses_exponential_backoff(self) -> None:
        strategy = ClassifiedRecoveryStrategy(base_wait_seconds=2.0)
        decision = strategy.decide(
            ProviderInvocationError(status_code=503),
            AttemptState(attempt=2),  # second attempt → 2s * 2 = 4s
            _ctx(),
        )
        self.assertTrue(decision.retry)
        self.assertEqual(decision.wait_seconds, 4.0)
        self.assertEqual(decision.classified_reason, "overloaded")

    def test_server_error_500_uses_exponential_backoff(self) -> None:
        strategy = ClassifiedRecoveryStrategy(base_wait_seconds=1.5)
        decision = strategy.decide(
            ProviderInvocationError(status_code=500),
            AttemptState(attempt=3),  # third attempt → 1.5s * 4 = 6s
            _ctx(),
        )
        self.assertTrue(decision.retry)
        self.assertEqual(decision.wait_seconds, 6.0)
        self.assertEqual(decision.classified_reason, "server_error")

    def test_timeout_uses_exponential_backoff(self) -> None:
        decision = _decide(
            ProviderInvocationError(is_network_error=True, body_summary="timed out"),
        )
        self.assertTrue(decision.retry)
        self.assertEqual(decision.classified_reason, "timeout")


class BudgetExhaustionTests(unittest.TestCase):
    def test_budget_exhausted_carries_classification(self) -> None:
        strategy = ClassifiedRecoveryStrategy(max_attempts=2)
        decision = strategy.decide(
            ProviderInvocationError(status_code=429),
            AttemptState(attempt=2),  # at the cap
            _ctx(),
        )
        self.assertFalse(decision.retry)
        self.assertEqual(decision.classified_reason, "rate_limit")
        # Even on give-up, hints propagate so the engine knows to try
        # the next chain slot.
        self.assertTrue(decision.activate_fallback)


class UnknownDispatchTests(unittest.TestCase):
    def test_unknown_reason_retries_with_exponential_backoff(self) -> None:
        # ``ProviderInvocationError()`` carries no status / network
        # flag → classifier falls through to ``unknown``.  The
        # composite must still retry (per hermes-agent semantics —
        # ``ClassifiedError(retryable=True)`` for unknowns).  A
        # ``GenericBackoffStrategy`` would have given up here because
        # its ``_is_retriable`` check rejects unrecognised statuses;
        # this test pins the new behaviour.
        strategy = ClassifiedRecoveryStrategy(base_wait_seconds=2.0)
        decision = strategy.decide(
            ProviderInvocationError(),
            AttemptState(attempt=1),
            _ctx(),
        )
        self.assertTrue(decision.retry)
        self.assertEqual(decision.classified_reason, "unknown")
        self.assertEqual(decision.reason, "unknown:backoff")
        self.assertEqual(decision.wait_seconds, 2.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
