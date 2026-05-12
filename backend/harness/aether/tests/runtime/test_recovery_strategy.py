"""Sprint 0 / PR 0.3 — engine-side recovery strategy tests.

Two layers of coverage:

1. ``GenericBackoffStrategy`` / ``NoRetryStrategy`` unit tests — pin the
   decision logic without involving the engine.
2. ``AgentEngine`` integration tests — verify the run-loop:
   - calls the strategy on ``ProviderInvocationError``,
   - retries the provider after the strategy says so,
   - bumps ``provider_error_retries`` once per failed attempt,
   - records each decision in ``context.metadata['recovery_decisions']``,
   - bypasses the strategy for non-structured exceptions (preserving the
     pre-Sprint-0 behaviour for ScriptedProvider exhaustion etc.),
   - aborts the wait when the session is interrupted mid-retry.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any, Iterable, List

from aether import AgentEngine
from aether.config.schema import ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineStatus,
    NormalizedResponse,
    StreamDeltaCallback,
    TurnContext,
)
from aether.runtime.control.interrupts import InterruptController
from aether.runtime.recovery.provider_errors import ProviderInvocationError
from aether.runtime.recovery.strategies import (
    AttemptState,
    GenericBackoffStrategy,
    NoRetryStrategy,
    RecoveryDecision,
    wait_interruptible,
)
from aether.tools.base import ToolDescriptor


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _ctx() -> TurnContext:
    return TurnContext(session_id="t", iteration=0, metadata={})


class _FailingProvider(ModelProvider):
    """Test double that yields a sequence of canned outcomes per call.

    Each entry is either a ``ProviderInvocationError`` (or any
    ``Exception``) — raised — or a ``NormalizedResponse`` — returned.
    Useful for asserting "fail twice, then succeed" patterns.
    """

    def __init__(self, script: Iterable) -> None:
        self._script = list(script)
        self.calls: int = 0

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: Any = None,  # noqa: ARG002
    ) -> NormalizedResponse:
        self.calls += 1
        if not self._script:
            raise RuntimeError("FailingProvider script exhausted")
        outcome = self._script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


# ---------------------------------------------------------------------------
# GenericBackoffStrategy unit tests
# ---------------------------------------------------------------------------


class GenericBackoffStrategyTests(unittest.TestCase):
    def test_retries_429_with_exponential_backoff(self) -> None:
        strategy = GenericBackoffStrategy(max_attempts=4, base_wait_seconds=2.0)
        state = AttemptState(attempt=1)

        decision = strategy.decide(
            ProviderInvocationError(status_code=429),
            state,
            _ctx(),
        )
        self.assertTrue(decision.retry)
        self.assertEqual(decision.wait_seconds, 2.0)

        state.attempt = 2
        decision = strategy.decide(
            ProviderInvocationError(status_code=503),
            state,
            _ctx(),
        )
        self.assertEqual(decision.wait_seconds, 4.0)

        state.attempt = 3
        decision = strategy.decide(
            ProviderInvocationError(status_code=502),
            state,
            _ctx(),
        )
        self.assertEqual(decision.wait_seconds, 8.0)

    def test_retry_after_header_overrides_exponential_backoff(self) -> None:
        strategy = GenericBackoffStrategy(max_attempts=3, base_wait_seconds=2.0)
        decision = strategy.decide(
            ProviderInvocationError(status_code=429, retry_after_seconds=12.0),
            AttemptState(attempt=1),
            _ctx(),
        )
        self.assertTrue(decision.retry)
        self.assertEqual(decision.wait_seconds, 12.0)
        self.assertIn("retry-after-header", decision.reason)

    def test_retry_after_capped_to_max_wait(self) -> None:
        strategy = GenericBackoffStrategy(max_wait_seconds=30.0)
        decision = strategy.decide(
            ProviderInvocationError(status_code=429, retry_after_seconds=600.0),
            AttemptState(attempt=1),
            _ctx(),
        )
        self.assertEqual(decision.wait_seconds, 30.0)

    def test_network_error_is_retriable(self) -> None:
        strategy = GenericBackoffStrategy()
        decision = strategy.decide(
            ProviderInvocationError(is_network_error=True, status_code=None),
            AttemptState(attempt=1),
            _ctx(),
        )
        self.assertTrue(decision.retry)

    def test_non_retriable_status_gives_up(self) -> None:
        strategy = GenericBackoffStrategy()
        for status in (400, 401, 403, 404, 422):
            decision = strategy.decide(
                ProviderInvocationError(status_code=status),
                AttemptState(attempt=1),
                _ctx(),
            )
            self.assertFalse(decision.retry, f"status {status} should not be retriable")
            self.assertIn("non-retriable", decision.reason)

    def test_budget_exhaustion_gives_up(self) -> None:
        strategy = GenericBackoffStrategy(max_attempts=2)
        decision = strategy.decide(
            ProviderInvocationError(status_code=500),
            AttemptState(attempt=2),
            _ctx(),
        )
        self.assertFalse(decision.retry)
        self.assertIn("budget-exhausted", decision.reason)


class NoRetryStrategyTests(unittest.TestCase):
    def test_always_gives_up(self) -> None:
        strategy = NoRetryStrategy()
        for status in (429, 500, 503, None):
            decision = strategy.decide(
                ProviderInvocationError(status_code=status, is_network_error=status is None),
                AttemptState(attempt=1),
                _ctx(),
            )
            self.assertFalse(decision.retry)
            self.assertIn("no-retry-policy", decision.reason)


# ---------------------------------------------------------------------------
# wait_interruptible unit tests
# ---------------------------------------------------------------------------


class WaitInterruptibleTests(unittest.TestCase):
    def test_wait_completes_when_uninterrupted(self) -> None:
        ic = InterruptController()
        start = time.monotonic()
        completed = wait_interruptible(0.2, interrupt_controller=ic, session_id="s")
        elapsed = time.monotonic() - start
        self.assertTrue(completed)
        self.assertGreaterEqual(elapsed, 0.18)  # allow tiny slack for timer

    def test_zero_seconds_short_circuits(self) -> None:
        ic = InterruptController()
        start = time.monotonic()
        completed = wait_interruptible(0, interrupt_controller=ic, session_id="s")
        elapsed = time.monotonic() - start
        self.assertTrue(completed)
        self.assertLess(elapsed, 0.05)

    def test_interrupt_during_wait_aborts(self) -> None:
        ic = InterruptController()

        def trip() -> None:
            time.sleep(0.05)
            ic.request("s", "user-cancel")

        threading.Thread(target=trip).start()
        start = time.monotonic()
        completed = wait_interruptible(2.0, interrupt_controller=ic, session_id="s")
        elapsed = time.monotonic() - start
        self.assertFalse(completed)
        # Should have woken well before the 2s deadline.
        self.assertLess(elapsed, 0.5)


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------


class EngineRecoveryIntegrationTests(unittest.TestCase):
    def test_engine_retries_then_succeeds_on_429(self) -> None:
        # Two 429 errors followed by a real response.  With max_attempts=3
        # and tiny backoff, the engine should retry twice and complete.
        provider = _FailingProvider(
            [
                ProviderInvocationError(status_code=429, retry_after_seconds=0.0),
                ProviderInvocationError(status_code=429, retry_after_seconds=0.0),
                NormalizedResponse(content="finally"),
            ]
        )
        engine = AgentEngine(
            provider,
            recovery_strategy=GenericBackoffStrategy(
                max_attempts=3, base_wait_seconds=0.0
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="r1", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "finally")
        self.assertEqual(provider.calls, 3)

        runtime = result.metadata["runtime"]
        # provider_error_retries = number of failed attempts (2).
        self.assertEqual(runtime["provider_error_retries"], 2)

        decisions = result.metadata["turn"]["recovery_decisions"]
        self.assertEqual(len(decisions), 2)
        self.assertTrue(all(d["retry"] for d in decisions))
        self.assertEqual(decisions[0]["status_code"], 429)

    def test_engine_gives_up_on_non_retriable_status(self) -> None:
        # 401 must NOT be retried — engine falls through to the middleware
        # path which (with default config) maps it to PROVIDER_ERROR.
        provider = _FailingProvider(
            [ProviderInvocationError(status_code=401, body_summary="bad key")]
        )
        engine = AgentEngine(
            provider,
            recovery_strategy=GenericBackoffStrategy(max_attempts=5),
        )

        result = engine.run_turn(EngineRequest(session_id="r2", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(provider.calls, 1)
        decisions = result.metadata["turn"]["recovery_decisions"]
        self.assertEqual(len(decisions), 1)
        self.assertFalse(decisions[0]["retry"])
        self.assertIn("non-retriable", decisions[0]["reason"])

    def test_engine_exhausts_budget_then_gives_up(self) -> None:
        # All attempts fail with 500 → engine exhausts budget then surfaces
        # PROVIDER_ERROR.  provider_error_retries counts each failed attempt.
        provider = _FailingProvider(
            [ProviderInvocationError(status_code=500) for _ in range(5)]
        )
        engine = AgentEngine(
            provider,
            recovery_strategy=GenericBackoffStrategy(
                max_attempts=3, base_wait_seconds=0.0
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="r3", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(provider.calls, 3)
        self.assertEqual(result.metadata["runtime"]["provider_error_retries"], 3)
        decisions = result.metadata["turn"]["recovery_decisions"]
        # First two retry, third gives up due to budget.
        self.assertEqual([d["retry"] for d in decisions], [True, True, False])
        self.assertIn("budget-exhausted", decisions[-1]["reason"])

    def test_non_provider_invocation_error_bypasses_strategy(self) -> None:
        # RuntimeError (i.e. anything other than ProviderInvocationError)
        # must NOT trigger retry — preserves pre-Sprint-0 behaviour for
        # non-network bugs and ScriptedProvider exhaustion.
        provider = _FailingProvider([RuntimeError("scripted bug")])
        engine = AgentEngine(
            provider,
            recovery_strategy=GenericBackoffStrategy(
                max_attempts=5, base_wait_seconds=0.0
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="r4", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(provider.calls, 1)
        # No recovery decisions should have been recorded — strategy never ran.
        decisions = result.metadata["turn"].get("recovery_decisions", [])
        self.assertEqual(decisions, [])

    def test_no_retry_strategy_disables_recovery(self) -> None:
        # Even on retriable 429, if the engine is configured with
        # NoRetryStrategy it must give up after the first attempt.
        provider = _FailingProvider(
            [ProviderInvocationError(status_code=429, retry_after_seconds=0.0)]
        )
        engine = AgentEngine(provider, recovery_strategy=NoRetryStrategy())

        result = engine.run_turn(EngineRequest(session_id="r5", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(provider.calls, 1)

    def test_interrupt_during_retry_wait_aborts_turn(self) -> None:
        # A 429 with a long retry-after.  We trip the interrupt mid-wait;
        # the engine must abandon the turn rather than continuing to retry.
        provider = _FailingProvider(
            [
                ProviderInvocationError(status_code=429, retry_after_seconds=5.0),
                # Should never be reached
                NormalizedResponse(content="should not run"),
            ]
        )
        ic = InterruptController()
        engine = AgentEngine(
            provider,
            interrupt_controller=ic,
            recovery_strategy=GenericBackoffStrategy(
                max_attempts=3, base_wait_seconds=5.0, max_wait_seconds=10.0
            ),
        )

        def trip() -> None:
            time.sleep(0.05)
            ic.request("rint", "user-cancel")

        threading.Thread(target=trip).start()
        start = time.monotonic()
        result = engine.run_turn(EngineRequest(session_id="rint", user_message="hi"))
        elapsed = time.monotonic() - start

        self.assertEqual(result.status, EngineStatus.INTERRUPTED)
        self.assertEqual(provider.calls, 1)
        # Should have aborted within a fraction of a second, not waited 5s.
        self.assertLess(elapsed, 1.5)


# ---------------------------------------------------------------------------
# RecoveryDecision constructors
# ---------------------------------------------------------------------------


class RecoveryDecisionTests(unittest.TestCase):
    def test_give_up_factory(self) -> None:
        d = RecoveryDecision.give_up("nope")
        self.assertFalse(d.retry)
        self.assertEqual(d.wait_seconds, 0.0)
        self.assertEqual(d.reason, "give-up:nope")

    def test_retry_after_factory_clamps_negative(self) -> None:
        d = RecoveryDecision.retry_after(-1.0, reason="weird")
        self.assertTrue(d.retry)
        self.assertEqual(d.wait_seconds, 0.0)
        self.assertEqual(d.reason, "weird")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
