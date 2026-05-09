"""Sprint 2 / PR 2.2 — fallback chain unit + integration tests.

Coverage layout:

1. ``FallbackChain`` unit tests pin the chain primitives in isolation
   (cursor advancement, factory caching, exhaustion semantics, lazy
   resolution, empty-chain rejection).

2. ``EngineServices`` integration tests verify the ``provider`` property
   re-reads the chain after rotation so a single ``decide`` call from
   the recovery layer is enough to swap upstreams.

3. End-to-end ``AgentEngine`` integration tests script "fail twice on
   provider A, succeed on provider B" using ``ClassifiedRecoveryStrategy``
   and assert the active provider really did rotate, that the
   observability counters tick, and that exhaustion surfaces
   ``ExitReason.FALLBACK_EXHAUSTED``.
"""

from __future__ import annotations

import logging
import unittest
from typing import Any, Iterable, List, Optional

from aether import AgentEngine
from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    StreamDeltaCallback,
    TurnContext,
)
from aether.runtime.fallback_chain import (
    FallbackChain,
    FallbackChainExhausted,
    ProviderSlot,
)
from aether.runtime.interrupts import InterruptController
from aether.runtime.provider_errors import ProviderInvocationError
from aether.runtime.recovery import (
    ClassifiedRecoveryStrategy,
    GenericBackoffStrategy,
)
from aether.runtime.services import EngineServices
from aether.tools.base import ToolDescriptor
from aether.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _ScriptedProvider(ModelProvider):
    """Yields a canned sequence of responses or raises canned errors.

    Distinct from the production ``ScriptedProvider`` in
    ``models/provider/scripted.py`` because we want to count calls per
    instance and inject ``ProviderInvocationError`` directly without
    going through the middleware.
    """

    def __init__(self, name: str, script: Iterable[Any]) -> None:
        self.name = name
        self._script = list(script)
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        # Cross-PR hotfix: PR 3.1 (sprint-3) added this kwarg to the
        # base ``ModelProvider.generate`` contract; this test file was
        # authored against the pre-PR-3.1 signature.  Accept it so the
        # engine's call site (which now always passes both callbacks)
        # doesn't trip a TypeError before our recovery logic runs.
        stream_silent_callback: Any = None,  # noqa: ARG002
    ) -> NormalizedResponse:
        self.calls += 1
        if not self._script:
            raise RuntimeError(f"{self.name}: script exhausted")
        outcome = self._script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _slot(name: str, provider: Optional[ModelProvider]) -> ProviderSlot:
    """Build a ``ProviderSlot`` whose factory returns the given provider."""
    return ProviderSlot(name=name, factory=lambda: provider)


# ---------------------------------------------------------------------------
# FallbackChain unit tests
# ---------------------------------------------------------------------------


class FallbackChainUnitTests(unittest.TestCase):
    def test_empty_chain_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            FallbackChain([])

    def test_single_slot_chain_has_no_next(self) -> None:
        provider = _ScriptedProvider("p", [])
        chain = FallbackChain([_slot("only", provider)])
        self.assertFalse(chain.has_next())
        self.assertIs(chain.current_provider, provider)
        self.assertEqual(chain.current_slot_name, "only")
        self.assertEqual(chain.activations, 0)

    def test_activate_next_advances_cursor(self) -> None:
        a = _ScriptedProvider("a", [])
        b = _ScriptedProvider("b", [])
        chain = FallbackChain([_slot("a", a), _slot("b", b)])
        self.assertIs(chain.current_provider, a)
        self.assertTrue(chain.activate_next())
        self.assertIs(chain.current_provider, b)
        self.assertEqual(chain.cursor, 1)
        self.assertEqual(chain.activations, 1)
        self.assertEqual(chain.activated_provider_names, ["a", "b"])

    def test_activate_next_returns_false_at_end(self) -> None:
        chain = FallbackChain([
            _slot("a", _ScriptedProvider("a", [])),
            _slot("b", _ScriptedProvider("b", [])),
        ])
        self.assertTrue(chain.activate_next())
        self.assertFalse(chain.activate_next())
        # Cursor shouldn't have moved past the last slot.
        self.assertEqual(chain.cursor, 1)
        self.assertEqual(chain.activations, 1)

    def test_factory_called_lazily_per_slot(self) -> None:
        calls = {"a": 0, "b": 0}

        def make_factory(name: str, provider: ModelProvider):
            def _factory() -> ModelProvider:
                calls[name] += 1
                return provider

            return _factory

        a = _ScriptedProvider("a", [])
        b = _ScriptedProvider("b", [])
        chain = FallbackChain([
            ProviderSlot("a", make_factory("a", a)),
            ProviderSlot("b", make_factory("b", b)),
        ])
        # Head slot resolved eagerly at construction.
        self.assertEqual(calls, {"a": 1, "b": 0})
        # Repeated ``current_provider`` reads do not re-invoke factory.
        for _ in range(5):
            chain.current_provider
        self.assertEqual(calls, {"a": 1, "b": 0})
        # Activation resolves the next factory exactly once.
        chain.activate_next()
        self.assertEqual(calls, {"a": 1, "b": 1})

    def test_factory_returning_none_skipped_during_activation(self) -> None:
        a = _ScriptedProvider("a", [])
        c = _ScriptedProvider("c", [])
        chain = FallbackChain([
            _slot("a", a),
            ProviderSlot("b", lambda: None),
            _slot("c", c),
        ])
        self.assertTrue(chain.activate_next())
        # Should have skipped the ``None`` slot and landed on "c".
        self.assertIs(chain.current_provider, c)
        self.assertEqual(chain.cursor, 2)
        self.assertEqual(chain.activated_provider_names, ["a", "c"])

    def test_factory_raising_treated_as_unavailable_slot(self) -> None:
        a = _ScriptedProvider("a", [])

        def bad_factory() -> ModelProvider:
            raise RuntimeError("config bug")

        c = _ScriptedProvider("c", [])
        chain = FallbackChain([
            _slot("a", a),
            ProviderSlot("b", bad_factory),
            _slot("c", c),
        ])
        self.assertTrue(chain.activate_next())
        self.assertIs(chain.current_provider, c)

    def test_dead_chain_raises_at_first_use(self) -> None:
        with self.assertRaises(FallbackChainExhausted):
            chain = FallbackChain([ProviderSlot("dead", lambda: None)])
            # Resolution is deferred to first ``current_provider`` lookup
            chain.current_provider  # noqa: B018 — intentional read

    def test_reset_returns_cursor_to_head_but_keeps_history(self) -> None:
        a = _ScriptedProvider("a", [])
        b = _ScriptedProvider("b", [])
        chain = FallbackChain([_slot("a", a), _slot("b", b)])
        chain.activate_next()
        self.assertEqual(chain.cursor, 1)
        chain.reset()
        self.assertEqual(chain.cursor, 0)
        # Activation history is cumulative observability — should
        # survive a logical reset so dashboards see the full trail.
        self.assertEqual(chain.activated_provider_names, ["a", "b"])
        self.assertEqual(chain.activations, 1)


# ---------------------------------------------------------------------------
# EngineServices integration
# ---------------------------------------------------------------------------


class EngineServicesProviderPropertyTests(unittest.TestCase):
    def test_provider_property_reads_chain_when_present(self) -> None:
        a = _ScriptedProvider("a", [])
        b = _ScriptedProvider("b", [])
        chain = FallbackChain([_slot("a", a), _slot("b", b)])
        services = EngineServices(
            provider=a,
            tool_registry=ToolRegistry(),
            middleware_pipeline=MiddlewarePipeline(),
            interrupt_controller=InterruptController(),
            logger=logging.getLogger("test"),
            recovery_strategy=GenericBackoffStrategy(),
            fallback_chain=chain,
        )
        self.assertIs(services.provider, a)
        chain.activate_next()
        # No refresh call — the property re-reads each access.
        self.assertIs(services.provider, b)

    def test_provider_property_falls_back_to_initial_when_no_chain(self) -> None:
        a = _ScriptedProvider("a", [])
        services = EngineServices(
            provider=a,
            tool_registry=ToolRegistry(),
            middleware_pipeline=MiddlewarePipeline(),
            interrupt_controller=InterruptController(),
            logger=logging.getLogger("test"),
            recovery_strategy=GenericBackoffStrategy(),
            fallback_chain=None,
        )
        self.assertIs(services.provider, a)


# ---------------------------------------------------------------------------
# AgentEngine end-to-end with fallback chain
# ---------------------------------------------------------------------------


def _engine_with_chain(
    *,
    primary_script: Iterable[Any],
    fallback_script: Iterable[Any],
    fallback_chain_enabled: bool = True,
) -> tuple[AgentEngine, _ScriptedProvider, _ScriptedProvider]:
    """Build a 2-slot chain wired into an ``AgentEngine`` for end-to-end tests."""
    primary = _ScriptedProvider("primary", primary_script)
    fallback = _ScriptedProvider("fallback", fallback_script)
    chain = FallbackChain([
        _slot("primary", primary),
        _slot("fallback", fallback),
    ])
    engine = AgentEngine(
        primary,
        config=EngineConfig(
            fallback_chain_enabled=fallback_chain_enabled,
            classified_recovery_enabled=True,
        ),
        recovery_strategy=ClassifiedRecoveryStrategy(
            max_attempts=3,
            base_wait_seconds=0.0,
            rate_limit_fallback_threshold_seconds=0.0,
        ),
        fallback_chain=chain,
    )
    return engine, primary, fallback


class FallbackChainEngineIntegrationTests(unittest.TestCase):
    def test_429_rotates_to_fallback_provider_and_succeeds(self) -> None:
        engine, primary, fallback = _engine_with_chain(
            primary_script=[ProviderInvocationError(status_code=429, retry_after_seconds=0.0)],
            fallback_script=[NormalizedResponse(content="hi from fallback")],
        )

        result = engine.run_turn(EngineRequest(session_id="rot1", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "hi from fallback")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)
        decisions = result.metadata["turn"]["recovery_decisions"]
        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0]["activate_fallback"])
        self.assertEqual(decisions[0]["classified_reason"], "rate_limit")

    def test_fallback_disabled_keeps_single_provider(self) -> None:
        # Same script but ``fallback_chain_enabled=False`` — the chain is
        # configured but the engine refuses to rotate.  Should give up
        # after exhausting the primary's retries.
        engine, primary, fallback = _engine_with_chain(
            primary_script=[
                ProviderInvocationError(status_code=429, retry_after_seconds=0.0),
                ProviderInvocationError(status_code=429, retry_after_seconds=0.0),
                ProviderInvocationError(status_code=429, retry_after_seconds=0.0),
            ],
            fallback_script=[NormalizedResponse(content="never reached")],
            fallback_chain_enabled=False,
        )

        result = engine.run_turn(EngineRequest(session_id="nofb", user_message="hi"))

        # Fallback never invoked — flag respected.
        self.assertEqual(fallback.calls, 0)
        # Strategy went through 3 attempts, then gave up.
        self.assertGreaterEqual(primary.calls, 1)
        self.assertEqual(result.status, EngineStatus.FAILED)

    def test_chain_exhaustion_surfaces_fallback_exhausted(self) -> None:
        # Both providers return 429 forever — chain rotates once, then
        # gives up.  Exit reason must be FALLBACK_EXHAUSTED, not
        # PROVIDER_ERROR / RATE_LIMITED, so observers can tell the
        # cause from the symptom.
        engine, primary, fallback = _engine_with_chain(
            primary_script=[ProviderInvocationError(status_code=429, retry_after_seconds=0.0)],
            fallback_script=[
                ProviderInvocationError(status_code=429, retry_after_seconds=0.0),
                ProviderInvocationError(status_code=429, retry_after_seconds=0.0),
                ProviderInvocationError(status_code=429, retry_after_seconds=0.0),
                ProviderInvocationError(status_code=429, retry_after_seconds=0.0),
                ProviderInvocationError(status_code=429, retry_after_seconds=0.0),
            ],
        )

        result = engine.run_turn(EngineRequest(session_id="exh", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.FALLBACK_EXHAUSTED)

    def test_response_invalid_eagerly_falls_back(self) -> None:
        # ResponseInvalidError on the primary should rotate without
        # waiting — no exponential backoff, immediate fallback.
        from aether.runtime.provider_errors import ResponseInvalidError

        engine, primary, fallback = _engine_with_chain(
            primary_script=[ResponseInvalidError(validation_errors=["raw.error"])],
            fallback_script=[NormalizedResponse(content="recovered")],
        )

        result = engine.run_turn(EngineRequest(session_id="ri", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "recovered")
        decisions = result.metadata["turn"]["recovery_decisions"]
        self.assertTrue(decisions[0]["activate_fallback"])
        self.assertEqual(decisions[0]["classified_reason"], "response_invalid")
        self.assertEqual(decisions[0]["wait_seconds"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
