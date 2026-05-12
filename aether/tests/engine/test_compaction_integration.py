"""Sprint 3 / PR 3.4 — end-to-end compaction integration tests.

These tests script a custom ``ModelProvider`` that disambiguates the
*main loop* call from the *summariser fork* sub-call (the fork passes
``tools=[]`` and an inner ``TurnContext`` carrying
``_compaction_in_progress=True``).  They then drive the full
``AgentEngine`` to assert that:

* Group L — preflight compaction triggers Tier 5 when the seed history
  exceeds the model window.
* Group M — recovery-driven compaction:
    M1: 400 + ``context_overflow`` → fork → retry succeeds.
    M2: pipeline can't reduce → ``COMPRESSION_EXHAUSTED`` terminal.
    M3: 413 + ``payload_too_large`` → exhausted → ``PAYLOAD_TOO_LARGE``.
* Group N — interaction with the fallback chain (Sprint 2): a 429 on
  provider 1 rotates to provider 2, which then triggers compression.
* Group O — per-turn isolation of the consecutive-failures counter.
"""

from __future__ import annotations

import logging
import unittest
from collections import deque
from typing import Any, Iterable, List, Optional

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.runtime.recovery.fallback_chain import FallbackChain, ProviderSlot
from aether.runtime.recovery.provider_errors import ProviderInvocationError
from aether.runtime.recovery.strategies import ClassifiedRecoveryStrategy
from aether.tools.base import ToolDescriptor


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CompactionProvider(ModelProvider):
    """Provider that branches on compaction sub-calls vs main loop calls.

    Sub-call detection: the summariser builds a fresh ``TurnContext``
    with ``_compaction_in_progress=True``, so any ``generate`` whose
    context carries that flag is the fork.  Main-loop calls take from
    ``main_script`` (errors raise, ``NormalizedResponse`` returns).
    Sub-calls take from ``fork_script``.
    """

    provider_name = "openai"
    api_mode = "chat"

    def __init__(
        self,
        *,
        main_script: Iterable[Any],
        fork_script: Iterable[Any] | None = None,
        model: str = "test-model",
        context_window: int = 100,
    ) -> None:
        self.model = model
        self.context_window = context_window
        self._main: deque[Any] = deque(main_script)
        self._fork: deque[Any] = deque(fork_script or [])
        self.main_calls = 0
        self.fork_calls = 0
        self.last_main_messages: list[dict] | None = None
        self.last_fork_messages: list[dict] | None = None

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,  # noqa: ARG002
    ) -> NormalizedResponse:
        if context.metadata.get("_compaction_in_progress"):
            self.fork_calls += 1
            self.last_fork_messages = list(messages)
            if not self._fork:
                raise RuntimeError(f"{type(self).__name__}: fork script exhausted")
            outcome = self._fork.popleft()
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        self.main_calls += 1
        self.last_main_messages = list(messages)
        if not self._main:
            raise RuntimeError(f"{type(self).__name__}: main script exhausted")
        outcome = self._main.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _seed_messages(count: int = 6, body_chars: int = 800) -> list[dict[str, Any]]:
    """Build ``count`` user messages large enough to flood a small window.

    With ``context_window=100`` (the L-tests' setup), even a single
    800-char message pushes the estimator past 0.85 * 100 = 85 tokens.
    We keep multiple seed messages so the summariser has a middle slice
    to compress (``protect_first_n + protect_last_n = 2`` in the test
    config means we need at least 3 messages for fork to do anything).
    """
    body = "x" * body_chars
    return [{"role": "user", "content": f"seed-{i} {body}"} for i in range(count)]


def _engine(
    provider: ModelProvider,
    *,
    config_overrides: dict[str, Any] | None = None,
    fallback_chain: FallbackChain | None = None,
) -> AgentEngine:
    """Build an engine wired for compaction integration tests.

    The default config overrides force every tier to attempt to fire:

    * ``compression_pre_llm_pct=0.0`` makes the pipeline tier loop
      never short-circuit (``tier_after <= 0`` is almost never true),
      so the pipeline always reaches the trailing AutoCompactor.
    * ``compression_autocompact_pct=0.0`` makes AutoCompactor's
      threshold gate (``pre_compaction_tokens >= window * pct``) trip
      on any non-zero token count.
    * The 0.0 preflight threshold also means *preflight fires every
      turn* — each test scripts both a preflight fork response and the
      recovery fork response (or two error fixtures) accordingly.
    """
    base = dict(
        max_iterations=4,
        use_builtin_tools=False,
        tool_use_contract_enabled=False,
        compression_enabled=True,
        autocompact_enabled=True,
        compression_pre_llm_pct=0.0,
        compression_autocompact_pct=0.0,
        compression_protect_first_n=1,
        compression_protect_last_n=1,
        compression_target_summary_tokens=100,
        compression_max_failures=3,
        classified_recovery_enabled=True,
        fallback_chain_enabled=fallback_chain is not None,
    )
    if config_overrides:
        base.update(config_overrides)
    return AgentEngine(
        provider,
        config=EngineConfig(**base),
        recovery_strategy=ClassifiedRecoveryStrategy(
            max_attempts=3,
            base_wait_seconds=0.0,
            rate_limit_fallback_threshold_seconds=0.0,
        ),
        fallback_chain=fallback_chain,
    )


# ---------------------------------------------------------------------------
# Group L — preflight compaction
# ---------------------------------------------------------------------------


class CompactionPreflightIntegrationTests(unittest.TestCase):
    def test_l1_preflight_triggers_tier5_then_succeeds(self) -> None:
        """Seed history exceeds window → preflight forks → main returns text.

        Also asserts the **usage-bridge fix**: the fork response below
        carries an explicit ``usage`` payload, and we verify that *both*
        the fork and the main call land on the parent turn's
        accumulator (so cost dashboards see the true spend, not just
        the main-loop slice).
        """
        provider = _CompactionProvider(
            main_script=[
                NormalizedResponse(
                    content="final answer",
                    metadata={"usage": {"prompt_tokens": 50, "completion_tokens": 5}},
                )
            ],
            fork_script=[
                NormalizedResponse(
                    content="## summary\n\n* compressed",
                    metadata={"usage": {"prompt_tokens": 4_000, "completion_tokens": 200}},
                )
            ],
        )
        engine = _engine(provider)
        result = engine.run_turn(
            EngineRequest(
                session_id="L1",
                messages=_seed_messages(count=6, body_chars=800),
                user_message="please answer",
            )
        )
        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(result.final_response, "final answer")
        # Exactly one fork (preflight) and one main call (the actual answer).
        self.assertEqual(provider.fork_calls, 1)
        self.assertEqual(provider.main_calls, 1)
        # The main call must have seen the *compressed* messages (much
        # shorter than the seed history).  We assert the compact summary
        # marker landed in the body the model actually saw.
        self.assertIsNotNone(provider.last_main_messages)
        joined = "\n".join(
            str(m.get("content", "")) for m in provider.last_main_messages or []
        )
        self.assertIn("[compact_boundary]", joined)
        compaction_meta = result.metadata.get("compaction") or {}
        self.assertEqual(
            compaction_meta.get("tier5_summaries_generated", 0),
            1,
            f"expected 1 tier5 summary, got compaction meta {compaction_meta!r}",
        )
        # ---- usage-bridge assertions (Issue #2 regression cover) ----
        # Total accumulator must include BOTH the main call and the
        # fork.  Without the bridge we'd see only 50/5 here (main only).
        usage = result.metadata.get("usage") or {}
        self.assertEqual(
            usage.get("input_tokens"),
            4_050,
            f"input tokens should sum main+fork; got usage={usage!r}",
        )
        self.assertEqual(
            usage.get("output_tokens"),
            205,
            f"output tokens should sum main+fork; got usage={usage!r}",
        )
        # Top-level api_calls must count both calls (main + fork).
        api_calls = (result.metadata.get("runtime") or {}).get("api_calls")
        if api_calls is None:
            api_calls = result.metadata.get("api_calls")
        self.assertEqual(
            api_calls,
            2,
            f"api_calls should include the fork; got metadata={result.metadata!r}",
        )
        # Fork-only counter is also surfaced for compaction-specific
        # observability (so dashboards can split "compaction cost"
        # out from "main turn cost" without re-deriving it).
        turn_meta = result.metadata.get("turn") or {}
        self.assertEqual(turn_meta.get("compaction_fork_api_calls"), 1)

    def test_l2_compression_disabled_skips_preflight(self) -> None:
        """``compression_enabled=False`` → preflight is inert, no fork called."""
        provider = _CompactionProvider(
            main_script=[NormalizedResponse(content="hello")],
            fork_script=[NormalizedResponse(content="must-not-be-called")],
        )
        engine = _engine(
            provider,
            config_overrides={"compression_enabled": False},
        )
        result = engine.run_turn(
            EngineRequest(
                session_id="L2",
                messages=_seed_messages(count=6, body_chars=800),
                user_message="please answer",
            )
        )
        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(provider.fork_calls, 0)
        self.assertEqual(provider.main_calls, 1)


# ---------------------------------------------------------------------------
# Group M — recovery-driven compaction
# ---------------------------------------------------------------------------


def _context_overflow_error() -> ProviderInvocationError:
    return ProviderInvocationError(
        status_code=400,
        body_summary="prompt is too long: context length exceeded",
    )


def _payload_too_large_error() -> ProviderInvocationError:
    return ProviderInvocationError(
        status_code=413,
        body_summary="payload too large",
    )


class CompactionRecoveryIntegrationTests(unittest.TestCase):
    def test_m1_context_overflow_compresses_then_retries(self) -> None:
        """400 ``context_overflow`` → pipeline → fork → retry succeeds.

        With the test config's pre_llm_pct=0.0 preflight also fires, so
        the script provides a long preflight summary and a short
        recovery summary — the recovery fork has to *strictly* reduce
        the token estimate for the engine to retry the main call
        (see the ``tokens_after < tokens_before`` guard in agent.py).
        """
        provider = _CompactionProvider(
            main_script=[
                _context_overflow_error(),
                NormalizedResponse(content="ok after compress"),
            ],
            fork_script=[
                NormalizedResponse(content="## summary\n\n* preflight " + ("x " * 400)),
                NormalizedResponse(content="## summary\n\n* recovery"),
            ],
        )
        engine = _engine(provider)
        result = engine.run_turn(
            EngineRequest(
                session_id="M1",
                messages=_seed_messages(count=6, body_chars=800),
                user_message="please answer",
            )
        )
        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "ok after compress")
        # Two forks (preflight + recovery), two main calls (error + retry).
        self.assertEqual(provider.fork_calls, 2)
        self.assertEqual(provider.main_calls, 2)
        compaction_meta = result.metadata.get("compaction") or {}
        self.assertEqual(compaction_meta.get("tier5_summaries_generated"), 2)

    def test_m2_pipeline_exhausted_emits_compression_exhausted(self) -> None:
        """Fork raises → pipeline exhausted → ``COMPRESSION_EXHAUSTED``.

        Both the preflight and the recovery fork raise; the second
        failure pushes ``compaction_consecutive_failures`` to 2, the
        pipeline cannot reduce tokens, and the engine surfaces the
        dedicated terminal exit reason.
        """
        provider = _CompactionProvider(
            main_script=[
                _context_overflow_error(),
                NormalizedResponse(content="must-not-be-called"),
            ],
            fork_script=[
                RuntimeError("preflight summariser blew up"),
                RuntimeError("recovery summariser blew up"),
            ],
        )
        engine = _engine(provider)
        result = engine.run_turn(
            EngineRequest(
                session_id="M2",
                messages=_seed_messages(count=6, body_chars=800),
                user_message="answer",
            )
        )
        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.COMPRESSION_EXHAUSTED)
        # Two forks attempted (preflight + recovery), one main call.
        self.assertEqual(provider.fork_calls, 2)
        self.assertEqual(provider.main_calls, 1)
        # Per-turn failure counter reflects both fork failures.
        turn_meta = result.metadata.get("turn") or {}
        self.assertEqual(turn_meta.get("compaction_consecutive_failures"), 2)

    def test_m3_payload_too_large_exhausted_emits_payload_too_large(self) -> None:
        """413 path → exhausted → ``PAYLOAD_TOO_LARGE`` (not COMPRESSION_EXHAUSTED)."""
        provider = _CompactionProvider(
            main_script=[
                _payload_too_large_error(),
                NormalizedResponse(content="must-not-be-called"),
            ],
            fork_script=[
                RuntimeError("preflight summariser blew up"),
                RuntimeError("recovery summariser blew up"),
            ],
        )
        engine = _engine(provider)
        result = engine.run_turn(
            EngineRequest(
                session_id="M3",
                messages=_seed_messages(count=6, body_chars=800),
                user_message="answer",
            )
        )
        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.PAYLOAD_TOO_LARGE)


# ---------------------------------------------------------------------------
# Group N — interaction with the fallback chain
# ---------------------------------------------------------------------------


def _slot(name: str, provider: ModelProvider) -> ProviderSlot:
    return ProviderSlot(name=name, factory=lambda: provider)


class _RateLimitedProvider(_CompactionProvider):
    """Always returns 429 on the main path so the chain rotates."""

    provider_name = "openai"


class CompactionWithFallbackIntegrationTests(unittest.TestCase):
    def test_n1_rate_limit_then_context_overflow_compresses(self) -> None:
        """Provider 1 throttles, chain rotates, provider 2 hits context overflow → compress → success.

        Preflight is allowed to fire (it always does in the test config);
        the primary therefore needs a preflight fork response too.
        After the chain rotates, ``_preflight_compaction_done`` is
        already set on the turn so the secondary skips its preflight
        and only forks during the context-overflow recovery.
        """
        primary = _RateLimitedProvider(
            main_script=[ProviderInvocationError(status_code=429, retry_after_seconds=0.0)],
            # Long preflight summary so the secondary's recovery fork can
            # measurably shrink the prompt afterwards (engine requires a
            # strict token reduction to retry).
            fork_script=[
                NormalizedResponse(content="## summary\n\n* primary preflight " + ("x " * 400))
            ],
        )
        secondary = _CompactionProvider(
            main_script=[
                _context_overflow_error(),
                NormalizedResponse(content="ok from secondary"),
            ],
            fork_script=[NormalizedResponse(content="## summary\n\n* secondary recovery")],
        )
        chain = FallbackChain([_slot("primary", primary), _slot("secondary", secondary)])
        engine = _engine(
            primary,
            config_overrides={"fallback_chain_enabled": True},
            fallback_chain=chain,
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="N1",
                messages=_seed_messages(count=6, body_chars=800),
                user_message="answer",
            )
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "ok from secondary")
        # Primary: 1 preflight fork + 1 main call (the 429).
        self.assertEqual(primary.main_calls, 1)
        self.assertEqual(primary.fork_calls, 1)
        # Secondary: 2 main calls (overflow + retry) + 1 recovery fork.
        self.assertEqual(secondary.main_calls, 2)
        self.assertEqual(secondary.fork_calls, 1)


# ---------------------------------------------------------------------------
# Group O — per-turn isolation of consecutive-failures counter
# ---------------------------------------------------------------------------


class CompactionPerTurnIsolationTests(unittest.TestCase):
    def test_o1_failure_counter_is_per_turn_not_per_session(self) -> None:
        """Two back-to-back turns on the same engine: the second is not penalised by the first's failure.

        Each turn fires preflight + recovery (2 forks).  Turn 1 has both
        fork attempts raise → COMPRESSION_EXHAUSTED with counter=2.  Turn
        2 has preflight raise (counter=1) but recovery succeed (counter
        reset to 0) so the engine retries the main call and completes.
        """
        provider = _CompactionProvider(
            main_script=[
                _context_overflow_error(),       # turn 1: only call
                _context_overflow_error(),       # turn 2: triggers recovery
                NormalizedResponse(content="ok turn-2"),
            ],
            fork_script=[
                RuntimeError("turn-1 preflight blew up"),
                RuntimeError("turn-1 recovery blew up"),
                RuntimeError("turn-2 preflight blew up"),
                NormalizedResponse(content="## summary\n\n* turn-2 recovery"),
            ],
        )
        engine = _engine(provider)

        first = engine.run_turn(
            EngineRequest(
                session_id="O1",
                messages=_seed_messages(count=6, body_chars=800),
                user_message="turn one",
            )
        )
        self.assertEqual(first.status, EngineStatus.FAILED)
        self.assertEqual(first.exit_reason, ExitReason.COMPRESSION_EXHAUSTED)
        first_meta = first.metadata.get("turn") or {}
        self.assertEqual(first_meta.get("compaction_consecutive_failures"), 2)

        second = engine.run_turn(
            EngineRequest(
                session_id="O1",
                messages=_seed_messages(count=6, body_chars=800),
                user_message="turn two",
            )
        )
        self.assertEqual(second.status, EngineStatus.COMPLETED)
        self.assertEqual(second.final_response, "ok turn-2")
        # Recovery succeeded → counter reset to 0 in turn 2's metadata.
        second_meta = second.metadata.get("turn") or {}
        self.assertEqual(second_meta.get("compaction_consecutive_failures"), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
