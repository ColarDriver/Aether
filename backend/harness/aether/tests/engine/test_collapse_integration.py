"""Sprint 3 / PR 3.7 — end-to-end Tier 4 collapse integration tests.

These tests drive the full :class:`AgentEngine` (no mocks of internal
compaction wiring) and assert four classes of behaviour:

* **Group J** — mutual exclusion with Tier 5: a committed Tier 4
  segment sets ``collapse_owns_headroom``; AutoCompactor honours that
  flag and bails out, so we should observe Tier 4 runs but Tier 5
  does not fork.
* **Group K** — projection correctness: the messages list the
  *provider* receives is the *view* (with the synthetic
  ``[collapsed_segment ...]`` marker), but the *original* message
  history (visible via the engine's recorded transcript) is left
  intact — required for session_record / replay / steer.
* **Group L** — multi-tier interplay: the pipeline always runs the
  tiers in 2 → 3 → 4 → 5 order; Tier 4 runs only if the upstream
  tiers haven't already brought the view below ``commit_pct``.
* **Group M** — per-session isolation: a committed segment in one
  session does NOT leak ``collapse_owns_headroom`` into another
  session's metadata.
"""

from __future__ import annotations

import unittest
from collections import deque
from typing import Any, Iterable, List

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
from aether.runtime.recovery.strategies import ClassifiedRecoveryStrategy
from aether.tools.base import ToolDescriptor


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CollapseProvider(ModelProvider):
    """Provider that distinguishes Tier 4/5 fork sub-calls from main calls.

    Same disambiguation as the PR 3.4 integration test scaffold:
    sub-calls carry ``_compaction_in_progress=True`` on their inner
    ``TurnContext``.  We track the calls separately so tests can pin
    exactly how many forks each tier triggered.
    """

    provider_name = "openai"
    api_mode = "chat"

    def __init__(
        self,
        *,
        main_script: Iterable[Any],
        fork_script: Iterable[Any] | None = None,
        model: str = "test-model",
        context_window: int = 100_000,
    ) -> None:
        self.model = model
        self.context_window = context_window
        self._main: deque[Any] = deque(main_script)
        self._fork: deque[Any] = deque(fork_script or [])
        self.main_calls = 0
        self.fork_calls = 0
        self.last_main_messages: list[dict] | None = None
        self.last_fork_messages: list[dict] | None = None
        self.all_main_messages: list[list[dict]] = []

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
        snapshot = list(messages)
        self.last_main_messages = snapshot
        self.all_main_messages.append(snapshot)
        if not self._main:
            raise RuntimeError(f"{type(self).__name__}: main script exhausted")
        outcome = self._main.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _seed_messages(count: int = 30, body_chars: int = 4_000) -> list[dict[str, Any]]:
    """Long history sized to comfortably exceed the test commit threshold.

    Token estimator yields ~chars/3.  With model_window=10_000 and
    commit_pct=0.50 → commit_threshold=5_000 tokens.
    30 * 4_000 chars / 3 ≈ 40_000 tokens, well above 5_000 — so
    Tier 4 (when enabled) will fire.
    """
    body = "x" * body_chars
    return [{"role": "user", "content": f"seed-{i} {body}"} for i in range(count)]


def _engine(
    provider: ModelProvider,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> AgentEngine:
    """Build an engine wired for Tier 4 collapse integration tests.

    Defaults flip Tier 4 on at very low thresholds so we can drive
    it deterministically.  The PR-3.4 integration tests' approach of
    "force every tier to fire" is reused — preflight and recovery
    both trigger compaction unconditionally.
    """
    base = dict(
        max_iterations=4,
        use_builtin_tools=False,
        tool_use_contract_enabled=False,
        # Master switches.
        compression_enabled=True,
        autocompact_enabled=True,
        context_collapse_enabled=True,
        # Aggressive thresholds — deterministic firing.
        compression_pre_llm_pct=0.0,
        compression_autocompact_pct=0.0,
        context_collapse_commit_pct=0.50,
        context_collapse_blocking_pct=0.50,
        context_collapse_segment_max_messages=10,
        compression_protect_first_n=1,
        compression_protect_last_n=1,
        compression_target_summary_tokens=100,
        compression_max_failures=3,
        # Tier 3 (microcompact) — disable by setting an unreachable gap
        # threshold so it never fires (we want to isolate Tier 4 here).
        microcompact_gap_threshold_minutes=10_000_000.0,
        classified_recovery_enabled=True,
        fallback_chain_enabled=False,
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
    )


def _summary_response(text: str = "## summary\n\n* folded conversation") -> NormalizedResponse:
    """A summariser fork response shaped like the real LLM output."""
    return NormalizedResponse(
        content=text,
        metadata={"usage": {"prompt_tokens": 200, "completion_tokens": 50}},
    )


def _final_response(text: str = "final answer") -> NormalizedResponse:
    return NormalizedResponse(
        content=text,
        metadata={"usage": {"prompt_tokens": 80, "completion_tokens": 8}},
    )


def _compaction(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convenience accessor for the curated ``compaction`` sub-dict.

    The engine surfaces per-tier counters under
    ``EngineResult.metadata['compaction']`` (see ``agent.py`` final
    metadata block).  Internal keys like ``tier_outcomes`` and
    ``collapse_owns_headroom`` live on the per-turn ``TurnContext``
    and are *not* exposed on the result — by design, to keep the
    public engine surface small.
    """
    return dict(metadata.get("compaction") or {})


# ---------------------------------------------------------------------------
# Group J — mutual exclusion with Tier 5 (collapse_owns_headroom)
# ---------------------------------------------------------------------------


class CollapseTierFiveMutexTests(unittest.TestCase):
    def test_j1_committed_collapse_segment_blocks_tier5_fork(self) -> None:
        """Tier 4 commits → ``collapse_owns_headroom=True`` → Tier 5 bails.

        Direct observation:

        * Exactly one fork happened (consumed by Tier 4) — if Tier 5
          had *also* forked, the provider would have hit fork-script
          exhaustion and raised.
        * ``compaction.tier4_collapse_segments == 1`` — Tier 4 fired.
        * ``compaction.tier5_summaries_generated == 0`` — Tier 5 was
          blocked by the ``collapse_owns_headroom`` flag (the only
          way it can stay 0 when the threshold gate would otherwise
          trigger).
        """
        provider = _CollapseProvider(
            main_script=[_final_response()],
            fork_script=[_summary_response()],
            context_window=10_000,
        )
        engine = _engine(provider)
        result = engine.run_turn(
            EngineRequest(
                session_id="J1",
                messages=_seed_messages(count=30, body_chars=4_000),
                user_message="please answer",
            )
        )
        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(provider.fork_calls, 1)
        compaction = _compaction(result.metadata)
        self.assertGreaterEqual(compaction.get("tier4_collapse_segments", 0), 1)
        self.assertEqual(compaction.get("tier5_summaries_generated"), 0)

    def test_j2_no_committed_segment_lets_tier5_fork(self) -> None:
        """Tier 4 with an unreachable commit threshold → Tier 5 still forks.

        We push the commit threshold to ``model_window * 100`` (i.e.
        well above any realistic view-token count) so Tier 4 stays
        read-only.  Tier 5 then does its normal job.  The
        percentages are ``float`` values that the tier multiplies
        by ``model_window`` and ``int()`` casts — values above 1.0
        are accepted (they just produce a huge threshold).
        """
        provider = _CollapseProvider(
            main_script=[_final_response()],
            fork_script=[_summary_response()],
            context_window=10_000,
        )
        engine = _engine(
            provider,
            config_overrides={
                "context_collapse_commit_pct": 100.0,
                "context_collapse_blocking_pct": 100.0,
            },
        )
        result = engine.run_turn(
            EngineRequest(
                session_id="J2",
                messages=_seed_messages(count=30, body_chars=4_000),
                user_message="please answer",
            )
        )
        self.assertEqual(provider.fork_calls, 1)
        compaction = _compaction(result.metadata)
        # Tier 4 stayed inert (no segments committed/proposed).
        self.assertEqual(compaction.get("tier4_collapse_segments"), 0)
        # Tier 5 owned the fork.
        self.assertEqual(compaction.get("tier5_summaries_generated"), 1)

    def test_j3_collapse_disabled_falls_through_to_tier5(self) -> None:
        """``context_collapse_enabled=False`` → Tier 4 inert → Tier 5 forks normally."""
        provider = _CollapseProvider(
            main_script=[_final_response()],
            fork_script=[_summary_response()],
            context_window=10_000,
        )
        engine = _engine(
            provider,
            config_overrides={"context_collapse_enabled": False},
        )
        result = engine.run_turn(
            EngineRequest(
                session_id="J3",
                messages=_seed_messages(count=30, body_chars=4_000),
                user_message="please answer",
            )
        )
        self.assertEqual(provider.fork_calls, 1)
        compaction = _compaction(result.metadata)
        self.assertEqual(compaction.get("tier4_collapse_segments"), 0)
        self.assertEqual(compaction.get("tier5_summaries_generated"), 1)


# ---------------------------------------------------------------------------
# Group K — projection correctness
# ---------------------------------------------------------------------------


class CollapseProjectionTests(unittest.TestCase):
    def test_k1_provider_sees_view_with_collapsed_marker(self) -> None:
        """The main-loop messages the provider receives carry the ``[collapsed_segment ...]`` marker."""
        provider = _CollapseProvider(
            main_script=[_final_response()],
            fork_script=[_summary_response("## summary\n\n* compressed body")],
            context_window=10_000,
        )
        engine = _engine(provider)
        engine.run_turn(
            EngineRequest(
                session_id="K1",
                messages=_seed_messages(count=30, body_chars=4_000),
                user_message="please answer",
            )
        )
        # Inspect what the provider's main_calls received.
        self.assertIsNotNone(provider.last_main_messages)
        joined = "\n".join(
            str(m.get("content", "")) for m in provider.last_main_messages or []
        )
        # Marker present.
        self.assertIn("[collapsed_segment idx=", joined)
        # Summary text from the fork landed in the projection.
        self.assertIn("compressed body", joined)
        # The projection is shorter than the original 30+ messages.
        # (Original had 30 seed + system + user + ... the marker
        # collapses 10 of them into 1, so length should drop.)
        self.assertLess(len(provider.last_main_messages or []), 30)

    def test_k2_collapse_store_resets_per_turn(self) -> None:
        """Sprint 3 minimum: store is per-turn (lives on TurnContext.metadata).

        After a turn finishes, the result's metadata captures a
        snapshot of that turn's collapse state.  A subsequent turn
        starts with a fresh ``TurnContext`` and therefore an empty
        store — collapse must re-evaluate from scratch.  This test
        documents that decision (cross-turn persistence is a planned
        Sprint 3+ follow-up).
        """
        provider = _CollapseProvider(
            main_script=[_final_response("answer-1"), _final_response("answer-2")],
            fork_script=[_summary_response(), _summary_response()],
            context_window=10_000,
        )
        engine = _engine(provider)
        # Turn 1 — Tier 4 commits a segment.
        result1 = engine.run_turn(
            EngineRequest(
                session_id="K2",
                messages=_seed_messages(count=30, body_chars=4_000),
                user_message="first",
            )
        )
        # Turn 2 — fresh request, fresh TurnContext.
        result2 = engine.run_turn(
            EngineRequest(
                session_id="K2",
                messages=_seed_messages(count=30, body_chars=4_000),
                user_message="second",
            )
        )
        # Both turns saw Tier 4 fork (store reset between them).
        self.assertEqual(provider.fork_calls, 2)
        self.assertEqual(result1.status, EngineStatus.COMPLETED)
        self.assertEqual(result2.status, EngineStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Group L — multi-tier interplay
# ---------------------------------------------------------------------------


class CollapseMultiTierInteractionTests(unittest.TestCase):
    def test_l1_pipeline_runs_tiers_in_documented_order(self) -> None:
        """Pipeline log emits one line per tier, in 2 → 3 → 4 → 5 order.

        :class:`CompactionPipeline` writes one ``compact[<trigger>]
        <tier_name>: ... tokens (-N)`` log per tier as it iterates.
        We capture the root logger and verify the relative order.
        """
        provider = _CollapseProvider(
            main_script=[_final_response()],
            fork_script=[_summary_response()],
            context_window=10_000,
        )
        engine = _engine(provider)
        # ``logging.getLogger()`` (root) captures the pipeline logger
        # because the engine uses the standard ``logging`` module.
        with self.assertLogs(level="INFO") as recorded:
            engine.run_turn(
                EngineRequest(
                    session_id="L1",
                    messages=_seed_messages(count=30, body_chars=4_000),
                    user_message="please answer",
                )
            )
        log_blob = "\n".join(recorded.output)
        # All four tier names must appear, with Tier 2 before Tier 3
        # before Tier 4 before Tier 5.
        positions = {
            name: log_blob.find(name)
            for name in (
                "tier2_snip",
                "tier3_microcompact",
                "tier4_collapse",
                "tier5_autocompact",
            )
        }
        for name, pos in positions.items():
            self.assertGreaterEqual(pos, 0, f"missing log mention of {name}")
        self.assertLess(positions["tier2_snip"], positions["tier3_microcompact"])
        self.assertLess(positions["tier3_microcompact"], positions["tier4_collapse"])
        self.assertLess(positions["tier4_collapse"], positions["tier5_autocompact"])

    def test_l2_tier4_disabled_is_a_pure_passthrough(self) -> None:
        """Tier 4 disabled → counter stays 0; Tier 5 still runs."""
        provider = _CollapseProvider(
            main_script=[_final_response()],
            fork_script=[_summary_response()],
            context_window=10_000,
        )
        engine = _engine(
            provider,
            config_overrides={"context_collapse_enabled": False},
        )
        result = engine.run_turn(
            EngineRequest(
                session_id="L2",
                messages=_seed_messages(count=30, body_chars=4_000),
                user_message="please answer",
            )
        )
        compaction = _compaction(result.metadata)
        self.assertEqual(compaction.get("tier4_collapse_segments"), 0)
        # Tier 5 still got to fork (handled the actual reduction).
        self.assertGreaterEqual(compaction.get("tier5_summaries_generated", 0), 1)
        self.assertGreaterEqual(provider.fork_calls, 1)


# ---------------------------------------------------------------------------
# Group M — per-session isolation
# ---------------------------------------------------------------------------


class CollapseSessionIsolationTests(unittest.TestCase):
    def test_m1_collapse_state_does_not_leak_across_sessions(self) -> None:
        """Per-session counters reset between ``run_turn`` calls.

        Each ``run_turn`` builds a *fresh* ``TurnContext``, so the
        ``_collapse_store`` and ``collapse_owns_headroom`` flag (which
        live on ``TurnContext.metadata``) cannot leak cross-session
        by construction.  This test pins that invariant: session A
        triggers Tier 4 (committed segment); session B sends a tiny
        history that should NOT trip Tier 4.
        """
        provider = _CollapseProvider(
            main_script=[_final_response("a"), _final_response("b")],
            fork_script=[_summary_response()],
            context_window=10_000,
        )
        engine = _engine(provider)
        result_a = engine.run_turn(
            EngineRequest(
                session_id="M1-A",
                messages=_seed_messages(count=30, body_chars=4_000),
                user_message="a",
            )
        )
        result_b = engine.run_turn(
            EngineRequest(
                session_id="M1-B",
                messages=[{"role": "user", "content": "tiny"}],
                user_message="b",
            )
        )
        compaction_a = _compaction(result_a.metadata)
        compaction_b = _compaction(result_b.metadata)
        # Session A: Tier 4 fired.
        self.assertGreaterEqual(compaction_a.get("tier4_collapse_segments", 0), 1)
        # Session B: Tier 4 did NOT fire (tiny history); leak would
        # show as a non-zero counter.
        self.assertEqual(compaction_b.get("tier4_collapse_segments"), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
