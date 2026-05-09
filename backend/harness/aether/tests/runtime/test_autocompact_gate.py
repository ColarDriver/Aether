"""Sprint 3 / PR 3.4 — ``AutoCompactor`` (Tier 5) gate + circuit breaker.

The auto-compactor's job is to decide *whether* to fork an LLM
summariser, then run it.  Five conditions must all be true to fire:

  1. recursion guard — caller is not already inside a compaction
  2. compression master switch is on
  3. autocompact tier toggle is on
  4. context-collapse (Tier 4) does not own headroom
  5. token usage above the Tier 5 threshold

A circuit breaker on top: after ``compression_max_failures`` consecutive
summariser failures in this turn, refuse to call again — the test
fixture proves both directions of the gate, then sets each condition
false in isolation to verify the gate composition.
"""

from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass
from typing import Any

from aether.runtime.contracts import TurnContext
from aether.services.compact.autocompact import AutoCompactor
from aether.services.compact.compactor import CompactionContext


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    """Duck-typed ``EngineConfig`` slice the autocompactor consults."""

    compression_enabled: bool = True
    autocompact_enabled: bool = True
    compression_autocompact_pct: float = 0.85
    compression_max_failures: int = 3


class _RecordingSummarizer:
    """Stand-in for ``LLMForkSummarizer`` that records calls."""

    def __init__(
        self,
        *,
        return_messages: list[dict] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self._return_messages = return_messages
        self._raises = raises
        self.calls: list[tuple[list[dict], str, TurnContext]] = []

    def summarise(
        self,
        messages: list[dict],
        *,
        model: str,
        turn_context: TurnContext,
    ) -> list[dict]:
        self.calls.append((messages, model, turn_context))
        if self._raises is not None:
            raise self._raises
        if self._return_messages is None:
            # Default: collapse to two messages so freed > 0.
            return [
                {"role": "system", "content": "[boundary]"},
                {"role": "user", "content": "summary"},
            ]
        return list(self._return_messages)


def _ctx(metadata: dict | None = None) -> TurnContext:
    return TurnContext(
        session_id="sess-x",
        iteration=0,
        metadata=metadata if metadata is not None else {},
    )


def _compaction_ctx(
    *,
    pre_tokens: int = 200_000,
    window: int = 200_000,
    failures: int = 0,
    trigger: str = "context_overflow",
) -> CompactionContext:
    return CompactionContext(
        session_id="sess-x",
        model="claude-sonnet",
        model_window=window,
        pre_compaction_tokens=pre_tokens,
        target_pct=0.85,
        trigger_reason=trigger,
        consecutive_failures=failures,
    )


def _build(
    *,
    config: _Config | None = None,
    summarizer: _RecordingSummarizer | None = None,
) -> tuple[AutoCompactor, _RecordingSummarizer, _Config]:
    cfg = config or _Config()
    summer = summarizer or _RecordingSummarizer()
    return (
        AutoCompactor(
            config=cfg,
            summarizer=summer,  # type: ignore[arg-type]
            logger=logging.getLogger("test.autocompact"),
        ),
        summer,
        cfg,
    )


def _ten_messages() -> list[dict]:
    return [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "first user"},
        *[{"role": "assistant", "content": f"assistant body {i}"} for i in range(8)],
    ]


# ---------------------------------------------------------------------------
# Group D — five-condition gate
# ---------------------------------------------------------------------------


class AutoCompactorGroupDGateTests(unittest.TestCase):
    def test_d1_recursion_guard_blocks_summariser(self) -> None:
        """Condition 1: ``_compaction_in_progress=True`` → no-op."""
        compactor, summer, _ = _build()
        turn = _ctx({"_compaction_in_progress": True})
        out, freed = compactor.maybe_run(_ten_messages(), _compaction_ctx(), turn)
        self.assertEqual(freed, 0)
        self.assertEqual(out, _ten_messages())
        self.assertEqual(summer.calls, [])

    def test_d2_compression_master_switch_off(self) -> None:
        """Condition 2: ``compression_enabled=False`` → no-op."""
        compactor, summer, _ = _build(config=_Config(compression_enabled=False))
        turn = _ctx()
        out, freed = compactor.maybe_run(_ten_messages(), _compaction_ctx(), turn)
        self.assertEqual(freed, 0)
        self.assertEqual(out, _ten_messages())
        self.assertEqual(summer.calls, [])

    def test_d3_autocompact_tier_switch_off(self) -> None:
        """Condition 3: ``autocompact_enabled=False`` → no-op."""
        compactor, summer, _ = _build(config=_Config(autocompact_enabled=False))
        out, freed = compactor.maybe_run(_ten_messages(), _compaction_ctx(), _ctx())
        self.assertEqual(freed, 0)
        self.assertEqual(out, _ten_messages())
        self.assertEqual(summer.calls, [])

    def test_d4_collapse_owns_headroom_blocks_summariser(self) -> None:
        """Condition 4: Tier 4 owns headroom → no-op."""
        compactor, summer, _ = _build()
        turn = _ctx({"collapse_owns_headroom": True})
        out, freed = compactor.maybe_run(_ten_messages(), _compaction_ctx(), turn)
        self.assertEqual(freed, 0)
        self.assertEqual(out, _ten_messages())
        self.assertEqual(summer.calls, [])

    def test_d5_below_token_threshold_short_circuits(self) -> None:
        """Condition 5: ``pre_compaction_tokens < threshold`` → no-op."""
        compactor, summer, _ = _build()
        # Threshold = 200_000 * 0.85 = 170_000.  Pre-tokens < threshold.
        ctx = _compaction_ctx(pre_tokens=100_000, window=200_000)
        out, freed = compactor.maybe_run(_ten_messages(), ctx, _ctx())
        self.assertEqual(freed, 0)
        self.assertEqual(out, _ten_messages())
        self.assertEqual(summer.calls, [])

    def test_d6_all_conditions_satisfied_invokes_summariser(self) -> None:
        """All conditions met → summariser called once and result returned."""
        replacement = [
            {"role": "system", "content": "[boundary]"},
            {"role": "user", "content": "compact summary"},
        ]
        compactor, summer, _ = _build(
            summarizer=_RecordingSummarizer(return_messages=replacement)
        )
        turn = _ctx()
        out, freed = compactor.maybe_run(_ten_messages(), _compaction_ctx(), turn)
        self.assertEqual(out, replacement)
        # Original 10 - new 2 = 8 freed (rough proxy).
        self.assertEqual(freed, 10 - 2)
        self.assertEqual(len(summer.calls), 1)
        # Summariser was invoked with the active model from CompactionContext.
        self.assertEqual(summer.calls[0][1], "claude-sonnet")
        # Recursion guard was lifted in the finally block.
        self.assertNotIn("_compaction_in_progress", turn.metadata)


# ---------------------------------------------------------------------------
# Group E — circuit breaker
# ---------------------------------------------------------------------------


class AutoCompactorGroupECircuitBreakerTests(unittest.TestCase):
    def test_e1_below_max_failures_still_runs(self) -> None:
        """``failures < max_failures`` → summariser still gets a chance."""
        compactor, summer, _ = _build(config=_Config(compression_max_failures=3))
        ctx = _compaction_ctx(failures=2)
        compactor.maybe_run(_ten_messages(), ctx, _ctx())
        self.assertEqual(len(summer.calls), 1)

    def test_e2_at_max_failures_skips_summariser(self) -> None:
        """``failures >= max_failures`` → no summariser, no-op return."""
        compactor, summer, _ = _build(config=_Config(compression_max_failures=3))
        ctx = _compaction_ctx(failures=3)
        out, freed = compactor.maybe_run(_ten_messages(), ctx, _ctx())
        self.assertEqual(freed, 0)
        self.assertEqual(summer.calls, [])
        self.assertEqual(out, _ten_messages())

    def test_e3_summariser_exception_increments_failure_counter(self) -> None:
        """Exception → metadata counter +1, no crash."""
        compactor, _, _ = _build(
            summarizer=_RecordingSummarizer(raises=RuntimeError("boom")),
        )
        turn = _ctx()
        out, freed = compactor.maybe_run(_ten_messages(), _compaction_ctx(), turn)
        self.assertEqual(freed, 0)
        self.assertEqual(out, _ten_messages())
        self.assertEqual(turn.metadata.get("compaction_consecutive_failures"), 1)
        # finally block must clean up the recursion guard.
        self.assertNotIn("_compaction_in_progress", turn.metadata)

    def test_e4_success_resets_failure_counter(self) -> None:
        """Successful summary → counter reset to 0 + ``tier5`` counter ticks."""
        compactor, summer, _ = _build()
        turn = _ctx({"compaction_consecutive_failures": 2})
        out, freed = compactor.maybe_run(_ten_messages(), _compaction_ctx(), turn)
        self.assertGreater(freed, 0)
        self.assertEqual(turn.metadata["compaction_consecutive_failures"], 0)
        self.assertEqual(turn.metadata.get("tier5_summaries_generated"), 1)
        self.assertEqual(len(summer.calls), 1)
        self.assertNotEqual(out, _ten_messages())

    def test_e5_three_failures_then_blocks_subsequent_attempts(self) -> None:
        """Simulate three consecutive failures from a single fixture, then fourth call must skip."""
        compactor, summer, _ = _build(
            summarizer=_RecordingSummarizer(raises=RuntimeError("nope")),
            config=_Config(compression_max_failures=3),
        )
        turn = _ctx()
        for failures_before in range(3):
            ctx = _compaction_ctx(failures=failures_before)
            compactor.maybe_run(_ten_messages(), ctx, turn)
        # Now metadata reflects 3 failures.  Fourth attempt must short-circuit.
        self.assertEqual(turn.metadata["compaction_consecutive_failures"], 3)
        ctx = _compaction_ctx(
            failures=int(turn.metadata["compaction_consecutive_failures"])
        )
        out, freed = compactor.maybe_run(_ten_messages(), ctx, turn)
        self.assertEqual(out, _ten_messages())
        self.assertEqual(freed, 0)
        # Still 3 calls — the breaker prevented the 4th.
        self.assertEqual(len(summer.calls), 3)

    def test_e6_failure_increment_reads_latest_metadata_not_snapshot(self) -> None:
        """If something else bumped ``compaction_consecutive_failures``
        between snapshot and write, the increment uses the latest
        value — never overwrites concurrent updates with stale +1.

        This pins the Issue #7 fix: the autocompactor now reads the
        current value from ``turn_context.metadata`` at write time
        instead of trusting the ``ctx.consecutive_failures`` snapshot
        captured at pipeline entry.  Today no other tier writes the
        counter, but the contract is "writers compose without losing
        each other's increments" so future tiers (e.g. a summariser
        retry layer) don't silently regress this counter.
        """
        compactor, _, _ = _build(
            summarizer=_RecordingSummarizer(raises=RuntimeError("boom")),
        )
        # Snapshot says 0, but metadata jumped to 5 mid-flight (e.g.
        # because a hypothetical sibling tier reported its own batch
        # of failures).  After our increment we expect 6, not 1.
        turn = _ctx({"compaction_consecutive_failures": 5})
        ctx = _compaction_ctx(failures=0)  # stale snapshot
        compactor.maybe_run(_ten_messages(), ctx, turn)
        self.assertEqual(turn.metadata["compaction_consecutive_failures"], 6)


# ---------------------------------------------------------------------------
# Group F — observability invariants
# ---------------------------------------------------------------------------


class AutoCompactorGroupFObservabilityTests(unittest.TestCase):
    def test_f1_success_increments_tier5_summaries_generated(self) -> None:
        """Each successful summary increments the public counter."""
        compactor, _, _ = _build()
        turn = _ctx()
        for expected in (1, 2, 3):
            compactor.maybe_run(_ten_messages(), _compaction_ctx(), turn)
            self.assertEqual(turn.metadata["tier5_summaries_generated"], expected)

    def test_f2_recursion_guard_cleared_even_on_summariser_failure(self) -> None:
        """If the summariser throws, the recursion guard MUST still be cleared."""
        compactor, _, _ = _build(
            summarizer=_RecordingSummarizer(raises=RuntimeError("nope")),
        )
        turn = _ctx()
        compactor.maybe_run(_ten_messages(), _compaction_ctx(), turn)
        self.assertNotIn("_compaction_in_progress", turn.metadata)

    def test_f3_summariser_returning_same_object_is_treated_as_no_op(self) -> None:
        """If summariser is too short-circuited and returns input identity, do not bump counters."""
        original = _ten_messages()
        compactor, _, _ = _build(summarizer=_IdentitySummarizer())
        turn = _ctx({"compaction_consecutive_failures": 1})
        out, freed = compactor.maybe_run(original, _compaction_ctx(), turn)
        self.assertIs(out, original)
        self.assertEqual(freed, 0)
        # Counter not reset (treated as no-op, not success), and not incremented.
        self.assertEqual(turn.metadata["compaction_consecutive_failures"], 1)
        self.assertNotIn("tier5_summaries_generated", turn.metadata)


class _IdentitySummarizer:
    """Returns the *same* list object so the gate can detect "did nothing"."""

    def summarise(self, messages, *, model, turn_context):  # type: ignore[no-untyped-def]
        return messages


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
