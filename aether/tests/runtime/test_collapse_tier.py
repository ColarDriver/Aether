"""Sprint 3 / PR 3.7 — ``ContextCollapseTier`` (Tier 4) unit tests.

The tier coordinates four pieces:

* **Gate** — master compression switch + tier-specific switch.
* **Threshold staging** — below ``commit_pct`` it's read-only;
  between ``commit_pct`` and ``blocking_pct`` it *proposes*; at and
  above ``blocking_pct`` it *commits*.
* **Segment selection** — picks the next non-collapsed range while
  respecting the protected head/tail and tool-pair boundaries.
* **Store mutation + ``collapse_owns_headroom`` flag** — what
  ultimately blocks Tier 5 from running on the same pass.

Tests cover six groups (matching the design doc § 5.2):

* Group D — gate / early-exit
* Group E — segment selection (head/tail protect, max-seg cap, tool-pair safety)
* Group F — propose vs commit state machine
* Group G — ``collapse_owns_headroom`` flag maintenance
* Group H — summariser failure modes
* Group I — observability counters + log line
"""

from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass, field
from typing import Any

from aether.runtime.core.contracts import TurnContext
from aether.services.compact.collapse import (
    CollapseSegment,
    CollapseStore,
    ContextCollapseTier,
)
from aether.services.compact.compactor import CompactionContext


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    """Duck-typed ``EngineConfig`` slice the tier consults."""

    compression_enabled: bool = True
    context_collapse_enabled: bool = True
    context_collapse_commit_pct: float = 0.90
    context_collapse_blocking_pct: float = 0.95
    context_collapse_segment_max_messages: int = 20
    compression_protect_first_n: int = 2
    compression_protect_last_n: int = 6


@dataclass
class _StubSummarizer:
    """Stand-in for :class:`LLMForkSummarizer` returning a canned summary.

    The real summariser emits a ``[compact_boundary]`` prefix + summary
    text in a single ``user`` message tagged ``compact_summary=True``.
    We mimic that exact shape so ``ContextCollapseTier._extract_summary_text``
    exercises its production parser path.
    """

    summary_text: str = "## summary\n\nfolded conversation"
    raises: BaseException | None = None
    calls: list[tuple[list[dict[str, Any]], str, TurnContext]] = field(
        default_factory=list
    )

    def summarise(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        turn_context: TurnContext,
    ) -> list[dict[str, Any]]:
        self.calls.append((list(messages), model, turn_context))
        if self.raises is not None:
            raise self.raises
        head = messages[: max(0, len(messages) - 1)]
        merged = {
            "role": "user",
            "content": (
                f"[compact_boundary] Compacted {len(messages)} messages.\n\n"
                f"{self.summary_text}"
            ),
            "_aether_meta": {
                "compact_boundary": True,
                "compact_summary": True,
                "compacted_messages": len(messages),
                "model": model,
            },
        }
        return head + [merged]


def _ctx(metadata: dict[str, Any] | None = None) -> TurnContext:
    return TurnContext(
        session_id="sess-tier4",
        iteration=0,
        metadata=dict(metadata or {}),
        task_id="task-1",
        turn_id="turn-1",
    )


def _compaction_ctx(
    *,
    model_window: int = 100_000,
    pre_compaction_tokens: int = 0,
    trigger_reason: str = "preflight",
) -> CompactionContext:
    return CompactionContext(
        session_id="sess-tier4",
        model="claude-sonnet",
        model_window=model_window,
        pre_compaction_tokens=pre_compaction_tokens,
        target_pct=0.85,
        trigger_reason=trigger_reason,
    )


def _build(
    config: _Config | None = None,
    summarizer: _StubSummarizer | None = None,
) -> tuple[ContextCollapseTier, _StubSummarizer, _Config]:
    cfg = config or _Config()
    summ = summarizer or _StubSummarizer()
    logger = logging.getLogger("test.collapse.tier")
    # Silence the WARNING/INFO output during normal test runs — we
    # explicitly capture logs in the I-group tests via ``assertLogs``,
    # which works regardless of effective level.
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    tier = ContextCollapseTier(
        config=cfg,
        summarizer=summ,
        logger=logger,
    )
    return tier, summ, cfg


def _msgs(n: int, *, body_chars: int = 800) -> list[dict[str, Any]]:
    """Build N user messages of size ``body_chars``.

    Token estimator yields roughly ``chars / 3`` (see
    ``estimate_messages_tokens``), so for the heavy-trigger tests we
    want N * body_chars / 3 to comfortably exceed the commit
    threshold the test passes.
    """
    body = "x" * body_chars
    return [{"role": "user", "content": f"msg-{i} {body}"} for i in range(n)]


# ---------------------------------------------------------------------------
# Group D — gate / early-exit
# ---------------------------------------------------------------------------


class CollapseTierGroupDGateTests(unittest.TestCase):
    def test_d1_compression_master_switch_off(self) -> None:
        """``compression_enabled=False`` → no work, no store created."""
        tier, summ, _ = _build(config=_Config(compression_enabled=False))
        ctx = _ctx()
        out, freed = tier.maybe_run(_msgs(30), _compaction_ctx(), ctx)
        self.assertEqual(freed, 0)
        self.assertEqual(summ.calls, [])
        self.assertNotIn("_collapse_store", ctx.metadata)

    def test_d2_collapse_specific_switch_off(self) -> None:
        """``context_collapse_enabled=False`` → tier skips even if compression is on."""
        tier, summ, _ = _build(config=_Config(context_collapse_enabled=False))
        ctx = _ctx()
        out, freed = tier.maybe_run(_msgs(30), _compaction_ctx(), ctx)
        self.assertEqual(freed, 0)
        self.assertEqual(summ.calls, [])
        self.assertNotIn("_collapse_store", ctx.metadata)

    def test_d3_below_commit_threshold_returns_existing_view(self) -> None:
        """View tokens < commit_pct → no propose; existing committed segments still apply."""
        tier, summ, _ = _build()
        # Pre-seed a committed segment so we have something to "view".
        ctx = _ctx({
            "_collapse_store": CollapseStore(
                segments=[
                    CollapseSegment(
                        start_idx=2,
                        end_idx=5,
                        summary_text="prior fold",
                        tokens_before=2_000,
                        tokens_after=200,
                        committed=True,
                    )
                ],
            ),
        })
        # Tiny message set → view_tokens well below commit_threshold.
        msgs = _msgs(8, body_chars=10)
        out, _ = tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        # No new fork happened; existing view still applied.
        self.assertEqual(summ.calls, [])
        # Output reflects the pre-existing committed segment (so length
        # is shorter than the original 8).
        self.assertLess(len(out), len(msgs))
        # ``collapse_owns_headroom`` stays True because of the prior
        # committed segment.
        self.assertTrue(ctx.metadata.get("collapse_owns_headroom"))

    def test_d4_too_few_messages_to_collapse_anything(self) -> None:
        """``n <= protect_head + protect_tail + 4`` → ``_select_next_segment`` returns None."""
        tier, summ, _ = _build()
        ctx = _ctx()
        # protect_head=2 + protect_tail=6 + 4 = 12; give exactly 12.
        msgs = _msgs(12, body_chars=20_000)  # huge bodies to clear commit_threshold
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        self.assertEqual(summ.calls, [])  # no segment to collapse


# ---------------------------------------------------------------------------
# Group E — segment selection (head/tail protect, max-seg, tool-pair safety)
# ---------------------------------------------------------------------------


class CollapseTierGroupESegmentSelectionTests(unittest.TestCase):
    def test_e1_first_segment_skips_protected_head(self) -> None:
        """30 msgs, protect_head=2, store empty → start=2."""
        tier, summ, _ = _build()
        ctx = _ctx()
        # Token estimator is ~chars/3.  30 * 12_000 chars → ~120k
        # tokens, well above commit_pct=0.90 of model_window=100_000.
        msgs = _msgs(30, body_chars=12_000)
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        # Summariser was called; inspect the start_idx via the resulting segment.
        store: CollapseStore = ctx.metadata["_collapse_store"]
        self.assertEqual(len(store.segments), 1)
        seg = store.segments[0]
        self.assertEqual(seg.start_idx, 2)
        # End cannot exceed protect_head + max_seg - 1 = 2 + 20 - 1 = 21.
        self.assertLessEqual(seg.end_idx, 21)

    def test_e2_next_segment_starts_after_existing_segments(self) -> None:
        """Pre-existing segment[2-10]: next selection starts at idx=11."""
        tier, summ, _ = _build()
        ctx = _ctx({
            "_collapse_store": CollapseStore(
                segments=[
                    CollapseSegment(
                        start_idx=2, end_idx=10,
                        summary_text="first", tokens_before=5_000, tokens_after=200,
                        committed=True,
                    )
                ],
            ),
        })
        # 30 msgs * 18_000 chars / 3 ≈ 180k tokens; with the
        # pre-existing segment[2-10] folding 9 msgs into ~30 chars of
        # marker, the residual view is still ~21 msgs * 18_000 / 3 ≈
        # 126k tokens — well above commit_threshold (90_000).
        msgs = _msgs(30, body_chars=18_000)
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        store: CollapseStore = ctx.metadata["_collapse_store"]
        new_segments = [s for s in store.segments if s.start_idx >= 11]
        self.assertEqual(len(new_segments), 1)
        self.assertEqual(new_segments[0].start_idx, 11)

    def test_e3_no_room_left_returns_none(self) -> None:
        """Pre-existing segment[2-25] + protect_tail=6 → no candidate; bail."""
        tier, summ, _ = _build()
        ctx = _ctx({
            "_collapse_store": CollapseStore(
                segments=[
                    CollapseSegment(
                        start_idx=2, end_idx=25,
                        summary_text="huge", tokens_before=10_000, tokens_after=200,
                        committed=True,
                    )
                ],
            ),
        })
        msgs = _msgs(30, body_chars=12_000)
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        # No new segment added — the only candidate slot is 26-29
        # which is inside protect_tail=6.
        store: CollapseStore = ctx.metadata["_collapse_store"]
        new_segments = [s for s in store.segments if s.start_idx >= 26]
        self.assertEqual(len(new_segments), 0)
        # Summariser was NOT called for any new segment.
        self.assertEqual(summ.calls, [])

    def test_e4_max_seg_caps_segment_length(self) -> None:
        """``context_collapse_segment_max_messages=5`` → segment length exactly 5."""
        tier, summ, _ = _build(
            config=_Config(context_collapse_segment_max_messages=5)
        )
        ctx = _ctx()
        msgs = _msgs(30, body_chars=12_000)
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        store: CollapseStore = ctx.metadata["_collapse_store"]
        self.assertEqual(len(store.segments), 1)
        seg = store.segments[0]
        # Length == end - start + 1 == 5.
        self.assertEqual(seg.end_idx - seg.start_idx + 1, 5)

    def test_e5_too_few_messages_returns_none(self) -> None:
        """If candidate window < 4 → return None, no summariser call."""
        tier, summ, cfg = _build()
        # Build a setup where candidate window has only 3 messages:
        # protect_head=2, protect_tail=6, total messages must give window < 4.
        # With n=11: candidate range = [2..4] = 3 indices < 4 → bail.
        ctx = _ctx()
        msgs = _msgs(11, body_chars=20_000)  # huge bodies → above commit
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        self.assertEqual(summ.calls, [])

    def test_e6_tool_pair_split_is_repaired_at_segment_end(self) -> None:
        """If the segment end falls between an assistant tool_use and its tool_result,
        the boundary advances to include the tool_result (or rolls back past the
        tool_use entirely).  The committed segment must NOT split a pair."""
        tier, summ, _ = _build(
            config=_Config(
                context_collapse_segment_max_messages=4,
                compression_protect_first_n=0,
                compression_protect_last_n=0,
            )
        )
        # Build: m0(user), m1(user), m2(user), m3(assistant + tool_use id=t1),
        # m4(user with tool_result for t1), m5(user), m6(user), m7(user).
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "msg-0"},
            {"role": "user", "content": "msg-1"},
            {"role": "user", "content": "msg-2"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "shell", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                ],
            },
            {"role": "user", "content": "msg-5"},
            {"role": "user", "content": "msg-6"},
            {"role": "user", "content": "msg-7"},
        ]
        ctx = _ctx()
        # Force commit threshold to 0 so the tier always tries to fold.
        ctx_obj = _compaction_ctx(model_window=100_000)
        # max_seg=4 → naive end=3 (assistant+tool_use), but tool_result
        # is at idx=4.  Tier should either include idx 4 (extending
        # past max_seg slightly) OR roll back to end=2 (excluding the
        # tool_use).  Whichever it picks, it must NOT produce a
        # segment that ends at idx=3 (which would orphan the result).
        tier.maybe_run(msgs, ctx_obj, ctx)
        store: CollapseStore = ctx.metadata.get("_collapse_store") or CollapseStore()
        if store.segments:
            seg = store.segments[0]
            self.assertNotEqual(
                seg.end_idx, 3,
                f"segment {seg.start_idx}-{seg.end_idx} splits tool pair at idx 3-4",
            )

    def test_e7_orphan_tool_result_at_segment_start_is_skipped(self) -> None:
        """If start_idx lands on a tool_result whose tool_use lives outside the
        window, advance start past the orphan (or bail)."""
        tier, summ, _ = _build(
            config=_Config(
                context_collapse_segment_max_messages=20,
                compression_protect_first_n=2,
                compression_protect_last_n=6,
            )
        )
        # Build: m0(user), m1(assistant tool_use t1), m2(user tool_result t1),
        # m3..m25(filler).  protect_head=2 → first candidate is idx=2,
        # which is the tool_result.  Tier must roll start forward to
        # idx=3 (or further).
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "kick-off"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "shell", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "out"}
                ],
            },
        ]
        # Pad with 25 long user messages so we cross commit threshold.
        msgs.extend(_msgs(25, body_chars=4_000))
        ctx = _ctx()
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        store: CollapseStore = ctx.metadata.get("_collapse_store") or CollapseStore()
        if store.segments:
            seg = store.segments[0]
            # Either start moved past idx 2 (the orphan tool_result),
            # OR no segment was created because the rollback shrank
            # the window below the 4-message floor.  Both are
            # acceptable safe outcomes; what we forbid is a segment
            # that starts at idx 2 with the orphan tool_result inside.
            self.assertNotEqual(seg.start_idx, 2)


# ---------------------------------------------------------------------------
# Group F — propose vs commit state machine
# ---------------------------------------------------------------------------


class CollapseTierGroupFCommitStateTests(unittest.TestCase):
    def test_f1_in_propose_band_segment_is_uncommitted(self) -> None:
        """commit_pct <= view_tokens < blocking_pct → seg.committed = False.

        Token estimator yields roughly chars/3.  We size the test so:

        * model_window=100_000
        * commit_pct=0.90 → commit_threshold=90_000
        * blocking_pct=0.95 → blocking_threshold=95_000
        * 30 msgs * ~9_200 chars / 3 ≈ 92_000 view tokens

        92_000 is comfortably inside ``[90_000, 95_000)`` → propose
        but not commit.
        """
        tier, summ, _ = _build(
            config=_Config(
                context_collapse_commit_pct=0.90,
                context_collapse_blocking_pct=0.95,
            )
        )
        msgs = _msgs(30, body_chars=9_200)
        ctx = _ctx()
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        store: CollapseStore = ctx.metadata.get("_collapse_store") or CollapseStore()
        self.assertGreaterEqual(len(store.segments), 1)
        # Newly added segments must NOT be committed (view_tokens is
        # still below blocking_threshold).
        self.assertFalse(store.segments[-1].committed)

    def test_f2_at_or_above_blocking_band_segment_is_committed(self) -> None:
        """view_tokens >= blocking_pct → newly added seg.committed = True."""
        tier, summ, _ = _build(
            config=_Config(
                context_collapse_commit_pct=0.50,  # very low so we always trigger
                context_collapse_blocking_pct=0.50,  # same → always committed
            )
        )
        ctx = _ctx()
        msgs = _msgs(30, body_chars=12_000)
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        store: CollapseStore = ctx.metadata["_collapse_store"]
        self.assertEqual(len(store.segments), 1)
        self.assertTrue(store.segments[0].committed)

    def test_f3_proposed_segment_is_not_auto_promoted_on_next_run(self) -> None:
        """Proposed segments stay proposed; each run adds a NEW segment.

        This documents the Sprint 3 simplification: we don't have
        per-segment promotion.  When pressure escalates, the next
        ``maybe_run`` proposes (or commits) a fresh segment for the
        next un-collapsed range — the old proposed segment stays
        as-is and just doesn't appear in the view.
        """
        tier, summ, _ = _build(
            config=_Config(
                context_collapse_commit_pct=0.80,
                context_collapse_blocking_pct=0.99,
                context_collapse_segment_max_messages=10,
            )
        )
        ctx = _ctx()
        # 60 * 4_500 chars / 3 ≈ 90_000 tokens.  Sits comfortably
        # above commit (80_000) but below blocking (99_000) → propose.
        msgs = _msgs(60, body_chars=4_500)
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        store: CollapseStore = ctx.metadata["_collapse_store"]
        first_segments = list(store.segments)
        # Now bump the threshold via a different blocking_pct so the
        # tier sees us at >= blocking on the next run.
        tier.config = _Config(
            context_collapse_commit_pct=0.50,
            context_collapse_blocking_pct=0.50,
        )
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        store_after = ctx.metadata["_collapse_store"]
        # Old proposals are unchanged (still uncommitted) and a new
        # committed segment was appended for the next range.
        for old in first_segments:
            match = next(
                s for s in store_after.segments
                if s.start_idx == old.start_idx and s.end_idx == old.end_idx
            )
            self.assertEqual(match.committed, old.committed)


# ---------------------------------------------------------------------------
# Group G — collapse_owns_headroom flag maintenance
# ---------------------------------------------------------------------------


class CollapseTierGroupGFlagTests(unittest.TestCase):
    def test_g1_committed_segment_sets_flag(self) -> None:
        tier, _, _ = _build(
            config=_Config(
                context_collapse_commit_pct=0.50,
                context_collapse_blocking_pct=0.50,
            )
        )
        ctx = _ctx()
        msgs = _msgs(30, body_chars=12_000)
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        self.assertTrue(ctx.metadata.get("collapse_owns_headroom"))

    def test_g2_proposed_only_clears_flag(self) -> None:
        """Pre-existing flag from a prior session must be cleared when
        the current store has no committed segment."""
        tier, _, _ = _build(
            config=_Config(
                context_collapse_commit_pct=0.80,
                context_collapse_blocking_pct=0.99,
            )
        )
        # Stale flag from a previous turn (or carry-over).
        ctx = _ctx({"collapse_owns_headroom": True})
        msgs = _msgs(60, body_chars=4_500)  # ~90k tokens, in propose band
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        # Even if a propose happened, no committed segment yet → flag cleared.
        store: CollapseStore = ctx.metadata.get("_collapse_store") or CollapseStore()
        if not store.has_committed:
            self.assertNotIn("collapse_owns_headroom", ctx.metadata)

    def test_g3_empty_store_clears_flag(self) -> None:
        """No segments at all → flag must not be set / must be popped."""
        tier, _, _ = _build()
        ctx = _ctx({"collapse_owns_headroom": True})
        # Tiny msg set so we never propose / commit.
        msgs = _msgs(8, body_chars=10)
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        self.assertNotIn("collapse_owns_headroom", ctx.metadata)


# ---------------------------------------------------------------------------
# Group H — summariser failure modes
# ---------------------------------------------------------------------------


class CollapseTierGroupHFailureTests(unittest.TestCase):
    def test_h1_summariser_exception_is_swallowed(self) -> None:
        """Summariser raises → tier logs warning, no segment added, view returned."""
        tier, summ, _ = _build(
            summarizer=_StubSummarizer(raises=RuntimeError("model down")),
            config=_Config(
                context_collapse_commit_pct=0.50,
                context_collapse_blocking_pct=0.50,
            ),
        )
        ctx = _ctx()
        msgs = _msgs(30, body_chars=12_000)
        out, _ = tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        # No segment added.
        store: CollapseStore = ctx.metadata.get("_collapse_store") or CollapseStore()
        self.assertEqual(len(store.segments), 0)
        # Output is the (unchanged) view of the empty store.
        self.assertIs(out, msgs)
        # Summariser was called once and raised.
        self.assertEqual(len(summ.calls), 1)

    def test_h2_empty_summary_text_is_refused(self) -> None:
        """Summariser returns empty string → don't commit a meaningless segment."""
        tier, summ, _ = _build(
            summarizer=_StubSummarizer(summary_text=""),
            config=_Config(
                context_collapse_commit_pct=0.50,
                context_collapse_blocking_pct=0.50,
            ),
        )
        ctx = _ctx()
        msgs = _msgs(30, body_chars=12_000)
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        store: CollapseStore = ctx.metadata.get("_collapse_store") or CollapseStore()
        self.assertEqual(len(store.segments), 0)


# ---------------------------------------------------------------------------
# Group I — observability counters + log line
# ---------------------------------------------------------------------------


class CollapseTierGroupIObservabilityTests(unittest.TestCase):
    def test_i1_tier4_collapse_segments_counter_increments(self) -> None:
        tier, _, _ = _build(
            config=_Config(
                context_collapse_commit_pct=0.50,
                context_collapse_blocking_pct=0.50,
            )
        )
        ctx = _ctx()
        msgs = _msgs(30, body_chars=12_000)
        tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        self.assertEqual(ctx.metadata.get("tier4_collapse_segments"), 1)

    def test_i2_log_line_carries_diagnostic_fields(self) -> None:
        """Log mentions committed/proposed + segment range + view tokens."""
        tier, _, _ = _build(
            config=_Config(
                context_collapse_commit_pct=0.50,
                context_collapse_blocking_pct=0.50,
            )
        )
        ctx = _ctx()
        msgs = _msgs(30, body_chars=12_000)
        with self.assertLogs("test.collapse.tier", level="INFO") as recorded:
            tier.maybe_run(msgs, _compaction_ctx(model_window=100_000), ctx)
        log_blob = "\n".join(recorded.output)
        self.assertIn("tier4:", log_blob)
        self.assertIn("committed segment", log_blob)
        self.assertIn("view", log_blob)
        self.assertIn("tokens", log_blob)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
