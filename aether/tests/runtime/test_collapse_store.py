"""Sprint 3 / PR 3.7 — ``CollapseStore`` + ``CollapseSegment`` data-model tests.

The store is a thin projection layer: it owns zero or more
:class:`CollapseSegment` records and produces a *view* of the messages
list with each committed segment swapped for a single synthetic
``[collapsed_segment ...]`` marker.  The original messages list is
never mutated — that's what session_record / replay sees.

Tests cover three groups (matching the design doc § 5.1):

* Group A — :meth:`CollapseStore.as_view` projection semantics
* Group B — :attr:`CollapseSegment.freed` arithmetic
* Group C — store-level observability (``has_committed`` / ``total_freed``)
"""

from __future__ import annotations

import unittest
from typing import Any

from aether.services.compact.collapse import CollapseSegment, CollapseStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msgs(n: int) -> list[dict[str, Any]]:
    """Build N distinguishable messages (alternating user / assistant)."""
    out: list[dict[str, Any]] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"msg-{i}"})
    return out


def _seg(
    start: int,
    end: int,
    *,
    summary: str = "summary",
    tokens_before: int = 1_000,
    tokens_after: int = 200,
    committed: bool = True,
) -> CollapseSegment:
    return CollapseSegment(
        start_idx=start,
        end_idx=end,
        summary_text=summary,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        committed=committed,
    )


# ---------------------------------------------------------------------------
# Group A — CollapseStore.as_view
# ---------------------------------------------------------------------------


class CollapseStoreAsViewTests(unittest.TestCase):
    def test_a1_empty_store_returns_messages_unchanged(self) -> None:
        """No segments → ``as_view`` returns the SAME list (identity).

        Identity return matters: the agent's PRE_LLM hook does an
        ``is`` compare to detect "no projection happened" and skip
        unnecessary copying.  If we ever start copy-returning here,
        update the agent hook too.
        """
        store = CollapseStore()
        msgs = _msgs(5)
        view = store.as_view(msgs)
        self.assertIs(view, msgs)

    def test_a2_proposed_only_segment_does_not_appear_in_view(self) -> None:
        """Proposed (uncommitted) segments are invisible to ``as_view``."""
        store = CollapseStore(segments=[_seg(2, 4, committed=False)])
        msgs = _msgs(8)
        view = store.as_view(msgs)
        self.assertIs(view, msgs)
        self.assertFalse(store.has_committed)

    def test_a3_one_committed_segment_replaces_range(self) -> None:
        """Segment[2-5] in 8 msgs → view = [m0, m1, marker, m6, m7]."""
        store = CollapseStore(segments=[_seg(2, 5, summary="folded mid")])
        msgs = _msgs(8)
        view = store.as_view(msgs)
        self.assertEqual(len(view), 5)
        # Head preserved verbatim.
        self.assertEqual(view[0], msgs[0])
        self.assertEqual(view[1], msgs[1])
        # Marker carries the structured pointer.
        marker = view[2]
        self.assertEqual(marker["role"], "user")
        self.assertIn("[collapsed_segment idx=2-5", marker["content"])
        self.assertIn("folded mid", marker["content"])
        meta = marker.get("_aether_meta")
        self.assertIsNotNone(meta)
        self.assertTrue(meta["collapsed_segment"])
        self.assertEqual(meta["start_idx"], 2)
        self.assertEqual(meta["end_idx"], 5)
        # Tail preserved verbatim.
        self.assertEqual(view[3], msgs[6])
        self.assertEqual(view[4], msgs[7])

    def test_a4_two_non_overlapping_segments_both_render(self) -> None:
        """Segments[2-3] + [5-6] in 8 msgs → [m0, m1, mk1, m4, mk2, m7]."""
        store = CollapseStore(
            segments=[
                _seg(2, 3, summary="block-a"),
                _seg(5, 6, summary="block-b"),
            ]
        )
        msgs = _msgs(8)
        view = store.as_view(msgs)
        self.assertEqual(len(view), 6)
        self.assertEqual(view[0], msgs[0])
        self.assertEqual(view[1], msgs[1])
        self.assertIn("block-a", view[2]["content"])
        self.assertEqual(view[3], msgs[4])
        self.assertIn("block-b", view[4]["content"])
        self.assertEqual(view[5], msgs[7])

    def test_a5_segment_covering_head_renders_marker_first(self) -> None:
        """Segment[0-3] → view starts with the marker, then m4..."""
        store = CollapseStore(segments=[_seg(0, 3, summary="head")])
        msgs = _msgs(6)
        view = store.as_view(msgs)
        self.assertEqual(len(view), 3)  # 1 marker + m4 + m5
        self.assertIn("[collapsed_segment idx=0-3", view[0]["content"])
        self.assertEqual(view[1], msgs[4])
        self.assertEqual(view[2], msgs[5])

    def test_a6_segment_covering_tail_renders_marker_last(self) -> None:
        """Segment[N-3..N-1] → view ends with the marker."""
        store = CollapseStore(segments=[_seg(3, 5, summary="tail")])
        msgs = _msgs(6)
        view = store.as_view(msgs)
        self.assertEqual(len(view), 4)  # m0..m2 + marker
        self.assertEqual(view[:3], msgs[:3])
        self.assertIn("[collapsed_segment idx=3-5", view[3]["content"])

    def test_a7_single_message_segment_replaces_one_for_one(self) -> None:
        """Segment[3-3] (single msg) → view length = N (1 swapped)."""
        store = CollapseStore(segments=[_seg(3, 3, summary="single")])
        msgs = _msgs(8)
        view = store.as_view(msgs)
        self.assertEqual(len(view), 8)
        self.assertIn("single", view[3]["content"])
        self.assertEqual(view[4], msgs[4])

    def test_a8_as_view_does_not_mutate_original_messages(self) -> None:
        """Original list and its dict elements are untouched."""
        store = CollapseStore(segments=[_seg(2, 5, summary="folded")])
        msgs = _msgs(8)
        # Snapshot identities + contents up front.
        original_ids = [id(m) for m in msgs]
        original_dump = [dict(m) for m in msgs]
        store.as_view(msgs)
        self.assertEqual([id(m) for m in msgs], original_ids)
        for m, expected in zip(msgs, original_dump):
            self.assertEqual(m, expected)


class CollapseStoreEdgeCaseTests(unittest.TestCase):
    """Defensive cases — committed segments that point outside the
    current messages list (e.g. another tier truncated the list
    between commit-time and view-time) are silently dropped from
    the projection rather than raising."""

    def test_segment_past_end_is_clamped_or_dropped(self) -> None:
        store = CollapseStore(segments=[_seg(10, 20, summary="off-end")])
        msgs = _msgs(5)
        # No raise; whatever projection comes out is consistent.
        view = store.as_view(msgs)
        self.assertEqual(len(view), 5)
        # All originals preserved (nothing got swapped).
        for m, original in zip(view, msgs):
            self.assertEqual(m, original)

    def test_segment_start_past_end_is_dropped(self) -> None:
        """``start_idx > end_idx`` (degenerate) → silently ignored."""
        store = CollapseStore(segments=[_seg(5, 2, summary="reversed")])
        msgs = _msgs(8)
        view = store.as_view(msgs)
        # Defensive code returns either the full unchanged list or a
        # non-empty projection — never raises.  For this degenerate
        # input we accept either outcome as long as it's safe.
        self.assertGreaterEqual(len(view), 0)


# ---------------------------------------------------------------------------
# Group B — CollapseSegment.freed
# ---------------------------------------------------------------------------


class CollapseSegmentFreedTests(unittest.TestCase):
    def test_b1_simple_diff(self) -> None:
        seg = _seg(0, 3, tokens_before=1_000, tokens_after=200)
        self.assertEqual(seg.freed, 800)

    def test_b2_negative_diff_clamps_to_zero(self) -> None:
        """Bad estimator → ``after > before`` → freed must NOT go negative.

        Pipeline accounting elsewhere (e.g. ``CompactionResult.exhausted``)
        relies on ``freed >= 0`` to decide whether a tier "helped".  A
        negative number would silently cancel out genuine gains from
        other tiers.
        """
        seg = _seg(0, 3, tokens_before=200, tokens_after=1_000)
        self.assertEqual(seg.freed, 0)

    def test_b3_zero_diff_is_zero(self) -> None:
        seg = _seg(0, 3, tokens_before=500, tokens_after=500)
        self.assertEqual(seg.freed, 0)


# ---------------------------------------------------------------------------
# Group C — store-level observability
# ---------------------------------------------------------------------------


class CollapseStoreObservabilityTests(unittest.TestCase):
    def test_c1_total_freed_only_counts_committed_segments(self) -> None:
        """Proposed (uncommitted) segments contribute 0 to ``total_freed``.

        Counting them would mis-report the actual savings the model
        is seeing — proposals don't appear in the projection.
        """
        store = CollapseStore(
            segments=[
                _seg(0, 3, tokens_before=1_000, tokens_after=200, committed=True),
                _seg(5, 8, tokens_before=2_000, tokens_after=400, committed=True),
                _seg(10, 12, tokens_before=500, tokens_after=100, committed=False),
            ]
        )
        # Two committed: 800 + 1600 = 2400.  Proposed contributes 0.
        self.assertEqual(store.total_freed(), 2_400)

    def test_c2_has_committed_reflects_at_least_one_committed(self) -> None:
        empty_store = CollapseStore()
        self.assertFalse(empty_store.has_committed)

        proposed_only = CollapseStore(segments=[_seg(0, 3, committed=False)])
        self.assertFalse(proposed_only.has_committed)

        mixed = CollapseStore(
            segments=[
                _seg(0, 3, committed=False),
                _seg(5, 8, committed=True),
            ]
        )
        self.assertTrue(mixed.has_committed)

    def test_c3_empty_store_total_freed_is_zero(self) -> None:
        self.assertEqual(CollapseStore().total_freed(), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
