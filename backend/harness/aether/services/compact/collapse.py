"""Tier 4 context collapse — projection-based folding.

Self-developed: claude-code's ``contextCollapse/*`` is shipped as a
``@generated stub``, so we own the design end-to-end.  Key invariants:

* The original messages list is **never mutated**.  That's what
  ``session_record`` / replay / steer see.  Mutating in place would
  break the "load yesterday's session and replay" workflow.
* A :class:`CollapseStore` (per-turn, lives on
  ``TurnContext.metadata['_collapse_store']``) holds zero or more
  :class:`CollapseSegment` records — each a ``(start_idx, end_idx)``
  range plus the LLM-generated summary text that replaces it in the
  projection.
* A "view" function (:meth:`CollapseStore.as_view`) is applied at
  PRE_LLM time to swap the segment ranges for their summary — only
  the *payload sent to the provider* sees the projection.

Two-stage trigger:

* ``commit_pct`` (default 0.90) — start *proposing* segments.  A
  proposed segment is summarised but not yet committed; if pressure
  drops back down before ``blocking_pct``, no harm done.
* ``blocking_pct`` (default 0.95) — newly-proposed segments are
  committed (i.e. take effect on subsequent ``as_view`` calls).
  This is the "we can't afford to wait any longer" line.

Mutual exclusion with Tier 5:

* When this tier has at least one *committed* segment in the current
  turn, we set ``turn_context.metadata['collapse_owns_headroom'] = True``.
* :class:`AutoCompactor` checks that flag (condition 4) and bails — so
  Tier 4 wins over Tier 5 once it has visibly helped.  Sprint 3
  trade-off: we'd rather keep the model's recent fine-grained context
  than re-summarise it through Tier 5's fork.

Sprint 3 store-lifetime trade-off:

The store currently lives on ``turn_context.metadata`` which resets
per turn (``TurnContext`` is constructed fresh per ``run_turn``).
This is the *minimum-viable* version called out in the PR doc § 5.3
T-M2 — cross-turn persistence (serialise the store to ``session_record``)
is a planned follow-up.  Per-turn rebuild still works in practice
because:

* Long sessions trigger Tier 4 inside a single turn (preflight or
  recovery), and the view is applied for *every* provider call within
  that same turn.
* On the next turn the local messages list still carries any
  ``[collapsed_segment]`` annotations the previous turn's view emitted
  — so even if the store resets, the model's view of the past is not
  silently expanded back to the original messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext
from aether.services.compact.llm_fork import LLMForkSummarizer
from aether.services.compact.token_estimation import estimate_messages_tokens


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CollapseSegment:
    """One folded slice of the original messages.

    ``start_idx`` / ``end_idx`` are **inclusive** indices into the
    original (un-projected) messages list.  ``committed=False`` means
    "proposed but not yet binding"; only committed segments are honoured
    by :meth:`CollapseStore.as_view`.
    """

    start_idx: int
    end_idx: int
    summary_text: str
    tokens_before: int
    tokens_after: int
    committed: bool = False

    @property
    def freed(self) -> int:
        """Token delta this segment claims it freed.

        ``max(0, ...)`` guards against the heuristic estimator
        producing a *larger* "after" count for a degenerate summary —
        we never want freed accounting to go negative because that
        breaks the pipeline's "did this tier help?" logic.
        """
        return max(0, self.tokens_before - self.tokens_after)


@dataclass
class CollapseStore:
    """All collapse state for one turn.

    Lives on ``turn_context.metadata['_collapse_store']``.  Per-turn
    lifetime is the Sprint 3 minimum (see module docstring).
    """

    segments: list[CollapseSegment] = field(default_factory=list)

    @property
    def has_committed(self) -> bool:
        return any(s.committed for s in self.segments)

    def total_freed(self) -> int:
        """Sum of ``freed`` across committed segments only.

        Proposed-but-not-committed segments contribute 0 because they
        don't actually appear in the projection yet — counting them
        would mis-report the savings to anyone reading the metadata.
        """
        return sum(s.freed for s in self.segments if s.committed)

    def as_view(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a NEW list with each committed segment replaced by a marker.

        Original ``messages`` is not mutated.  Adjacent segments are
        emitted as separate marker messages (no merging) so per-segment
        observability stays clean.

        The synthetic marker uses ``role='user'`` (not ``system``)
        because mid-conversation ``system`` messages are rejected /
        silently folded by several providers (Anthropic Messages,
        Bedrock, vLLM with strict alternation) — the same lesson PR
        3.4 learned for the LLM-fork boundary message.  The
        ``[collapsed_segment ...]`` text prefix lets text-only consumers
        (transcript dumps, ``/context`` view) still locate the
        collapse boundary; ``_aether_meta.collapsed_segment=True`` is
        the structured pointer downstream code can pivot on.
        """
        if not self.has_committed:
            # Identity short-circuit — caller can ``is`` compare to
            # detect "no projection happened".  Saves an allocation
            # on the cold path (every PRE_LLM call when collapse is
            # enabled but nothing committed yet).
            return messages

        committed = sorted(
            (s for s in self.segments if s.committed),
            key=lambda s: s.start_idx,
        )

        out: list[dict[str, Any]] = []
        cursor = 0
        n = len(messages)
        for seg in committed:
            # Defensive clamp — a segment whose indices fell outside
            # the current message list (e.g. the messages list was
            # truncated by another tier between commit and view) is
            # silently dropped from the projection.  Better than
            # raising — the worst case is the model sees the original
            # messages it would have seen without the tier.
            start = max(cursor, min(seg.start_idx, n))
            end = max(start - 1, min(seg.end_idx, n - 1))
            if start > cursor:
                out.extend(messages[cursor:start])
            if start <= end:
                out.append(self._marker_message(seg))
                cursor = end + 1
            else:
                cursor = max(cursor, start)
        if cursor < n:
            out.extend(messages[cursor:])
        return out

    @staticmethod
    def _marker_message(seg: CollapseSegment) -> dict[str, Any]:
        return {
            "role": "user",
            "content": (
                f"[collapsed_segment idx={seg.start_idx}-{seg.end_idx} "
                f"freed~{seg.freed} tokens]\n{seg.summary_text}"
            ),
            "_aether_meta": {
                "collapsed_segment": True,
                "start_idx": seg.start_idx,
                "end_idx": seg.end_idx,
                "freed": seg.freed,
            },
        }


# ---------------------------------------------------------------------------
# Tier
# ---------------------------------------------------------------------------


# Synthesised assistant marker name for the second tool-pair guard
# (see ``_select_next_segment``) — kept as a module constant so tests
# can match on it without depending on the exact string layout.
_TOOL_USE_BLOCK_TYPE = "tool_use"
_TOOL_RESULT_BLOCK_TYPE = "tool_result"


class ContextCollapseTier:
    """Tier 4 — projection-based collapse.

    Conforms to the ``CompactorTier`` protocol.  Returns the *view*
    (with any committed segments applied) — so even on read-only
    invocations the pipeline's per-tier accounting reflects what the
    next provider call would actually see.
    """

    name: str = "tier4_collapse"

    def __init__(
        self,
        *,
        config: Any,
        summarizer: LLMForkSummarizer,
        logger: Any,
    ) -> None:
        self.config = config
        self.summarizer = summarizer
        self.logger = logger

    def maybe_run(
        self,
        messages: list[dict[str, Any]],
        ctx: CompactionContext,
        turn_context: TurnContext,
    ) -> tuple[list[dict[str, Any]], int]:
        if not getattr(self.config, "compression_enabled", False):
            return messages, 0
        if not getattr(self.config, "context_collapse_enabled", False):
            return messages, 0

        store = self._get_or_create_store(turn_context)

        # Apply existing committed segments first so the rest of the
        # analysis (token accounting, propose/commit decision) sees
        # the projection — otherwise we'd double-count un-collapsed
        # ranges and over-react.
        view = store.as_view(messages)
        view_tokens = estimate_messages_tokens(view)

        commit_threshold = int(
            ctx.model_window
            * float(getattr(self.config, "context_collapse_commit_pct", 0.90))
        )
        blocking_threshold = int(
            ctx.model_window
            * float(getattr(self.config, "context_collapse_blocking_pct", 0.95))
        )

        if view_tokens < commit_threshold:
            # Below pressure — read-only path.  Still update the
            # ``collapse_owns_headroom`` flag so previously-committed
            # segments keep blocking Tier 5 even on cool-down passes.
            self._update_collapse_owns_flag(turn_context, store)
            return view, max(0, ctx.pre_compaction_tokens - view_tokens)

        # Pressure: try to commit / propose one more segment.
        segment_range = self._select_next_segment(messages, store)
        if segment_range is None:
            # Nothing more to collapse — Tier 5 will pick up the slack.
            self._update_collapse_owns_flag(turn_context, store)
            return view, max(0, ctx.pre_compaction_tokens - view_tokens)

        start_idx, end_idx = segment_range
        segment_msgs = messages[start_idx : end_idx + 1]

        try:
            summarised = self.summarizer.summarise(
                segment_msgs,
                model=ctx.model,
                turn_context=turn_context,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("tier4: summarise failed: %s", exc)
            self._update_collapse_owns_flag(turn_context, store)
            return view, max(0, ctx.pre_compaction_tokens - view_tokens)

        summary_text = self._extract_summary_text(summarised)
        if not summary_text:
            # Empty summary is worse than no summary — refuse to add
            # an empty segment and fall back to the existing view.
            self.logger.warning(
                "tier4: summariser returned empty text for segment[%d-%d]; "
                "skipping commit",
                start_idx,
                end_idx,
            )
            self._update_collapse_owns_flag(turn_context, store)
            return view, max(0, ctx.pre_compaction_tokens - view_tokens)

        tokens_before = estimate_messages_tokens(segment_msgs)
        tokens_after = estimate_messages_tokens(
            [{"role": "user", "content": summary_text}]
        )
        seg = CollapseSegment(
            start_idx=start_idx,
            end_idx=end_idx,
            summary_text=summary_text,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            committed=(view_tokens >= blocking_threshold),
        )
        store.segments.append(seg)

        turn_context.metadata["tier4_collapse_segments"] = (
            int(turn_context.metadata.get("tier4_collapse_segments", 0)) + 1
        )

        new_view = store.as_view(messages)
        new_view_tokens = estimate_messages_tokens(new_view)
        self._update_collapse_owns_flag(turn_context, store)

        self.logger.info(
            "tier4: %s segment[%d-%d], view %d->%d tokens (commit_pct=%.2f, "
            "blocking_pct=%.2f)",
            "committed" if seg.committed else "proposed",
            start_idx,
            end_idx,
            view_tokens,
            new_view_tokens,
            float(getattr(self.config, "context_collapse_commit_pct", 0.90)),
            float(getattr(self.config, "context_collapse_blocking_pct", 0.95)),
        )
        return new_view, max(0, view_tokens - new_view_tokens)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_or_create_store(turn_context: TurnContext) -> CollapseStore:
        store = turn_context.metadata.get("_collapse_store")
        if not isinstance(store, CollapseStore):
            store = CollapseStore()
            turn_context.metadata["_collapse_store"] = store
        return store

    def _select_next_segment(
        self,
        messages: list[dict[str, Any]],
        store: CollapseStore,
    ) -> Optional[tuple[int, int]]:
        """Pick the next contiguous range to collapse.

        Strategy (Sprint 3 simplest version):

        * Skip the protected head (``compression_protect_first_n``).
        * Skip already-collapsed indices (any prior segment in store).
        * Skip the protected tail (``compression_protect_last_n``).
        * From what remains, take a contiguous range of up to
          ``context_collapse_segment_max_messages``.
        * Bail with ``None`` if fewer than 4 messages are available
          in the candidate window — too small to bother summarising,
          let Tier 5 handle it.

        **Tool-pair safety** (high-risk item from PR doc § 9): the
        end of the segment is rolled back if it would split an
        ``assistant + tool_use`` from its matching ``tool_result``,
        and the start is rolled forward if it would orphan a
        ``tool_result`` whose ``tool_use`` lives outside the segment.
        Otherwise the projection produces a tool-result without a
        matching tool-call, which providers like Anthropic reject
        outright.
        """
        protect_head = max(0, int(getattr(self.config, "compression_protect_first_n", 2)))
        protect_tail = max(0, int(getattr(self.config, "compression_protect_last_n", 6)))
        max_seg = max(
            1,
            int(getattr(self.config, "context_collapse_segment_max_messages", 20)),
        )

        n = len(messages)
        if n <= protect_head + protect_tail + 4:
            return None

        covered: set[int] = set()
        for s in store.segments:
            for i in range(s.start_idx, s.end_idx + 1):
                covered.add(i)

        start: Optional[int] = None
        for i in range(protect_head, n - protect_tail):
            if i not in covered:
                start = i
                break
        if start is None:
            return None

        end = start
        while (
            end + 1 < n - protect_tail
            and (end + 1) not in covered
            and (end - start + 1) < max_seg
        ):
            end += 1

        # Tool-pair safety pass.  Order matters: roll the end back
        # *before* checking the start, because a roll-back might
        # collapse the window below the 4-message floor.
        end = self._roll_end_back_to_complete_pair(messages, start, end)
        if end is None or end < start:
            return None
        start = self._roll_start_forward_past_orphan_results(messages, start, end)
        if start is None or start > end:
            return None

        if end - start + 1 < 4:
            return None

        return (start, end)

    @staticmethod
    def _roll_end_back_to_complete_pair(
        messages: list[dict[str, Any]],
        start: int,
        end: int,
    ) -> Optional[int]:
        """Walk ``end`` backwards until it doesn't split a tool pair.

        A tool pair is ``(assistant with tool_use blocks, user/tool
        message carrying the matching tool_results)``.  If ``end``
        lands on the assistant half but the result half is at
        ``end+1``, we either include the result (if it's still inside
        the window) or roll back past the assistant.  Returns the
        adjusted ``end`` or ``None`` if the rollback eats the whole
        window.
        """
        cur = end
        while cur >= start:
            msg = messages[cur]
            if not isinstance(msg, dict):
                cur -= 1
                continue
            tool_use_ids = ContextCollapseTier._extract_tool_use_ids(msg)
            if not tool_use_ids:
                return cur
            # This message is an assistant turn that emitted tool_use
            # blocks.  We need its results to either be inside the
            # window or excluded entirely.
            unmatched = set(tool_use_ids)
            scan = cur + 1
            while unmatched and scan < len(messages):
                resolved = ContextCollapseTier._extract_tool_result_ids(messages[scan])
                if not resolved:
                    break
                unmatched -= resolved
                scan += 1
            if not unmatched:
                # All tool results follow contiguously — try to extend
                # the segment to include them.  Capped by the original
                # window growth (we don't extend past the configured
                # tail / max_seg here; if the results don't fit, we
                # roll back instead).
                last_result = scan - 1
                # We only extend within the *current* end's vicinity
                # — i.e. if the very next messages after ``cur`` are
                # the matching results.  If they are, advance ``cur``
                # to ``last_result`` and we're done.
                if last_result > cur:
                    return last_result
                return cur
            # Tool results spill outside the window — roll back past
            # this assistant entirely.
            cur -= 1
        return None

    @staticmethod
    def _roll_start_forward_past_orphan_results(
        messages: list[dict[str, Any]],
        start: int,
        end: int,
    ) -> Optional[int]:
        """If ``start`` lands on a tool-result without its tool_use, advance.

        Symmetric to ``_roll_end_back_to_complete_pair``: a window
        whose first message is a tool_result orphaned from its
        tool_use breaks the model's view of "what tool produced
        this result".  Roll forward until ``messages[start]`` is
        either non-tool-result or the matching tool_use is also
        inside the window (which it can't be by construction here,
        because we're walking forward from the original start).
        """
        cur = start
        while cur <= end:
            msg = messages[cur]
            if not isinstance(msg, dict):
                cur += 1
                continue
            if not ContextCollapseTier._extract_tool_result_ids(msg):
                return cur
            cur += 1
        return None

    @staticmethod
    def _extract_tool_use_ids(msg: dict[str, Any]) -> set[str]:
        """IDs of ``tool_use`` blocks emitted by an assistant message.

        Handles both Anthropic-style (``content`` is a list with
        ``{"type": "tool_use", "id": ...}`` blocks) and OpenAI-style
        (``tool_calls`` list with ``{"id": ..., "function": {...}}``).
        """
        if msg.get("role") != "assistant":
            return set()
        ids: set[str] = set()
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == _TOOL_USE_BLOCK_TYPE
                    and isinstance(block.get("id"), str)
                ):
                    ids.add(block["id"])
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict) and isinstance(call.get("id"), str):
                    ids.add(call["id"])
        return ids

    @staticmethod
    def _extract_tool_result_ids(msg: dict[str, Any]) -> set[str]:
        """IDs that this message resolves (Anthropic ``tool_result`` or OpenAI ``role=tool``)."""
        if not isinstance(msg, dict):
            return set()
        ids: set[str] = set()
        if msg.get("role") == "tool":
            tcid = msg.get("tool_call_id")
            if isinstance(tcid, str):
                ids.add(tcid)
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == _TOOL_RESULT_BLOCK_TYPE
                        and isinstance(block.get("tool_use_id"), str)
                    ):
                        ids.add(block["tool_use_id"])
        return ids

    @staticmethod
    def _extract_summary_text(summarised_messages: list[dict[str, Any]]) -> str:
        """Pull the summary text out of ``LLMForkSummarizer``'s output.

        After PR 3.4's boundary-merge fix, the summariser emits a
        single ``user`` message bearing
        ``_aether_meta.compact_summary=True``.  Its ``content`` starts
        with the ``[compact_boundary] ...`` prefix; we strip that
        prefix (boundary and summary are merged in PR 3.4 but the
        boundary text is meta-noise here, not part of the summary the
        model needs to read).
        """
        for msg in summarised_messages:
            meta = msg.get("_aether_meta") or {}
            if not isinstance(meta, dict) or not meta.get("compact_summary"):
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                return ""
            # Drop the merged boundary prefix if present (form:
            # ``"[compact_boundary] ...\n\n<actual summary>"``).
            if content.startswith("[compact_boundary]"):
                _, _, tail = content.partition("\n\n")
                return tail.strip()
            return content.strip()
        return ""

    @staticmethod
    def _update_collapse_owns_flag(
        turn_context: TurnContext,
        store: CollapseStore,
    ) -> None:
        """Maintain the Tier 5 mutex flag.

        Set when the store has at least one *committed* segment;
        cleared otherwise.  Clearing on cool-down is what lets Tier 5
        kick in if Tier 4 only managed proposals (none committed).
        """
        if store.has_committed:
            turn_context.metadata["collapse_owns_headroom"] = True
        else:
            turn_context.metadata.pop("collapse_owns_headroom", None)
