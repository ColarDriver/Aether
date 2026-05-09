"""Sprint 3 / PR 3.5 — ``TimeBasedMicrocompactor`` (Tier 3) unit tests.

Tier 3 is the cheap, local "user came back after a break" path:

* gate on ``compression_enabled`` + a configurable gap threshold;
* preserve the last ``keep_recent`` compactable tool calls verbatim;
* replace older ``tool_result`` payloads with a fixed placeholder
  string so the prompt shrinks without losing the call markers
  themselves (the model still sees that ``read_file('foo.py')``
  happened, just not the content);
* stamp ``tier3_cleared_count`` on ``turn_context.metadata`` for
  observability.

Tests are organised into the eight groups listed in the design doc
(``docs/sprint-3-compaction-pipeline/05_pr3_5_tier3_microcompact.md``
§ 5):

* Group A — gate disabled / early-exit semantics
* Group B — basic clearing behaviour
* Group C — tool-name matching (case / namespace / config respected)
* Group D — timestamp resolution (per-message + session fallback)
* Group E — ``keep_recent`` boundary behaviour (floor / equality)
* Group F — message-structure tolerance (malformed / missing fields)
* Group G — observability counters + log line format
* Group I — :class:`CachedMicrocompactor` stub contract

(Group H — pipeline integration with Tier 5 — lives in
``test_compaction_integration.py`` because it needs the full pipeline
wiring, not just the bare tier.)
"""

from __future__ import annotations

import logging
import time
import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext
from aether.services.compact.microcompact import (
    DEFAULT_COMPACTABLE_TOOLS,
    TIME_BASED_MC_CLEARED_MESSAGE,
    CachedMicrocompactor,
    TimeBasedMicrocompactor,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    """Duck-typed ``EngineConfig`` slice the microcompactor consults."""

    compression_enabled: bool = True
    microcompact_gap_threshold_minutes: float = 5.0
    microcompact_keep_recent: int = 5
    microcompact_compactable_tools: tuple[str, ...] = DEFAULT_COMPACTABLE_TOOLS


def _ctx(metadata: dict[str, Any] | None = None) -> TurnContext:
    return TurnContext(
        session_id="sess-tier3",
        iteration=0,
        metadata=dict(metadata or {}),
        task_id="task-1",
        turn_id="turn-1",
    )


def _compaction_ctx(*, model_window: int = 200_000) -> CompactionContext:
    return CompactionContext(
        session_id="sess-tier3",
        model="any",
        model_window=model_window,
        pre_compaction_tokens=0,
        target_pct=0.85,
        trigger_reason="preflight",
    )


def _build(config: _Config | None = None) -> TimeBasedMicrocompactor:
    return TimeBasedMicrocompactor(
        config=config or _Config(),
        logger=logging.getLogger("test.microcompact"),
    )


def _now() -> float:
    return time.time()


def _ago(minutes: float) -> float:
    """Wall-clock timestamp ``minutes`` ago."""
    return _now() - minutes * 60.0


def _assistant_with_tools(
    tool_calls: list[tuple[str, str, str]],
    *,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Build an assistant message with N tool_use blocks.

    ``tool_calls`` items are ``(tool_id, tool_name, input_str)`` triples.
    """
    blocks: list[dict[str, Any]] = []
    for tool_id, tool_name, payload in tool_calls:
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": {"args": payload},
            }
        )
    msg: dict[str, Any] = {"role": "assistant", "content": blocks}
    if timestamp is not None:
        msg["_aether_meta"] = {"timestamp": timestamp}
    return msg


def _user_tool_results(
    results: list[tuple[str, str]],
) -> dict[str, Any]:
    """Build a user message carrying tool_result blocks.

    ``results`` items are ``(tool_use_id, content)`` tuples.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
            for tool_use_id, content in results
        ],
    }


def _conversation(
    *,
    n_calls: int,
    tool_name: str = "read_file",
    body_chars: int = 400,
    last_assistant_minutes_ago: float | None = 10.0,
) -> list[dict[str, Any]]:
    """Build ``n_calls`` paired assistant→tool_result rounds.

    The final assistant message (the one Tier 3 measures from) is
    emitted with ``timestamp = _ago(last_assistant_minutes_ago)`` if
    that argument is not ``None``.
    """
    msgs: list[dict[str, Any]] = [{"role": "user", "content": "kick off"}]
    for i in range(n_calls):
        is_last = i == n_calls - 1
        ts: float | None = None
        if is_last and last_assistant_minutes_ago is not None:
            ts = _ago(last_assistant_minutes_ago)
        msgs.append(
            _assistant_with_tools(
                [(f"call-{i}", tool_name, "x" * body_chars)],
                timestamp=ts,
            )
        )
        msgs.append(
            _user_tool_results([(f"call-{i}", "y" * body_chars)])
        )
    return msgs


# ---------------------------------------------------------------------------
# Group A — gate disabled / early-exit semantics
# ---------------------------------------------------------------------------


class TimeBasedMicrocompactorGroupAGateTests(unittest.TestCase):
    def test_a1_compression_disabled_returns_unchanged(self) -> None:
        """Master switch off → no work, no metadata side effects."""
        compactor = _build(config=_Config(compression_enabled=False))
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=60)
        ctx = _ctx()
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), ctx)
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)
        self.assertNotIn("tier3_cleared_count", ctx.metadata)

    def test_a2_no_timestamp_anywhere_returns_unchanged(self) -> None:
        """If gap can't be computed, Tier 3 is a no-op (does NOT crash)."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=None)
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)

    def test_a3_gap_below_threshold_returns_unchanged(self) -> None:
        """gap=4min < threshold=5min → no clearing."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=4.0)
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)

    def test_a4_too_few_compactable_tools_returns_unchanged(self) -> None:
        """gap met but only 3 tools < keep_recent=5 → nothing to clear."""
        compactor = _build()
        msgs = _conversation(n_calls=3, last_assistant_minutes_ago=60.0)
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)


# ---------------------------------------------------------------------------
# Group B — basic clearing behaviour
# ---------------------------------------------------------------------------


class TimeBasedMicrocompactorGroupBClearingTests(unittest.TestCase):
    def test_b1_clears_oldest_keeps_recent(self) -> None:
        """10 tools + keep_recent=5 → first 5 cleared, last 5 untouched."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertGreater(freed, 0)
        # Pick out the 10 user messages bearing tool_result blocks.
        result_msgs = [m for m in out if m.get("role") == "user" and isinstance(m.get("content"), list)]
        self.assertEqual(len(result_msgs), 10)
        # First 5 cleared.
        for msg in result_msgs[:5]:
            block = msg["content"][0]
            self.assertEqual(block["content"], TIME_BASED_MC_CLEARED_MESSAGE)
        # Last 5 preserve their original ``y...`` body.
        for msg in result_msgs[5:]:
            block = msg["content"][0]
            self.assertNotEqual(block["content"], TIME_BASED_MC_CLEARED_MESSAGE)
            self.assertTrue(block["content"].startswith("y"))

    def test_b2_cleared_block_preserves_all_other_fields(self) -> None:
        """Replacing ``content`` must not drop ``type`` / ``tool_use_id``."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        out, _ = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        first_cleared = [m for m in out if m.get("role") == "user" and isinstance(m.get("content"), list)][0]
        block = first_cleared["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "call-0")
        self.assertEqual(block["content"], TIME_BASED_MC_CLEARED_MESSAGE)

    def test_b3_idempotent_when_run_twice(self) -> None:
        """Re-running on already-cleared messages is a no-op (freed=0)."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        out1, freed1 = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertGreater(freed1, 0)
        # Re-run on the already-compacted output.  No new clears should
        # happen — the implementation detects that every targeted block
        # already carries the placeholder and short-circuits with
        # freed=0 (so the pipeline doesn't double-count).
        ctx2 = _ctx()
        out2, freed2 = compactor.maybe_run(out1, _compaction_ctx(), ctx2)
        self.assertIs(out2, out1)
        self.assertEqual(freed2, 0)
        self.assertNotIn("tier3_cleared_count", ctx2.metadata)

    def test_b4_freed_token_estimate_is_positive_and_meaningful(self) -> None:
        """``freed`` reflects token count delta from clearing the bodies."""
        compactor = _build()
        msgs = _conversation(
            n_calls=10, body_chars=2_000, last_assistant_minutes_ago=10.0
        )
        _, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        # 5 cleared × ~2000 chars / 4 chars-per-token ≈ 2500 tokens.
        # Allow generous slack — heuristic isn't promised to be exact.
        self.assertGreater(freed, 1_000)


# ---------------------------------------------------------------------------
# Group C — tool name matching
# ---------------------------------------------------------------------------


class TimeBasedMicrocompactorGroupCNameMatchingTests(unittest.TestCase):
    def test_c1_case_and_dash_normalised_tool_name_matches(self) -> None:
        """``Read-File`` (mixed case + dash) normalises to ``read_file`` and matches.

        Note: ``_normalize_name`` does case-fold + dash↔underscore +
        namespace strip — but NOT CamelCase splitting (``ReadFile`` →
        ``readfile`` ≠ ``read_file``).  The Tier 3 whitelist therefore
        accepts every variant the rest of the agent's tool-resolution
        path also accepts: ``READ_FILE`` / ``read-file`` /
        ``Read_File``.  Models that emit pure CamelCase tool names
        already lose those calls at dispatch time, so adding extra
        normalisation here would just paper over a wider issue.
        """
        compactor = _build()
        msgs = _conversation(
            n_calls=10, tool_name="Read-File", last_assistant_minutes_ago=10.0
        )
        _, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertGreater(freed, 0)

    def test_c2_mcp_namespace_prefix_strips_via_normalisation(self) -> None:
        """``mcp__router__shell`` is recognised as ``shell``."""
        compactor = _build()
        msgs = _conversation(
            n_calls=10,
            tool_name="mcp__router__shell",
            last_assistant_minutes_ago=10.0,
        )
        _, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertGreater(freed, 0)

    def test_c3_tool_not_in_compactable_set_is_skipped(self) -> None:
        """``update_todo`` (not in default whitelist) is never cleared."""
        compactor = _build()
        msgs = _conversation(
            n_calls=10, tool_name="update_todo", last_assistant_minutes_ago=10.0
        )
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(freed, 0)
        self.assertIs(out, msgs)

    def test_c4_empty_compactable_set_disables_clearing(self) -> None:
        """``microcompact_compactable_tools=()`` short-circuits before walking."""
        compactor = _build(
            config=_Config(microcompact_compactable_tools=())
        )
        msgs = _conversation(
            n_calls=10, tool_name="read_file", last_assistant_minutes_ago=10.0
        )
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(freed, 0)
        self.assertIs(out, msgs)

    def test_c5_mixed_batch_only_clears_whitelisted_tools(self) -> None:
        """5 read_file + 5 update_todo: read_file count <= keep_recent → no clear."""
        compactor = _build()
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        # 5 read_file rounds ...
        for i in range(5):
            msgs.append(
                _assistant_with_tools(
                    [(f"r-{i}", "read_file", f"file{i}")],
                    timestamp=None,
                )
            )
            msgs.append(_user_tool_results([(f"r-{i}", f"contents-{i}")]))
        # ... 5 update_todo rounds, with the final assistant carrying timestamp.
        for i in range(5):
            is_last = i == 4
            msgs.append(
                _assistant_with_tools(
                    [(f"t-{i}", "update_todo", "[]")],
                    timestamp=_ago(10.0) if is_last else None,
                )
            )
            msgs.append(_user_tool_results([(f"t-{i}", "ok")]))
        # Only 5 read_file calls exist (not in update_todo's whitelist
        # entry) — that's exactly == keep_recent=5, so Tier 3 finds
        # nothing to clear and bails.
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(freed, 0)
        self.assertIs(out, msgs)


# ---------------------------------------------------------------------------
# Group D — timestamp resolution
# ---------------------------------------------------------------------------


class TimeBasedMicrocompactorGroupDTimestampTests(unittest.TestCase):
    def test_d1_per_message_timestamp_is_used(self) -> None:
        """``_aether_meta.timestamp`` 5min ago → gap ≈ 5.0min."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        # Confirm we go through ``_compute_gap_minutes`` and not the
        # fallback by stripping any session_record_meta.
        ctx = _ctx({})
        _, freed = compactor.maybe_run(msgs, _compaction_ctx(), ctx)
        self.assertGreater(freed, 0)

    def test_d2_session_record_fallback_when_no_per_msg_timestamp(self) -> None:
        """Falls back to ``session_record_meta.last_assistant_timestamp``."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=None)
        ctx = _ctx(
            {
                "session_record_meta": {
                    "last_assistant_timestamp": _ago(10.0),
                }
            }
        )
        _, freed = compactor.maybe_run(msgs, _compaction_ctx(), ctx)
        self.assertGreater(freed, 0)

    def test_d3_no_source_returns_none_no_clear(self) -> None:
        """Both sources missing → bail without clearing."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=None)
        ctx = _ctx({})
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), ctx)
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)

    def test_d4_string_timestamp_is_treated_as_missing(self) -> None:
        """Bad type for timestamp → degrade to fallback / no-op, never raise."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=None)
        # Hand-craft an assistant with a *string* timestamp (corrupt).
        msgs[-2]["_aether_meta"] = {"timestamp": "not-a-float"}
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)

    def test_d5_only_last_assistant_consulted_for_timestamp(self) -> None:
        """Walk in reverse and stop at the FIRST assistant — even if it
        lacks a timestamp.  An older assistant's timestamp is *not*
        consulted; we don't want stale gap measurements."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=None)
        # Stamp an old timestamp on assistant #0 (the first round); the
        # last assistant (round 9) has no timestamp.  Tier 3 should see
        # "no timestamp on last assistant" → bail.
        first_assistant = next(m for m in msgs if m.get("role") == "assistant")
        first_assistant["_aether_meta"] = {"timestamp": _ago(60.0)}
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)

    def test_d6_clock_skew_clamps_negative_gap_to_zero(self) -> None:
        """Future timestamp (system clock rewind) → gap clamped to 0,
        below threshold, no clear.  Must NOT crash with a negative
        comparison."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=None)
        # 10 minutes in the future.
        msgs[-2]["_aether_meta"] = {"timestamp": _now() + 600}
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)


# ---------------------------------------------------------------------------
# Group E — keep_recent boundary behaviour
# ---------------------------------------------------------------------------


class TimeBasedMicrocompactorGroupEKeepRecentTests(unittest.TestCase):
    def test_e1_keep_recent_zero_floors_to_one(self) -> None:
        """``keep_recent=0`` would strand the model — must floor to 1."""
        compactor = _build(config=_Config(microcompact_keep_recent=0))
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertGreater(freed, 0)
        result_msgs = [m for m in out if m.get("role") == "user" and isinstance(m.get("content"), list)]
        # First 9 cleared, last 1 preserved (because floor=1).
        cleared = sum(
            1 for m in result_msgs
            if m["content"][0]["content"] == TIME_BASED_MC_CLEARED_MESSAGE
        )
        self.assertEqual(cleared, 9)

    def test_e2_keep_recent_one_clears_all_but_last(self) -> None:
        compactor = _build(config=_Config(microcompact_keep_recent=1))
        msgs = _conversation(n_calls=6, last_assistant_minutes_ago=10.0)
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertGreater(freed, 0)
        result_msgs = [m for m in out if m.get("role") == "user" and isinstance(m.get("content"), list)]
        cleared = sum(
            1 for m in result_msgs
            if m["content"][0]["content"] == TIME_BASED_MC_CLEARED_MESSAGE
        )
        self.assertEqual(cleared, 5)

    def test_e3_keep_recent_above_total_returns_unchanged(self) -> None:
        """``keep_recent=100`` with 6 calls → all preserved, freed=0."""
        compactor = _build(config=_Config(microcompact_keep_recent=100))
        msgs = _conversation(n_calls=6, last_assistant_minutes_ago=10.0)
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)


# ---------------------------------------------------------------------------
# Group F — message structure tolerance
# ---------------------------------------------------------------------------


class TimeBasedMicrocompactorGroupFToleranceTests(unittest.TestCase):
    def test_f1_user_with_string_content_is_left_alone(self) -> None:
        """user msg whose ``content`` is plain string (not list) is skipped."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        # Inject an extra string-content user message; it must survive
        # untouched alongside the cleared tool_result blocks.
        msgs.insert(1, {"role": "user", "content": "stop being so literal"})
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertGreater(freed, 0)
        # The injected string-content message is still there.
        self.assertIn(
            {"role": "user", "content": "stop being so literal"}, out
        )

    def test_f2_tool_result_missing_tool_use_id_is_left_alone(self) -> None:
        """tool_result without ``tool_use_id`` doesn't match any clear set entry."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        # Drop tool_use_id from the second tool_result message
        # (call-1).  That block should NOT be cleared (no ID match).
        target_msg = msgs[4]  # 1=user/2=assistant/3=user(call-0)/4=assistant(call-1)... wait
        # Robustly find the tool_result for call-1 instead of indexing.
        for msg in msgs:
            if (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), list)
                and msg["content"]
                and msg["content"][0].get("tool_use_id") == "call-1"
            ):
                target_msg = msg
                break
        del target_msg["content"][0]["tool_use_id"]
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        # Other clears still happen (so freed > 0); but the no-id block
        # is unchanged.
        self.assertGreater(freed, 0)
        self.assertNotIn(
            "tool_use_id", target_msg["content"][0],
            f"missing tool_use_id should remain absent, got {target_msg['content'][0]!r}",
        )
        self.assertNotEqual(
            target_msg["content"][0].get("content"),
            TIME_BASED_MC_CLEARED_MESSAGE,
        )

    def test_f3_tool_use_missing_id_is_skipped(self) -> None:
        """tool_use without ``id`` doesn't enter the candidate list."""
        compactor = _build()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "go"},
        ]
        # 6 normal calls (so we exceed keep_recent=5) ...
        for i in range(6):
            is_last = i == 5
            msgs.append(
                _assistant_with_tools(
                    [(f"call-{i}", "read_file", "x")],
                    timestamp=_ago(10.0) if is_last else None,
                )
            )
            msgs.append(_user_tool_results([(f"call-{i}", "y" * 200)]))
        # ... plus one assistant whose tool_use lacks ``id``.  That call
        # MUST NOT enter the candidate list (otherwise we'd have an
        # entry whose ID is missing and the clear set would be wrong).
        msgs.insert(
            1,
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "read_file", "input": {}}],
            },
        )
        compactor.maybe_run(msgs, _compaction_ctx(), _ctx())  # must not raise

    def test_f4_empty_messages_returns_empty(self) -> None:
        compactor = _build()
        out, freed = compactor.maybe_run([], _compaction_ctx(), _ctx())
        self.assertEqual(out, [])
        self.assertEqual(freed, 0)


# ---------------------------------------------------------------------------
# Group G — observability counters + log line format
# ---------------------------------------------------------------------------


class TimeBasedMicrocompactorGroupGObservabilityTests(unittest.TestCase):
    def test_g1_tier3_cleared_count_is_set(self) -> None:
        """Counter equals number of clear-set entries on first run."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        ctx = _ctx()
        compactor.maybe_run(msgs, _compaction_ctx(), ctx)
        # 10 calls, keep_recent=5 → 5 cleared.
        self.assertEqual(ctx.metadata.get("tier3_cleared_count"), 5)

    def test_g2_counter_accumulates_across_invocations(self) -> None:
        """Two independent runs against fresh inputs accumulate the count."""
        compactor = _build()
        ctx = _ctx()
        compactor.maybe_run(
            _conversation(n_calls=10, last_assistant_minutes_ago=10.0),
            _compaction_ctx(),
            ctx,
        )
        compactor.maybe_run(
            _conversation(n_calls=8, last_assistant_minutes_ago=10.0),
            _compaction_ctx(),
            ctx,
        )
        # 10-keep5=5 + 8-keep5=3 = 8 total.
        self.assertEqual(ctx.metadata.get("tier3_cleared_count"), 8)

    def test_g3_log_line_contains_diagnostic_fields(self) -> None:
        """Log payload mentions gap / threshold / cleared / kept / freed."""
        compactor = _build()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        with self.assertLogs("test.microcompact", level="INFO") as recorded:
            compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        log_blob = "\n".join(recorded.output)
        self.assertIn("tier3:", log_blob)
        self.assertIn("gap=", log_blob)
        self.assertIn("min", log_blob)
        self.assertIn("cleared 5", log_blob)
        self.assertIn("kept last 5", log_blob)
        self.assertIn("freed", log_blob)


# ---------------------------------------------------------------------------
# Group I — CachedMicrocompactor stub contract
# ---------------------------------------------------------------------------


class CachedMicrocompactorStubTests(unittest.TestCase):
    def test_i1_maybe_run_is_noop_returns_unchanged(self) -> None:
        cached = CachedMicrocompactor()
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        out, freed = cached.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)

    def test_i2_name_is_distinct_from_time_based_variant(self) -> None:
        """Distinct name lets log analysis tell the two paths apart."""
        self.assertEqual(CachedMicrocompactor.name, "tier3_microcompact_cached")
        self.assertNotEqual(
            CachedMicrocompactor.name, TimeBasedMicrocompactor.name
        )


# ---------------------------------------------------------------------------
# Bonus — tool-style (flat OpenAI ``role=tool``) results are also cleared
# ---------------------------------------------------------------------------


class TimeBasedMicrocompactorToolRoleTests(unittest.TestCase):
    """OpenAI-style flat tool-result messages live under ``role=tool``;
    the implementation must clear those too (so providers using the
    chat-completions tool API benefit from Tier 3 just like Anthropic
    Messages users do)."""

    def test_tool_role_results_get_cleared(self) -> None:
        compactor = _build()
        # Build a hybrid history with role=tool messages.
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        for i in range(6):
            is_last = i == 5
            msgs.append(
                _assistant_with_tools(
                    [(f"call-{i}", "read_file", f"f{i}")],
                    timestamp=_ago(10.0) if is_last else None,
                )
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call-{i}",
                    "content": "y" * 500,
                }
            )
        out, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertGreater(freed, 0)
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 6)
        # 6 calls, keep_recent=5 → first 1 cleared, last 5 preserved.
        cleared = [
            m for m in tool_msgs if m.get("content") == TIME_BASED_MC_CLEARED_MESSAGE
        ]
        self.assertEqual(len(cleared), 1)
        # And the last one's body is preserved.
        self.assertNotEqual(tool_msgs[-1]["content"], TIME_BASED_MC_CLEARED_MESSAGE)


# ---------------------------------------------------------------------------
# Time-control utility test — pin a fixed ``time.time()`` for determinism
# ---------------------------------------------------------------------------


class TimeBasedMicrocompactorDeterministicTimeTests(unittest.TestCase):
    """Sanity-check that all gap math goes through ``time.time()``;
    by patching it we get a fully deterministic gap regardless of
    system load between message construction and tier execution."""

    def test_patched_time_yields_predictable_gap(self) -> None:
        compactor = _build(
            config=_Config(microcompact_gap_threshold_minutes=5.0)
        )
        # Build messages with timestamp = 1_000_000.0; then patch
        # ``time.time`` to return 1_000_000.0 + 6*60 → gap = 6 min.
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
        for i in range(10):
            is_last = i == 9
            msgs.append(
                _assistant_with_tools(
                    [(f"c{i}", "read_file", "x")],
                    timestamp=1_000_000.0 if is_last else None,
                )
            )
            msgs.append(_user_tool_results([(f"c{i}", "y" * 400)]))
        with patch(
            "aether.services.compact.microcompact.time.time",
            return_value=1_000_000.0 + 360.0,
        ):
            _, freed = compactor.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertGreater(freed, 0)


# ---------------------------------------------------------------------------
# Group H — pipeline integration with Tier 5
# ---------------------------------------------------------------------------


@dataclass
class _SpyTier:
    """Minimal CompactorTier double for Group H pipeline integration.

    We don't reuse the ``_FakeTier`` from ``test_compaction_pipeline.py``
    because importing it across test modules creates a hidden coupling;
    keeping a tiny purpose-built spy here is cheaper than reorganising
    the test layout.
    """

    name: str = "tier5_spy"
    freed: int = 0
    transform: Any = None
    calls: int = 0

    def maybe_run(
        self,
        messages: list[dict[str, Any]],
        ctx: CompactionContext,
        turn_context: TurnContext,
    ) -> tuple[list[dict[str, Any]], int]:
        self.calls += 1
        if self.transform is not None:
            return self.transform(messages), self.freed
        return messages, self.freed


@dataclass
class _PipelineConfig:
    """Duck-typed config that satisfies BOTH ``CompactionPipeline`` and ``TimeBasedMicrocompactor``."""

    compression_enabled: bool = True
    compression_pre_llm_pct: float = 0.85
    microcompact_gap_threshold_minutes: float = 5.0
    microcompact_keep_recent: int = 5
    microcompact_compactable_tools: tuple[str, ...] = DEFAULT_COMPACTABLE_TOOLS


class TimeBasedMicrocompactorGroupHPipelineIntegrationTests(unittest.TestCase):
    """Tier 3 + Tier 5 cooperate correctly inside the real pipeline.

    These tests use the production ``CompactionPipeline`` orchestrator
    with a ``TimeBasedMicrocompactor`` (real) plus a tiny Tier 5 spy
    so we can assert "did Tier 5 also fire?" without standing up the
    full LLM-fork machinery.
    """

    @staticmethod
    def _build_pipeline(
        *,
        tier3: TimeBasedMicrocompactor,
        tier5: _SpyTier,
        estimator: Any,
        config: _PipelineConfig,
    ) -> Any:
        from aether.services.compact.compactor import CompactionPipeline

        return CompactionPipeline(
            tiers=[tier3, tier5],
            token_estimator=estimator,
            config=config,
            logger=logging.getLogger("test.microcompact.pipeline"),
        )

    def test_h1_tier3_meets_target_short_circuits_tier5(self) -> None:
        """Preflight + Tier 3 drops estimate below target → Tier 5 skipped."""
        config = _PipelineConfig()
        tier3 = TimeBasedMicrocompactor(
            config=config,
            logger=logging.getLogger("test.microcompact.h1"),
        )
        tier5 = _SpyTier(name="tier5_autocompact_spy")
        # Estimator script:
        #   1) initial estimate (above target),
        #   2) tier3 before, 3) tier3 after (already at target),
        #   4) final estimate.
        # Since respect_threshold_break=True for preflight, Tier 5
        # never gets reached.
        sequence = iter([200_000, 200_000, 50_000, 50_000])
        pipeline = self._build_pipeline(
            tier3=tier3,
            tier5=tier5,
            estimator=lambda msgs: next(sequence),
            config=config,
        )
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        result = pipeline.maybe_compress(
            msgs,
            turn_context=_ctx(),
            model="x",
            model_window=100_000,
            trigger_reason="preflight",
        )
        self.assertEqual(result.tiers_run, [tier3.name])
        self.assertEqual(tier5.calls, 0)
        self.assertFalse(result.exhausted)

    def test_h2_tier3_partial_relief_falls_through_to_tier5(self) -> None:
        """Tier 3 frees only a little → Tier 5 still runs."""
        config = _PipelineConfig()
        tier3 = TimeBasedMicrocompactor(
            config=config,
            logger=logging.getLogger("test.microcompact.h2"),
        )
        tier5 = _SpyTier(name="tier5_autocompact_spy", freed=10_000)
        # Tier 3 brings us from 200k to 180k (still above 85k target);
        # Tier 5 then runs, brings us down to 70k.
        sequence = iter([200_000, 200_000, 180_000, 180_000, 70_000, 70_000])
        pipeline = self._build_pipeline(
            tier3=tier3,
            tier5=tier5,
            estimator=lambda msgs: next(sequence),
            config=config,
        )
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=10.0)
        result = pipeline.maybe_compress(
            msgs,
            turn_context=_ctx(),
            model="x",
            model_window=100_000,
            trigger_reason="preflight",
        )
        self.assertEqual(result.tiers_run, [tier3.name, tier5.name])
        self.assertEqual(tier5.calls, 1)

    def test_h3_gap_below_threshold_skips_tier3_runs_tier5(self) -> None:
        """Tier 3 inert (gap < threshold) but tokens > target → Tier 5 fires anyway."""
        config = _PipelineConfig()
        tier3 = TimeBasedMicrocompactor(
            config=config,
            logger=logging.getLogger("test.microcompact.h3"),
        )
        tier5 = _SpyTier(name="tier5_autocompact_spy")
        sequence = iter([200_000, 200_000, 200_000, 200_000, 50_000, 50_000])
        pipeline = self._build_pipeline(
            tier3=tier3,
            tier5=tier5,
            estimator=lambda msgs: next(sequence),
            config=config,
        )
        # gap=2min < threshold=5min — Tier 3 returns unchanged, freed=0.
        msgs = _conversation(n_calls=10, last_assistant_minutes_ago=2.0)
        result = pipeline.maybe_compress(
            msgs,
            turn_context=_ctx(),
            model="x",
            model_window=100_000,
            trigger_reason="preflight",
        )
        # Pipeline records BOTH tiers as having "run" (tier3 ran but
        # was a no-op; tier5 actually did the work).  This asserts
        # we don't accidentally short-circuit at tier3 when freed=0.
        self.assertIn(tier5.name, result.tiers_run)
        self.assertEqual(tier5.calls, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
