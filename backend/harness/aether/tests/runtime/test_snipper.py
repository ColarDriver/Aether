"""Sprint 3 / PR 3.6 — :class:`Snipper` (Tier 2) unit tests.

Covers all nine test groups from
``docs/sprint-3-compaction-pipeline/06_pr3_6_tier2_snip.md`` § 5.1:

* Group A — gates / early exit
* Group B — Rule 1 (DUPE)
* Group C — Rule 2 (FAIL)
* Group D — Rule 3 (EMPTY assistant)
* Group E — pairing invariants
* Group F — multi-rule combinations
* Group G — observability
* Group H — pipeline interplay (the doc's "with Tier 5" group; here
            we exercise the Snipper alone but assert the pipeline-
            visible signals — counter, freed, no orphans)
* Group I — boundary / fault tolerance
"""

from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass
from typing import Any

from aether.runtime.contracts import TurnContext
from aether.services.compact.compactor import CompactionContext
from aether.services.compact.snip import SNIP_DUPE_RULE_TOOLS, Snipper, _SnipPlan


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    """Duck-typed ``EngineConfig`` slice the snipper consults."""

    compression_enabled: bool = True
    snip_enabled: bool = True
    snip_dupe_enabled: bool = True
    snip_fail_enabled: bool = True
    snip_empty_enabled: bool = True


def _ctx(metadata: dict[str, Any] | None = None) -> TurnContext:
    return TurnContext(
        session_id="snip-tests",
        iteration=0,
        metadata=dict(metadata or {}),
        task_id="t",
        turn_id="r",
    )


def _compaction_ctx() -> CompactionContext:
    return CompactionContext(
        session_id="snip-tests",
        model="claude-sonnet",
        model_window=100_000,
        pre_compaction_tokens=0,
        target_pct=0.85,
        trigger_reason="preflight",
    )


def _build(config: _Config | None = None) -> tuple[Snipper, _Config, logging.Logger]:
    cfg = config or _Config()
    logger = logging.getLogger("test.snipper")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return Snipper(config=cfg, logger=logger), cfg, logger


# ---------------------------------------------------------------------------
# Message builders — keep tests dense + readable
# ---------------------------------------------------------------------------


def _u(text: str) -> dict[str, Any]:
    """User text message."""
    return {"role": "user", "content": text}


def _a_text(text: str) -> dict[str, Any]:
    """Assistant message with text content (string form)."""
    return {"role": "assistant", "content": text}


def _a_blocks(*blocks: dict[str, Any]) -> dict[str, Any]:
    """Assistant message with structured content blocks."""
    return {"role": "assistant", "content": list(blocks)}


def _u_results(*blocks: dict[str, Any]) -> dict[str, Any]:
    """User message carrying tool_result blocks."""
    return {"role": "user", "content": list(blocks)}


def _tool_use(name: str, *, id_: str, input_: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": id_,
        "name": name,
        "input": dict(input_ or {}),
    }


def _tool_result(
    *,
    id_: str,
    content: str = "ok",
    is_error: bool = False,
) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": id_,
        "content": content,
        "is_error": is_error,
    }


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _thinking_block(text: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": text}


def _ids_in(messages: list[dict[str, Any]]) -> list[str]:
    """Extract tool_use ids in encounter order — handy for assertions."""
    out: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    out.append(b.get("id"))
    return out


def _result_ids_in(messages: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out.append(b.get("tool_use_id"))
    return out


# ---------------------------------------------------------------------------
# Group A — gates / early exit
# ---------------------------------------------------------------------------


class SnipperGroupAGateTests(unittest.TestCase):
    def test_a1_compression_disabled(self) -> None:
        """``compression_enabled=False`` → snipper returns input untouched."""
        snipper, _, _ = _build(_Config(compression_enabled=False))
        msgs = [_a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"}))]
        out, freed = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)

    def test_a2_snip_disabled(self) -> None:
        """``snip_enabled=False`` short-circuits even if compression is on."""
        snipper, _, _ = _build(_Config(snip_enabled=False))
        msgs = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        out, freed = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)

    def test_a3_empty_messages(self) -> None:
        snipper, _, _ = _build()
        out, freed = snipper.maybe_run([], _compaction_ctx(), _ctx())
        self.assertEqual(out, [])
        self.assertEqual(freed, 0)

    def test_a4_no_tool_use_no_change(self) -> None:
        """All-text conversation produces no plan → identity return."""
        snipper, _, _ = _build()
        msgs = [_u("hello"), _a_text("hi back"), _u("ok")]
        out, freed = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIs(out, msgs)
        self.assertEqual(freed, 0)


# ---------------------------------------------------------------------------
# Group B — Rule 1 (DUPE)
# ---------------------------------------------------------------------------


class SnipperGroupBDupeRuleTests(unittest.TestCase):
    def test_b1_three_reads_same_path_keeps_last_only(self) -> None:
        """3x read_file path=/x.txt → keep only the last; delete first 2 pairs."""
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A", content="v1")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B", content="v2")),
            _a_blocks(_tool_use("read_file", id_="C", input_={"path": "/x"})),
            _u_results(_tool_result(id_="C", content="v3")),
        ]
        out, freed = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        # Only id=C survives.
        self.assertEqual(_ids_in(out), ["C"])
        self.assertEqual(_result_ids_in(out), ["C"])
        self.assertGreater(freed, 0)

    def test_b2_two_distinct_paths_each_dedupe_independently(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/y"})),
            _u_results(_tool_result(id_="B")),
            _a_blocks(_tool_use("read_file", id_="C", input_={"path": "/x"})),
            _u_results(_tool_result(id_="C")),
            _a_blocks(_tool_use("read_file", id_="D", input_={"path": "/y"})),
            _u_results(_tool_result(id_="D")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        # A (path=/x) and B (path=/y) deleted; C and D survive.
        self.assertEqual(set(_ids_in(out)), {"C", "D"})

    def test_b3_input_dict_key_order_does_not_matter(self) -> None:
        """``{a:1,b:2}`` and ``{b:2,a:1}`` are the SAME key (sort_keys)."""
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("grep", id_="A", input_={"path": "/x", "pattern": "p"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("grep", id_="B", input_={"pattern": "p", "path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["B"])

    def test_b4_shell_is_excluded_from_dupe_rule(self) -> None:
        """Same shell command twice → both kept (side-effect possibility)."""
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("shell", id_="A", input_={"command": "ls"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("shell", id_="B", input_={"command": "ls"})),
            _u_results(_tool_result(id_="B")),
        ]
        out, freed = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["A", "B"])
        # No plan work → freed=0.
        self.assertEqual(freed, 0)

    def test_b5_write_file_is_excluded_from_dupe_rule(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("write_file", id_="A", input_={"path": "/x", "content": "a"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("write_file", id_="B", input_={"path": "/x", "content": "a"})),
            _u_results(_tool_result(id_="B")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["A", "B"])

    def test_b6_grep_with_different_paths_not_treated_as_duplicate(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("grep", id_="A", input_={"path": "/x", "pattern": "p"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("grep", id_="B", input_={"path": "/y", "pattern": "p"})),
            _u_results(_tool_result(id_="B")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["A", "B"])

    def test_b7_dupe_rule_disabled(self) -> None:
        snipper, _, _ = _build(_Config(snip_dupe_enabled=False))
        msgs = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["A", "B"])


# ---------------------------------------------------------------------------
# Group C — Rule 2 (FAIL)
# ---------------------------------------------------------------------------


class SnipperGroupCFailRuleTests(unittest.TestCase):
    def test_c1_failed_then_success_deletes_failed(self) -> None:
        # Use shell so Rule 1 doesn't also fire (so we isolate Rule 2's effect).
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("shell", id_="A", input_={"command": "x"})),
            _u_results(_tool_result(id_="A", is_error=True, content="ENOENT")),
            _a_blocks(_tool_use("shell", id_="B", input_={"command": "x"})),
            _u_results(_tool_result(id_="B", is_error=False, content="hello")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        # A deleted (failed, superseded by B success); B preserved.
        self.assertEqual(_ids_in(out), ["B"])

    def test_c2_pure_failure_with_no_success_is_preserved(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("shell", id_="A", input_={"command": "fail"})),
            _u_results(_tool_result(id_="A", is_error=True, content="boom")),
        ]
        out, freed = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        # Failure kept (no later success to supersede).
        self.assertEqual(_ids_in(out), ["A"])
        self.assertEqual(freed, 0)

    def test_c3_three_calls_error_error_success_deletes_both_errors(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("shell", id_="A", input_={"command": "x"})),
            _u_results(_tool_result(id_="A", is_error=True)),
            _a_blocks(_tool_use("shell", id_="B", input_={"command": "x"})),
            _u_results(_tool_result(id_="B", is_error=True)),
            _a_blocks(_tool_use("shell", id_="C", input_={"command": "x"})),
            _u_results(_tool_result(id_="C", is_error=False)),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["C"])

    def test_c4_rule2_applies_to_all_tools_including_shell(self) -> None:
        """Rule 2 doesn't share Rule 1's whitelist; shell counts here."""
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("shell", id_="A", input_={"command": "x"})),
            _u_results(_tool_result(id_="A", is_error=True)),
            _a_blocks(_tool_use("shell", id_="B", input_={"command": "x"})),
            _u_results(_tool_result(id_="B", is_error=False)),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["B"])

    def test_c5_orphan_tool_use_without_result_is_left_alone(self) -> None:
        """Interrupted call (tool_use without result) → don't touch."""
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("shell", id_="A", input_={"command": "x"})),
            # Note: no tool_result for A.
            _a_blocks(_tool_use("shell", id_="B", input_={"command": "x"})),
            _u_results(_tool_result(id_="B", is_error=False)),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["A", "B"])

    def test_c6_rule2_disabled(self) -> None:
        snipper, _, _ = _build(_Config(snip_fail_enabled=False))
        msgs = [
            _a_blocks(_tool_use("shell", id_="A", input_={"command": "x"})),
            _u_results(_tool_result(id_="A", is_error=True)),
            _a_blocks(_tool_use("shell", id_="B", input_={"command": "x"})),
            _u_results(_tool_result(id_="B", is_error=False)),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["A", "B"])


# ---------------------------------------------------------------------------
# Group D — Rule 3 (EMPTY ASSISTANT)
# ---------------------------------------------------------------------------


class SnipperGroupDEmptyAssistantRuleTests(unittest.TestCase):
    def test_d1_assistant_empty_string_content(self) -> None:
        snipper, _, _ = _build()
        msgs = [_u("q"), _a_text(""), _u("more")]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual([m["role"] for m in out], ["user", "user"])

    def test_d2_assistant_whitespace_only_content(self) -> None:
        snipper, _, _ = _build()
        msgs = [_u("q"), _a_text("  \n\t  "), _u("more")]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual([m["role"] for m in out], ["user", "user"])

    def test_d3_assistant_content_none(self) -> None:
        snipper, _, _ = _build()
        msgs = [_u("q"), {"role": "assistant", "content": None}, _u("more")]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual([m["role"] for m in out], ["user", "user"])

    def test_d4_blocks_all_empty_text_and_thinking_dropped(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _u("q"),
            _a_blocks(_text_block(""), _thinking_block("  ")),
            _u("more"),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual([m["role"] for m in out], ["user", "user"])

    def test_d5_real_thinking_block_preserved(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _u("q"),
            _a_blocks(_thinking_block("I should think hard")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(len(out), 2)

    def test_d6_real_text_block_preserved(self) -> None:
        snipper, _, _ = _build()
        msgs = [_u("q"), _a_blocks(_text_block("hi"))]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(len(out), 2)

    def test_d7_tool_use_alone_not_treated_as_empty(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _u("q"),
            _a_blocks(_tool_use("shell", id_="A", input_={"command": "x"})),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(len(out), 2)
        self.assertEqual(_ids_in(out), ["A"])

    def test_d8_unknown_block_type_keeps_message_conservatively(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _u("q"),
            _a_blocks({"type": "redacted_thinking", "data": "..."}),
            _u("ok"),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(len(out), 3)

    def test_d9_rule3_disabled(self) -> None:
        snipper, _, _ = _build(_Config(snip_empty_enabled=False))
        msgs = [_u("q"), _a_text(""), _u("more")]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(len(out), 3)


# ---------------------------------------------------------------------------
# Group E — pairing invariants
# ---------------------------------------------------------------------------


class SnipperGroupEPairingTests(unittest.TestCase):
    def test_e1_use_and_result_in_separate_messages_both_dropped(self) -> None:
        """Standard case: tool_use in assistant, tool_result in next user."""
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        # A's tool_use AND its tool_result both removed.
        self.assertEqual(_ids_in(out), ["B"])
        self.assertEqual(_result_ids_in(out), ["B"])

    def test_e2_partial_block_deletion_keeps_text_and_remaining_uses(self) -> None:
        """Assistant has text + 2 tool_uses; Rule 1 deletes 1 → text + other use kept."""
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(
                _text_block("running 2 reads"),
                _tool_use("read_file", id_="A", input_={"path": "/x"}),
                _tool_use("read_file", id_="B", input_={"path": "/y"}),
            ),
            _u_results(_tool_result(id_="A"), _tool_result(id_="B")),
            _a_blocks(_tool_use("read_file", id_="C", input_={"path": "/x"})),
            _u_results(_tool_result(id_="C")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        # A is deleted (Rule 1), B and C survive.
        self.assertEqual(_ids_in(out), ["B", "C"])
        # First user message keeps only B's result.
        first_user = next(m for m in out if m.get("role") == "user")
        result_ids = [
            b["tool_use_id"] for b in first_user["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        self.assertEqual(result_ids, ["B"])
        # Original assistant text is preserved.
        first_a = next(m for m in out if m.get("role") == "assistant")
        text_blocks = [b for b in first_a["content"] if b.get("type") == "text"]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0]["text"], "running 2 reads")

    def test_e3_assistant_with_only_deleted_tool_uses_dropped_whole(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(
                _tool_use("read_file", id_="A", input_={"path": "/x"}),
                _tool_use("read_file", id_="B", input_={"path": "/x"}),
            ),
            _u_results(_tool_result(id_="A"), _tool_result(id_="B")),
            _a_blocks(_tool_use("read_file", id_="C", input_={"path": "/x"})),
            _u_results(_tool_result(id_="C")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        # Both tool_uses in the first assistant deleted (A and B,
        # both pre-final dupes of /x), so the whole assistant message
        # disappears.  The first user (which only carried A's and B's
        # results) likewise disappears.
        roles = [m["role"] for m in out]
        self.assertEqual(roles, ["assistant", "user"])
        self.assertEqual(_ids_in(out), ["C"])

    def test_e4_user_with_only_deleted_tool_results_dropped_whole(self) -> None:
        snipper, _, _ = _build()
        # Assistant with text + 1 tool_use, its result + 1 unrelated text.
        msgs = [
            _a_blocks(_text_block("doing the thing"),
                      _tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        # First user (only carried A's result) is dropped whole.
        users = [m for m in out if m.get("role") == "user"]
        self.assertEqual(len(users), 1)
        self.assertEqual(_result_ids_in(out), ["B"])

    def test_e5_use_and_result_separated_by_unrelated_messages_still_paired(self) -> None:
        """Rule 1 should still remove both halves when separated by text turns."""
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u("interim text"),
            _a_text("interim assistant"),
            _u_results(_tool_result(id_="A")),
            # ... 5 messages ago ...
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["B"])
        self.assertEqual(_result_ids_in(out), ["B"])


# ---------------------------------------------------------------------------
# Group F — multi-rule combinations
# ---------------------------------------------------------------------------


class SnipperGroupFCombinationTests(unittest.TestCase):
    def test_f1_rule1_and_rule2_target_same_pair_dedupe_via_set(self) -> None:
        """Both rules vote to delete the same id → set removes the dup; pair gone exactly once."""
        snipper, _, _ = _build()
        msgs = [
            # Failed read of /x
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A", is_error=True)),
            # Successful read of /x — supersedes A by both Rule 1 (last wins)
            # and Rule 2 (first success after failure).
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B", is_error=False)),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(_ids_in(out), ["B"])
        # No orphaned tool_result of A anywhere.
        self.assertEqual(_result_ids_in(out), ["B"])

    def test_f2_rule3_and_rule1_independent(self) -> None:
        """Rule 3 dropping an empty assistant doesn't affect Rule 1's pairing."""
        snipper, _, _ = _build()
        msgs = [
            _a_text("  "),  # empty → dropped by Rule 3
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        # Empty assistant is gone; Rule 1 still keeps only B.
        roles = [m["role"] for m in out]
        self.assertEqual(roles, ["assistant", "user"])
        self.assertEqual(_ids_in(out), ["B"])

    def test_f3_large_message_list_processes_in_one_pass(self) -> None:
        """100-msg list with 10 dupe groups + 5 failed pairs + 3 empties."""
        snipper, _, _ = _build()
        msgs: list[dict[str, Any]] = []
        # 10 dupe groups of 3 reads each (30 pair messages → 60 msgs of A/U).
        for g in range(10):
            for k in range(3):
                tid = f"d{g}-{k}"
                msgs.append(_a_blocks(_tool_use(
                    "read_file", id_=tid, input_={"path": f"/g{g}.txt"},
                )))
                msgs.append(_u_results(_tool_result(id_=tid)))
        # 5 failed-then-success pairs (20 msgs).
        for f in range(5):
            tid_a = f"f{f}-err"
            tid_b = f"f{f}-ok"
            msgs.append(_a_blocks(_tool_use(
                "shell", id_=tid_a, input_={"command": f"x{f}"},
            )))
            msgs.append(_u_results(_tool_result(id_=tid_a, is_error=True)))
            msgs.append(_a_blocks(_tool_use(
                "shell", id_=tid_b, input_={"command": f"x{f}"},
            )))
            msgs.append(_u_results(_tool_result(id_=tid_b, is_error=False)))
        # 3 empty assistants.
        msgs.extend([_a_text(""), _a_text("  "), {"role": "assistant", "content": None}])

        before = len(msgs)
        out, freed = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        after = len(out)
        # Survived ids per dupe group: 1 each (10 ids); per fail group: 1 each (5 ids).
        surviving = set(_ids_in(out))
        for g in range(10):
            self.assertIn(f"d{g}-2", surviving)
        for f in range(5):
            self.assertIn(f"f{f}-ok", surviving)
        self.assertNotIn(f"d0-0", surviving)
        self.assertNotIn(f"f0-err", surviving)
        # Lots of messages dropped overall.
        self.assertLess(after, before)
        self.assertGreater(freed, 0)


# ---------------------------------------------------------------------------
# Group G — observability
# ---------------------------------------------------------------------------


class SnipperGroupGObservabilityTests(unittest.TestCase):
    def test_g1_counter_records_pairs_plus_empty_messages(self) -> None:
        snipper, _, _ = _build()
        ctx = _ctx()
        msgs = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
            _a_blocks(_tool_use("read_file", id_="C", input_={"path": "/x"})),
            _u_results(_tool_result(id_="C")),
            _a_blocks(_tool_use("read_file", id_="D", input_={"path": "/x"})),
            _u_results(_tool_result(id_="D")),
            _a_text(""),  # 1 empty assistant
        ]
        snipper.maybe_run(msgs, _compaction_ctx(), ctx)
        # 3 dupes deleted (A, B, C — D survives) + 1 empty = 4.
        self.assertEqual(ctx.metadata.get("tier2_snipped_count"), 4)

    def test_g2_counter_accumulates_across_runs(self) -> None:
        """Second invocation adds to the counter rather than overwriting."""
        snipper, _, _ = _build()
        ctx = _ctx()
        first = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        snipper.maybe_run(first, _compaction_ctx(), ctx)
        c1 = int(ctx.metadata.get("tier2_snipped_count", 0))
        second = [
            _a_blocks(_tool_use("shell", id_="C", input_={"command": "x"})),
            _u_results(_tool_result(id_="C", is_error=True)),
            _a_blocks(_tool_use("shell", id_="D", input_={"command": "x"})),
            _u_results(_tool_result(id_="D", is_error=False)),
        ]
        snipper.maybe_run(second, _compaction_ctx(), ctx)
        c2 = int(ctx.metadata.get("tier2_snipped_count", 0))
        self.assertGreater(c2, c1)
        self.assertEqual(c1, 1)  # A deleted
        self.assertEqual(c2, 2)  # A + C deleted

    def test_g3_freed_matches_estimator_delta(self) -> None:
        from aether.services.compact.token_estimation import estimate_messages_tokens

        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use(
                "read_file", id_="A",
                input_={"path": "/x", "padding": "x" * 4_000},
            )),
            _u_results(_tool_result(id_="A", content="x" * 4_000)),
            _a_blocks(_tool_use(
                "read_file", id_="B",
                input_={"path": "/x", "padding": "x" * 4_000},
            )),
            _u_results(_tool_result(id_="B", content="x" * 4_000)),
        ]
        before = estimate_messages_tokens(msgs)
        out, freed = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        after = estimate_messages_tokens(out)
        self.assertEqual(freed, max(0, before - after))
        self.assertGreater(freed, 0)

    def test_g4_log_line_carries_diagnostic_fields(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        with self.assertLogs("test.snipper", level="INFO") as recorded:
            snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        blob = "\n".join(recorded.output)
        self.assertIn("tier2:", blob)
        self.assertIn("snipped", blob)
        self.assertIn("pairs", blob)
        self.assertIn("freed", blob)


# ---------------------------------------------------------------------------
# Group H — pipeline-visible signals (no orphans, freed accurate)
# ---------------------------------------------------------------------------


class SnipperGroupHPipelineSignalTests(unittest.TestCase):
    def test_h1_no_orphan_tool_results_after_aggressive_dedup(self) -> None:
        """Every surviving tool_result has a matching surviving tool_use."""
        snipper, _, _ = _build()
        msgs = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
            _a_blocks(_tool_use("read_file", id_="C", input_={"path": "/x"})),
            _u_results(_tool_result(id_="C")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        use_ids = set(_ids_in(out))
        result_ids = set(_result_ids_in(out))
        # Surviving result ids subset of surviving use ids (and vice versa).
        self.assertEqual(use_ids, result_ids)

    def test_h2_idempotent_second_run_makes_no_more_changes(self) -> None:
        """Run once → run again → second run sees clean input → no extra freed."""
        snipper, _, _ = _build()
        ctx = _ctx()
        msgs = [
            _a_blocks(_tool_use("read_file", id_="A", input_={"path": "/x"})),
            _u_results(_tool_result(id_="A")),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        out1, freed1 = snipper.maybe_run(msgs, _compaction_ctx(), ctx)
        out2, freed2 = snipper.maybe_run(out1, _compaction_ctx(), ctx)
        self.assertGreater(freed1, 0)
        self.assertEqual(freed2, 0)
        # Same content the second time around.
        self.assertEqual(out2, out1)


# ---------------------------------------------------------------------------
# Group I — boundary / fault tolerance
# ---------------------------------------------------------------------------


class SnipperGroupIBoundaryTests(unittest.TestCase):
    def test_i1_tool_use_without_id_is_skipped_silently(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _a_blocks({"type": "tool_use", "name": "read_file", "input": {"path": "/x"}}),
            _a_blocks(_tool_use("read_file", id_="B", input_={"path": "/x"})),
            _u_results(_tool_result(id_="B")),
        ]
        # Should not crash; the malformed block is left as-is.
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertIn("B", _ids_in(out))

    def test_i2_string_content_message_is_passed_through(self) -> None:
        snipper, _, _ = _build()
        msgs = [
            _u("hello"),  # content is str, not list
            _a_text("world"),
            _u("ok"),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        self.assertEqual(out, msgs)

    def test_i3_non_json_input_falls_back_safely(self) -> None:
        """Non-JSON-friendly input (e.g. a set) → key falls back to str()."""
        snipper, _, _ = _build()
        # A set is not JSON-encodable; the helper must not crash.
        msgs = [
            _a_blocks({
                "type": "tool_use",
                "id": "A",
                "name": "read_file",
                "input": {"weird": {"a", "b"}},
            }),
            _u_results(_tool_result(id_="A")),
            _a_blocks({
                "type": "tool_use",
                "id": "B",
                "name": "read_file",
                "input": {"weird": {"a", "b"}},
            }),
            _u_results(_tool_result(id_="B")),
        ]
        out, _ = snipper.maybe_run(msgs, _compaction_ctx(), _ctx())
        # Whatever happens, no exception and at least B survives.
        self.assertIn("B", _ids_in(out))


# ---------------------------------------------------------------------------
# _SnipPlan smoke (helps coverage of the dataclass surface)
# ---------------------------------------------------------------------------


class SnipPlanSmokeTests(unittest.TestCase):
    def test_has_work_false_when_empty(self) -> None:
        self.assertFalse(_SnipPlan().has_work)

    def test_has_work_true_when_either_set_populated(self) -> None:
        p = _SnipPlan(delete_tool_use_ids={"A"})
        self.assertTrue(p.has_work)
        p2 = _SnipPlan(delete_msg_indices={3})
        self.assertTrue(p2.has_work)


# ---------------------------------------------------------------------------
# Module constant pin
# ---------------------------------------------------------------------------


class SnipDupeRuleToolsPinTests(unittest.TestCase):
    def test_whitelist_is_read_only_tools_only(self) -> None:
        """Operators rely on shell/write_file NOT being in the dupe whitelist."""
        self.assertNotIn("shell", SNIP_DUPE_RULE_TOOLS)
        self.assertNotIn("write_file", SNIP_DUPE_RULE_TOOLS)
        self.assertIn("read_file", SNIP_DUPE_RULE_TOOLS)
        self.assertIn("list_dir", SNIP_DUPE_RULE_TOOLS)
        self.assertIn("glob", SNIP_DUPE_RULE_TOOLS)
        self.assertIn("grep", SNIP_DUPE_RULE_TOOLS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
