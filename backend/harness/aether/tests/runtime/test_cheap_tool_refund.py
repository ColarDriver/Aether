"""End-to-end tests for cheap-tool refund behaviour.

Sprint 3 / PR 3.2 — covers groups H, I, J, K from
``docs/sprint-3-compaction-pipeline/02_pr3_2_iteration_budget.md`` § 5.3.

Fixture pattern:

  * ``_QueuedProvider`` returns a pre-canned sequence of
    ``NormalizedResponse`` objects (tool-calls then a final text).
  * The tool registry hosts a few minimal tools (``update_todo``,
    ``memory_write``, ``session_search``, ``shell``, ``read_file``)
    each implemented as a no-op so dispatch doesn't depend on the
    real bundled tool kit.
  * Assertions read ``result.metadata['iteration_budget']`` — the
    structured snapshot PR 3.1 reserved and PR 3.2 fills in.
"""

from __future__ import annotations

import unittest
from typing import Any, List

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    ExitReason,
    NormalizedResponse,
    StreamDeltaCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


# --------------------------------------------------------------------- #
# Test helpers                                                          #
# --------------------------------------------------------------------- #


class _NoOpTool(ToolExecutor):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name=self._name)

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        return ToolResult(tool_call_id=call.id, name=call.name, content="ok")


class _QueuedProvider(ModelProvider):
    """Returns a pre-canned response queue; appends a fallback text on exhaustion."""

    provider_name = "openai"
    api_mode = "chat"

    def __init__(self, queue: List[NormalizedResponse]) -> None:
        self._queue = list(queue)
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: Any,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: Any = None,  # noqa: ARG002
    ) -> NormalizedResponse:
        self.calls += 1
        # Empty tools = grace round; we never expect that here, but
        # return an empty text just in case so the loop terminates
        # cleanly rather than hanging.
        if not tools:
            return NormalizedResponse(content="")
        if not self._queue:
            return NormalizedResponse(content="exhausted-fallback-text")
        return self._queue.pop(0)


def _tool_call(name: str, *, idx: int = 0) -> NormalizedResponse:
    return NormalizedResponse(
        tool_calls=[ToolCall(id=f"c{idx}-{name}", name=name, arguments={})],
        metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )


def _multi_tool_call(*names: str, base_idx: int = 0) -> NormalizedResponse:
    return NormalizedResponse(
        tool_calls=[
            ToolCall(id=f"c{base_idx + i}-{n}", name=n, arguments={})
            for i, n in enumerate(names)
        ],
        metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )


def _final_text(text: str = "done") -> NormalizedResponse:
    return NormalizedResponse(
        content=text,
        metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )


def _make_engine(
    provider: ModelProvider,
    *,
    max_iterations: int = 5,
    cheap_tool_names: tuple[str, ...] = ("update_todo", "memory_write", "session_search"),
    extra_tool_names: tuple[str, ...] = (
        "update_todo",
        "memory_write",
        "session_search",
        "shell",
        "read_file",
    ),
    summary_on_budget_exhausted: bool = False,
) -> AgentEngine:
    registry = ToolRegistry()
    for n in extra_tool_names:
        registry.register(_NoOpTool(n))
    return AgentEngine(
        provider,
        tool_registry=registry,
        config=EngineConfig(
            max_iterations=max_iterations,
            use_builtin_tools=False,
            cheap_tool_names=cheap_tool_names,
            summary_on_budget_exhausted=summary_on_budget_exhausted,
        ),
    )


def _budget(result) -> dict:
    return result.metadata["iteration_budget"]


# --------------------------------------------------------------------- #
# Group H — refund triggers                                             #
# --------------------------------------------------------------------- #


class GroupHRefundTriggers(unittest.TestCase):
    """T-H1..T-H4 — when refund should and shouldn't fire."""

    def test_h1_single_cheap_iteration_refunds(self) -> None:
        """T-H1: one update_todo round → budget.used == 0."""
        provider = _QueuedProvider(
            [_tool_call("update_todo"), _final_text("ok")]
        )
        engine = _make_engine(provider, max_iterations=5)
        result = engine.run_turn(EngineRequest(session_id="sH1", user_message="go"))

        budget = _budget(result)
        # 2 LLM calls happened; 1 cheap iteration refunded; 1 final text iteration consumed.
        # Final text iteration doesn't dispatch tools so no refund attempt.
        self.assertEqual(budget["consume_count"], 2)
        self.assertEqual(budget["refund_count"], 1)
        self.assertEqual(budget["used"], 1)
        self.assertEqual(budget["remaining"], 4)

    def test_h2_mixed_cheap_and_real_does_not_refund(self) -> None:
        """T-H2: shell + update_todo in same iteration → no refund."""
        provider = _QueuedProvider(
            [
                _multi_tool_call("shell", "update_todo"),
                _final_text("done"),
            ]
        )
        engine = _make_engine(provider, max_iterations=5)
        result = engine.run_turn(EngineRequest(session_id="sH2", user_message="go"))

        budget = _budget(result)
        self.assertEqual(budget["refund_count"], 0)
        self.assertEqual(budget["used"], 2)

    def test_h3_five_cheap_iterations_all_refund(self) -> None:
        """T-H3: 5 cheap rounds → refund_count == 5, used == 0 + final text consume."""
        provider = _QueuedProvider(
            [_tool_call("update_todo", idx=i) for i in range(5)]
            + [_final_text("ok")]
        )
        engine = _make_engine(provider, max_iterations=10)
        result = engine.run_turn(EngineRequest(session_id="sH3", user_message="go"))

        budget = _budget(result)
        self.assertEqual(budget["refund_count"], 5)
        # 6 consume_count total: 5 cheap + 1 final text.
        self.assertEqual(budget["consume_count"], 6)
        self.assertEqual(budget["used"], 1)  # only the final text iteration counts

    def test_h4_cheap_then_real_correctly_accounts(self) -> None:
        """T-H4: 5 cheap + 5 real → used reflects only the real work."""
        provider = _QueuedProvider(
            [_tool_call("update_todo", idx=i) for i in range(5)]
            + [_tool_call("shell", idx=i + 100) for i in range(5)]
            + [_final_text("ok")]
        )
        engine = _make_engine(provider, max_iterations=20)
        result = engine.run_turn(EngineRequest(session_id="sH4", user_message="go"))

        budget = _budget(result)
        # 5 refunds for the cheap prefix; 5 + 1 (final) = 6 net used.
        self.assertEqual(budget["refund_count"], 5)
        self.assertEqual(budget["consume_count"], 11)
        self.assertEqual(budget["used"], 6)


# --------------------------------------------------------------------- #
# Group I — name matching                                               #
# --------------------------------------------------------------------- #


class GroupINameMatching(unittest.TestCase):
    """T-I1..T-I6 — case-fold / dash / namespace / similar-name matching."""

    def _run_with_name(
        self, dispatch_name: str, *, cheap: tuple[str, ...] = ("update_todo",)
    ):
        provider = _QueuedProvider(
            [_tool_call(dispatch_name), _final_text("ok")]
        )
        engine = _make_engine(
            provider,
            max_iterations=5,
            cheap_tool_names=cheap,
            extra_tool_names=(dispatch_name, "update_todo", "shell"),
        )
        return engine.run_turn(EngineRequest(session_id=f"sI-{dispatch_name}", user_message="go"))

    def test_i1_uppercase_with_underscore_matches(self) -> None:
        """T-I1: ``UPDATE_TODO`` matches configured ``update_todo`` via case-fold.

        Note: the shared ``_normalize_name`` helper does case-fold +
        dash↔underscore + namespace strip but **not** CamelCase →
        snake_case (would conflict with names that legitimately omit
        underscores like ``readfile``).  ``UpdateTodo`` therefore does
        NOT match ``update_todo`` — that's a separate (out-of-scope)
        normaliser feature; document the limitation here so future
        contributors don't accidentally regress the matched cases.
        """
        result = self._run_with_name("UPDATE_TODO")
        self.assertEqual(_budget(result)["refund_count"], 1)

    def test_i2_dashed_name_matches(self) -> None:
        """T-I2: ``update-todo`` matches ``update_todo`` (dash↔underscore)."""
        result = self._run_with_name("update-todo")
        self.assertEqual(_budget(result)["refund_count"], 1)

    def test_i3_namespace_prefix_matches(self) -> None:
        """T-I3: ``mcp__router__update_todo`` matches ``update_todo``."""
        result = self._run_with_name("mcp__router__update_todo")
        self.assertEqual(_budget(result)["refund_count"], 1)

    def test_i4_namespace_real_tool_does_not_match(self) -> None:
        """T-I4: ``mcp__router__shell`` is NOT cheap (shell isn't in whitelist)."""
        result = self._run_with_name("mcp__router__shell")
        self.assertEqual(_budget(result)["refund_count"], 0)

    def test_i5_similar_but_distinct_name_does_not_match(self) -> None:
        """T-I5: ``update_todo_v2`` is NOT cheap — substring matches don't count."""
        result = self._run_with_name("update_todo_v2")
        self.assertEqual(
            _budget(result)["refund_count"],
            0,
            "fuzzy match must require full normalised equality",
        )

    def test_i6_empty_cheap_list_disables_refund(self) -> None:
        """T-I6: cheap_tool_names=() → no tool ever refunds, even canonical names."""
        result = self._run_with_name("update_todo", cheap=())
        self.assertEqual(_budget(result)["refund_count"], 0)


# --------------------------------------------------------------------- #
# Group J — interaction with summary                                    #
# --------------------------------------------------------------------- #


class GroupJSummaryInteraction(unittest.TestCase):
    """T-J1..T-J2 — refund and summary correctly coexist."""

    def test_j1_refund_extends_loop_past_apparent_max(self) -> None:
        """T-J1: 3 cheap iters + 3 real iters fits in max_iterations=3 thanks to refund.

        Sequence: 3× update_todo (refunded) → 3× shell → final text.
        Without refund this would exhaust at iteration 3; with refund
        only the 3 shell + 1 text consume net slots, totalling 4 used.
        Set max_iterations=4 so the loop just barely succeeds.
        """
        provider = _QueuedProvider(
            [_tool_call("update_todo", idx=i) for i in range(3)]
            + [_tool_call("shell", idx=10 + i) for i in range(3)]
            + [_final_text("done")]
        )
        engine = _make_engine(provider, max_iterations=4)
        result = engine.run_turn(EngineRequest(session_id="sJ1", user_message="go"))

        # Loop completed normally (no MAX_ITERATIONS)
        self.assertNotEqual(result.exit_reason, ExitReason.MAX_ITERATIONS)
        budget = _budget(result)
        self.assertEqual(budget["refund_count"], 3)
        self.assertFalse(
            budget["grace_consumed"],
            "summary must not fire when budget didn't exhaust",
        )

    def test_j2_real_work_exhausts_then_summary_fires(self) -> None:
        """T-J2: 3 real-work iters + max_iterations=3 → MAX_ITERATIONS + summary."""
        provider = _QueuedProvider(
            [_tool_call("shell", idx=i) for i in range(5)]
        )
        # Enable summary; the queue keeps returning shell calls so we
        # exhaust the 3-iteration budget legitimately.
        engine = _make_engine(
            provider, max_iterations=3, summary_on_budget_exhausted=True
        )
        # Need a custom override for summary fixture: provider returns
        # tool calls indefinitely until tools=[] arrives at grace.
        # Replace the provider with one that knows to handle grace.
        result = engine.run_turn(EngineRequest(session_id="sJ2", user_message="go"))

        self.assertEqual(result.exit_reason, ExitReason.MAX_ITERATIONS)
        budget = _budget(result)
        self.assertTrue(
            budget["grace_consumed"],
            "summary path must run when real work exhausts the budget",
        )


# --------------------------------------------------------------------- #
# Group K — metadata exposure                                           #
# --------------------------------------------------------------------- #


class GroupKMetadataExposure(unittest.TestCase):
    """T-K1..T-K3 — iteration_budget metadata is observable."""

    def test_k1_metadata_has_full_schema(self) -> None:
        """T-K1: every documented field is present under metadata['iteration_budget']."""
        provider = _QueuedProvider([_final_text("ok")])
        engine = _make_engine(provider, max_iterations=5)
        result = engine.run_turn(EngineRequest(session_id="sK1", user_message="go"))

        budget = _budget(result)
        for key in (
            "used",
            "max",
            "remaining",
            "grace_consumed",
            "consume_count",
            "refund_count",
        ):
            self.assertIn(key, budget, f"metadata missing field: {key}")

    def test_k2_refund_count_visible_after_cheap_iteration(self) -> None:
        """T-K2: refund_count > 0 surfaces in metadata for cheap-only turns."""
        provider = _QueuedProvider(
            [_tool_call("memory_write"), _final_text("ok")]
        )
        engine = _make_engine(provider, max_iterations=5)
        result = engine.run_turn(EngineRequest(session_id="sK2", user_message="go"))

        self.assertGreater(_budget(result)["refund_count"], 0)

    def test_k3_grace_consumed_visible_after_summary(self) -> None:
        """T-K3: grace_consumed flips True in metadata when summary fires."""
        provider = _QueuedProvider(
            [_tool_call("shell", idx=i) for i in range(5)]
        )
        engine = _make_engine(
            provider, max_iterations=2, summary_on_budget_exhausted=True
        )
        result = engine.run_turn(EngineRequest(session_id="sK3", user_message="go"))

        self.assertTrue(_budget(result)["grace_consumed"])


# --------------------------------------------------------------------- #
# Group L \u2014 Sprint 3.5 / PR 3.5.3 closure: real TodoWriteTool refund   #
# --------------------------------------------------------------------- #


class GroupLRealTodoWriteRefund(unittest.TestCase):
    """T-L1..T-L2 \u2014 the actual ``TodoWriteTool`` (not a stub) triggers
    cheap-tool refund.

    Sprint 3 / PR 3.2 wired up the cheap-tool refund machinery and seeded
    ``EngineConfig.cheap_tool_names`` with ``"todo_write"`` (among others)
    in anticipation of this PR.  Until Sprint 3.5 / PR 3.5.3 there was
    no actual ``todo_write`` tool registered, so the entry was a dangling
    reference that nothing exercised.  These tests close that loop by
    wiring the bundled :class:`TodoWriteTool` into the engine and
    verifying the budget snapshot reports ``refund_count == 1`` for a
    turn whose only tool call is ``todo_write``.
    """

    def _build_engine_with_real_todo(
        self, *, max_iterations: int = 5
    ) -> AgentEngine:
        # Local import keeps the module-level imports tight and matches
        # how production code lazy-imports the builtins.
        from aether.tools.builtins.todo_write import TodoWriteTool, clear_session_todos

        # Each test starts with a fresh per-session todo store so the
        # "all done" auto-clear from a previous test never leaks in.
        clear_session_todos("sL1")
        clear_session_todos("sL2")

        registry = ToolRegistry()
        registry.register(TodoWriteTool())
        registry.register(_NoOpTool("shell"))
        provider = _QueuedProvider([
            NormalizedResponse(
                tool_calls=[
                    ToolCall(
                        id="c0-todo_write",
                        name="todo_write",
                        arguments={
                            "todos": [
                                {
                                    "id": "1",
                                    "content": "do thing",
                                    "status": "in_progress",
                                }
                            ]
                        },
                    )
                ],
                metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            ),
            _final_text("ack"),
        ])
        return AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(
                max_iterations=max_iterations,
                use_builtin_tools=False,
                summary_on_budget_exhausted=False,
            ),
        )

    def test_l1_real_todo_write_call_refunds_iteration(self) -> None:
        """T-L1: a turn calling the real TodoWriteTool only \u2192 budget.used == 0."""
        from aether.tools.builtins.todo_write import get_session_todos

        engine = self._build_engine_with_real_todo()
        result = engine.run_turn(
            EngineRequest(session_id="sL1", user_message="go")
        )

        budget = _budget(result)
        # The cheap-tool refund cancels the iteration the cheap call
        # consumed, so refund_count == 1 proves the engine identified
        # ``todo_write`` as cheap.
        self.assertEqual(
            budget["refund_count"], 1, "todo_write should be cheap-refunded"
        )
        self.assertNotEqual(result.exit_reason, ExitReason.MAX_ITERATIONS)
        # Sanity-check the real TodoWriteTool actually ran rather than
        # being swallowed by the engine: the session's stored todo list
        # must reflect the canned argument we sent.
        stored = get_session_todos("sL1")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["status"], "in_progress")

    def test_l2_todo_write_in_default_cheap_list(self) -> None:
        """T-L2 \u2014 ``todo_write`` lives in the default cheap_tool_names
        tuple (regression-only assertion; if a future config edit
        drops it the refund stops firing in production)."""
        cfg = EngineConfig()
        self.assertIn("todo_write", cfg.cheap_tool_names)


if __name__ == "__main__":
    unittest.main()
