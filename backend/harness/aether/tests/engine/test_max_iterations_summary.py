"""End-to-end tests for the max-iterations summary fallback.

Sprint 3 / PR 3.2 — covers groups E, F, G from
``docs/sprint-3-compaction-pipeline/02_pr3_2_iteration_budget.md`` § 5.2.

The fixture pattern is consistent across cases:

  * ``_TwoPhaseProvider`` returns tool-only responses while ``tools`` is
    non-empty (the regular loop path), then switches to a scripted
    summary text response the first time the engine calls
    ``generate(tools=[])`` (the grace path).
  * A no-op ``noop`` tool keeps the dispatch loop alive without
    side effects.
  * ``EngineConfig.max_iterations`` is set deliberately small (1-3)
    so the budget exhausts after a handful of LLM calls.
"""

from __future__ import annotations

import unittest
from typing import Any, List

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    StreamDeltaCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.provider_errors import ProviderInvocationError
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


# --------------------------------------------------------------------- #
# Test helpers                                                          #
# --------------------------------------------------------------------- #


class _NoOpTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="noop")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        return ToolResult(tool_call_id=call.id, name=call.name, content="ok")


class _TwoPhaseProvider(ModelProvider):
    """Returns tool-call responses on the loop path, summary text on grace.

    Detects the grace path via ``len(tools) == 0`` (the engine's
    ``_handle_max_iterations`` always issues with empty tool list).
    The recorded ``last_summary_messages`` and ``last_summary_tools``
    fields are inspected by group F to verify the prompt shape.
    """

    provider_name = "openai"
    api_mode = "chat"

    def __init__(
        self,
        *,
        loop_response: NormalizedResponse | None = None,
        summary_response: NormalizedResponse | None = None,
        grace_error: Exception | None = None,
    ) -> None:
        self._loop_response = loop_response or NormalizedResponse(
            tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
            metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )
        self._summary_response = summary_response
        self._grace_error = grace_error
        self.calls = 0
        self.grace_calls = 0
        self.last_summary_messages: list[dict] | None = None
        self.last_summary_tools: list[Any] | None = None

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
        # Detect grace round (engine always passes tools=[] for summary)
        if not tools:
            self.grace_calls += 1
            self.last_summary_messages = list(messages)
            self.last_summary_tools = list(tools)
            if self._grace_error is not None:
                raise self._grace_error
            return self._summary_response or NormalizedResponse(content="")
        # Regular loop path — keep firing tool calls to drain budget.
        return NormalizedResponse(
            tool_calls=list(self._loop_response.tool_calls),
            metadata=dict(self._loop_response.metadata),
        )


def _make_engine(
    provider: ModelProvider,
    *,
    max_iterations: int = 3,
    summary_on_budget_exhausted: bool = True,
) -> AgentEngine:
    registry = ToolRegistry()
    registry.register(_NoOpTool())
    return AgentEngine(
        provider,
        tool_registry=registry,
        config=EngineConfig(
            max_iterations=max_iterations,
            use_builtin_tools=False,
            summary_on_budget_exhausted=summary_on_budget_exhausted,
            cheap_tool_names=(),  # disable refund so noop counts as real work
        ),
    )


# --------------------------------------------------------------------- #
# Group E — basic generation                                            #
# --------------------------------------------------------------------- #


class GroupEBasicGeneration(unittest.TestCase):
    """T-E1..T-E4 — summary text replaces empty final_response."""

    def test_e1_summary_replaces_empty_final_response(self) -> None:
        """T-E1: budget exhausts, grace returns text, final_response = text."""
        provider = _TwoPhaseProvider(
            summary_response=NormalizedResponse(content="Summary text"),
        )
        engine = _make_engine(provider, max_iterations=3)
        result = engine.run_turn(EngineRequest(session_id="sE1", user_message="go"))

        self.assertEqual(result.exit_reason, ExitReason.MAX_ITERATIONS)
        self.assertEqual(result.status, EngineStatus.MAX_ITERATIONS)
        self.assertEqual(result.final_response, "Summary text")
        self.assertTrue(result.metadata["iteration_budget"]["grace_consumed"])
        self.assertEqual(provider.grace_calls, 1)

    def test_e2_summary_disabled_returns_empty(self) -> None:
        """T-E2: rollback switch — empty final_response, no grace call."""
        provider = _TwoPhaseProvider(
            summary_response=NormalizedResponse(content="Summary text"),
        )
        engine = _make_engine(
            provider,
            max_iterations=3,
            summary_on_budget_exhausted=False,
        )
        result = engine.run_turn(EngineRequest(session_id="sE2", user_message="go"))

        self.assertEqual(result.exit_reason, ExitReason.MAX_ITERATIONS)
        # Pre-PR-3.2 behaviour preserved: empty/None final_response on exhaust.
        self.assertFalse(
            result.final_response,
            f"expected falsy final_response, got {result.final_response!r}",
        )
        self.assertEqual(provider.grace_calls, 0)
        self.assertFalse(
            result.metadata["iteration_budget"]["grace_consumed"],
            "grace must remain unconsumed when feature is disabled",
        )

    def test_e3_grace_provider_error_does_not_crash(self) -> None:
        """T-E3: provider raises during summary → empty final_response, no crash."""
        provider = _TwoPhaseProvider(
            grace_error=ProviderInvocationError(
                status_code=500,
                body_summary="upstream broke",
            ),
        )
        engine = _make_engine(provider, max_iterations=2)
        result = engine.run_turn(EngineRequest(session_id="sE3", user_message="go"))

        self.assertEqual(result.exit_reason, ExitReason.MAX_ITERATIONS)
        self.assertFalse(
            result.final_response,
            f"failed grace must leave final_response falsy, got {result.final_response!r}",
        )
        self.assertEqual(provider.grace_calls, 1)
        # grace was attempted (consumed) even though it failed — that
        # flag must be True so observability reflects "we tried".
        self.assertTrue(result.metadata["iteration_budget"]["grace_consumed"])

    def test_e4_grace_returns_empty_string(self) -> None:
        """T-E4: provider returns empty content → final_response stays empty."""
        provider = _TwoPhaseProvider(
            summary_response=NormalizedResponse(content=""),
        )
        engine = _make_engine(provider, max_iterations=2)
        result = engine.run_turn(EngineRequest(session_id="sE4", user_message="go"))

        self.assertFalse(
            result.final_response,
            f"empty grace must leave final_response falsy, got {result.final_response!r}",
        )
        self.assertEqual(provider.grace_calls, 1)
        self.assertNotIn(
            "summary_provided",
            _filter_metadata(result.metadata.get("turn", {})),
        )


# --------------------------------------------------------------------- #
# Group F — prompt content                                              #
# --------------------------------------------------------------------- #


class GroupFPromptContent(unittest.TestCase):
    """T-F1..T-F3 — verify the grace prompt shape."""

    def _run_to_grace(self) -> _TwoPhaseProvider:
        provider = _TwoPhaseProvider(
            summary_response=NormalizedResponse(content="Done."),
        )
        engine = _make_engine(provider, max_iterations=2)
        engine.run_turn(EngineRequest(session_id="sF", user_message="initial-user-msg"))
        self.assertEqual(provider.grace_calls, 1, "fixture must trigger grace")
        return provider

    def test_f1_summary_prompt_appends_system_nudge(self) -> None:
        """T-F1: last user message of grace prompt asks for the summary."""
        provider = self._run_to_grace()
        msgs = provider.last_summary_messages
        self.assertIsNotNone(msgs)
        self.assertGreaterEqual(len(msgs), 2, "grace prompt must include history + nudge")
        last = msgs[-1]
        self.assertEqual(last["role"], "user")
        self.assertIn("[System: You've used your iteration budget", last["content"])
        self.assertIn("What you accomplished", last["content"])
        self.assertIn("What remains to be done", last["content"])

    def test_f2_summary_call_passes_empty_tools(self) -> None:
        """T-F2: tools=[] forces a pure text response (no further tool churn)."""
        provider = self._run_to_grace()
        self.assertEqual(provider.last_summary_tools, [])

    def test_f3_summary_prompt_preserves_full_history(self) -> None:
        """T-F3: the prior conversation (system + user + assistant turns) survives."""
        provider = self._run_to_grace()
        msgs = provider.last_summary_messages
        # The grace prompt is "everything we had + 1 new user nudge",
        # so initial user message must still be in there.
        roles = [m.get("role") for m in msgs]
        self.assertIn("user", roles)
        # And the original user content was "initial-user-msg" — assert
        # it survived end-to-end into the grace round.
        self.assertTrue(
            any(
                isinstance(m.get("content"), str)
                and "initial-user-msg" in m["content"]
                for m in msgs
            ),
            f"original user prompt missing from grace history; messages={msgs!r}",
        )


# --------------------------------------------------------------------- #
# Group G — grace one-shot contract                                     #
# --------------------------------------------------------------------- #


class GroupGGraceOneShot(unittest.TestCase):
    """T-G1 — verify grace is one-shot per turn."""

    def test_g1_second_handle_call_in_same_turn_is_noop(self) -> None:
        """T-G1: calling _handle_max_iterations twice yields one provider call.

        Defensive — the engine should never double-call this helper, but
        the IterationBudget grace_call() guard must absorb a duplicate
        without firing the provider a second time.
        """
        provider = _TwoPhaseProvider(
            summary_response=NormalizedResponse(content="Summary v1"),
        )
        engine = _make_engine(provider, max_iterations=2)
        request = EngineRequest(session_id="sG1", user_message="go")
        # Run a normal turn — grace fires once.
        result = engine.run_turn(request)
        self.assertEqual(provider.grace_calls, 1)
        self.assertEqual(result.final_response, "Summary v1")

        # Second invocation in the SAME turn (synthesised manually) must be no-op.
        # We need access to the budget for this — ``_handle_max_iterations``
        # reads ``context.metadata['_iteration_budget_obj']`` so we craft a
        # context carrying an already-grace-consumed budget.
        from aether.runtime.iteration_budget import IterationBudget

        spent_budget = IterationBudget(max_total=2, used=2)
        spent_budget.grace_call()  # consume the one-shot

        ctx = TurnContext(session_id="sG1-synth", iteration=0)
        ctx.metadata["_iteration_budget_obj"] = spent_budget
        before = provider.grace_calls
        result_text = engine._handle_max_iterations(request, [], ctx)
        self.assertIsNone(
            result_text,
            "second grace call must return None (no second provider hit)",
        )
        self.assertEqual(
            provider.grace_calls,
            before,
            "second grace call must not invoke the provider",
        )


def _filter_metadata(d: Any) -> dict:
    """Tiny helper — guard tests against context.metadata being a non-dict."""
    return d if isinstance(d, dict) else {}


if __name__ == "__main__":
    unittest.main()
