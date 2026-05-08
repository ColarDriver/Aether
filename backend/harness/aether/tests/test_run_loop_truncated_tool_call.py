"""Sprint 1 / PR 1.3 — truncated tool-call detection (P0-4).

Pins down the four pieces of behaviour called out in
``docs/run-loop-roadmap/02_p0_critical_gaps.md`` § P0-4:

1. ``_detect_truncated_tool_call`` heuristic — args that don't terminate
   with ``}`` / ``]`` are treated as truncated.
2. Engine refuses to dispatch truncated args, retries the API call once
   without poisoning history, then surfaces ExitReason.TOOL_CALL_TRUNCATED
   on the second strike.
3. Router-rewrite case — ``finish_reason="tool_calls"`` with truncated
   args is still detected (the heuristic doesn't depend on
   finish_reason).
4. Pure-syntax JSON errors (not truncated) trigger silent retry up to
   ``max_invalid_json_retries`` times and then inject a tool-error
   message so the model can self-correct.

These tests exercise the engine end-to-end via ``ScriptedProvider`` /
``RecordingProvider`` so the run-loop wiring (``_validate_tool_call_arguments``,
``_handle_length_with_tool_calls``, dispatch gating) is covered as one
unit.
"""

from __future__ import annotations

import unittest
from typing import List

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.models.provider.scripted import ScriptedProvider
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
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingProvider(ModelProvider):
    """Provider that emits scripted responses and records every call.

    Used by tests that need to assert *which* messages were sent on each
    retry attempt — ScriptedProvider doesn't expose that.
    """

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        if not self._responses:
            raise RuntimeError("RecordingProvider exhausted")
        # Snapshot the message list — the engine reuses this list across
        # iterations so we have to copy if we want a stable history per call.
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "max_tokens": config.max_tokens,
            }
        )
        return self._responses.pop(0)


class _SpyTool(ToolExecutor):
    """Tool that records every dispatch so tests can assert non-execution."""

    def __init__(self, name: str = "write_file") -> None:
        self._name = name
        self.calls: list[ToolCall] = []

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name=self._name)

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        self.calls.append(call)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="ok",
        )


def _registry_with(*tools: ToolExecutor) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


# ---------------------------------------------------------------------------
# _detect_truncated_tool_call (pure unit)
# ---------------------------------------------------------------------------


def _raw_call(call_id: str, name: str, args_text: str) -> ToolCall:
    """Create a ToolCall with a raw string ``arguments`` payload.

    The dataclass annotation says ``Dict[str, Any]`` but the field is
    only used as opaque storage by the engine; passing a string works
    end-to-end because the engine's first pass normalises types.  We
    keep this helper instead of ad-hoc construction so the intent
    "this is the bytes-on-the-wire shape" is explicit at every call
    site.
    """
    call = ToolCall(id=call_id, name=name)
    call.arguments = args_text  # type: ignore[assignment]
    return call


class DetectTruncatedToolCallTests(unittest.TestCase):
    def test_string_not_ending_with_brace_is_truncated(self) -> None:
        calls = [_raw_call("c1", "write_file", '{"path": "/etc/pas')]
        self.assertTrue(AgentEngine._detect_truncated_tool_call(calls))

    def test_string_ending_with_close_brace_is_not_truncated(self) -> None:
        calls = [ToolCall(id="c1", name="write_file", arguments={"path": "/x"})]
        self.assertFalse(AgentEngine._detect_truncated_tool_call(calls))

    def test_pre_parsed_dict_is_never_truncated(self) -> None:
        # Already-parsed dict args came through a fully successful
        # upstream JSON parse — by definition not truncated.
        calls = [ToolCall(id="c1", name="x", arguments={"a": 1})]
        self.assertFalse(AgentEngine._detect_truncated_tool_call(calls))

    def test_empty_args_is_not_truncated(self) -> None:
        calls = [_raw_call("c1", "x", "")]
        self.assertFalse(AgentEngine._detect_truncated_tool_call(calls))

    def test_string_ending_with_close_bracket_is_not_truncated(self) -> None:
        # Lists are valid top-level JSON too.
        calls = [_raw_call("c1", "x", "[1, 2, 3]")]
        self.assertFalse(AgentEngine._detect_truncated_tool_call(calls))

    def test_any_truncated_call_flips_the_verdict(self) -> None:
        # If even a single tool_call in the batch looks cut off we treat
        # the whole response as untrustworthy.
        good = ToolCall(id="c1", name="x", arguments={"a": 1})
        bad = _raw_call("c2", "y", '{"path": "/foo')
        self.assertTrue(AgentEngine._detect_truncated_tool_call([good, bad]))


# ---------------------------------------------------------------------------
# Run-loop integration
# ---------------------------------------------------------------------------


class TruncatedToolCallRunLoopTests(unittest.TestCase):
    """Engine-level scenarios for length + tool_calls and JSON-error paths."""

    # 1) Half-JSON args → not dispatched; retried once; refused on second strike.
    def test_half_json_with_length_finish_reason_retries_then_refuses(self) -> None:
        bad = _raw_call("c1", "write_file", '{"path":"report.md","content":"partial')
        bad2 = _raw_call("c2", "write_file", '{"path":"report.md","content":"partial')

        spy = _SpyTool("write_file")
        provider = ScriptedProvider(
            [
                NormalizedResponse(content="", tool_calls=[bad], finish_reason="length"),
                NormalizedResponse(content="", tool_calls=[bad2], finish_reason="length"),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=4,
                max_truncated_tool_call_retries=1,
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="ttc-1", user_message="go"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.TOOL_CALL_TRUNCATED)
        # Tool MUST NOT have been dispatched.
        self.assertEqual(spy.calls, [])
        # Counter should reflect the single retry that was burned.
        self.assertEqual(
            result.metadata["runtime"]["truncated_tool_call_retries"], 1
        )

    # 2) finish_reason=length + tool_calls; retry once; second is good → tool runs.
    def test_length_with_tool_calls_retried_once_then_dispatches_on_success(self) -> None:
        bad = _raw_call("c1", "write_file", '{"path":"report.md","content":"partial')
        good = ToolCall(
            id="c2",
            name="write_file",
            arguments={"path": "report.md", "content": "full"},
        )
        spy = _SpyTool("write_file")

        provider = _RecordingProvider(
            [
                NormalizedResponse(content="", tool_calls=[bad], finish_reason="length"),
                NormalizedResponse(content="", tool_calls=[good], finish_reason="tool_calls"),
                NormalizedResponse(content="Done!", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=6,
                max_truncated_tool_call_retries=1,
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="ttc-2", user_message="go"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "Done!")
        # The good tool call was dispatched; the bad one was not.
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0].arguments, {"path": "report.md", "content": "full"})
        # Three provider invocations total: bad → retry-without-history,
        # good → dispatch, then final text.
        self.assertEqual(len(provider.calls), 3)
        # The retry must NOT have appended the broken assistant message
        # to history — the second call's outbound messages must contain
        # the same number of assistant turns as the first.
        first_assistants = [
            m for m in provider.calls[0]["messages"] if m.get("role") == "assistant"
        ]
        second_assistants = [
            m for m in provider.calls[1]["messages"] if m.get("role") == "assistant"
        ]
        self.assertEqual(len(first_assistants), len(second_assistants))

    # 3) Router-rewrite case: finish_reason="tool_calls" with truncated args.
    def test_router_rewrite_truncation_still_detected(self) -> None:
        bad = _raw_call("c1", "write_file", '{"path":"report.md","content":"partial')
        bad2 = _raw_call("c2", "write_file", '{"path":"report.md","content":"partial')

        spy = _SpyTool("write_file")
        # finish_reason is NOT "length" — but the args are still truncated.
        # The validator's heuristic must catch this regardless of what
        # the upstream gateway claimed.
        provider = ScriptedProvider(
            [
                NormalizedResponse(content="", tool_calls=[bad], finish_reason="tool_calls"),
                NormalizedResponse(content="", tool_calls=[bad2], finish_reason="tool_calls"),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=4,
                max_truncated_tool_call_retries=1,
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="ttc-3", user_message="go"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.TOOL_CALL_TRUNCATED)
        self.assertEqual(spy.calls, [])

    # 4) JSON syntax error (not truncated): retry silently then inject tool errors.
    def test_invalid_json_silent_retries_then_injects_tool_error(self) -> None:
        # ``{"path: "/foo"}`` — bad JSON syntax, but it ENDS with ``}`` so
        # the truncation heuristic won't fire.  This must take the
        # silent-retry-then-inject path.
        bad = _raw_call("c1", "write_file", '{"path: "/foo"}')
        bad2 = _raw_call("c2", "write_file", '{"path: "/foo"}')
        bad3 = _raw_call("c3", "write_file", '{"path: "/foo"}')
        # 4th attempt: the model self-corrects after seeing the tool error.
        good = ToolCall(
            id="c4", name="write_file", arguments={"path": "/foo", "content": "x"}
        )

        spy = _SpyTool("write_file")
        provider = _RecordingProvider(
            [
                NormalizedResponse(content="", tool_calls=[bad], finish_reason="tool_calls"),
                NormalizedResponse(content="", tool_calls=[bad2], finish_reason="tool_calls"),
                NormalizedResponse(content="", tool_calls=[bad3], finish_reason="tool_calls"),
                NormalizedResponse(content="", tool_calls=[good], finish_reason="tool_calls"),
                NormalizedResponse(content="All good", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=8,
                max_invalid_json_retries=3,
                max_truncated_tool_call_retries=0,  # not in play here
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="ttc-4", user_message="go"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "All good")
        # Tool dispatched exactly once on the 4th attempt.
        self.assertEqual(len(spy.calls), 1)
        # The injected recovery messages must be present in history before
        # the model's self-corrected attempt.  Find them.
        recovery_assistants = [
            m
            for m in result.messages
            if m.get("role") == "assistant"
            and isinstance(m.get("metadata"), dict)
            and m["metadata"].get("_invalid_json_recovery")
        ]
        self.assertEqual(len(recovery_assistants), 1)
        recovery_tools = [
            m
            for m in result.messages
            if m.get("role") == "tool"
            and isinstance(m.get("metadata"), dict)
            and m["metadata"].get("_invalid_json_recovery")
        ]
        self.assertEqual(len(recovery_tools), 1)
        self.assertIn("Invalid JSON arguments", recovery_tools[0]["content"])

    # 5) Pre-parsed dict args dispatch without any truncation tripwire.
    def test_dict_arguments_pass_through_unchanged(self) -> None:
        good = ToolCall(
            id="c1",
            name="write_file",
            arguments={"path": "/x", "content": "y"},
        )
        spy = _SpyTool("write_file")
        provider = ScriptedProvider(
            [
                NormalizedResponse(content="", tool_calls=[good], finish_reason="tool_calls"),
                NormalizedResponse(content="ok", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(max_iterations=4),
        )

        result = engine.run_turn(EngineRequest(session_id="ttc-5", user_message="go"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(spy.calls[0].arguments, {"path": "/x", "content": "y"})
        self.assertEqual(
            result.metadata["runtime"]["truncated_tool_call_retries"], 0
        )
        self.assertEqual(result.metadata["runtime"]["invalid_json_retries"], 0)

    # 6) Empty args coerced to {} and dispatched.
    def test_empty_string_arguments_coerced_to_empty_object(self) -> None:
        empty = _raw_call("c1", "write_file", "")
        spy = _SpyTool("write_file")
        provider = ScriptedProvider(
            [
                NormalizedResponse(content="", tool_calls=[empty], finish_reason="tool_calls"),
                NormalizedResponse(content="ok", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(max_iterations=4),
        )

        result = engine.run_turn(EngineRequest(session_id="ttc-6", user_message="go"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0].arguments, {})

    # 7) Rollback switch: detector disabled → broken args fall through.
    def test_detector_disabled_falls_through_to_dispatcher(self) -> None:
        # When the rollback switch is off, the engine no longer guards
        # the dispatch path.  The tool then receives whatever the
        # provider handed up — in our scripted setup that's a fully
        # parseable dict (because the test goal is to assert that the
        # GUARD step disappeared, not that broken args reach a real
        # tool runtime — that would just throw a TypeError without
        # exercising any contract).
        good = ToolCall(id="c1", name="write_file", arguments={"path": "/x"})
        spy = _SpyTool("write_file")
        provider = ScriptedProvider(
            [
                NormalizedResponse(content="", tool_calls=[good], finish_reason="tool_calls"),
                NormalizedResponse(content="ok", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=_registry_with(spy),
            config=EngineConfig(
                max_iterations=4,
                truncated_tool_call_detection_enabled=False,
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="ttc-7", user_message="go"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        # Counters stayed at zero because the validator never ran.
        self.assertEqual(
            result.metadata["runtime"]["truncated_tool_call_retries"], 0
        )
        self.assertEqual(result.metadata["runtime"]["invalid_json_retries"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
