"""``EngineResult.metadata`` schema tests — Sprint 3 / PR 3.1.

Pin the **stable v1 schema** documented on ``EngineResult``: every turn
must surface ``usage / api_calls / iteration_budget / exit / reasoning /
compaction`` at metadata's top level, in their documented shapes, and be
JSON-serialisable end-to-end.

Test groups:

  F — usage accumulation across LLM calls
  G — exit / reasoning / compaction shape
  H — cross-session isolation
  I — co-existence with Sprint 1.5 / 2 ad-hoc fields
"""

from __future__ import annotations

import json
import unittest
from typing import Any, List

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.base import ModelProvider
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.contracts import (
    EngineRequest,
    NormalizedResponse,
    StreamDeltaCallback,
    ToolCall,
    TurnContext,
)
from aether.runtime.provider_errors import ProviderInvocationError
from aether.tools.base import ToolDescriptor


# --------------------------------------------------------------------- #
# Test helpers                                                          #
# --------------------------------------------------------------------- #


class _MockProviderWithUsage(ModelProvider):
    """Like ScriptedProvider but each NormalizedResponse pre-baked with usage."""

    provider_name = "openai"
    api_mode = "chat"

    def __init__(self, responses: List[NormalizedResponse]) -> None:
        self._queue = list(responses)
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
        if not self._queue:
            raise RuntimeError("MockProviderWithUsage has no remaining responses")
        return self._queue.pop(0)


class _RaisingThenSucceedingProvider(ModelProvider):
    """First call raises ProviderInvocationError; subsequent calls return scripted responses."""

    provider_name = "openai"
    api_mode = "chat"

    def __init__(self, error: Exception, ok_responses: List[NormalizedResponse]) -> None:
        self._error: Exception | None = error
        self._queue = list(ok_responses)
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
        if self._error is not None:
            err = self._error
            self._error = None
            raise err
        if not self._queue:
            raise RuntimeError("queue exhausted")
        return self._queue.pop(0)


def _resp_with_usage(text: str, prompt: int, completion: int) -> NormalizedResponse:
    return NormalizedResponse(
        content=text,
        metadata={
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        },
    )


def _make_engine(
    provider: ModelProvider,
    *,
    max_iterations: int = 4,
) -> AgentEngine:
    return AgentEngine(
        provider,
        config=EngineConfig(max_iterations=max_iterations, use_builtin_tools=False),
    )


# --------------------------------------------------------------------- #
# Group F — usage accumulation                                          #
# --------------------------------------------------------------------- #


class UsageAccumulationTests(unittest.TestCase):
    """T-F1..T-F4."""

    def test_T_F1_single_call_usage_recorded(self) -> None:
        provider = _MockProviderWithUsage(
            [_resp_with_usage("hello", prompt=100, completion=50)]
        )
        engine = _make_engine(provider)
        result = engine.run_turn(EngineRequest(session_id="sF1", user_message="hi"))

        usage = result.metadata["usage"]
        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 150)
        self.assertEqual(result.metadata["api_calls"], 1)

    def test_T_F2_five_call_accumulation(self) -> None:
        # Force 5 LLM calls: 4 tool-calling + 1 final text.
        responses = [
            NormalizedResponse(
                tool_calls=[ToolCall(id=f"c{i}", name="noop", arguments={})],
                metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
            )
            for i in range(4)
        ] + [_resp_with_usage("done", prompt=100, completion=50)]
        provider = _MockProviderWithUsage(responses)

        # Simple noop tool so dispatch loop progresses.
        from aether.tools.base import ToolExecutor
        from aether.runtime.contracts import ToolResult

        class _NoOpTool(ToolExecutor):
            @property
            def descriptor(self) -> ToolDescriptor:
                return ToolDescriptor(name="noop")

            def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
                return ToolResult(tool_call_id=call.id, name=call.name, content="ok")

        from aether.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(_NoOpTool())
        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(max_iterations=10, use_builtin_tools=False),
        )
        result = engine.run_turn(EngineRequest(session_id="sF2", user_message="go"))

        usage = result.metadata["usage"]
        self.assertEqual(usage["total_tokens"], 750)
        self.assertEqual(result.metadata["api_calls"], 5)

    def test_T_F3_failed_call_not_counted(self) -> None:
        # First provider call raises (recovery exhausts immediately because
        # ScriptedProvider raises a non-recoverable RuntimeError → routed
        # straight to FAILED, no usage to count).
        provider = _RaisingThenSucceedingProvider(
            error=RuntimeError("boom"),
            ok_responses=[_resp_with_usage("after", prompt=100, completion=50)],
        )
        engine = _make_engine(provider)
        result = engine.run_turn(EngineRequest(session_id="sF3", user_message="hi"))

        # Whatever the exit reason, usage reflects only successful calls.
        # In this scenario the failure is fatal so api_calls == 0 and usage zero.
        self.assertEqual(result.metadata["api_calls"], 0)
        self.assertEqual(result.metadata["usage"]["total_tokens"], 0)

    def test_T_F4_provider_omits_usage_field(self) -> None:
        provider = _MockProviderWithUsage(
            [NormalizedResponse(content="hi", metadata={})]
        )
        engine = _make_engine(provider)
        result = engine.run_turn(EngineRequest(session_id="sF4", user_message="hi"))

        # Schema present, all zero.
        self.assertEqual(result.metadata["usage"]["total_tokens"], 0)
        self.assertEqual(result.metadata["api_calls"], 1)


# --------------------------------------------------------------------- #
# Group G — exit / reasoning / compaction shape                         #
# --------------------------------------------------------------------- #


class MetadataSchemaTests(unittest.TestCase):
    """T-G1..T-G6."""

    def _typical_result(self):
        provider = _MockProviderWithUsage(
            [_resp_with_usage("hello", prompt=10, completion=5)]
        )
        engine = _make_engine(provider)
        return engine.run_turn(EngineRequest(session_id="sG", user_message="hi"))

    def test_T_G1_all_stable_keys_present(self) -> None:
        result = self._typical_result()
        for key in (
            "usage",
            "api_calls",
            "iteration_budget",
            "exit",
            "reasoning",
            "compaction",
        ):
            self.assertIn(key, result.metadata, f"missing stable key: {key}")

    def test_T_G2_exit_reason_value(self) -> None:
        result = self._typical_result()
        self.assertEqual(result.metadata["exit"]["reason"], result.exit_reason.value)

    def test_T_G3_stuck_after_tool_when_last_msg_is_tool(self) -> None:
        # Construct a turn that ends with a tool message: budget=1, single
        # tool call, then engine hits MAX_ITERATIONS before producing text.
        from aether.tools.base import ToolExecutor
        from aether.runtime.contracts import ToolResult
        from aether.tools.registry import ToolRegistry

        class _NoOpTool(ToolExecutor):
            @property
            def descriptor(self) -> ToolDescriptor:
                return ToolDescriptor(name="noop")

            def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
                return ToolResult(tool_call_id=call.id, name=call.name, content="ok")

        provider = _MockProviderWithUsage(
            [
                NormalizedResponse(
                    tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
                    metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                ),
            ]
        )
        registry = ToolRegistry()
        registry.register(_NoOpTool())
        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(max_iterations=1, use_builtin_tools=False),
        )
        result = engine.run_turn(EngineRequest(session_id="sG3", user_message="go"))
        self.assertEqual(result.metadata["exit"]["last_msg_role"], "tool")
        self.assertTrue(result.metadata["exit"]["stuck_after_tool"])

    def test_T_G4_not_stuck_when_last_msg_is_assistant(self) -> None:
        result = self._typical_result()
        self.assertEqual(result.metadata["exit"]["last_msg_role"], "assistant")
        self.assertFalse(result.metadata["exit"]["stuck_after_tool"])

    def test_T_G5_compaction_counters_zero_in_pr_3_1(self) -> None:
        result = self._typical_result()
        comp = result.metadata["compaction"]
        for key in (
            "tier1_spilled_count",
            "tier2_snipped_count",
            "tier3_cleared_count",
            "tier4_collapse_segments",
            "tier5_summaries_generated",
        ):
            self.assertEqual(comp[key], 0, f"{key} should be 0 before PR 3.3")

    def test_T_G6_metadata_is_json_serialisable(self) -> None:
        result = self._typical_result()
        # Must round-trip through json with no fallback encoder.
        encoded = json.dumps(result.metadata)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["usage"]["total_tokens"], 15)

    def test_T_G7_iteration_budget_shape(self) -> None:
        result = self._typical_result()
        budget = result.metadata["iteration_budget"]
        self.assertIn("used", budget)
        self.assertIn("max", budget)
        self.assertIn("remaining", budget)
        self.assertIn("grace_consumed", budget)
        # PR 3.1: max == EngineConfig.max_iterations (4)
        self.assertEqual(budget["max"], 4)
        # used == 1 (one LLM call, terminal text response)
        self.assertEqual(budget["used"], 1)
        self.assertEqual(budget["remaining"], 3)
        self.assertFalse(budget["grace_consumed"])

    def test_T_G8_reasoning_block_present_with_none_default(self) -> None:
        result = self._typical_result()
        self.assertIn("reasoning", result.metadata)
        self.assertIn("last_reasoning", result.metadata["reasoning"])
        self.assertIsNone(result.metadata["reasoning"]["last_reasoning"])


# --------------------------------------------------------------------- #
# Group H — cross-session isolation                                     #
# --------------------------------------------------------------------- #


class CrossSessionIsolationTests(unittest.TestCase):
    """T-H1."""

    def test_T_H1_two_sessions_independent_usage(self) -> None:
        provider = _MockProviderWithUsage(
            [
                _resp_with_usage("aA", prompt=100, completion=50),
                _resp_with_usage("bB", prompt=200, completion=10),
            ]
        )
        engine = _make_engine(provider)

        result_a = engine.run_turn(EngineRequest(session_id="sH-A", user_message="hi"))
        result_b = engine.run_turn(EngineRequest(session_id="sH-B", user_message="hi"))

        self.assertEqual(result_a.metadata["usage"]["total_tokens"], 150)
        self.assertEqual(result_b.metadata["usage"]["total_tokens"], 210)
        # Each turn's api_calls is independent — no cross-session bleed.
        self.assertEqual(result_a.metadata["api_calls"], 1)
        self.assertEqual(result_b.metadata["api_calls"], 1)


# --------------------------------------------------------------------- #
# Group I — coexistence with legacy ad-hoc keys                         #
# --------------------------------------------------------------------- #


class LegacyKeysCoexistenceTests(unittest.TestCase):
    """T-I1..T-I2."""

    def test_T_I1_stable_keys_dont_collide_with_runtime_block(self) -> None:
        result = _MockProviderWithUsage(
            [_resp_with_usage("hi", prompt=10, completion=5)]
        )
        engine = _make_engine(result)
        out = engine.run_turn(EngineRequest(session_id="sI1", user_message="hi"))

        # Stable v1 schema lives at the top.
        self.assertIn("usage", out.metadata)
        self.assertIn("compaction", out.metadata)
        # Sprint 1+ runtime counters live under `runtime`.
        self.assertIn("runtime", out.metadata)
        self.assertIn("phantom_tool_synthesized", out.metadata["runtime"])

    def test_T_I2_internal_keys_excluded_from_turn_snapshot(self) -> None:
        result = _MockProviderWithUsage(
            [_resp_with_usage("hi", prompt=10, completion=5)]
        )
        engine = _make_engine(result)
        out = engine.run_turn(EngineRequest(session_id="sI2", user_message="hi"))

        # `usage_accumulator` is an internal CanonicalUsage object —
        # MUST NOT leak into the JSON-serialisable `turn` snapshot
        # (its public dict form is at metadata["usage"]).
        self.assertNotIn("usage_accumulator", out.metadata["turn"])


if __name__ == "__main__":
    unittest.main()
