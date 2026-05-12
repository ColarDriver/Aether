from __future__ import annotations

import unittest
from typing import Any, List

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    StreamDeltaCallback,
    ToolCall,
    TurnContext,
)
from aether.tools.base import ToolDescriptor


class RecordingProvider(ModelProvider):
    """Provider that returns scripted responses and records each invocation.

    Used here to assert the exact continuation scaffolding the engine feeds
    back into LLM_CALL after a `finish_reason="length"` response.
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
        stream_silent_callback: Any = None,  # noqa: ARG002
    ) -> NormalizedResponse:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": config.max_tokens,
            }
        )
        if not self._responses:
            raise RuntimeError("RecordingProvider exhausted")
        return self._responses.pop(0)


class LengthContinuationTests(unittest.TestCase):
    def test_length_once_then_success_continues_with_prompt_and_higher_budget(self) -> None:
        provider = RecordingProvider(
            [
                NormalizedResponse(content="Hello wor", finish_reason="length"),
                NormalizedResponse(content="ld!", finish_reason="stop"),
            ]
        )
        engine = AgentEngine(
            provider,
            config=EngineConfig(max_iterations=4, max_length_continue_retries=3),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="len-1",
                user_message="say hello",
                model_config=ModelCallConfig(max_tokens=10),
            )
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.LENGTH_RECOVERED)
        self.assertEqual(result.final_response, "Hello wor ld!")
        self.assertEqual(result.metadata["turn"].get("length_continue_attempts"), 1)
        self.assertEqual(len(provider.calls), 2)

        second_call = provider.calls[1]
        # Continuation path must include both the partial assistant message
        # and a synthetic user continuation instruction somewhere in the
        # outbound history.  We don't assert exact tail indexes because the
        # provider records the same mutable message list object the engine
        # later appends the final stitched assistant response to.
        assistant_scaffolds = [m for m in second_call["messages"] if m.get("role") == "assistant" and m.get("finish_reason") == "length"]
        self.assertTrue(assistant_scaffolds)
        self.assertEqual(assistant_scaffolds[0]["content"], "Hello wor")
        continuation_prompts = [m for m in second_call["messages"] if m.get("role") == "user" and "Continue exactly where you left off" in m.get("content", "")]
        self.assertTrue(continuation_prompts)
        # Ephemeral max token budget should be raised to base * 2 on the first
        # continuation retry.
        self.assertEqual(second_call["max_tokens"], 20)

    def test_length_retries_exhausted_rolls_back_and_returns_partial(self) -> None:
        provider = RecordingProvider(
            [
                NormalizedResponse(content="alpha ", finish_reason="length"),
                NormalizedResponse(content="beta ", finish_reason="length"),
                NormalizedResponse(content="gamma", finish_reason="length"),
            ]
        )
        engine = AgentEngine(
            provider,
            config=EngineConfig(max_iterations=6, max_length_continue_retries=2),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="len-2",
                user_message="continue",
                model_config=ModelCallConfig(max_tokens=8),
            )
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.LENGTH_EXHAUSTED)
        self.assertTrue(result.metadata["turn"].get("partial"))
        self.assertEqual(result.metadata["turn"].get("length_exit_reason"), "retries_exhausted")
        self.assertEqual(result.metadata["turn"].get("length_continue_attempts"), 2)
        # We should return the stitched visible prefix rather than nothing.
        self.assertEqual(result.final_response, "alpha beta gamma")
        self.assertEqual(result.messages[-1]["role"], "assistant")
        self.assertEqual(result.messages[-1]["content"], "alpha beta gamma")

    def test_thinking_only_length_returns_friendly_partial_exit(self) -> None:
        provider = RecordingProvider(
            [
                NormalizedResponse(
                    content="<think>I need more tokens</think>",
                    finish_reason="length",
                )
            ]
        )
        engine = AgentEngine(
            provider,
            config=EngineConfig(max_iterations=4, max_length_continue_retries=3),
        )

        result = engine.run_turn(
            EngineRequest(session_id="len-3", user_message="solve")
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.LENGTH_EXHAUSTED)
        self.assertTrue(result.metadata["turn"].get("partial"))
        self.assertEqual(result.metadata["turn"].get("length_exit_reason"), "thinking_budget")
        self.assertIn("lower reasoning effort", result.final_response or "")
        self.assertEqual(len(provider.calls), 1)

    def test_length_continuation_can_be_disabled(self) -> None:
        provider = RecordingProvider(
            [NormalizedResponse(content="truncated", finish_reason="length")]
        )
        engine = AgentEngine(
            provider,
            config=EngineConfig(length_continuation_enabled=False),
        )

        result = engine.run_turn(EngineRequest(session_id="len-4", user_message="x"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "truncated")
        self.assertEqual(len(provider.calls), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
