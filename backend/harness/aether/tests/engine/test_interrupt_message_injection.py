from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import (
    EngineRequest,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.interrupts import InterruptController
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _InterruptingProvider(ModelProvider):
    def __init__(self, controller: InterruptController) -> None:
        self._controller = controller

    def generate(
        self,
        messages: list[dict],
        tools,
        config,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        assert stream_callback is not None
        stream_callback("hello")
        self._controller.request(context.session_id, reason="user-interrupt")
        stream_callback(" world")
        return NormalizedResponse(content="ignored")


class InterruptMessageInjectionTests(unittest.TestCase):
    def test_partial_and_marker_are_preserved(self) -> None:
        controller = InterruptController()
        engine = AgentEngine(_InterruptingProvider(controller), interrupt_controller=controller)
        result = engine.run_turn(EngineRequest(session_id="s1", user_message="hi"))
        self.assertEqual(result.status.value, "INTERRUPTED")
        self.assertGreaterEqual(len(result.messages), 3)
        self.assertEqual(result.messages[-2]["role"], "assistant")
        self.assertEqual(result.messages[-2]["content"], "hello")
        self.assertEqual(result.messages[-1]["role"], "user")
        self.assertIn("[Request interrupted by user]", result.messages[-1]["content"])
        interrupt_meta = result.metadata.get("interrupt") or {}
        self.assertEqual(interrupt_meta.get("marker"), "[Request interrupted by user]")
        self.assertEqual(interrupt_meta.get("partial_assistant_chars"), 5)

    def test_interrupted_tool_turn_gets_synthetic_results_for_pending_calls(self) -> None:
        class _ToolTurnProvider(ModelProvider):
            def __init__(self) -> None:
                self.calls = 0

            def generate(
                self,
                messages: list[dict],
                tools,
                config,
                context: TurnContext,
                stream_callback: StreamDeltaCallback | None = None,
                stream_silent_callback: StreamSilentCallback | None = None,
            ) -> NormalizedResponse:
                self.calls += 1
                if self.calls == 1:
                    return NormalizedResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="call-1", name="demo", arguments={"n": 1}),
                            ToolCall(id="call-2", name="demo", arguments={"n": 2}),
                        ],
                    )

                unresolved: list[str] = []
                for idx, message in enumerate(messages):
                    if not isinstance(message, dict) or not message.get("tool_calls"):
                        continue
                    responded = {
                        str(candidate.get("tool_call_id") or "")
                        for candidate in messages[idx + 1 :]
                        if isinstance(candidate, dict) and candidate.get("role") == "tool"
                    }
                    for call in message.get("tool_calls") or []:
                        call_id = str(call.get("id") or "")
                        if call_id and call_id not in responded:
                            unresolved.append(call_id)
                if unresolved:
                    raise AssertionError(f"unresolved tool calls: {unresolved}")
                return NormalizedResponse(content="safe to continue")

        class _InterruptingTool(ToolExecutor):
            interrupt_behavior = "cancel"

            @property
            def descriptor(self) -> ToolDescriptor:
                return ToolDescriptor(name="demo")

            def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
                assert context.interrupt_signal is not None
                context.interrupt_signal.abort("user-interrupt")
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content="partial tool output",
                    is_error=True,
                    metadata={"interrupted": True},
                )

        registry = ToolRegistry()
        registry.register(_InterruptingTool())
        engine = AgentEngine(_ToolTurnProvider(), tool_registry=registry)

        first = engine.run_turn(EngineRequest(session_id="tool-int", user_message="inspect"))
        self.assertEqual(first.status.value, "INTERRUPTED")
        self.assertEqual(first.messages[-3]["role"], "tool")
        self.assertEqual(first.messages[-3]["tool_call_id"], "call-1")
        self.assertTrue(first.messages[-3]["metadata"]["interrupted"])
        self.assertEqual(first.messages[-2]["role"], "tool")
        self.assertEqual(first.messages[-2]["tool_call_id"], "call-2")
        self.assertTrue(first.messages[-2]["metadata"]["synthetic_interrupt_result"])
        self.assertIn("Tool execution interrupted by user", first.messages[-2]["content"])
        self.assertEqual(first.messages[-1]["role"], "user")
        self.assertIn(
            "[Request interrupted by user for tool use]",
            first.messages[-1]["content"],
        )

        second = engine.run_turn(
            EngineRequest(
                session_id="tool-int",
                user_message="continue",
                messages=list(first.messages),
            )
        )
        self.assertEqual(second.status.value, "COMPLETED")
        self.assertEqual(second.final_response, "safe to continue")


if __name__ == "__main__":
    unittest.main()
