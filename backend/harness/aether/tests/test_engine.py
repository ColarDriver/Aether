from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.contracts import EngineRequest, EngineStatus, ExitReason, NormalizedResponse, ToolCall, ToolResult
from aether.runtime.interrupts import InterruptController
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class SumTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="sum")

    def execute(self, call: ToolCall, context) -> ToolResult:
        total = int(call.arguments.get("a", 0)) + int(call.arguments.get("b", 0))
        return ToolResult(tool_call_id=call.id, name=call.name, content=str(total))


class EngineTests(unittest.TestCase):
    def test_completes_with_text_response(self) -> None:
        provider = ScriptedProvider([NormalizedResponse(content="hello world")])
        engine = AgentEngine(provider, config=EngineConfig(max_iterations=4))

        result = engine.run_turn(EngineRequest(session_id="s1", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(result.final_response, "hello world")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.messages[0]["role"], "user")
        self.assertEqual(result.messages[-1]["role"], "assistant")

    def test_tool_call_round_trip_then_text(self) -> None:
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="sum", arguments={"a": 2, "b": 3})],
                ),
                NormalizedResponse(content="done"),
            ]
        )
        registry = ToolRegistry()
        registry.register(SumTool())

        engine = AgentEngine(provider, tool_registry=registry, config=EngineConfig(max_iterations=5))
        result = engine.run_turn(EngineRequest(session_id="s2", user_message="calculate"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "done")
        self.assertEqual(result.iterations, 2)
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["content"], "5")

    def test_interrupt_before_turn(self) -> None:
        provider = ScriptedProvider([NormalizedResponse(content="should not run")])
        interrupts = InterruptController()
        engine = AgentEngine(provider, interrupt_controller=interrupts)
        engine.interrupt("s3", "stop")

        result = engine.run_turn(EngineRequest(session_id="s3", user_message="hello"))

        self.assertEqual(result.status, EngineStatus.INTERRUPTED)
        self.assertEqual(result.exit_reason, ExitReason.INTERRUPTED)
        self.assertEqual(result.iterations, 0)

    def test_provider_error_fails(self) -> None:
        provider = ScriptedProvider([])
        engine = AgentEngine(provider)

        result = engine.run_turn(EngineRequest(session_id="s4", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.PROVIDER_ERROR)
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
