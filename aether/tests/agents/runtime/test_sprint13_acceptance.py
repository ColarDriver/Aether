from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _ReadOnlyTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="read_only")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=f"read:{call.arguments.get('path', '')}",
        )


class Sprint13AcceptanceSmokeTests(unittest.TestCase):
    def test_text_and_tool_turn_preserve_stable_metadata(self) -> None:
        registry = ToolRegistry()
        registry.register(_ReadOnlyTool())
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="read_only",
                            arguments={"path": "README.md"},
                        )
                    ],
                    metadata={"usage": {"prompt_tokens": 4, "completion_tokens": 2}},
                ),
                NormalizedResponse(
                    content="done",
                    metadata={"usage": {"prompt_tokens": 5, "completion_tokens": 3}},
                ),
            ]
        )
        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(use_builtin_tools=False, max_iterations=4),
        )

        result = engine.run_turn(
            EngineRequest(session_id="sprint13-smoke", user_message="read")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(result.final_response, "done")
        self.assertEqual(result.metadata["api_calls"], 2)
        self.assertEqual(result.metadata["usage"]["total_tokens"], 14)
        for key in (
            "turn",
            "runtime",
            "memory",
            "compaction",
            "resource_cleanup",
            "iteration_budget",
            "exit",
            "reasoning",
        ):
            self.assertIn(key, result.metadata)
        tool_messages = [m for m in result.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["content"], "read:README.md")


if __name__ == "__main__":
    unittest.main()

