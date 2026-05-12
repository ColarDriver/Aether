from __future__ import annotations

import unittest
from typing import Any, Iterable, List

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineStatus,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _ScriptedProvider(ModelProvider):
    def __init__(self, responses: Iterable[NormalizedResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.seen_messages: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,  # noqa: ARG002
    ) -> NormalizedResponse:
        self.seen_messages.append(list(messages))
        if self.calls >= len(self._responses):
            raise RuntimeError("script exhausted")
        response = self._responses[self.calls]
        self.calls += 1
        return response


class _ReadFileTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="read_file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                },
                "required": ["path"],
            },
            required=["path"],
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=f"content:{call.arguments.get('path')}",
        )


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_ReadFileTool())
    return registry


def _engine(provider: ModelProvider) -> AgentEngine:
    return AgentEngine(
        provider,
        tool_registry=_registry(),
        config=EngineConfig(
            max_iterations=8,
            use_builtin_tools=False,
            tool_use_contract_enabled=False,
            fail_on_unknown_tool=False,
            max_invalid_json_retries=1,
            tool_error_structured_format_enabled=True,
            tool_schema_precheck_enabled=True,
        ),
    )


def _tool_messages(result) -> list[dict[str, Any]]:
    return [message for message in result.messages if message.get("role") == "tool"]


class JsonTolerancePR4Tests(unittest.TestCase):
    def test_invalid_json_injection_uses_structured_formatter(self) -> None:
        provider = _ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="bad-json",
                            name="read_file",
                            arguments='{"path": "/tmp/x" "bad": 1}',
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                NormalizedResponse(content="fixed"),
            ]
        )
        engine = _engine(provider)

        result = engine.run_turn(EngineRequest(session_id="json", user_message="read"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        error_messages = [
            message for message in _tool_messages(result)
            if message.get("metadata", {}).get("_invalid_json_recovery")
        ]
        self.assertEqual(len(error_messages), 1)
        self.assertIn("Invalid JSON arguments for tool `read_file`", error_messages[0]["content"])
        self.assertIn("Hint:", error_messages[0]["content"])
        self.assertEqual(
            error_messages[0]["metadata"].get("_tool_error_category"),
            "json_syntax",
        )
        self.assertEqual(result.metadata["tool_errors"]["by_category"]["json_syntax"], 1)

    def test_schema_missing_injection_prevents_dispatch_until_model_corrects(self) -> None:
        provider = _ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[ToolCall(id="missing", name="read_file", arguments="{}")],
                    finish_reason="tool_calls",
                ),
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="ok",
                            name="read_file",
                            arguments='{"path": "/tmp/x"}',
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                NormalizedResponse(content="done"),
            ]
        )
        engine = _engine(provider)

        result = engine.run_turn(EngineRequest(session_id="schema", user_message="read"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        schema_errors = [
            message for message in _tool_messages(result)
            if message.get("metadata", {}).get("_schema_error_recovery")
        ]
        self.assertEqual(len(schema_errors), 1)
        self.assertIn("The required parameter `path` is missing", schema_errors[0]["content"])
        self.assertEqual(
            schema_errors[0]["metadata"].get("_tool_error_category"),
            "schema_missing",
        )
        successful_results = [
            message for message in _tool_messages(result)
            if message.get("tool_call_id") == "ok"
        ]
        self.assertEqual(successful_results[0]["content"], "content:/tmp/x")

    def test_unknown_tool_synthetic_message_uses_structured_content(self) -> None:
        provider = _ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(id="unknown", name="totally_not_a_tool", arguments={})
                    ],
                    finish_reason="tool_calls",
                ),
                NormalizedResponse(content="recovered"),
            ]
        )
        engine = _engine(provider)

        result = engine.run_turn(EngineRequest(session_id="unknown", user_message="use tool"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        unknown_errors = [
            message for message in _tool_messages(result)
            if message.get("metadata", {}).get("_unknown_tool_recovery")
        ]
        self.assertEqual(len(unknown_errors), 1)
        self.assertIn("Unknown tool", unknown_errors[0]["content"])
        self.assertIn("Available tools:", unknown_errors[0]["content"])
        self.assertIn("read_file", unknown_errors[0]["content"])
        self.assertEqual(
            unknown_errors[0]["metadata"].get("_tool_error_category"),
            "unknown_tool",
        )


if __name__ == "__main__":
    unittest.main()
