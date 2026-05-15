from __future__ import annotations

import copy
import unittest
from typing import Any

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _RecordingProvider(ModelProvider):
    provider_name: str = "test-provider"
    api_mode: str = "chat"

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self.model: str = "test-model"
        self.responses: list[NormalizedResponse] = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del tools, config, context, stream_callback, stream_silent_callback
        self.calls.append({"messages": copy.deepcopy(messages)})
        if not self.responses:
            raise RuntimeError("no scripted response")
        return self.responses.pop(0)


class _EditMarkerTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="fake_edit")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        path = (call.arguments or {}).get("path", "/tmp/edited.py")
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="edited",
            metadata={"edited_paths": [str(path)]},
        )


class _TaskMarkerTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="task")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        return ToolResult(tool_call_id=call.id, name=call.name, content="verifier spawned")


def _engine(provider: _RecordingProvider, *, verifier_gate_enabled: bool = True) -> AgentEngine:
    registry = ToolRegistry()
    registry.register(_EditMarkerTool())
    registry.register(_TaskMarkerTool())
    return AgentEngine(
        provider,
        tool_registry=registry,
        config=EngineConfig(
            use_builtin_tools=False,
            tool_use_contract_enabled=False,
            verification_directive_enabled=False,
            faithful_reporting_enabled=False,
            verifier_gate_enabled=verifier_gate_enabled,
            verifier_gate_file_threshold=3,
            max_iterations=5,
        ),
    )


def _verifier_gate_blocks(messages: list[dict[str, Any]]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message.get("content"), str)
        and message.get("metadata", {}).get("source") == "verifier_gate"
    )


class VerifierGateReminderTests(unittest.TestCase):
    def test_reminder_is_injected_after_threshold_edits(self) -> None:
        provider = _RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="e1", name="fake_edit", arguments={"path": "/tmp/a.py"}),
                        ToolCall(id="e2", name="fake_edit", arguments={"path": "/tmp/b.py"}),
                        ToolCall(id="e3", name="fake_edit", arguments={"path": "/tmp/c.py"}),
                    ],
                ),
                NormalizedResponse(content="done"),
            ]
        )

        _engine(provider).run_turn(EngineRequest(session_id="gate-1", user_message="go"))

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(_verifier_gate_blocks(provider.calls[0]["messages"]), 0)
        self.assertEqual(_verifier_gate_blocks(provider.calls[1]["messages"]), 1)
        content = provider.calls[1]["messages"][-1]["content"]
        self.assertIn('task(subagent_type="Verifier"', content)

    def test_disabled_gate_never_injects_reminder(self) -> None:
        provider = _RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="e1", name="fake_edit", arguments={"path": "/tmp/a.py"}),
                        ToolCall(id="e2", name="fake_edit", arguments={"path": "/tmp/b.py"}),
                        ToolCall(id="e3", name="fake_edit", arguments={"path": "/tmp/c.py"}),
                    ],
                ),
                NormalizedResponse(content="done"),
            ]
        )

        _engine(provider, verifier_gate_enabled=False).run_turn(
            EngineRequest(session_id="gate-2", user_message="go")
        )

        self.assertEqual(_verifier_gate_blocks(provider.calls[1]["messages"]), 0)

    def test_verifier_invocation_prevents_repeated_reminders(self) -> None:
        provider = _RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="e1", name="fake_edit", arguments={"path": "/tmp/a.py"}),
                        ToolCall(id="e2", name="fake_edit", arguments={"path": "/tmp/b.py"}),
                        ToolCall(id="e3", name="fake_edit", arguments={"path": "/tmp/c.py"}),
                    ],
                ),
                NormalizedResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="task",
                            arguments={"subagent_type": "Verifier", "prompt": "verify"},
                        )
                    ],
                ),
                NormalizedResponse(content="done"),
            ]
        )

        _engine(provider).run_turn(EngineRequest(session_id="gate-3", user_message="go"))

        self.assertEqual(_verifier_gate_blocks(provider.calls[1]["messages"]), 1)
        self.assertEqual(_verifier_gate_blocks(provider.calls[2]["messages"]), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
