from __future__ import annotations

import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from aether import AgentEngine, SteerInbox
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import (
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


class RecordingProvider(ModelProvider):
    provider_name = "test-provider"
    api_mode = "chat"

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self.model = "test-model"
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del tools, config, context, stream_callback, stream_silent_callback
        self.calls.append(copy.deepcopy(messages))
        if not self.responses:
            raise RuntimeError("no scripted response")
        return self.responses.pop(0)


class SteeringTool(ToolExecutor):
    def __init__(self) -> None:
        self.engine: AgentEngine | None = None

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="steer_tool")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        assert self.engine is not None
        accepted = self.engine.send_steer(context.session_id, "prefer concise answer")
        assert accepted
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="tool output",
        )


class SteerInboxTests(unittest.TestCase):
    def test_inbox_rejects_empty_text_and_merges_by_session(self) -> None:
        inbox = SteerInbox()

        self.assertFalse(inbox.append("s1", "   "))
        self.assertTrue(inbox.append("s1", "first"))
        self.assertTrue(inbox.append("s1", "second"))
        self.assertTrue(inbox.append("s2", "other"))

        self.assertEqual(inbox.drain("s1"), "first\nsecond")
        self.assertEqual(inbox.drain("s2"), "other")
        self.assertIsNone(inbox.drain("s1"))

    def test_inbox_threaded_appends_do_not_drop_text(self) -> None:
        inbox = SteerInbox()
        values = [f"msg-{i}" for i in range(40)]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda text: inbox.append("s1", text), values))

        self.assertTrue(all(results))
        drained = inbox.drain("s1")
        self.assertIsNotNone(drained)
        drained_values = set(drained.splitlines())
        self.assertEqual(drained_values, set(values))

    def test_send_steer_appends_to_last_tool_result_before_next_llm_call(self) -> None:
        provider = RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="steer_tool",
                            arguments={},
                        )
                    ],
                ),
                NormalizedResponse(content="done"),
            ]
        )
        tool = SteeringTool()
        registry = ToolRegistry()
        registry.register(tool)
        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(max_iterations=4, use_builtin_tools=False),
        )
        tool.engine = engine

        result = engine.run_turn(
            EngineRequest(session_id="steer-run", user_message="use tool")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(len(provider.calls), 2)
        second_call_tool_messages = [
            message for message in provider.calls[1] if message.get("role") == "tool"
        ]
        self.assertEqual(len(second_call_tool_messages), 1)
        self.assertIn(
            "User guidance: prefer concise answer",
            second_call_tool_messages[0]["content"],
        )
        roles = [message.get("role") for message in provider.calls[1]]
        self.assertNotEqual(roles[-1], "user")
        self.assertEqual(result.metadata["pending_steer"], None)
        self.assertEqual(result.metadata["turn"].get("steer_injected_count"), 1)

    def test_unconsumed_steer_is_returned_in_result_metadata(self) -> None:
        provider = RecordingProvider([NormalizedResponse(content="ok")])
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False),
        )
        self.assertTrue(engine.send_steer("pending-s", "next turn please"))

        result = engine.run_turn(
            EngineRequest(session_id="pending-s", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.metadata["pending_steer"], "next turn please")
        self.assertEqual(result.metadata["turn"].get("pending_steer"), "next turn please")
        self.assertIsNone(engine.services.steer_inbox.drain("pending-s"))

    def test_interrupt_clears_pending_steer(self) -> None:
        provider = RecordingProvider([NormalizedResponse(content="should not run")])
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False),
        )
        self.assertTrue(engine.send_steer("interrupt-s", "discard me"))

        engine.interrupt("interrupt-s", "stop")
        result = engine.run_turn(
            EngineRequest(session_id="interrupt-s", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.INTERRUPTED)
        self.assertIsNone(result.metadata["pending_steer"])
        self.assertIsNone(engine.services.steer_inbox.drain("interrupt-s"))

    def test_multimodal_tool_content_appends_text_block(self) -> None:
        provider = RecordingProvider([])
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False),
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "vision_tool",
                "content": [{"type": "text", "text": "tool output"}],
            }
        ]
        context = TurnContext(session_id="multi-s", iteration=1, metadata={})
        self.assertTrue(engine.send_steer("multi-s", "look at the top left"))

        engine._apply_pending_steer_to_tool_results(
            messages,
            session_id="multi-s",
            start_idx=0,
            context=context,
        )

        content = messages[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[-1]["type"], "text")
        self.assertIn("User guidance: look at the top left", content[-1]["text"])


if __name__ == "__main__":
    unittest.main()
