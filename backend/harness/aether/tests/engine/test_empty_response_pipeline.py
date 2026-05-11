from __future__ import annotations

import unittest
from typing import Any, Iterable, List

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _ScriptedStreamProvider(ModelProvider):
    provider_name = "openai"
    api_mode = "chat"

    def __init__(
        self,
        responses: Iterable[NormalizedResponse],
        *,
        stream_chunks: Iterable[Iterable[str]] | None = None,
        provider_name: str = "openai",
        api_mode: str = "chat",
    ) -> None:
        self._responses = list(responses)
        self._stream_chunks = [list(chunks) for chunks in (stream_chunks or [])]
        self.provider_name = provider_name
        self.api_mode = api_mode
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
        index = self.calls
        self.calls += 1
        if index >= len(self._responses):
            raise RuntimeError("script exhausted")
        if stream_callback and index < len(self._stream_chunks):
            for chunk in self._stream_chunks[index]:
                stream_callback(chunk)
        return self._responses[index]


class _TodoTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="update_todo",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        return ToolResult(tool_call_id=call.id, name=call.name, content="todo updated")


def _engine(
    provider: ModelProvider,
    *,
    config: EngineConfig | None = None,
    registry: ToolRegistry | None = None,
) -> AgentEngine:
    return AgentEngine(
        provider,
        config=config
        or EngineConfig(
            max_iterations=8,
            use_builtin_tools=False,
            tool_use_contract_enabled=False,
            empty_response_max_retries=0,
        ),
        tool_registry=registry,
    )


class EmptyResponsePipelineTests(unittest.TestCase):
    def test_legitimate_end_turn_passthrough_does_not_surface_empty_response(self) -> None:
        provider = _ScriptedStreamProvider([NormalizedResponse(content="", finish_reason="stop")])
        engine = _engine(provider)

        result = engine.run_turn(EngineRequest(session_id="legit", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(result.final_response, "")
        self.assertEqual(result.metadata["empty_recovery"]["last_step"], "legitimate_empty_passthrough")

    def test_terminal_empty_response_when_no_recovery_step_matches(self) -> None:
        provider = _ScriptedStreamProvider([NormalizedResponse(content="", finish_reason="length")])
        engine = _engine(provider)

        result = engine.run_turn(EngineRequest(session_id="terminal", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.EMPTY_RESPONSE)
        self.assertEqual(result.metadata["empty_recovery"]["last_step"], "terminal_empty")

    def test_partial_stream_recovery_returns_streamed_text(self) -> None:
        provider = _ScriptedStreamProvider(
            [NormalizedResponse(content="", finish_reason="length")],
            stream_chunks=[["Hello ", "world"]],
        )
        seen: list[str] = []
        engine = _engine(provider)

        result = engine.run_turn(
            EngineRequest(session_id="partial", user_message="hi", stream_callback=seen.append)
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.PARTIAL_STREAM_RECOVERY)
        self.assertEqual(result.final_response, "Hello world")
        self.assertEqual(seen, ["Hello ", "world"])
        self.assertEqual(result.metadata["empty_recovery"]["last_step"], "partial_stream_recovery")

    def test_truncated_prefix_has_priority_over_partial_stream(self) -> None:
        provider = _ScriptedStreamProvider(
            [NormalizedResponse(content="", finish_reason="length")],
            stream_chunks=[["Continuation"]],
        )
        engine = _engine(provider)

        result = engine.run_turn(
            EngineRequest(
                session_id="prefix",
                user_message="continue",
                metadata={"truncated_response_prefix": "Previous "},
            )
        )

        self.assertEqual(result.exit_reason, ExitReason.LENGTH_RECOVERED)
        self.assertEqual(result.final_response, "Previous Continuation")
        self.assertEqual(result.metadata["empty_recovery"]["last_step"], "truncated_prefix_concat")
        self.assertEqual(result.metadata["turn"].get("truncated_response_prefix"), "")

    def test_thinking_prefill_continues_then_cleans_prefill_messages(self) -> None:
        provider = _ScriptedStreamProvider(
            [
                NormalizedResponse(
                    content="",
                    finish_reason="length",
                    metadata={"reasoning_content": "work"},
                ),
                NormalizedResponse(content="answer"),
            ]
        )
        engine = _engine(
            provider,
            config=EngineConfig(
                max_iterations=8,
                use_builtin_tools=False,
                tool_use_contract_enabled=False,
                empty_response_max_retries=0,
                thinking_prefill_max_retries=2,
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="thinking", user_message="solve"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "answer")
        self.assertEqual(result.metadata["empty_recovery"]["last_step"], "thinking_prefill")
        self.assertEqual(result.metadata["empty_recovery"]["thinking_prefill_cleaned"], 1)
        self.assertTrue(
            any(message.get("_thinking_prefill") for message in provider.seen_messages[1])
        )

    def test_codex_intermediate_ack_finalise_hook_continues(self) -> None:
        provider = _ScriptedStreamProvider(
            [
                NormalizedResponse(content="好的，我马上读文件"),
                NormalizedResponse(content="done"),
            ],
            provider_name="codex",
            api_mode="responses",
        )
        engine = _engine(
            provider,
            config=EngineConfig(
                max_iterations=8,
                use_builtin_tools=False,
                tool_use_contract_enabled=False,
                codex_intermediate_ack_max_retries=2,
            ),
        )

        result = engine.run_turn(EngineRequest(session_id="codex", user_message="read file"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "done")
        self.assertEqual(result.metadata["empty_recovery"]["last_step"], "codex_intermediate_ack")
        self.assertTrue(
            any(
                message.get("metadata", {}).get("_codex_intermediate_ack")
                for message in provider.seen_messages[1]
            )
        )

    def test_housekeeping_fallback_reads_session_runtime_state_across_turns(self) -> None:
        provider = _ScriptedStreamProvider(
            [NormalizedResponse(content="", finish_reason="length")]
        )
        engine = _engine(
            provider,
            config=EngineConfig(
                max_iterations=4,
                use_builtin_tools=False,
                tool_use_contract_enabled=False,
                empty_response_max_retries=0,
                housekeeping_fallback_enabled=True,
                summary_on_budget_exhausted=False,
            ),
        )
        engine._record_last_content_for_housekeeping_fallback(
            messages=[
                {
                    "role": "assistant",
                    "content": "Updated todo",
                    "tool_calls": [
                        {
                            "id": "todo-1",
                            "type": "function",
                            "function": {"name": "update_todo", "arguments": {"text": "x"}},
                        }
                    ],
                }
            ],
            context=TurnContext(session_id="house", iteration=0, metadata={}),
        )

        second = engine.run_turn(EngineRequest(session_id="house", user_message="what happened?"))

        self.assertEqual(second.status, EngineStatus.COMPLETED)
        self.assertEqual(second.exit_reason, ExitReason.FALLBACK_PRIOR_TURN_CONTENT)
        self.assertEqual(second.final_response, "Updated todo")
        self.assertEqual(second.metadata["empty_recovery"]["last_step"], "housekeeping_fallback")


if __name__ == "__main__":
    unittest.main()
