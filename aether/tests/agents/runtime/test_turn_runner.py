from __future__ import annotations

import unittest
from typing import Any, List

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.control.interrupts import InterruptController
from aether.runtime.core.contracts import (
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
from aether.runtime.recovery.provider_errors import ProviderInvocationError
from aether.runtime.recovery.strategies import NoRetryStrategy
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _CountingProvider(ModelProvider):
    provider_name = "openai"
    api_mode = "chat"

    def __init__(self, response: NormalizedResponse | None = None, *, error: Exception | None = None) -> None:
        self.response = response or NormalizedResponse(content="ok")
        self.error = error
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del messages, tools, config, context, stream_callback, stream_silent_callback
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


class _EchoTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="echo")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=str(call.arguments.get("text", "")),
        )


def _config(**overrides: Any) -> EngineConfig:
    defaults = {
        "use_builtin_tools": False,
        "verification_directive_enabled": False,
        "faithful_reporting_enabled": False,
        "verifier_gate_enabled": False,
        "memory_enabled": False,
        "summary_on_budget_exhausted": False,
        "max_iterations": 4,
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)


class TurnRunnerIntegrationTests(unittest.TestCase):
    def test_text_response_path(self) -> None:
        engine = AgentEngine(
            ScriptedProvider([NormalizedResponse(content="hello")]),
            config=_config(),
        )

        result = engine.run_turn(EngineRequest(session_id="runner-text", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.exit_reason, ExitReason.TEXT_RESPONSE)
        self.assertEqual(result.final_response, "hello")

    def test_tool_response_path(self) -> None:
        registry = ToolRegistry()
        registry.register(_EchoTool())
        engine = AgentEngine(
            ScriptedProvider(
                [
                    NormalizedResponse(
                        tool_calls=[
                            ToolCall(id="call-1", name="echo", arguments={"text": "ok"})
                        ],
                        finish_reason="tool_calls",
                    ),
                    NormalizedResponse(content="done"),
                ]
            ),
            tool_registry=registry,
            config=_config(),
        )

        result = engine.run_turn(EngineRequest(session_id="runner-tool", user_message="echo"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "done")
        self.assertTrue(any(message.get("role") == "tool" for message in result.messages))

    def test_provider_error_path(self) -> None:
        provider = _CountingProvider(
            error=ProviderInvocationError(status_code=500, body_summary="boom")
        )
        engine = AgentEngine(
            provider,
            config=_config(max_iterations=1),
            recovery_strategy=NoRetryStrategy(),
        )

        result = engine.run_turn(EngineRequest(session_id="runner-error", user_message="hi"))

        self.assertEqual(provider.calls, 1)
        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.PROVIDER_ERROR)
        self.assertIn("boom", result.error or "")

    def test_interrupt_before_provider_call(self) -> None:
        interrupt_controller = InterruptController()
        interrupt_controller.request("runner-interrupt", "stop")
        provider = _CountingProvider()
        engine = AgentEngine(
            provider,
            config=_config(),
            interrupt_controller=interrupt_controller,
        )

        result = engine.run_turn(EngineRequest(session_id="runner-interrupt", user_message="hi"))

        self.assertEqual(provider.calls, 0)
        self.assertEqual(result.status, EngineStatus.INTERRUPTED)
        self.assertEqual(result.exit_reason, ExitReason.INTERRUPTED)

    def test_max_iteration_exit_after_tool_round(self) -> None:
        registry = ToolRegistry()
        registry.register(_EchoTool())
        engine = AgentEngine(
            ScriptedProvider(
                [
                    NormalizedResponse(
                        tool_calls=[
                            ToolCall(id="call-1", name="echo", arguments={"text": "ok"})
                        ],
                        finish_reason="tool_calls",
                    )
                ]
            ),
            tool_registry=registry,
            config=_config(max_iterations=1, summary_on_budget_exhausted=False),
        )

        result = engine.run_turn(EngineRequest(session_id="runner-max", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.MAX_ITERATIONS)
        self.assertEqual(result.exit_reason, ExitReason.MAX_ITERATIONS)


if __name__ == "__main__":
    unittest.main()
