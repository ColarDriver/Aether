"""Tests for the per-turn plan-mode reminder — Sprint 12 PR 12.3."""

from __future__ import annotations

import copy
import unittest
from typing import Any

from aether import AgentEngine
from aether.agents.core.system_prompt import (
    _PLAN_MODE_REMINDER,  # noqa: PLC2701 - test wants the exact text
    append_plan_mode_reminder,
    build_plan_mode_reminder,
)
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    TurnContext,
)
from aether.runtime.session import clear_mode, set_mode
from aether.runtime.session.session_state import SessionMode
from aether.tools.base import ToolDescriptor


class RecordingProvider(ModelProvider):
    provider_name = "test-provider"
    api_mode = "chat"

    def __init__(self, responses: list[NormalizedResponse]) -> None:
        self.model = "test-model"
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"messages": copy.deepcopy(messages)})
        if not self.responses:
            raise RuntimeError("no scripted response")
        return self.responses.pop(0)


# ----------------------------------------------------------------- helper

class HelperTests(unittest.TestCase):
    def test_append_to_existing_prompt(self) -> None:
        out = append_plan_mode_reminder("you are helpful")
        assert out is not None
        self.assertTrue(out.startswith("you are helpful\n\n"))
        self.assertIn("<plan_mode_reminder>", out)

    def test_append_to_none(self) -> None:
        out = append_plan_mode_reminder(None)
        assert out is not None
        self.assertEqual(out, _PLAN_MODE_REMINDER)

    def test_append_to_blank(self) -> None:
        out = append_plan_mode_reminder("   ")
        self.assertEqual(out, _PLAN_MODE_REMINDER)


# ----------------------------------------------------------------- injection

class InjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Each test gets a fresh process-level session state.
        self.session_id = "plan-mode-test"
        self.addCleanup(clear_mode, self.session_id)

    def _engine(self, provider: RecordingProvider) -> AgentEngine:
        return AgentEngine(
            provider,
            config=EngineConfig(
                use_builtin_tools=False,
                # disable the directive sections so the system message
                # is small and the reminder is easy to assert on.
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
                verifier_gate_enabled=False,
            ),
        )

    def test_plan_mode_injects_reminder(self) -> None:
        set_mode(self.session_id, SessionMode.PLAN)
        provider = RecordingProvider([NormalizedResponse(content="ok")])
        engine = self._engine(provider)
        engine.run_turn(
            EngineRequest(
                session_id=self.session_id,
                user_message="hi",
                system_message="you are helpful",
            )
        )
        system = provider.calls[0]["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("<plan_mode_reminder>", system["content"])
        self.assertIn("Plan file:", system["content"])
        self.assertIn("The only file you may write or edit", system["content"])
        self.assertIn("exit_plan_mode", system["content"])

    def test_agent_mode_does_not_inject_reminder(self) -> None:
        # Default mode is "agent" — never set_mode(plan).
        provider = RecordingProvider([NormalizedResponse(content="ok")])
        engine = self._engine(provider)
        engine.run_turn(
            EngineRequest(
                session_id=self.session_id,
                user_message="hi",
                system_message="you are helpful",
            )
        )
        system = provider.calls[0]["messages"][0]
        self.assertNotIn("<plan_mode_reminder>", system["content"])

    def test_reminder_stable_across_turns(self) -> None:
        # PR 12.3 promises stable text so the prompt-cache prefix
        # doesn't churn; assert the reminder block is identical on
        # consecutive LLM calls.
        set_mode(self.session_id, SessionMode.PLAN)
        provider = RecordingProvider(
            [
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
                ),
                NormalizedResponse(content="done"),
            ]
        )
        # Add a no-op tool so the first call's tool_call is dispatched
        # and the loop reaches a second LLM iteration.
        from aether.runtime.core.contracts import ToolResult
        from aether.tools.base import ToolExecutor
        from aether.tools.registry import ToolRegistry

        class _NoopTool(ToolExecutor):
            @property
            def descriptor(self) -> ToolDescriptor:
                return ToolDescriptor(name="noop")

            def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
                del context
                return ToolResult(tool_call_id=call.id, name=call.name, content="ok")

        registry = ToolRegistry()
        registry.register(_NoopTool())
        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(
                use_builtin_tools=False,
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
                verifier_gate_enabled=False,
            ),
        )
        engine.run_turn(
            EngineRequest(session_id=self.session_id, user_message="go")
        )
        self.assertGreaterEqual(len(provider.calls), 2)
        first = provider.calls[0]["messages"][0]["content"]
        second = provider.calls[1]["messages"][0]["content"]
        self.assertEqual(first, second)
        self.assertIn("<plan_mode_reminder>", first)
        self.assertIn("Plan file:", first)

    def test_build_reminder_mentions_plan_path(self) -> None:
        from aether.runtime.session.plan_artifact import get_plan_path

        out = build_plan_mode_reminder(self.session_id)
        self.assertIn(str(get_plan_path(self.session_id)), out)
        self.assertIn("no plan file exists yet", out)

    def test_plan_mode_active_metadata_set(self) -> None:
        # context.metadata['plan_mode_active'] should be set so the
        # downstream observability layer can detect the mode without
        # re-reading session_state.  Use a custom provider that records
        # the live context.
        captured: dict[str, Any] = {}

        class _CtxCaptureProvider(ModelProvider):
            provider_name = "ctx-test"
            api_mode = "chat"
            model = "ctx"

            def generate(
                self,
                messages: list[dict],
                tools: list[ToolDescriptor],
                config: ModelCallConfig,
                context: TurnContext,
                stream_callback: StreamDeltaCallback | None = None,
                stream_silent_callback: StreamSilentCallback | None = None,
            ) -> NormalizedResponse:
                del messages, tools, config, stream_callback, stream_silent_callback
                captured["plan_mode_active"] = context.metadata.get("plan_mode_active")
                return NormalizedResponse(content="ok")

        set_mode(self.session_id, SessionMode.PLAN)
        engine = AgentEngine(
            _CtxCaptureProvider(),
            config=EngineConfig(
                use_builtin_tools=False,
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
                verifier_gate_enabled=False,
            ),
        )
        engine.run_turn(
            EngineRequest(session_id=self.session_id, user_message="hi")
        )
        self.assertTrue(captured.get("plan_mode_active"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
