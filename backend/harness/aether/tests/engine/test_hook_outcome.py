from __future__ import annotations

import copy
import unittest
from typing import Any

from aether import AgentEngine, HookOutcome
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.runtime.hooks import EngineHooks
from aether.tools.base import ToolDescriptor


class RecordingProvider(ModelProvider):
    provider_name = "test-provider"
    api_mode = "chat"

    def __init__(
        self,
        responses: list[NormalizedResponse] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.model = "test-model"
        self.responses = list(responses or [])
        self.error = error
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
        del context, stream_callback, stream_silent_callback
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": list(tools),
                "config": config,
            }
        )
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise RuntimeError("no scripted response")
        return self.responses.pop(0)


class UserContextHook(EngineHooks):
    def pre_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        context_metadata: dict[str, Any],
    ) -> HookOutcome | None:
        del session_id, iteration, messages, context_metadata
        return HookOutcome(inject_user_context="Memory: foo")


class SystemAddendumHook(EngineHooks):
    def pre_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        context_metadata: dict[str, Any],
    ) -> HookOutcome | None:
        del session_id, iteration, messages, context_metadata
        return HookOutcome(inject_system_addendum="Use terse answers.")


class ShortCircuitHook(EngineHooks):
    def pre_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        context_metadata: dict[str, Any],
    ) -> HookOutcome | None:
        del session_id, iteration, messages, context_metadata
        return HookOutcome(
            short_circuit_response=NormalizedResponse(content="from hook")
        )


class FailingPreLlmHook(EngineHooks):
    def pre_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        context_metadata: dict[str, Any],
    ) -> HookOutcome | None:
        del session_id, iteration, messages, context_metadata
        raise RuntimeError("hook failed")


class ApiHookRecorder(EngineHooks):
    def __init__(self) -> None:
        self.pre_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []

    def pre_api_request(self, **kwargs: Any) -> None:
        self.pre_calls.append(dict(kwargs))

    def post_api_request(self, **kwargs: Any) -> None:
        self.post_calls.append(dict(kwargs))


class HookOutcomeTests(unittest.TestCase):
    def test_inject_user_context_only_affects_provider_payload(self) -> None:
        provider = RecordingProvider([NormalizedResponse(content="ok")])
        engine = AgentEngine(
            provider,
            hooks=UserContextHook(),
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(session_id="hook-user", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(len(provider.calls), 1)
        provider_user = provider.calls[0]["messages"][-1]
        self.assertEqual(provider_user["role"], "user")
        self.assertIn("<hook_context>", provider_user["content"])
        self.assertIn("Memory: foo", provider_user["content"])
        self.assertEqual(result.messages[0]["content"], "hello")

    def test_inject_system_addendum_only_affects_provider_payload(self) -> None:
        provider = RecordingProvider([NormalizedResponse(content="ok")])
        engine = AgentEngine(
            provider,
            hooks=SystemAddendumHook(),
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="hook-system",
                user_message="hello",
                system_message="Base system",
            )
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        provider_system = provider.calls[0]["messages"][0]
        self.assertEqual(provider_system["role"], "system")
        self.assertIn("Base system", provider_system["content"])
        self.assertIn("<hook_system_addendum>", provider_system["content"])
        self.assertIn("Use terse answers.", provider_system["content"])
        self.assertEqual(result.messages[0]["content"], "Base system")

    def test_short_circuit_response_skips_provider_generate(self) -> None:
        provider = RecordingProvider(error=AssertionError("must not call provider"))
        engine = AgentEngine(
            provider,
            hooks=ShortCircuitHook(),
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(session_id="hook-short", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "from hook")
        self.assertEqual(provider.calls, [])

    def test_api_hooks_fire_on_success_with_request_metadata(self) -> None:
        hooks = ApiHookRecorder()
        provider = RecordingProvider([NormalizedResponse(content="ok")])
        engine = AgentEngine(
            provider,
            hooks=hooks,
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(session_id="hook-api", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(len(hooks.pre_calls), 1)
        self.assertEqual(len(hooks.post_calls), 1)
        pre = hooks.pre_calls[0]
        post = hooks.post_calls[0]
        self.assertEqual(pre["session_id"], "hook-api")
        self.assertEqual(pre["iteration"], 1)
        self.assertEqual(pre["model"], "test-model")
        self.assertEqual(pre["provider"], "test-provider")
        self.assertEqual(pre["api_mode"], "chat")
        self.assertEqual(pre["api_call_count"], 1)
        self.assertEqual(pre["message_count"], 1)
        self.assertEqual(pre["tool_count"], 0)
        self.assertGreater(pre["request_char_count"], 0)
        self.assertEqual(post["api_call_count"], 1)
        self.assertEqual(post["response_finish_reason"], "stop")
        self.assertIsNone(post["error"])

    def test_api_post_hook_receives_provider_error(self) -> None:
        hooks = ApiHookRecorder()
        error = RuntimeError("provider exploded")
        provider = RecordingProvider(error=error)
        engine = AgentEngine(
            provider,
            hooks=hooks,
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(session_id="hook-api-error", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(len(hooks.pre_calls), 1)
        self.assertEqual(len(hooks.post_calls), 1)
        self.assertIs(hooks.post_calls[0]["error"], error)
        self.assertIsNone(hooks.post_calls[0]["response_finish_reason"])

    def test_pre_llm_hook_exception_is_isolated(self) -> None:
        provider = RecordingProvider([NormalizedResponse(content="ok")])
        engine = AgentEngine(
            provider,
            hooks=FailingPreLlmHook(),
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(session_id="hook-fails", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "ok")
        self.assertEqual(len(provider.calls), 1)


if __name__ == "__main__":
    unittest.main()
