from __future__ import annotations

import logging
import unittest
from typing import Any

from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.agents.runtime.provider_invocation import (
    ProviderInvocationController,
    ProviderInvocationRequest,
)
from aether.config.schema import ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.control.interrupts import InterruptController
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse, ToolCall, TurnContext
from aether.runtime.core.hooks import EngineHooks
from aether.runtime.core.services import EngineServices
from aether.runtime.recovery.fallback_chain import FallbackChain, ProviderSlot
from aether.runtime.recovery.provider_errors import ProviderInvocationError, ResponseInvalidError
from aether.runtime.recovery.strategies import GenericBackoffStrategy
from aether.tools.base import ToolDescriptor
from aether.tools.registry import ToolRegistry


class _ScriptedProvider(ModelProvider):
    provider_name = "openai"
    api_mode = "chat"

    def __init__(
        self,
        response: NormalizedResponse | None = None,
        *,
        error: Exception | None = None,
        valid: bool = True,
        model: str = "unit-model",
        silent_delta: str | None = None,
    ) -> None:
        self.response = response or NormalizedResponse(content="ok")
        self.error = error
        self.valid = valid
        self.model = model
        self.silent_delta = silent_delta
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: Any = None,
        stream_silent_callback: Any = None,
    ) -> NormalizedResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "config": config,
                "context": context,
            }
        )
        if self.silent_delta and stream_silent_callback is not None:
            stream_silent_callback(self.silent_delta)
        if self.error is not None:
            raise self.error
        return self.response

    def validate_response(self, response: NormalizedResponse) -> tuple[bool, list[str]]:
        del response
        if self.valid:
            return True, []
        return False, ["missing choices"]


class _HookRecorder(EngineHooks):
    def __init__(self) -> None:
        self.pre_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []

    def pre_api_request(self, **kwargs: Any) -> None:
        self.pre_calls.append(dict(kwargs))

    def post_api_request(self, **kwargs: Any) -> None:
        self.post_calls.append(dict(kwargs))


def _services(provider: ModelProvider, *, chain: FallbackChain | None = None) -> EngineServices:
    return EngineServices(
        provider=provider,
        tool_registry=ToolRegistry(),
        middleware_pipeline=MiddlewarePipeline(),
        interrupt_controller=InterruptController(),
        logger=logging.getLogger(__name__),
        recovery_strategy=GenericBackoffStrategy(),
        fallback_chain=chain,
    )


def _invocation(
    *,
    response_provider: ModelProvider,
    hooks: EngineHooks,
    chain: FallbackChain | None = None,
    tools: list[ToolDescriptor] | None = None,
    stream_silent_callback: Any = None,
) -> tuple[ProviderInvocationController, ProviderInvocationRequest, TurnContext]:
    controller = ProviderInvocationController(
        services=_services(response_provider, chain=chain),
        hooks=hooks,
    )
    context = TurnContext(session_id="provider-test", iteration=2, metadata={})
    request = EngineRequest(
        session_id="provider-test",
        model_config=ModelCallConfig(max_tokens=123, extra={"model": "override-model"}),
    )
    messages = [{"role": "user", "content": "hello"}]
    invocation = ProviderInvocationRequest(
        request=request,
        canonical_messages=list(messages),
        prepared_messages=list(messages),
        tools=tools or [],
        call_config=request.model_config,
        context=context,
        stream_silent_callback=stream_silent_callback,
    )
    return controller, invocation, context


class ProviderInvocationControllerTests(unittest.TestCase):
    def test_text_response_is_returned_and_hooks_fire_once(self) -> None:
        hooks = _HookRecorder()
        provider = _ScriptedProvider(NormalizedResponse(content="hello"))
        controller, invocation, context = _invocation(response_provider=provider, hooks=hooks)

        result = controller.invoke(invocation)

        self.assertIsNone(result.error)
        assert result.response is not None
        self.assertEqual(result.response.content, "hello")
        self.assertEqual(result.provider_name, "openai")
        self.assertEqual(result.api_mode, "chat")
        self.assertEqual(result.model, "override-model")
        self.assertEqual(context.metadata["_api_request_attempt_count"], 1)
        self.assertEqual(len(hooks.pre_calls), 1)
        self.assertEqual(len(hooks.post_calls), 1)
        self.assertIsNone(hooks.post_calls[0]["error"])
        self.assertEqual(hooks.pre_calls[0]["message_count"], 1)
        self.assertEqual(hooks.pre_calls[0]["max_tokens"], 123)

    def test_tool_calls_are_preserved(self) -> None:
        call = ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"})
        hooks = _HookRecorder()
        provider = _ScriptedProvider(NormalizedResponse(tool_calls=[call]))
        controller, invocation, _context = _invocation(response_provider=provider, hooks=hooks)

        result = controller.invoke(invocation)

        assert result.response is not None
        self.assertEqual(result.response.tool_calls, [call])

    def test_invalid_response_returns_response_invalid_error(self) -> None:
        hooks = _HookRecorder()
        provider = _ScriptedProvider(NormalizedResponse(content="bad"), valid=False)
        controller, invocation, _context = _invocation(response_provider=provider, hooks=hooks)

        result = controller.invoke(invocation)

        self.assertIsInstance(result.error, ResponseInvalidError)
        self.assertEqual(len(hooks.post_calls), 1)
        self.assertIs(hooks.post_calls[0]["error"], result.error)

    def test_provider_exception_is_returned_and_post_hook_sees_it(self) -> None:
        error = ProviderInvocationError(status_code=500, body_summary="boom")
        hooks = _HookRecorder()
        provider = _ScriptedProvider(error=error)
        controller, invocation, _context = _invocation(response_provider=provider, hooks=hooks)

        result = controller.invoke(invocation)

        self.assertIs(result.error, error)
        self.assertEqual(len(hooks.pre_calls), 1)
        self.assertEqual(len(hooks.post_calls), 1)
        self.assertIs(hooks.post_calls[0]["error"], error)

    def test_silent_stream_callback_is_forwarded(self) -> None:
        hooks = _HookRecorder()
        seen: list[str] = []
        provider = _ScriptedProvider(silent_delta="hidden-json")
        controller, invocation, _context = _invocation(
            response_provider=provider,
            hooks=hooks,
            stream_silent_callback=seen.append,
        )

        result = controller.invoke(invocation)

        self.assertIsNone(result.error)
        self.assertEqual(seen, ["hidden-json"])

    def test_controller_reads_new_active_provider_after_fallback_rotation(self) -> None:
        first = _ScriptedProvider(NormalizedResponse(content="first"), model="first-model")
        second = _ScriptedProvider(NormalizedResponse(content="second"), model="second-model")
        chain = FallbackChain(
            [
                ProviderSlot(name="first", factory=lambda: first),
                ProviderSlot(name="second", factory=lambda: second),
            ]
        )
        self.assertTrue(chain.activate_next())
        hooks = _HookRecorder()
        controller, invocation, _context = _invocation(
            response_provider=first,
            hooks=hooks,
            chain=chain,
        )

        result = controller.invoke(invocation)

        assert result.response is not None
        self.assertEqual(result.response.content, "second")
        self.assertEqual(len(first.calls), 0)
        self.assertEqual(len(second.calls), 1)

    def test_usage_accumulation_matches_public_contract(self) -> None:
        hooks = _HookRecorder()
        provider = _ScriptedProvider(
            NormalizedResponse(
                content="hello",
                metadata={
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                    }
                },
            )
        )
        controller, invocation, context = _invocation(response_provider=provider, hooks=hooks)
        result = controller.invoke(invocation)

        assert result.response is not None
        controller.accumulate_usage(result.response, context)

        self.assertEqual(context.metadata["api_calls"], 1)
        self.assertEqual(context.metadata["usage_accumulator"].total_tokens, 10)


if __name__ == "__main__":
    unittest.main()
