from __future__ import annotations

import copy
import logging
import unittest
from typing import Any

from aether import AgentEngine, HookOutcome
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    EngineStatus,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.runtime.core.hooks import EngineHooks
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


def _surrogate_error() -> UnicodeEncodeError:
    return UnicodeEncodeError("utf-8", "\udcff", 0, 1, "surrogates not allowed")


def _ascii_error() -> UnicodeEncodeError:
    return UnicodeEncodeError("ascii", "é", 0, 1, "ordinal not in range(128)")


def _contains_surrogate(value: Any) -> bool:
    if isinstance(value, str):
        return any(0xD800 <= ord(ch) <= 0xDFFF for ch in value)
    if isinstance(value, list):
        return any(_contains_surrogate(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_surrogate(k) or _contains_surrogate(v) for k, v in value.items())
    return False


def _contains_non_ascii(value: Any) -> bool:
    if isinstance(value, str):
        return any(ord(ch) >= 128 for ch in value)
    if isinstance(value, list):
        return any(_contains_non_ascii(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_non_ascii(k) or _contains_non_ascii(v) for k, v in value.items())
    if isinstance(value, ToolDescriptor):
        return _contains_non_ascii(
            {
                "name": value.name,
                "description": value.description,
                "parameters": value.parameters,
                "required": value.required,
            }
        )
    return False


class FlakyUnicodeProvider(ModelProvider):
    provider_name = "test-provider"
    api_mode = "chat"

    def __init__(
        self,
        errors: list[Exception],
        *,
        content: str = "ok",
        api_key: str = "sk-test",
    ) -> None:
        self.model = "test-model"
        self.errors = list(errors)
        self.content = content
        self.api_key = api_key
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
        del config, context, stream_callback, stream_silent_callback
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "api_key": self.api_key,
            }
        )
        if self.errors:
            raise self.errors.pop(0)
        return NormalizedResponse(content=self.content)


class SurrogateHook(EngineHooks):
    def pre_llm_call(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        context_metadata: dict[str, Any],
    ) -> HookOutcome | None:
        del session_id, iteration, messages, context_metadata
        return HookOutcome(inject_user_context="bad\udcffcontext")


class NonAsciiTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="lookup",
            description="lookup São Paulo",
            parameters={"properties": {"city": {"description": "München"}}},
        )

    def execute(self, call, context):  # pragma: no cover - not dispatched here
        raise AssertionError("not used")


class UnicodePayloadRecoveryTests(unittest.TestCase):
    def test_surrogate_error_sanitizes_provider_payload_and_retries(self) -> None:
        provider = FlakyUnicodeProvider([_surrogate_error()])
        engine = AgentEngine(
            provider,
            hooks=SurrogateHook(),
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(session_id="unicode-surrogate", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(_contains_surrogate(provider.calls[0]["messages"]))
        self.assertFalse(_contains_surrogate(provider.calls[1]["messages"]))
        self.assertEqual(result.metadata["turn"].get("unicode_recovery_passes"), 1)
        self.assertEqual(result.metadata["turn"].get("unicode_recovery_reason"), "surrogate")

    def test_ascii_error_sanitizes_provider_api_key_and_retries(self) -> None:
        logger = logging.getLogger("unicode-test")
        provider = FlakyUnicodeProvider([_ascii_error()], api_key="sk-ʋalue")
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False),
            logger=logger,
        )

        with self.assertLogs("unicode-test", level="WARNING") as logs:
            result = engine.run_turn(
                EngineRequest(session_id="unicode-ascii", user_message="hello")
            )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(provider.api_key, "sk-alue")
        self.assertEqual(provider.calls[1]["api_key"], "sk-alue")
        self.assertEqual(result.metadata["turn"].get("unicode_recovery_passes"), 1)
        self.assertTrue(result.metadata["turn"].get("force_ascii_payload"))
        self.assertIn("credential/header contained non-ASCII", "\n".join(logs.output))

    def test_ascii_error_sanitizes_tool_schema_before_retry(self) -> None:
        provider = FlakyUnicodeProvider([_ascii_error()])
        registry = ToolRegistry()
        registry.register(NonAsciiTool())
        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(session_id="unicode-tool", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(_contains_non_ascii(provider.calls[0]["tools"]))
        self.assertFalse(_contains_non_ascii(provider.calls[1]["tools"]))

    def test_unicode_recovery_is_limited_to_two_passes(self) -> None:
        provider = FlakyUnicodeProvider(
            [_surrogate_error(), _surrogate_error(), _surrogate_error()]
        )
        engine = AgentEngine(
            provider,
            hooks=SurrogateHook(),
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(session_id="unicode-limit", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(result.metadata["turn"].get("unicode_recovery_passes"), 2)

    def test_non_unicode_error_does_not_enter_unicode_recovery(self) -> None:
        provider = FlakyUnicodeProvider([RuntimeError("boom")])
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(session_id="unicode-non-match", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn("unicode_recovery_passes", result.metadata["turn"])


if __name__ == "__main__":
    unittest.main()
