from __future__ import annotations

import copy
import unittest
from dataclasses import asdict
from typing import Any

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
from aether.runtime.recovery.error_classifier import FailoverReason, classify_api_error
from aether.runtime.recovery.provider_errors import ProviderInvocationError
from aether.runtime.recovery.strategies import ClassifiedRecoveryStrategy
from aether.runtime.recovery.schema_sanitizer import (
    sanitize_tool_descriptors,
    strip_pattern_and_format,
    strip_pattern_and_format_with_count,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _PatternTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="pattern_tool",
            description="Tool with local-backend-hostile schema keywords.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "pattern": r"^/",
                        "format": "uri",
                    }
                },
                "required": ["path"],
            },
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        return ToolResult(tool_call_id=call.id, name=call.name, content="ok")


class _CapturingGrammarProvider(ModelProvider):
    def __init__(
        self,
        *,
        failures_before_success: int,
        body_summary: str | None = None,
    ) -> None:
        self.failures_before_success = failures_before_success
        self.body_summary = body_summary or (
            "llama.cpp grammar compile failed: unsupported JSON schema keyword 'pattern'"
        )
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
        del messages, config, context, stream_callback, stream_silent_callback
        self.calls.append([copy.deepcopy(asdict(tool)) for tool in tools])
        if len(self.calls) <= self.failures_before_success:
            raise ProviderInvocationError(
                status_code=400,
                body_summary=self.body_summary,
            )
        return NormalizedResponse(content="ok")


def _registry_with_pattern_tool() -> tuple[ToolRegistry, _PatternTool]:
    tool = _PatternTool()
    registry = ToolRegistry()
    registry.register(tool)
    return registry, tool


class SchemaSanitizerUnitTests(unittest.TestCase):
    def test_strip_pattern_and_format_recursively_without_mutating_original(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "format": "json-schema",
                        "properties": {
                            "query": {
                                "type": "string",
                                "pattern": "^[a-z]+$",
                                "format": "regex",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string", "pattern": "^#"},
                            },
                        },
                    },
                },
            }
        ]
        original = copy.deepcopy(tools)

        cleaned, changed, removed = strip_pattern_and_format_with_count(tools)

        self.assertTrue(changed)
        self.assertEqual(removed, 4)
        self.assertEqual(tools, original)
        params = cleaned[0]["function"]["parameters"]
        self.assertNotIn("format", params)
        self.assertNotIn("pattern", params["properties"]["query"])
        self.assertNotIn("format", params["properties"]["query"])
        self.assertNotIn("pattern", params["properties"]["tags"]["items"])

    def test_strip_pattern_and_format_reports_unchanged_when_no_keys_match(self) -> None:
        tools = [{"function": {"parameters": {"type": "object", "properties": {}}}}]

        cleaned, changed = strip_pattern_and_format(tools)

        self.assertFalse(changed)
        self.assertEqual(cleaned, tools)
        self.assertIsNot(cleaned, tools)

    def test_sanitize_tool_descriptors_clones_parameters(self) -> None:
        descriptor = ToolDescriptor(
            name="demo",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string", "format": "uuid"}},
            },
        )

        sanitized, removed = sanitize_tool_descriptors([descriptor])

        self.assertEqual(removed, 1)
        self.assertEqual(descriptor.parameters["properties"]["value"]["format"], "uuid")
        self.assertNotIn("format", sanitized[0].parameters["properties"]["value"])


class SchemaSanitizerClassifierTests(unittest.TestCase):
    def test_classifier_identifies_llama_cpp_grammar_pattern_error(self) -> None:
        classified = classify_api_error(
            ProviderInvocationError(
                status_code=400,
                body_summary=(
                    "llama.cpp grammar compile failed: unsupported JSON schema "
                    "keyword 'pattern'"
                ),
            )
        )

        self.assertEqual(classified.reason, FailoverReason.llama_cpp_grammar_pattern)
        self.assertTrue(classified.retryable)
        self.assertFalse(classified.should_fallback)

    def test_non_grammar_format_error_does_not_trigger_sanitizer_reason(self) -> None:
        classified = classify_api_error(
            ProviderInvocationError(
                status_code=400,
                body_summary="invalid request format: temperature must be a number",
            )
        )

        self.assertEqual(classified.reason, FailoverReason.format_error)


class SchemaSanitizerEngineTests(unittest.TestCase):
    def test_provider_grammar_error_sanitizes_tools_and_retries_once(self) -> None:
        registry, tool = _registry_with_pattern_tool()
        provider = _CapturingGrammarProvider(failures_before_success=1)
        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(use_builtin_tools=False, max_iterations=2),
            recovery_strategy=ClassifiedRecoveryStrategy(
                max_attempts=3,
                base_wait_seconds=0.0,
            ),
        )

        result = engine.run_turn(
            EngineRequest(session_id="schema-sanitize-ok", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "ok")
        self.assertEqual(len(provider.calls), 2)
        first_params = provider.calls[0][0]["parameters"]["properties"]["path"]
        second_params = provider.calls[1][0]["parameters"]["properties"]["path"]
        self.assertIn("pattern", first_params)
        self.assertIn("format", first_params)
        self.assertNotIn("pattern", second_params)
        self.assertNotIn("format", second_params)
        self.assertEqual(
            tool.descriptor.parameters["properties"]["path"]["pattern"],
            r"^/",
        )
        self.assertTrue(result.metadata["turn"]["schema_sanitizer_applied"])
        self.assertEqual(result.metadata["turn"]["schema_sanitizer_removed_count"], 2)
        self.assertEqual(
            result.metadata["recovery"]["cascade_log"],
            ["schema_sanitizer(removed=2)"],
        )

    def test_sanitizer_failure_then_normal_recovery_surfaces_provider_error(self) -> None:
        registry, _tool = _registry_with_pattern_tool()
        provider = _CapturingGrammarProvider(failures_before_success=10)
        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(use_builtin_tools=False, max_iterations=2),
            recovery_strategy=ClassifiedRecoveryStrategy(
                max_attempts=2,
                base_wait_seconds=0.0,
            ),
        )

        result = engine.run_turn(
            EngineRequest(session_id="schema-sanitize-fail", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(result.metadata["turn"]["schema_sanitizer_applied"])
        self.assertEqual(result.metadata["turn"]["schema_sanitizer_removed_count"], 2)
        decisions = result.metadata["turn"]["recovery_decisions"]
        self.assertEqual(decisions[0]["retry"], True)
        self.assertEqual(decisions[1]["retry"], False)
        self.assertEqual(
            decisions[1]["classified_reason"],
            FailoverReason.llama_cpp_grammar_pattern.value,
        )

    def test_non_grammar_provider_error_does_not_sanitize(self) -> None:
        registry, _tool = _registry_with_pattern_tool()
        provider = _CapturingGrammarProvider(
            failures_before_success=10,
            body_summary="invalid request format: temperature must be a number",
        )
        engine = AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(use_builtin_tools=False, max_iterations=2),
            recovery_strategy=ClassifiedRecoveryStrategy(
                max_attempts=3,
                base_wait_seconds=0.0,
            ),
        )

        result = engine.run_turn(
            EngineRequest(session_id="schema-no-sanitize", user_message="hello")
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn("schema_sanitizer_applied", result.metadata["turn"])


if __name__ == "__main__":
    unittest.main()
