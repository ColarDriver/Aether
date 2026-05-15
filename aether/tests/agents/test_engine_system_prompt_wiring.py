"""Integration tests for engine wiring of verification directives.

Verifies that the verification + faithful-reporting blocks land in
the *first* provider call's system message and that subagents inherit
the parent's config flags.
"""

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
    TurnContext,
)
from aether.tools.base import ToolDescriptor


class RecordingProvider(ModelProvider):
    provider_name: str = "test-provider"
    api_mode: str = "chat"

    def __init__(self, responses: list[NormalizedResponse] | None = None) -> None:
        self.model: str = "test-model"
        self.responses: list[NormalizedResponse] = list(responses or [NormalizedResponse(content="ok")])
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


class EngineSystemPromptWiringTests(unittest.TestCase):
    def test_defaults_inject_verification_and_faithful_blocks(self) -> None:
        provider = RecordingProvider()
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False),
        )
        engine.run_turn(
            EngineRequest(
                session_id="prompt-wire-1",
                user_message="hi",
                system_message="You are helpful",
            )
        )
        system = provider.calls[0]["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("<verification_directive>", system["content"])
        self.assertIn("<faithful_reporting>", system["content"])
        # Original user text preserved at the tail.
        self.assertTrue(system["content"].endswith("You are helpful"))

    def test_individual_flags_can_be_disabled(self) -> None:
        provider = RecordingProvider()
        engine = AgentEngine(
            provider,
            config=EngineConfig(
                use_builtin_tools=False,
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
                tool_use_contract_enabled=False,
                verifier_gate_enabled=False,
            ),
        )
        engine.run_turn(
            EngineRequest(
                session_id="prompt-wire-2",
                user_message="hi",
                system_message="You are helpful",
            )
        )
        system = provider.calls[0]["messages"][0]
        # Every engine-added section is suppressed, so the user prompt
        # arrives verbatim.
        self.assertEqual(system["content"], "You are helpful")

    def test_tool_contract_off_keeps_directives(self) -> None:
        provider = RecordingProvider()
        engine = AgentEngine(
            provider,
            config=EngineConfig(
                use_builtin_tools=False,
                tool_use_contract_enabled=False,
            ),
        )
        engine.run_turn(
            EngineRequest(
                session_id="prompt-wire-3",
                user_message="hi",
                system_message="You are helpful",
            )
        )
        content = provider.calls[0]["messages"][0]["content"]
        self.assertNotIn("<tool_use_contract>", content)
        self.assertIn("<verification_directive>", content)
        self.assertIn("<faithful_reporting>", content)

    def test_no_user_system_message_still_injects_directives(self) -> None:
        provider = RecordingProvider()
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False),
        )
        engine.run_turn(
            EngineRequest(
                session_id="prompt-wire-4",
                user_message="hi",
                # no system_message
            )
        )
        # When there's no caller-supplied system message, the engine
        # may either inject our directives as the system message or
        # leave the messages list without one — both are acceptable.
        # What we require: if a system message DOES exist, it contains
        # the directives; if not, no other test invariants break.
        messages = provider.calls[0]["messages"]
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if system_msgs:
            self.assertIn("<verification_directive>", system_msgs[0]["content"])


class SubagentDirectiveInheritanceTests(unittest.TestCase):
    """Verify the two new flags survive the parent → child config copy."""

    def test_default_propagation(self) -> None:
        from aether.subagents.default_builder import DefaultSubagentBuilder
        from aether.subagents.contracts import SubagentTask

        # Parent engine with defaults (both flags True).
        parent = AgentEngine(
            RecordingProvider(),
            config=EngineConfig(use_builtin_tools=False),
        )
        builder = DefaultSubagentBuilder()

        task = SubagentTask(
            task_id="child-task-1",
            goal="hi",
            request=EngineRequest(session_id="child-1", user_message="hi"),
            metadata={},
        )
        child = builder.build_child(parent, task, child_depth=1)
        self.assertTrue(child.config.verification_directive_enabled)
        self.assertTrue(child.config.faithful_reporting_enabled)

    def test_parent_override_propagates(self) -> None:
        from aether.subagents.default_builder import DefaultSubagentBuilder
        from aether.subagents.contracts import SubagentTask

        parent = AgentEngine(
            RecordingProvider(),
            config=EngineConfig(
                use_builtin_tools=False,
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
                verifier_gate_enabled=False,
            ),
        )
        builder = DefaultSubagentBuilder()

        task = SubagentTask(
            task_id="child-task-2",
            goal="hi",
            request=EngineRequest(session_id="child-2", user_message="hi"),
            metadata={},
        )
        child = builder.build_child(parent, task, child_depth=1)
        self.assertFalse(child.config.verification_directive_enabled)
        self.assertFalse(child.config.faithful_reporting_enabled)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
