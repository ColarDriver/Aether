from __future__ import annotations

import copy
import unittest
from pathlib import Path
from typing import Any

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse, StreamDeltaCallback, StreamSilentCallback, ToolCall, ToolResult, TurnContext
from aether.runtime.tools.skill_catalog import Skill, SkillCatalog
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _StaticCatalog(SkillCatalog):
    def __init__(self, skills: list[Skill]) -> None:
        super().__init__(search_paths=[])
        self._skills = {skill.name: skill for skill in skills}
        self._loaded = True


class _PingTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="ping")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        return ToolResult(tool_call_id=call.id, name=call.name, content="pong")


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


class SkillNudgeIntervalTests(unittest.TestCase):
    def _catalog(self) -> SkillCatalog:
        return _StaticCatalog(
            [Skill(name="alpha-skill", path=Path("/tmp/alpha/SKILL.md"), description="Alpha helper")]
        )

    def _provider_for_iterations(self, iteration_count: int) -> RecordingProvider:
        responses: list[NormalizedResponse] = []
        for idx in range(iteration_count - 1):
            responses.append(
                NormalizedResponse(
                    content="",
                    tool_calls=[ToolCall(id=f"call-{idx + 1}", name="ping", arguments={})],
                )
            )
        responses.append(NormalizedResponse(content="done"))
        return RecordingProvider(responses)

    def _engine(self, provider: RecordingProvider, *, interval: int) -> AgentEngine:
        registry = ToolRegistry()
        registry.register(_PingTool())
        return AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(
                max_iterations=8,
                skill_nudge_interval=interval,
                skill_listing_enabled=True,
                use_builtin_tools=False,
            ),
            skill_catalog=self._catalog(),
        )

    def test_nudge_appears_on_interval(self) -> None:
        provider = self._provider_for_iterations(5)
        engine = self._engine(provider, interval=2)

        result = engine.run_turn(EngineRequest(session_id="skill-nudge-1", user_message="go"))

        self.assertEqual(result.final_response, "done")
        nudge_indices: list[int] = []
        for idx, call in enumerate(provider.calls, start=1):
            messages = call["messages"]
            if messages and messages[-1].get("metadata", {}).get("source") == "skill_nudge":
                nudge_indices.append(idx)
                self.assertEqual(messages[-1]["role"], "user")
                self.assertIn("<system-reminder>", messages[-1]["content"])
        self.assertEqual(nudge_indices, [2, 4])

    def test_nudge_zero_means_off(self) -> None:
        provider = self._provider_for_iterations(5)
        engine = self._engine(provider, interval=0)

        result = engine.run_turn(EngineRequest(session_id="skill-nudge-2", user_message="go"))

        self.assertEqual(result.final_response, "done")
        self.assertFalse(
            any(
                messages
                and messages[-1].get("metadata", {}).get("source") == "skill_nudge"
                for messages in (call["messages"] for call in provider.calls)
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
