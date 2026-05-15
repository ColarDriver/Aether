from __future__ import annotations

import copy
import unittest
from pathlib import Path
from typing import Any

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse, StreamDeltaCallback, StreamSilentCallback, TurnContext
from aether.runtime.tools.skill_catalog import Skill, SkillCatalog
from aether.tools.base import ToolDescriptor


class _StaticCatalog(SkillCatalog):
    def __init__(self, skills: list[Skill]) -> None:
        super().__init__(search_paths=[])
        self._skills = {skill.name: skill for skill in skills}
        self._loaded = True


class RecordingProvider(ModelProvider):
    provider_name = "test-provider"
    api_mode = "chat"

    def __init__(self, responses: list[NormalizedResponse] | None = None) -> None:
        self.model = "test-model"
        self.responses = list(responses or [NormalizedResponse(content="ok")])
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


class SkillListingInjectionTests(unittest.TestCase):
    def _catalog(self) -> SkillCatalog:
        return _StaticCatalog(
            [
                Skill(name="alpha-skill", path=Path("/tmp/alpha/SKILL.md"), description="Alpha helper"),
                Skill(name="beta-skill", path=Path("/tmp/beta/SKILL.md"), description="Beta helper"),
            ]
        )

    def test_turn_0_system_includes_listing(self) -> None:
        provider = RecordingProvider()
        engine = AgentEngine(
            provider,
            config=EngineConfig(use_builtin_tools=False),
            skill_catalog=self._catalog(),
        )

        engine.run_turn(
            EngineRequest(session_id="skill-list-1", user_message="What skills are available?", system_message="You are helpful")
        )

        system = provider.calls[0]["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("<system-reminder>", system["content"])
        self.assertIn("alpha-skill", system["content"])
        self.assertIn("beta-skill", system["content"])

    def test_disabled_via_config(self) -> None:
        provider = RecordingProvider()
        engine = AgentEngine(
            provider,
            # Disable verification and faithful-reporting directives
            # so this test asserts skill-listing behaviour in
            # isolation, not directive injection.
            config=EngineConfig(
                use_builtin_tools=False,
                skill_listing_enabled=False,
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
                verifier_gate_enabled=False,
            ),
            skill_catalog=self._catalog(),
        )

        engine.run_turn(
            EngineRequest(session_id="skill-list-2", user_message="What skills are available?", system_message="You are helpful")
        )

        system = provider.calls[0]["messages"][0]
        self.assertEqual(system["content"], "You are helpful")

    def test_no_listing_when_catalog_empty(self) -> None:
        provider = RecordingProvider()
        engine = AgentEngine(
            provider,
            config=EngineConfig(
                use_builtin_tools=False,
                verification_directive_enabled=False,
                faithful_reporting_enabled=False,
                verifier_gate_enabled=False,
            ),
            skill_catalog=_StaticCatalog([]),
        )

        engine.run_turn(
            EngineRequest(session_id="skill-list-3", user_message="What skills are available?", system_message="You are helpful")
        )

        system = provider.calls[0]["messages"][0]
        self.assertEqual(system["content"], "You are helpful")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
