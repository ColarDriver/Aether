from __future__ import annotations

import unittest
from pathlib import Path

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse
from aether.runtime.tools.skill_catalog import Skill, SkillCatalog
from aether.subagents import SubagentManager, SubagentTask
from aether.subagents.default_builder import DefaultSubagentBuilder


class _StaticCatalog(SkillCatalog):
    def __init__(self, skills: list[Skill]) -> None:
        super().__init__(search_paths=[])
        self._skills = {skill.name: skill for skill in skills}
        self._loaded = True


class SkillListingSubagentTests(unittest.TestCase):
    def test_child_inherits_skill_listing_config_and_catalog(self) -> None:
        parent_catalog = _StaticCatalog(
            [Skill(name="alpha-skill", path=Path("/tmp/alpha/SKILL.md"), description="Alpha")]
        )
        parent = AgentEngine(
            provider=ScriptedProvider([NormalizedResponse(content="parent")]),
            config=EngineConfig(
                skill_listing_enabled=True,
                skill_listing_token_budget=1234,
            ),
            subagent_manager=SubagentManager(),
            skill_catalog=parent_catalog,
        )
        task = SubagentTask(
            task_id="child-task",
            goal="work",
            request=EngineRequest(session_id="child-session", user_message="do work"),
            provider=ScriptedProvider([NormalizedResponse(content="child")]),
        )

        child = DefaultSubagentBuilder().build_child(parent, task, 1)

        self.assertTrue(child.config.skill_listing_enabled)
        self.assertEqual(child.config.skill_listing_token_budget, 1234)
        self.assertIs(child._skill_catalog, parent_catalog)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
