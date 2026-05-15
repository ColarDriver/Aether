from __future__ import annotations

import unittest
from pathlib import Path

from aether.config.schema import EngineConfig
from aether.gateway.handlers.agent_methods import _build_engine_config
from aether.runtime.tools.skill_catalog import SkillCatalog, build_default_skill_catalog
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse
from aether.models.provider.scripted import ScriptedProvider
from aether import AgentEngine


class SkillListingWiringTests(unittest.TestCase):
    def test_build_default_skill_catalog_uses_configured_search_paths(self) -> None:
        config = EngineConfig(skill_search_paths=(Path("/tmp/a"), Path("/tmp/b")))

        catalog = build_default_skill_catalog(config)

        self.assertIsInstance(catalog, SkillCatalog)
        self.assertEqual(catalog.search_paths, [Path("/tmp/a"), Path("/tmp/b")])

    def test_gateway_engine_config_enables_skill_listing(self) -> None:
        config = _build_engine_config(None)
        self.assertTrue(config.skill_listing_enabled)

    def test_agent_engine_injects_listing_with_default_catalog_builder(self) -> None:
        skill_root = Path("/workspace/Aether/skills/creative/p5js")
        config = EngineConfig(use_builtin_tools=False, skill_listing_token_budget=20_000)
        provider = ScriptedProvider([NormalizedResponse(content="ok")])
        engine = AgentEngine(
            provider,
            config=config,
            skill_catalog=build_default_skill_catalog(config),
        )

        result = engine.run_turn(
            EngineRequest(session_id="wiring-1", user_message="What skills are available?", system_message="You are helpful")
        )

        self.assertEqual(result.messages[0]["role"], "system")
        self.assertIn("<system-reminder>", result.messages[0]["content"])
        self.assertIn("p5js", result.messages[0]["content"])
        self.assertTrue(skill_root.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
