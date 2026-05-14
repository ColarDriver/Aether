from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch
import aether.cli.main as cli_main

from aether.cli.main import cmd_chat
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

    def test_cli_cmd_chat_builds_engine_with_skill_catalog(self) -> None:
        captured: dict[str, object] = {}

        class _Provider:
            model = "test-model"

        class _Engine:
            def __init__(self, *args, **kwargs) -> None:
                del args
                captured["skill_catalog"] = kwargs.get("skill_catalog")

        args = argparse.Namespace(
            log_level="warning",
            provider="openai",
            model=None,
            api_key=None,
            base_url=None,
            max_iterations=4,
            no_builtin_tools=False,
            temperature=None,
            max_tokens=None,
            system=None,
            system_file=None,
            session=None,
            resume=None,
            verbose=False,
            no_banner=True,
        )

        with patch.object(cli_main, "_augment_system_prompt", return_value="sys"), \
             patch("aether.cli.providers.build_provider", return_value=_Provider()), \
             patch("aether.cli.repl.run_repl"), \
             patch("aether.subagents.SubagentManager"), \
             patch("aether.agents.core.agent.AgentEngine", _Engine):
            cmd_chat(args)

        self.assertIsInstance(captured.get("skill_catalog"), SkillCatalog)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
