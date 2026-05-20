from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse
from aether.runtime.core.turn_metadata import INTERNAL_METADATA_KEYS


class EngineMetadataContractTests(unittest.TestCase):
    def test_engine_result_turn_metadata_filters_runtime_refs(self) -> None:
        provider = ScriptedProvider([NormalizedResponse(content="ok")])
        engine = AgentEngine(provider)

        result = engine.run_turn(
            EngineRequest(
                session_id="metadata-contract",
                user_message="hello",
                metadata={"custom_observation": {"kept": True}},
            )
        )

        turn = result.metadata["turn"]
        self.assertEqual(turn["custom_observation"], {"kept": True})
        for key in INTERNAL_METADATA_KEYS:
            self.assertNotIn(key, turn)

    def test_engine_result_keeps_stable_top_level_metadata(self) -> None:
        provider = ScriptedProvider([NormalizedResponse(content="ok")])
        engine = AgentEngine(provider)

        result = engine.run_turn(
            EngineRequest(session_id="metadata-top-level", user_message="hello")
        )

        for key in (
            "request",
            "turn",
            "runtime",
            "usage",
            "api_calls",
            "memory",
            "trajectory",
            "resource_cleanup",
            "iteration_budget",
            "exit",
            "reasoning",
            "compaction",
        ):
            self.assertIn(key, result.metadata)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
