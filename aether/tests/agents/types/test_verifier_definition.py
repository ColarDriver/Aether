from __future__ import annotations

import unittest

from aether.agents.types import AgentTypeRegistry, VERIFIER_AGENT_TYPE


class VerifierDefinitionTests(unittest.TestCase):
    def test_verifier_builtin_is_registered(self) -> None:
        registry = AgentTypeRegistry(search_paths=[])
        definition = registry.get("verifier")

        self.assertIsNotNone(definition)
        assert definition is not None
        self.assertEqual(definition.agent_type, VERIFIER_AGENT_TYPE)
        self.assertEqual(definition.source, "builtin")

    def test_verifier_is_read_and_verify_only(self) -> None:
        registry = AgentTypeRegistry(search_paths=[])
        definition = registry.get(VERIFIER_AGENT_TYPE)
        assert definition is not None

        self.assertIn("read_file", definition.tools or ())
        self.assertIn("shell", definition.tools or ())
        self.assertIn("lsp", definition.tools or ())
        for name in ("file_edit", "write_file", "notebook_edit", "task", "send_message"):
            self.assertIn(name, definition.disallowed_tools)
            self.assertNotIn(name, definition.tools or ())

    def test_verifier_prompt_requires_verdict_and_evidence(self) -> None:
        registry = AgentTypeRegistry(search_paths=[])
        definition = registry.get(VERIFIER_AGENT_TYPE)
        assert definition is not None

        self.assertIn("PASS", definition.system_prompt)
        self.assertIn("PARTIAL", definition.system_prompt)
        self.assertIn("FAIL", definition.system_prompt)
        self.assertIn("Evidence", definition.system_prompt)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
