from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aether.agents.types import AgentTypeRegistry


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class AgentTypeRegistryTests(unittest.TestCase):
    def test_builtins_present_and_case_insensitive_lookup(self) -> None:
        registry = AgentTypeRegistry(search_paths=[])
        names = {definition.agent_type for definition in registry.list_all()}
        self.assertIn("general-purpose", names)
        self.assertIn("Explore", names)
        self.assertIsNotNone(registry.get("explore"))
        self.assertEqual(registry.get("explore").agent_type, "Explore")  # type: ignore[union-attr]

    def test_markdown_override_builtin_last_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".claude" / "agents"
            _write(
                root / "explore.md",
                """---
name: Explore
description: Project override
---
Custom prompt
""",
            )
            registry = AgentTypeRegistry(search_paths=[root])
            definition = registry.get("Explore")
            assert definition is not None
            self.assertEqual(definition.description, "Project override")
            self.assertEqual(definition.source, "project")

    def test_invalid_markdown_skipped_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".claude" / "agents"
            _write(root / "bad.md", "---\nname: bad\n---\nNo description\n")
            registry = AgentTypeRegistry(search_paths=[root])
            names = {definition.agent_type for definition in registry.list_all()}
            self.assertNotIn("bad", names)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
