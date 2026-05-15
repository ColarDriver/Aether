from __future__ import annotations

import json
import unittest
from pathlib import Path

from aether.agents.types import AgentTypeDefinition


class AgentTypeDefinitionTests(unittest.TestCase):
    def test_to_snapshot_is_json_safe(self) -> None:
        definition = AgentTypeDefinition(
            agent_type="code-reviewer",
            description="Review changes",
            system_prompt="Check correctness",
            tools=("read_file", "grep"),
            disallowed_tools=("write_file",),
            model="sonnet",
            skills=("security-review",),
            max_turns=25,
            isolation="worktree",
            background=True,
            source="project",
            source_path=Path("/tmp/.claude/agents/code-reviewer.md"),
        )

        snapshot = definition.to_snapshot()
        encoded = json.dumps(snapshot)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["agent_type"], "code-reviewer")
        self.assertEqual(decoded["tools"], ["read_file", "grep"])
        self.assertEqual(decoded["source"], "project")
        self.assertTrue(decoded["source_path"].endswith("code-reviewer.md"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
