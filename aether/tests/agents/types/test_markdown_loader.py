from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aether.agents.types.markdown_loader import load_markdown_agent_type


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class MarkdownLoaderTests(unittest.TestCase):
    def test_full_frontmatter_parses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                Path(tmp) / "reviewer.md",
                """---
name: code-reviewer
description: Reviews PRs
tools: [read_file, grep]
disallowed_tools: [write_file]
model: sonnet
skills: [security-review, tdd]
max_turns: 30
isolation: worktree
background: true
---
You are a code reviewer.
""",
            )
            definition = load_markdown_agent_type(path)
            assert definition is not None
            self.assertEqual(definition.agent_type, "code-reviewer")
            self.assertEqual(definition.tools, ("read_file", "grep"))
            self.assertEqual(definition.disallowed_tools, ("write_file",))
            self.assertEqual(definition.model, "sonnet")
            self.assertEqual(definition.skills, ("security-review", "tdd"))
            self.assertEqual(definition.max_turns, 30)
            self.assertEqual(definition.isolation, "worktree")
            self.assertTrue(definition.background)
            self.assertEqual(definition.system_prompt, "You are a code reviewer.")

    def test_filename_stem_fallback_for_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                Path(tmp) / "reviewer.md",
                """---
description: Reviews PRs
---
Body
""",
            )
            definition = load_markdown_agent_type(path)
            assert definition is not None
            self.assertEqual(definition.agent_type, "reviewer")

    def test_missing_description_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "reviewer.md", "---\nname: reviewer\n---\nBody\n")
            self.assertIsNone(load_markdown_agent_type(path))

    def test_list_field_bracket_and_comma_forms_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            one = _write(
                Path(tmp) / "one.md",
                """---
description: d
tools: [read_file, grep]
---
Body
""",
            )
            two = _write(
                Path(tmp) / "two.md",
                """---
description: d
tools: read_file, grep
---
Body
""",
            )
            def_one = load_markdown_agent_type(one)
            def_two = load_markdown_agent_type(two)
            assert def_one is not None and def_two is not None
            self.assertEqual(def_one.tools, def_two.tools)

    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(load_markdown_agent_type(Path("/no/such/agent.md")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
