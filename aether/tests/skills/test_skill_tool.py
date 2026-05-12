"""Tests for ``aether.tools.builtins.skill``.

Sprint 3.5 / PR-2 (PR 3.5.8).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aether.config.schema import EngineConfig
from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.runtime.tools.skill_catalog import SkillCatalog
from aether.tools.builtins.skill import SkillTool


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


_TDD_BODY = """\
---
name: tdd
description: Test-driven development guidance.
whenToUse: Before writing a feature.
version: 0.1.0
---
# TDD

Run the failing test first.  $ARGUMENTS will appear here.
Session: ${AETHER_SESSION_ID}
"""


def _ctx(session_id: str = "ses-skill", *, config: EngineConfig | None = None) -> TurnContext:
    return TurnContext(
        session_id=session_id,
        iteration=0,
        metadata={"_engine_config": config or EngineConfig()},
    )


def _build_tool() -> tuple[SkillTool, Path]:
    tmp = Path(tempfile.mkdtemp())
    _write(tmp / "tdd" / "SKILL.md", _TDD_BODY)
    _write(tmp / "huge" / "SKILL.md", "---\nname: huge\n---\n" + ("X" * 80_000))
    catalog = SkillCatalog(search_paths=[tmp])
    return SkillTool(catalog=catalog), tmp


class HappyPathTests(unittest.TestCase):
    def test_b1_loads_known_skill(self) -> None:
        tool, _ = _build_tool()
        result = tool.execute(
            ToolCall(id="c1", name="skill", arguments={"skill": "tdd"}),
            _ctx(),
        )
        self.assertFalse(result.is_error, result.content)
        self.assertIn("Loaded skill: tdd", result.content)
        self.assertIn("Run the failing test first", result.content)
        self.assertIn("BEGIN SKILL", result.content)
        self.assertIn("END SKILL", result.content)
        self.assertEqual(result.metadata["skill_name"], "tdd")
        self.assertEqual(result.metadata["skill_version"], "0.1.0")

    def test_b4_args_substitution(self) -> None:
        tool, _ = _build_tool()
        result = tool.execute(
            ToolCall(
                id="c1",
                name="skill",
                arguments={"skill": "tdd", "args": "feature X"},
            ),
            _ctx(),
        )
        self.assertIn("feature X", result.content)
        self.assertNotIn("$ARGUMENTS", result.content)
        self.assertIn("Arguments: feature X", result.content)

    def test_b5_session_id_substitution(self) -> None:
        tool, _ = _build_tool()
        result = tool.execute(
            ToolCall(id="c1", name="skill", arguments={"skill": "tdd"}),
            _ctx(session_id="my-cool-session"),
        )
        self.assertIn("Session: my-cool-session", result.content)
        self.assertNotIn("${AETHER_SESSION_ID}", result.content)

    def test_b6_oversize_skill_spills(self) -> None:
        tool, _ = _build_tool()
        result = tool.execute(
            ToolCall(id="c1", name="skill", arguments={"skill": "huge"}),
            _ctx(session_id="ses-skill-spill"),
        )
        self.assertFalse(result.is_error)
        self.assertIn("output truncated", result.content)


class FailureTests(unittest.TestCase):
    def test_b2_unknown_skill_lists_available(self) -> None:
        tool, _ = _build_tool()
        result = tool.execute(
            ToolCall(id="c1", name="skill", arguments={"skill": "does-not-exist"}),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("unknown skill", result.content)
        self.assertIn("tdd", result.content)

    def test_b3_empty_skill_name_rejected(self) -> None:
        tool, _ = _build_tool()
        result = tool.execute(
            ToolCall(id="c1", name="skill", arguments={"skill": "  "}),
            _ctx(),
        )
        self.assertTrue(result.is_error)

    def test_b7_disabled_via_config(self) -> None:
        cfg = EngineConfig(skill_tool_enabled=False)
        tool, _ = _build_tool()
        result = tool.execute(
            ToolCall(id="c1", name="skill", arguments={"skill": "tdd"}),
            _ctx(config=cfg),
        )
        self.assertTrue(result.is_error)
        self.assertIn("disabled", result.content)

    def test_b8_no_catalog_returns_error(self) -> None:
        tool = SkillTool()
        cfg = EngineConfig(skill_search_paths=(Path("/nonexistent/never/here"),))
        ctx = TurnContext(
            session_id="s", iteration=0, metadata={"_engine_config": cfg}
        )
        result = tool.execute(
            ToolCall(id="c1", name="skill", arguments={"skill": "anything"}),
            ctx,
        )
        # With a configured (but empty) path the catalog builds OK; the
        # skill itself is not found.  Either way it must be an error
        # without crashing.
        self.assertTrue(result.is_error)

    def test_b9_args_must_be_string(self) -> None:
        tool, _ = _build_tool()
        result = tool.execute(
            ToolCall(
                id="c1",
                name="skill",
                arguments={"skill": "tdd", "args": [1, 2]},
            ),
            _ctx(),
        )
        self.assertTrue(result.is_error)


class CatalogResolutionTests(unittest.TestCase):
    def test_c1_metadata_catalog_used_when_constructor_empty(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        _write(tmp / "x" / "SKILL.md", _TDD_BODY)
        catalog = SkillCatalog(search_paths=[tmp])
        tool = SkillTool()  # no constructor arg
        ctx = TurnContext(
            session_id="s",
            iteration=0,
            metadata={"_engine_config": EngineConfig(), "_skill_catalog": catalog},
        )
        result = tool.execute(
            ToolCall(id="c1", name="skill", arguments={"skill": "tdd"}),
            ctx,
        )
        self.assertFalse(result.is_error, result.content)
        self.assertIn("Loaded skill: tdd", result.content)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
