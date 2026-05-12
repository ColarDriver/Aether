"""Unit tests for ``aether.runtime.tools.skill_catalog``.

Sprint 3.5 / PR-2 (PR 3.5.8).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from aether.runtime.tools.skill_catalog import Skill, SkillCatalog


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


_FRONTMATTERED = """\
---
name: tdd
description: Test-driven development guidance.
whenToUse: Use before writing any new feature.
version: 1.2.0
---

# TDD body
Some markdown content.
"""

_NAKED = """\
# Just a markdown file
No frontmatter, no metadata.
"""

_WEIRD_FENCE = """\
---
name: weird
description: missing closing fence
This file has no closing --- so the parser must treat
it as having no frontmatter at all.
"""


class CatalogDiscoveryTests(unittest.TestCase):
    def test_a1_single_dir_two_skills_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "alpha" / "SKILL.md", _NAKED)
            _write(root / "beta" / "SKILL.md", _NAKED)
            cat = SkillCatalog(search_paths=[root])
            cat.discover()
            self.assertEqual(len(cat.list_all()), 2)

    def test_a2_nested_dir_name_uses_dash_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "a" / "b" / "c" / "SKILL.md", _NAKED)
            cat = SkillCatalog(search_paths=[root])
            cat.discover()
            names = {s.name for s in cat.list_all()}
            self.assertIn("a-b-c", names)

    def test_a3_frontmatter_overrides_path_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "ignored-path-name" / "SKILL.md", _FRONTMATTERED)
            cat = SkillCatalog(search_paths=[root])
            skill = cat.get("tdd")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(skill.name, "tdd")
            self.assertEqual(skill.description, "Test-driven development guidance.")
            self.assertEqual(skill.when_to_use, "Use before writing any new feature.")
            self.assertEqual(skill.version, "1.2.0")
            self.assertIn("# TDD body", skill.body)
            self.assertNotIn("---", skill.body)

    def test_a4_later_search_path_overrides_earlier_on_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r1 = root / "first"
            r2 = root / "second"
            _write(r1 / "shared" / "SKILL.md", "---\nname: shared\ndescription: from-first\n---\nold")
            _write(r2 / "shared" / "SKILL.md", "---\nname: shared\ndescription: from-second\n---\nnew")
            cat = SkillCatalog(search_paths=[r1, r2])
            skill = cat.get("shared")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(skill.description, "from-second")
            self.assertIn("new", skill.body)

    def test_a5_missing_search_path_skipped_silently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "x" / "SKILL.md", _NAKED)
            cat = SkillCatalog(search_paths=[root, root / "missing", Path("/nope/never/here")])
            cat.discover()
            names = {s.name for s in cat.list_all()}
            self.assertEqual(names, {"x"})

    def test_a6_discover_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "x" / "SKILL.md", _NAKED)
            cat = SkillCatalog(search_paths=[root])
            cat.discover()
            cat.discover()
            self.assertEqual(len(cat.list_all()), 1)

    def test_a7_force_rediscovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "x" / "SKILL.md", _NAKED)
            cat = SkillCatalog(search_paths=[root])
            cat.discover()
            _write(root / "y" / "SKILL.md", _NAKED)
            cat.discover()  # idempotent path: stale
            self.assertEqual(len(cat.list_all()), 1)
            cat.discover(force=True)
            self.assertEqual(len(cat.list_all()), 2)


class CatalogLookupTests(unittest.TestCase):
    def _build(self, tmp: str) -> SkillCatalog:
        root = Path(tmp)
        _write(root / "alpha" / "SKILL.md", _FRONTMATTERED)
        _write(root / "Beta-Camel" / "SKILL.md", _NAKED)
        return SkillCatalog(search_paths=[root])

    def test_b1_get_unknown_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cat = self._build(tmp)
            self.assertIsNone(cat.get("does-not-exist"))

    def test_b2_get_empty_string_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cat = self._build(tmp)
            self.assertIsNone(cat.get(""))

    def test_b3_get_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cat = self._build(tmp)
            self.assertIsNotNone(cat.get("TDD"))

    def test_b4_get_strips_leading_slash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cat = self._build(tmp)
            self.assertIsNotNone(cat.get("/tdd"))

    def test_b5_get_exact_match_case_sensitive_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cat = self._build(tmp)
            # "Beta-Camel" -> name from path with mixed case
            self.assertIsNotNone(cat.get("Beta-Camel"))


class FrontmatterEdgeCases(unittest.TestCase):
    def test_c1_no_frontmatter_returns_empty_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "x" / "SKILL.md", _NAKED)
            cat = SkillCatalog(search_paths=[root])
            skill = cat.get("x")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(skill.description, "")
            self.assertEqual(skill.when_to_use, "")
            self.assertIsNone(skill.version)

    def test_c2_unclosed_frontmatter_treated_as_no_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "weird-dir" / "SKILL.md", _WEIRD_FENCE)
            cat = SkillCatalog(search_paths=[root])
            skill = cat.get("weird-dir")
            # Fallback to path name; whole text becomes body since
            # fence parsing failed.
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(skill.name, "weird-dir")
            self.assertIn("missing closing fence", skill.body)
            self.assertEqual(skill.description, "")

    def test_c3_quoted_frontmatter_value_unwrapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md = '---\nname: q\ndescription: "wrapped in quotes"\n---\nbody\n'
            _write(root / "qdir" / "SKILL.md", md)
            cat = SkillCatalog(search_paths=[root])
            skill = cat.get("q")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(skill.description, "wrapped in quotes")

    def test_c4_nested_metadata_block_skipped_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md = (
                "---\n"
                "name: nested\n"
                "description: top level\n"
                "metadata:\n"
                "  tags: [a, b, c]\n"
                "  related_skills: [foo, bar]\n"
                "---\n"
                "body content\n"
            )
            _write(root / "nested" / "SKILL.md", md)
            cat = SkillCatalog(search_paths=[root])
            skill = cat.get("nested")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(skill.description, "top level")
            self.assertIn("body content", skill.body)


class FilesystemErrorTests(unittest.TestCase):
    def test_d1_unreadable_skill_skipped(self) -> None:
        # Permissions trick may not work as root (this CI env is root,
        # so we simulate unreadable by deleting between scan and parse
        # with a monkey-patched read_text).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok = _write(root / "good" / "SKILL.md", _NAKED)
            broken = _write(root / "broken" / "SKILL.md", _NAKED)
            original_read_text = Path.read_text

            def boom(self: Path, *args, **kwargs) -> str:
                if self == broken:
                    raise OSError("simulated read failure")
                return original_read_text(self, *args, **kwargs)

            from unittest.mock import patch

            with patch.object(Path, "read_text", boom):
                cat = SkillCatalog(search_paths=[root])
                cat.discover()
            names = {s.name for s in cat.list_all()}
            self.assertIn("good", names)
            self.assertNotIn("broken", names)
            self.assertEqual(len(cat.list_all()), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
