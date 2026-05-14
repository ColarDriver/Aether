from __future__ import annotations

import unittest
from pathlib import Path

from aether.runtime.tools.skill_catalog import Skill, SkillCatalog
from aether.runtime.tools.skill_listing import format_skill_listing


class _StaticCatalog(SkillCatalog):
    def __init__(self, skills: list[Skill]) -> None:
        super().__init__(search_paths=[])
        self._skills = {skill.name: skill for skill in skills}
        self._loaded = True


class SkillListingTests(unittest.TestCase):
    def test_empty_catalog_returns_empty_string(self) -> None:
        catalog = _StaticCatalog([])
        self.assertEqual(format_skill_listing(catalog), "")

    def test_lists_all_within_budget(self) -> None:
        catalog = _StaticCatalog(
            [
                Skill(name="alpha", path=Path("/tmp/alpha/SKILL.md"), description="Alpha desc"),
                Skill(name="beta", path=Path("/tmp/beta/SKILL.md"), description="Beta desc"),
                Skill(name="gamma", path=Path("/tmp/gamma/SKILL.md"), description="Gamma desc"),
            ]
        )

        rendered = format_skill_listing(catalog, budget_chars=10_000)

        self.assertIn("alpha", rendered)
        self.assertIn("beta", rendered)
        self.assertIn("gamma", rendered)
        self.assertTrue(rendered.startswith("<system-reminder>\n"))
        self.assertTrue(rendered.endswith("</system-reminder>"))

    def test_truncates_when_over_budget(self) -> None:
        skills = [
            Skill(
                name=f"skill-{idx:02d}",
                path=Path(f"/tmp/skill-{idx:02d}/SKILL.md"),
                description="x" * 40,
            )
            for idx in range(50)
        ]
        catalog = _StaticCatalog(skills)

        rendered = format_skill_listing(catalog, budget_chars=400)

        self.assertIn("more skills omitted", rendered)
        self.assertRegex(rendered, r"\.\.\. \((\d+) more skills omitted; ")
        self.assertLessEqual(len(rendered), 400)

    def test_too_small_budget_returns_empty(self) -> None:
        catalog = _StaticCatalog(
            [Skill(name="alpha", path=Path("/tmp/alpha/SKILL.md"), description="Alpha desc")]
        )

        rendered = format_skill_listing(catalog, budget_chars=40)

        self.assertEqual(rendered, "")

    def test_when_to_use_field_rendered(self) -> None:
        catalog = _StaticCatalog(
            [
                Skill(
                    name="tdd",
                    path=Path("/tmp/tdd/SKILL.md"),
                    description="Use tests first.",
                    when_to_use="Before implementing new behavior.",
                )
            ]
        )

        rendered = format_skill_listing(catalog)
        self.assertIn("(use when: Before implementing new behavior.)", rendered)

    def test_sort_is_alphabetical(self) -> None:
        catalog = _StaticCatalog(
            [
                Skill(name="zeta", path=Path("/tmp/zeta/SKILL.md"), description="z"),
                Skill(name="alpha", path=Path("/tmp/alpha/SKILL.md"), description="a"),
                Skill(name="middle", path=Path("/tmp/middle/SKILL.md"), description="m"),
            ]
        )

        rendered = format_skill_listing(catalog)
        alpha_idx = rendered.index("- alpha:")
        middle_idx = rendered.index("- middle:")
        zeta_idx = rendered.index("- zeta:")
        self.assertLess(alpha_idx, middle_idx)
        self.assertLess(middle_idx, zeta_idx)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
