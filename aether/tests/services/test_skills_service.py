from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from aether.services.skills import SkillService


def test_skill_service_lists_discovered_skills_without_body(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "python"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: python-review\n"
        "description: Review Python changes\n"
        "whenToUse: Use for Python review\n"
        "version: 1.2.3\n"
        "---\n"
        "private body\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(skill_search_paths=(str(tmp_path / "skills"),))

    result = SkillService(config=config).list_skills()

    assert [skill.name for skill in result.skills] == ["python-review"]
    assert result.skills[0].description == "Review Python changes"
    assert result.skills[0].when_to_use == "Use for Python review"
    assert result.skills[0].source.source == "local"
    assert result.skills[0].source.path is not None
    assert "private body" not in repr(result)
    assert "body" not in asdict(result.skills[0])


def test_skill_service_tolerates_missing_metadata_and_gets_by_name(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "plain"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("body only\n", encoding="utf-8")
    config = SimpleNamespace(skill_search_paths=(str(tmp_path / "skills"),))
    service = SkillService(config=config)

    summary = service.get_skill("plain")

    assert summary is not None
    assert summary.name == "plain"
    assert summary.description == ""
    assert summary.when_to_use == ""
    assert summary.version is None
    assert service.get_skill("missing") is None
