"""Skill service implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aether.services.skills.contracts import (
    SkillCatalogResult,
    SkillSource,
    SkillSummary,
)


CatalogFactory = Callable[[Any | None], Any]


class SkillService:
    """Read-only skill catalog service."""

    def __init__(
        self,
        *,
        config: Any | None = None,
        catalog: Any | None = None,
        catalog_factory: CatalogFactory | None = None,
    ) -> None:
        self._config = config
        self._catalog = catalog
        self._catalog_factory = catalog_factory

    def list_skills(self) -> SkillCatalogResult:
        skills = [_skill_to_summary(skill) for skill in self._catalog_instance().list_all()]
        skills.sort(key=lambda item: item.name.lower())
        return SkillCatalogResult(skills=skills)

    def get_skill(self, name: str) -> SkillSummary | None:
        normalized = name.strip()
        if not normalized:
            return None
        skill = self._catalog_instance().get(normalized)
        if skill is None:
            return None
        return _skill_to_summary(skill)

    def _catalog_instance(self) -> Any:
        if self._catalog is not None:
            return self._catalog
        if self._catalog_factory is not None:
            self._catalog = self._catalog_factory(self._config)
            return self._catalog
        from aether.runtime.tools.skill_catalog import build_default_skill_catalog

        self._catalog = build_default_skill_catalog(self._config)
        return self._catalog


def _skill_to_summary(skill: Any) -> SkillSummary:
    return SkillSummary(
        name=str(getattr(skill, "name", "") or ""),
        description=str(getattr(skill, "description", "") or ""),
        when_to_use=str(getattr(skill, "when_to_use", "") or ""),
        source=SkillSource(
            source=str(getattr(skill, "source", "") or "local"),
            path=str(getattr(skill, "path", "")) or None,
        ),
        version=getattr(skill, "version", None),
    )


__all__ = ["SkillService"]
