"""Skill service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SkillSource:
    source: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class SkillSummary:
    name: str
    description: str = ""
    when_to_use: str = ""
    source: SkillSource = field(default_factory=lambda: SkillSource(source="local"))
    version: str | None = None


@dataclass(frozen=True, slots=True)
class SkillCatalogResult:
    skills: list[SkillSummary] = field(default_factory=list)


__all__ = [
    "SkillCatalogResult",
    "SkillSource",
    "SkillSummary",
]
