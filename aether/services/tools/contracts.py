"""Tool service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolSummary:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    tools: list[ToolSummary] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolGroup:
    name: str
    tools: list[ToolSummary] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolAvailability:
    name: str
    enabled: bool
    reason: str | None = None


__all__ = [
    "ToolAvailability",
    "ToolCatalog",
    "ToolGroup",
    "ToolSummary",
]
