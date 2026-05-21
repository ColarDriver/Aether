"""Documentation service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DocSummary:
    path: str
    title: str
    size_bytes: int
    updated_at: float


@dataclass(frozen=True, slots=True)
class DocIndex:
    root: str
    default_path: str | None = None
    documents: list[DocSummary] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DocContent:
    path: str
    title: str
    content: str
    size_bytes: int
    updated_at: float


__all__ = ["DocContent", "DocIndex", "DocSummary"]
