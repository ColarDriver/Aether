"""Workspace browser service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WorkspaceEntry:
    path: str
    name: str
    kind: str
    size_bytes: int | None = None
    updated_at: float | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceTree:
    root: str
    path: str
    parent_path: str | None
    entries: list[WorkspaceEntry] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    root: str
    path: str
    name: str
    content: str
    size_bytes: int
    updated_at: float
    language: str
    mime_type: str | None = None
    truncated: bool = False
    binary: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceSearchResult:
    root: str
    query: str
    entries: list[WorkspaceEntry] = field(default_factory=list)


__all__ = [
    "WorkspaceEntry",
    "WorkspaceFile",
    "WorkspaceSearchResult",
    "WorkspaceTree",
]
