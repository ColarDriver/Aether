"""Log service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LogFileSummary:
    key: str
    name: str
    path: str
    exists: bool
    size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class LogReadResult:
    file: str
    path: str
    exists: bool
    lines: list[str] = field(default_factory=list)
    available_files: list[LogFileSummary] = field(default_factory=list)


__all__ = ["LogFileSummary", "LogReadResult"]
