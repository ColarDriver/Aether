"""Diagnostics service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DiagnosticSummary:
    path: str
    message: str
    severity: str
    line: int
    column: int
    source: str
    code: str | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticFileSummary:
    path: str
    diagnostic_count: int


@dataclass(frozen=True, slots=True)
class DiagnosticsStatus:
    enabled: bool
    pending_files: list[DiagnosticFileSummary] = field(default_factory=list)
    pending_count: int = 0


@dataclass(frozen=True, slots=True)
class LspStatus:
    enabled: bool
    connected: bool
    detail: str | None = None


__all__ = [
    "DiagnosticFileSummary",
    "DiagnosticSummary",
    "DiagnosticsStatus",
    "LspStatus",
]
