"""Diagnostics service implementation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aether.runtime.diagnostics.attachments import collect_pending_diagnostics
from aether.runtime.diagnostics.types import Diagnostic, DiagnosticFile
from aether.services.diagnostics.contracts import (
    DiagnosticFileSummary,
    DiagnosticSummary,
    DiagnosticsStatus,
    LspStatus,
)


class DiagnosticsService:
    """Public-safe diagnostics read service."""

    def __init__(self, *, tracker: Any | None = None) -> None:
        self._tracker = tracker

    def status(self) -> DiagnosticsStatus:
        if not self._enabled:
            return DiagnosticsStatus(enabled=False)
        files = _pending_file_summaries(self._tracker)
        return DiagnosticsStatus(
            enabled=True,
            pending_files=files,
            pending_count=sum(item.diagnostic_count for item in files),
        )

    def recent(self, *, paths: Iterable[str | Path] | None = None) -> list[DiagnosticSummary]:
        files = collect_pending_diagnostics(self._tracker, paths=paths)
        summaries: list[DiagnosticSummary] = []
        for file in files:
            summaries.extend(_diagnostic_to_summary(file.path, diagnostic) for diagnostic in file.diagnostics)
        summaries.sort(key=lambda item: (item.path, item.line, item.column, item.message))
        return summaries

    def lsp_status(self) -> LspStatus:
        if self._tracker is None:
            return LspStatus(enabled=False, connected=False, detail="no diagnostic tracker configured")
        if not self._enabled:
            return LspStatus(enabled=False, connected=False, detail="diagnostics disabled")
        return LspStatus(enabled=True, connected=True)

    @property
    def _enabled(self) -> bool:
        return bool(self._tracker is not None and getattr(self._tracker, "enabled", False))


def _diagnostic_to_summary(path: Path, diagnostic: Diagnostic) -> DiagnosticSummary:
    return DiagnosticSummary(
        path=str(path),
        message=diagnostic.message,
        severity=diagnostic.severity,
        line=diagnostic.line,
        column=diagnostic.column,
        source=diagnostic.source,
        code=diagnostic.code,
    )


def _pending_file_summaries(tracker: Any) -> list[DiagnosticFileSummary]:
    # DiagnosticTracker intentionally does not expose a public pending-count
    # API. When tests or future adapters provide a safe snapshot hook, consume
    # that; otherwise report enabled readiness without draining diagnostics.
    snapshot = getattr(tracker, "pending_snapshot", None)
    if not callable(snapshot):
        return []
    try:
        files = snapshot()
    except Exception:  # noqa: BLE001
        return []
    out: list[DiagnosticFileSummary] = []
    if not isinstance(files, (list, tuple)):
        return []
    for file in files:
        if isinstance(file, DiagnosticFile):
            out.append(
                DiagnosticFileSummary(
                    path=str(file.path),
                    diagnostic_count=len(file.diagnostics),
                )
            )
    out.sort(key=lambda item: item.path)
    return out


__all__ = ["DiagnosticsService"]
