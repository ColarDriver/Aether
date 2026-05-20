from __future__ import annotations

from pathlib import Path

from aether.runtime.diagnostics.types import Diagnostic, DiagnosticFile
from aether.services.diagnostics import DiagnosticsService


class _Tracker:
    enabled = True

    def __init__(self) -> None:
        self.calls = 0

    def get_new_diagnostics(self, paths=None):  # noqa: ANN001
        self.calls += 1
        target = Path("src/app.py") if paths is None else Path(next(iter(paths)))
        return [
            DiagnosticFile(
                path=target,
                diagnostics=(
                    Diagnostic(
                        message="bad import",
                        severity="error",
                        line=3,
                        column=5,
                        source="pyright",
                        code="reportMissingImports",
                    ),
                ),
            )
        ]


class _SnapshotTracker(_Tracker):
    def pending_snapshot(self):
        return [
            DiagnosticFile(
                path=Path("src/app.py"),
                diagnostics=(
                    Diagnostic(
                        message="bad import",
                        severity="error",
                        line=3,
                        column=5,
                        source="pyright",
                    ),
                ),
            )
        ]


def test_diagnostics_service_handles_missing_tracker() -> None:
    service = DiagnosticsService()

    assert service.status().enabled is False
    assert service.status().pending_count == 0
    assert service.lsp_status().connected is False
    assert service.recent() == []


def test_diagnostics_service_drains_recent_public_summaries() -> None:
    service = DiagnosticsService(tracker=_Tracker())

    recent = service.recent(paths=[Path("src/main.py")])

    assert recent[0].path == "src/main.py"
    assert recent[0].message == "bad import"
    assert recent[0].severity == "error"
    assert recent[0].code == "reportMissingImports"


def test_diagnostics_status_uses_safe_snapshot_when_available() -> None:
    status = DiagnosticsService(tracker=_SnapshotTracker()).status()

    assert status.enabled is True
    assert status.pending_count == 1
    assert status.pending_files[0].path == "src/app.py"
