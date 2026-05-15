"""Public diagnostic dataclasses, frozen for safe sharing across threads.

These shapes are language-server-agnostic; the LSP transport translates
``textDocument/publishDiagnostics`` payloads into ``Diagnostic`` /
``DiagnosticFile`` here, and downstream consumers (DiagnosticTracker,
attachment renderer) never look at raw LSP envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Severity = Literal["error", "warning", "info", "hint"]


@dataclass(slots=True, frozen=True)
class Diagnostic:
    """A single LSP diagnostic.

    ``line`` and ``column`` are 1-based for display.  ``source`` carries
    whichever LSP server flagged the issue (``"pyright"``, ``"tsserver"``,
    ``"ruff"``); ``code`` is the server-specific rule identifier when one
    is available (often ``None``).
    """

    message: str
    severity: Severity
    line: int
    column: int
    source: str
    code: str | None = None


@dataclass(slots=True, frozen=True)
class DiagnosticFile:
    """Diagnostics scoped to a single file."""

    path: Path
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)


__all__ = ["Diagnostic", "DiagnosticFile", "Severity"]
