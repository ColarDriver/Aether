"""Diagnostic tracking — baseline + diff for post-edit verification.

Public surface:

* :class:`Diagnostic` / :class:`DiagnosticFile` / :data:`Severity` —
  language-server-agnostic dataclasses (see :mod:`.types`).
* :class:`DiagnosticTracker` — service used by the engine to capture
  baselines, surface only NEW diagnostics, and dedup deliveries.
"""

from aether.runtime.diagnostics.attachments import (
    collect_pending_diagnostics,
    render_diagnostics_block,
)
from aether.runtime.diagnostics.tracker import DiagnosticTracker
from aether.runtime.diagnostics.types import Diagnostic, DiagnosticFile, Severity

__all__ = [
    "Diagnostic",
    "DiagnosticFile",
    "DiagnosticTracker",
    "Severity",
    "collect_pending_diagnostics",
    "render_diagnostics_block",
]
