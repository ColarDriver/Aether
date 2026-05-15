"""Render :class:`DiagnosticTracker` output into an LLM-visible block.

Parity with ``open-claude-code/src/utils/attachments.ts``: the engine
drains the tracker on every PRE_LLM boundary, formats new diagnostics
into a ``<diagnostics>`` system-reminder block, and appends it as a
user-role message to the provider payload.  Model contract is fixed by
the ``<verification_directive>`` text: the model is told to treat
these blocks as authoritative.

Two public callables:

* :func:`render_diagnostics_block` — pure str renderer; returns ``""``
  for an empty file list so the engine can no-op cheaply.
* :func:`collect_pending_diagnostics` — drain the tracker, optionally
  scoped to the paths the model just edited.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from aether.runtime.diagnostics.tracker import DiagnosticTracker
from aether.runtime.diagnostics.types import DiagnosticFile


_HEADER = (
    "The following diagnostics appeared after your most recent edits. "
    "These were introduced by your changes (or newly visible because of "
    "them). Fix them before reporting the task complete."
)


def render_diagnostics_block(files: list[DiagnosticFile]) -> str:
    """Render a ``<diagnostics>`` system-reminder block.

    Returns ``""`` when *files* is empty.  Diagnostics within each file
    are sorted by ``(line, column)`` so the output is stable across
    runs — the dedup set in :class:`DiagnosticTracker` already
    deduplicates by structural identity, but a stable sort makes diffs
    easier to read in logs.
    """
    if not files:
        return ""
    lines: list[str] = ["<diagnostics>", _HEADER, ""]
    for file in files:
        lines.append(f"## {file.path}")
        sorted_diags = sorted(
            file.diagnostics, key=lambda d: (d.line, d.column, d.message)
        )
        for diag in sorted_diags:
            code = f" [{diag.code}]" if diag.code else ""
            lines.append(
                f"  {diag.severity.upper():<7} {diag.line}:{diag.column}  "
                f"{diag.source}{code}: {diag.message}"
            )
        lines.append("")
    lines.append("</diagnostics>")
    return "\n".join(lines)


def collect_pending_diagnostics(
    tracker: DiagnosticTracker | None,
    *,
    paths: Iterable[str | Path] | None = None,
) -> list[DiagnosticFile]:
    """Drain any new diagnostics queued by the tracker.

    *paths* is the list of files touched in the *most recent* turn,
    drawn from ``ToolResult.metadata['edited_paths']``.  When ``None``
    the tracker walks every path that has a baseline — covering the
    case where the model has not edited anything this turn but the
    tracker is still holding undelivered diagnostics from an earlier
    edit.

    Returns an empty list when the tracker is disabled (no LSP).
    """
    if tracker is None or not tracker.enabled:
        return []
    if paths is None:
        return tracker.get_new_diagnostics()
    resolved = [Path(p) for p in paths]
    if not resolved:
        return []
    return tracker.get_new_diagnostics(resolved)


__all__ = ["collect_pending_diagnostics", "render_diagnostics_block"]
