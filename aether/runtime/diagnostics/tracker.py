"""Tracks per-file diagnostic state across edits.

Parity with ``open-claude-code/src/services/diagnosticTracking.ts``.

Flow:

  1. A write-tool decides "I'm about to edit ``foo.py``" → calls
     :meth:`before_file_edited` which freezes the current diagnostic
     list as the *baseline* for that file.
  2. The tool finishes; the engine fires ``post_tool_use`` and invokes
     :meth:`notify_file_changed`, which forwards new content to the
     language server via :class:`LSPManager`.
  3. On the next LLM turn, the attachment pipeline calls
     :meth:`get_new_diagnostics` which returns only diagnostics that
     are present *now* but were absent in the baseline AND have not
     already been delivered to the model.

The tracker is intentionally a passive cache: it never schedules
work, never calls the LSP directly except when one of its public
methods is invoked.  All blocking calls are bounded by
``settle_timeout_ms`` so a hung language server can never block the
edit pipeline.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from aether.runtime.diagnostics.types import Diagnostic, DiagnosticFile


_NO_DIAGNOSTICS: tuple[Diagnostic, ...] = ()


@runtime_checkable
class _LSPManagerProto(Protocol):
    """The slice of :class:`LSPManager` we actually use.

    Declared as a ``Protocol`` so the tracker accepts fakes in tests
    and any future alternative implementation (remote LSP, multi-server
    aggregator) without inheriting from :class:`LSPManager`.
    """

    def change_file(self, path: Path, content: str) -> None: ...
    def save_file(self, path: Path, *, content: str | None = ...) -> None: ...
    def pull_diagnostics(
        self, path: Path, *, deadline: float
    ) -> list[Diagnostic]: ...


class DiagnosticTracker:
    """Per-engine diagnostic tracker.

    Thread-safe via a single :class:`threading.Lock`; the lock holds
    only across dict mutations and never spans a network call, so
    contention is microseconds-scale.
    """

    def __init__(
        self,
        lsp_manager: "_LSPManagerProto | None",
        *,
        settle_timeout_ms: int = 1500,
    ) -> None:
        self._lsp = lsp_manager
        self._settle_timeout_ms = int(settle_timeout_ms)
        self._lock = threading.Lock()
        self._baselines: dict[Path, tuple[Diagnostic, ...]] = {}
        # Set of diagnostics already returned by ``get_new_diagnostics``
        # for each path — keyed by frozen ``Diagnostic`` instances so
        # equality is structural (message/severity/line/column/source/code).
        self._delivered: dict[Path, set[Diagnostic]] = {}

    @property
    def enabled(self) -> bool:
        """``False`` when no LSP manager was injected.

        Disabled trackers turn every public method into a no-op so
        callers (edit tools, attachment producers) can stay
        unconditional.
        """
        return self._lsp is not None

    # ------------------------------------------------------ baseline mgmt

    def before_file_edited(self, path: Path) -> None:
        """Snapshot current diagnostics for *path* as the baseline.

        No-op when :attr:`enabled` is ``False``.  Always overwrites the
        previous baseline for the path (last call wins) — the use case
        is "set baseline right before this specific write".
        """
        if self._lsp is None:
            return
        resolved = _resolve(path)
        current = self._fetch_blocking(resolved)
        with self._lock:
            self._baselines[resolved] = current

    def notify_file_changed(self, path: Path, content: str) -> None:
        """Tell the LSP server about new content + a save.

        Fire-and-forget by intent: exceptions raised by the manager
        are logged and swallowed so a flaky server never breaks the
        edit pipeline.  The combined didChange + didSave matches OCC's
        ``lspManager.changeFile`` + ``saveFile`` sequence.
        """
        if self._lsp is None:
            return
        resolved = _resolve(path)
        try:
            self._lsp.change_file(resolved, content)
            self._lsp.save_file(resolved, content=content)
        except Exception:
            # LSPManager already logs internally; this catch is the
            # last-resort guard against an upstream regression.
            return

    # ----------------------------------------------------- diff + delivery

    def get_new_diagnostics(
        self,
        paths: Iterable[Path] | None = None,
    ) -> list[DiagnosticFile]:
        """Return diagnostics introduced since the last baseline.

        When *paths* is ``None`` we walk every path that currently has
        a baseline.  Diagnostics already returned by a prior call are
        suppressed so the model only sees each issue once until it
        either fixes it (the diagnostic disappears) or the caller
        invokes :meth:`clear_delivered` to reset.
        """
        if self._lsp is None:
            return []

        with self._lock:
            if paths is None:
                target_paths = list(self._baselines.keys())
            else:
                target_paths = [_resolve(p) for p in paths]

        out: list[DiagnosticFile] = []
        for path in target_paths:
            current = self._fetch_blocking(path)
            with self._lock:
                baseline = self._baselines.get(path, _NO_DIAGNOSTICS)
                baseline_set: set[Diagnostic] = set(baseline)
                delivered = self._delivered.setdefault(path, set())
                new_items = tuple(
                    d for d in current if d not in baseline_set and d not in delivered
                )
                if new_items:
                    delivered.update(new_items)
            if new_items:
                out.append(DiagnosticFile(path=path, diagnostics=new_items))
        return out

    def clear_delivered(self, path: Path | None = None) -> None:
        """Drop the dedup set so prior diagnostics can resurface.

        Use case: the engine wants to re-remind the model of an
        outstanding issue at a periodic interval; the attachment
        producer calls this method first.
        """
        with self._lock:
            if path is None:
                self._delivered.clear()
            else:
                self._delivered.pop(_resolve(path), None)

    # ----------------------------------------------------------- internals

    def _fetch_blocking(self, path: Path) -> tuple[Diagnostic, ...]:
        """Pull the latest diagnostics, bounded by ``settle_timeout_ms``."""
        if self._lsp is None:
            return _NO_DIAGNOSTICS
        deadline = time.perf_counter() + self._settle_timeout_ms / 1000.0
        try:
            diagnostics = self._lsp.pull_diagnostics(path, deadline=deadline)
        except Exception:
            return _NO_DIAGNOSTICS
        return tuple(diagnostics)


def _resolve(path: Path) -> Path:
    return Path(path).resolve()


__all__ = ["DiagnosticTracker"]
