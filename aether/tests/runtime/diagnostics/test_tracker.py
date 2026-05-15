"""Unit tests for DiagnosticTracker."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from aether.runtime.diagnostics.tracker import DiagnosticTracker
from aether.runtime.diagnostics.types import Diagnostic


def _diag(message: str, *, line: int = 1, col: int = 1, source: str = "pyright") -> Diagnostic:
    return Diagnostic(
        message=message,
        severity="error",
        line=line,
        column=col,
        source=source,
    )


class _ScriptedLSPManager:
    """Stand-in for :class:`LSPManager` driven by an explicit response queue."""

    def __init__(
        self,
        diagnostics_per_path: dict[Path, list[list[Diagnostic]]] | None = None,
        *,
        block_forever: bool = False,
    ) -> None:
        self.change_calls: list[tuple[Path, str]] = []
        self.save_calls: list[tuple[Path, str | None]] = []
        self.pull_calls: list[Path] = []
        self._queue = diagnostics_per_path or {}
        self._block_forever = block_forever

    def change_file(self, path: Path, content: str) -> None:
        self.change_calls.append((Path(path), content))

    def save_file(self, path: Path, *, content: str | None = None) -> None:
        self.save_calls.append((Path(path), content))

    def pull_diagnostics(self, path: Path, *, deadline: float) -> list[Diagnostic]:
        if self._block_forever:
            # Honour deadline so the test doesn't hang.
            remaining = max(0.0, deadline - time.perf_counter())
            time.sleep(remaining)
            return []
        self.pull_calls.append(Path(path))
        queue = self._queue.get(Path(path).resolve())
        if not queue:
            return []
        return queue.pop(0)


class DisabledTrackerTests(unittest.TestCase):
    def test_no_lsp_means_disabled(self) -> None:
        tracker = DiagnosticTracker(None)
        self.assertFalse(tracker.enabled)
        tracker.before_file_edited(Path("/tmp/x.py"))  # no-op
        tracker.notify_file_changed(Path("/tmp/x.py"), "content")
        self.assertEqual(tracker.get_new_diagnostics(), [])
        tracker.clear_delivered()  # no-op


class BaselineDiffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path("/tmp/aether-tracker-test.py").resolve()

    def test_baseline_freezes_current_set(self) -> None:
        # Server returns [A] before edit, then [A, B] after.
        a = _diag("A")
        b = _diag("B", line=2)
        lsp = _ScriptedLSPManager({self.path: [[a], [a, b], [a, b]]})
        tracker = DiagnosticTracker(lsp, settle_timeout_ms=50)

        tracker.before_file_edited(self.path)  # consumes the [A] snapshot

        result = tracker.get_new_diagnostics([self.path])
        self.assertEqual(len(result), 1)
        diags = result[0].diagnostics
        self.assertEqual(diags, (b,))

    def test_dedup_within_session(self) -> None:
        a = _diag("A")
        b = _diag("B", line=2)
        lsp = _ScriptedLSPManager({self.path: [[a], [a, b], [a, b]]})
        tracker = DiagnosticTracker(lsp, settle_timeout_ms=50)

        tracker.before_file_edited(self.path)
        first = tracker.get_new_diagnostics([self.path])
        self.assertEqual(len(first), 1)

        # Second call: same diagnostic still present but already delivered.
        second = tracker.get_new_diagnostics([self.path])
        self.assertEqual(second, [])

    def test_clear_delivered_lets_diagnostic_resurface(self) -> None:
        a = _diag("A")
        b = _diag("B", line=2)
        # Three [a, b] responses needed: after baseline + after first
        # get + after second get.
        lsp = _ScriptedLSPManager({self.path: [[a], [a, b], [a, b]]})
        tracker = DiagnosticTracker(lsp, settle_timeout_ms=50)

        tracker.before_file_edited(self.path)
        tracker.get_new_diagnostics([self.path])
        tracker.clear_delivered(self.path)
        third = tracker.get_new_diagnostics([self.path])
        self.assertEqual(len(third), 1)
        self.assertEqual(third[0].diagnostics, (b,))

    def test_paths_none_walks_all_baselines(self) -> None:
        p1 = Path("/tmp/aether-tracker-1.py").resolve()
        p2 = Path("/tmp/aether-tracker-2.py").resolve()
        b1 = _diag("err1")
        b2 = _diag("err2")
        lsp = _ScriptedLSPManager(
            {
                p1: [[], [b1]],  # baseline empty, then [b1]
                p2: [[], [b2]],
            }
        )
        tracker = DiagnosticTracker(lsp, settle_timeout_ms=50)
        tracker.before_file_edited(p1)
        tracker.before_file_edited(p2)

        result = tracker.get_new_diagnostics()  # paths=None
        paths = {file.path for file in result}
        self.assertEqual(paths, {p1, p2})

    def test_fetch_timeout_returns_empty_baseline(self) -> None:
        """Hung LSP must not block the edit pipeline."""
        lsp = _ScriptedLSPManager(block_forever=True)
        tracker = DiagnosticTracker(lsp, settle_timeout_ms=20)
        t0 = time.perf_counter()
        tracker.before_file_edited(self.path)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Allow some scheduling slack; 250 ms is generous for a 20 ms cap.
        self.assertLess(elapsed_ms, 250.0)

    def test_notify_file_changed_invokes_lsp_helpers(self) -> None:
        lsp = _ScriptedLSPManager()
        tracker = DiagnosticTracker(lsp, settle_timeout_ms=50)
        tracker.notify_file_changed(self.path, "new content")
        self.assertEqual(len(lsp.change_calls), 1)
        self.assertEqual(len(lsp.save_calls), 1)
        change_path, change_content = lsp.change_calls[0]
        self.assertEqual(change_path, self.path)
        self.assertEqual(change_content, "new content")
        save_path, save_content = lsp.save_calls[0]
        self.assertEqual(save_path, self.path)
        # save_file is called with content too (lets servers re-lint).
        self.assertEqual(save_content, "new content")

    def test_notify_file_changed_swallows_lsp_exceptions(self) -> None:
        lsp = MagicMock()
        lsp.change_file.side_effect = RuntimeError("server down")
        tracker = DiagnosticTracker(lsp, settle_timeout_ms=50)
        # Should not raise.
        tracker.notify_file_changed(self.path, "ignored")

    def test_resolves_path_arg_to_absolute(self) -> None:
        """Relative paths handed to ``before_file_edited`` are resolved
        so subsequent ``get_new_diagnostics([same_rel_path])`` finds the
        baseline regardless of how each call spelled the path."""
        rel = Path("./tmp-rel.py")
        absolute = rel.resolve()
        a = _diag("A")
        lsp = _ScriptedLSPManager({absolute: [[], [a]]})
        tracker = DiagnosticTracker(lsp, settle_timeout_ms=50)
        tracker.before_file_edited(rel)
        result = tracker.get_new_diagnostics([rel])
        self.assertEqual(len(result), 1)


class ThreadSafetyTests(unittest.TestCase):
    """Two threads racing on the same tracker must produce sane state."""

    def test_concurrent_baselines_and_gets(self) -> None:
        lsp = _ScriptedLSPManager({Path("/tmp/race.py").resolve(): [[], []]})
        tracker = DiagnosticTracker(lsp, settle_timeout_ms=50)

        errors: list[BaseException] = []

        def worker(idx: int) -> None:
            try:
                tracker.before_file_edited(Path("/tmp/race.py"))
                tracker.get_new_diagnostics([Path("/tmp/race.py")])
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        self.assertEqual(errors, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
