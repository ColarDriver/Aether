"""Tests for diagnostic attachment rendering / collection."""

from __future__ import annotations

import unittest
from pathlib import Path

from aether.runtime.diagnostics.attachments import (
    collect_pending_diagnostics,
    render_diagnostics_block,
)
from aether.runtime.diagnostics.tracker import DiagnosticTracker
from aether.runtime.diagnostics.types import Diagnostic, DiagnosticFile


def _diag(message: str, *, line: int = 1, column: int = 1, code: str | None = None) -> Diagnostic:
    return Diagnostic(
        message=message,
        severity="error",
        line=line,
        column=column,
        source="pyright",
        code=code,
    )


class _StubManager:
    """Mimic the slice of ``LSPManager`` that DiagnosticTracker consumes."""

    def __init__(self, queues: dict[Path, list[list[Diagnostic]]]) -> None:
        self._queues = queues
        self.pull_calls: list[Path] = []

    def change_file(self, path: Path, content: str) -> None:  # pragma: no cover
        return None

    def save_file(self, path: Path, *, content: str | None = None) -> None:  # pragma: no cover
        return None

    def pull_diagnostics(self, path: Path, *, deadline: float) -> list[Diagnostic]:
        del deadline
        self.pull_calls.append(Path(path))
        queue = self._queues.get(Path(path).resolve())
        if not queue:
            return []
        return queue.pop(0)


class RenderTests(unittest.TestCase):
    def test_empty_returns_empty_string(self) -> None:
        self.assertEqual(render_diagnostics_block([]), "")

    def test_includes_path_severity_and_message(self) -> None:
        path = Path("/tmp/x.py")
        block = render_diagnostics_block(
            [
                DiagnosticFile(
                    path=path,
                    diagnostics=(
                        _diag("undefined name 'foo'", line=2, column=4, code="reportUndefinedVariable"),
                    ),
                )
            ]
        )
        self.assertIn("<diagnostics>", block)
        self.assertIn("</diagnostics>", block)
        self.assertIn(str(path), block)
        self.assertIn("ERROR", block)  # severity upper-cased
        self.assertIn("2:4", block)
        self.assertIn("reportUndefinedVariable", block)
        self.assertIn("undefined name 'foo'", block)

    def test_sorts_within_file_by_line_then_column(self) -> None:
        block = render_diagnostics_block(
            [
                DiagnosticFile(
                    path=Path("/tmp/x.py"),
                    diagnostics=(
                        _diag("late err", line=5, column=1),
                        _diag("early err", line=2, column=10),
                        _diag("early earlier", line=2, column=1),
                    ),
                )
            ]
        )
        # The earliest message should appear before later ones in source order.
        early_idx = block.index("early earlier")
        mid_idx = block.index("early err")
        late_idx = block.index("late err")
        self.assertLess(early_idx, mid_idx)
        self.assertLess(mid_idx, late_idx)

    def test_skips_code_brackets_when_no_code(self) -> None:
        block = render_diagnostics_block(
            [
                DiagnosticFile(
                    path=Path("/tmp/x.py"),
                    diagnostics=(_diag("missing import"),),
                )
            ]
        )
        # No "[code]" bracket since diagnostic has no code attached.
        self.assertNotIn("[None]", block)
        self.assertNotIn("[]", block)


class CollectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path("/tmp/aether-attach.py").resolve()

    def test_disabled_tracker_returns_empty(self) -> None:
        self.assertEqual(collect_pending_diagnostics(None), [])
        disabled = DiagnosticTracker(None)
        self.assertEqual(collect_pending_diagnostics(disabled), [])

    def test_paths_none_drains_all_baselines(self) -> None:
        a = _diag("A")
        b = _diag("B", line=2)
        manager = _StubManager({self.path: [[], [a, b]]})
        tracker = DiagnosticTracker(manager, settle_timeout_ms=10)
        tracker.before_file_edited(self.path)
        result = collect_pending_diagnostics(tracker)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].path, self.path)
        self.assertEqual(set(result[0].diagnostics), {a, b})

    def test_paths_filter_scopes_to_those_files(self) -> None:
        p1 = Path("/tmp/aether-attach-1.py").resolve()
        p2 = Path("/tmp/aether-attach-2.py").resolve()
        d1 = _diag("err in 1")
        d2 = _diag("err in 2")
        manager = _StubManager(
            {
                p1: [[], [d1]],
                p2: [[], [d2]],
            }
        )
        tracker = DiagnosticTracker(manager, settle_timeout_ms=10)
        tracker.before_file_edited(p1)
        tracker.before_file_edited(p2)

        # Only ask about p1 → p2's diagnostic stays pending.
        scoped = collect_pending_diagnostics(tracker, paths=[str(p1)])
        self.assertEqual({f.path for f in scoped}, {p1})

    def test_empty_paths_list_returns_empty(self) -> None:
        manager = _StubManager({})
        tracker = DiagnosticTracker(manager, settle_timeout_ms=10)
        # ``paths=[]`` is the "we know the model edited nothing" signal;
        # the caller wants an empty result, not an all-baselines drain.
        self.assertEqual(collect_pending_diagnostics(tracker, paths=[]), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
