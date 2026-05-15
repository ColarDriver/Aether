"""Diagnostic wiring coverage for write-capable tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.runtime.diagnostics.tracker import DiagnosticTracker
from aether.runtime.diagnostics.types import Diagnostic
from aether.tools.builtins.file_edit import FileEditTool
from aether.tools.builtins.notebook_edit import NotebookEditTool
from aether.tools.builtins.write_file import WriteFileTool


def _ctx(tracker: DiagnosticTracker | None = None) -> TurnContext:
    metadata: dict[str, object] = {}
    if tracker is not None:
        metadata["_diagnostic_tracker"] = tracker
    return TurnContext(session_id="diagnostic-wire", iteration=1, metadata=metadata)


def _call(name: str, **args: object) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=args)


class _StubManager:
    def __init__(self) -> None:
        self.pull_calls: list[Path] = []
        self.change_calls: list[tuple[Path, str]] = []
        self.save_calls: list[tuple[Path, str | None]] = []

    def change_file(self, path: Path, content: str) -> None:
        self.change_calls.append((Path(path).resolve(), content))

    def save_file(self, path: Path, *, content: str | None = None) -> None:
        self.save_calls.append((Path(path).resolve(), content))

    def pull_diagnostics(self, path: Path, *, deadline: float) -> list[Diagnostic]:
        del deadline
        self.pull_calls.append(Path(path).resolve())
        return []


class FileEditDiagnosticWireTests(unittest.TestCase):
    def test_file_edit_calls_before_file_edited_and_emits_edited_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "code.py"
            path.write_text("def f():\n    return 1\n", encoding="utf-8")
            manager = _StubManager()
            tracker = DiagnosticTracker(manager, settle_timeout_ms=20)
            tool = FileEditTool()

            result = tool.execute(
                _call(
                    "file_edit",
                    path=str(path),
                    old_string="return 1",
                    new_string="return 2",
                ),
                _ctx(tracker),
            )

        self.assertFalse(result.is_error, result.content)
        self.assertEqual(manager.pull_calls, [path.resolve()])
        self.assertEqual(manager.change_calls, [])
        self.assertEqual(manager.save_calls, [])
        self.assertEqual(result.metadata["edited_paths"], [str(path.resolve())])

    def test_file_edit_works_without_tracker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "code.py"
            path.write_text("x = 1\n", encoding="utf-8")
            tool = FileEditTool()

            result = tool.execute(
                _call(
                    "file_edit",
                    path=str(path),
                    old_string="1",
                    new_string="2",
                ),
                _ctx(),
            )

            self.assertFalse(result.is_error, result.content)
            self.assertEqual(path.read_text(encoding="utf-8"), "x = 2\n")


class WriteFileDiagnosticWireTests(unittest.TestCase):
    def test_write_file_calls_before_file_edited_and_emits_edited_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = _StubManager()
            tracker = DiagnosticTracker(manager, settle_timeout_ms=20)
            tool = WriteFileTool(default_cwd=Path(tmp))

            result = tool.execute(
                _call("write_file", path="out.txt", content="hello world"),
                _ctx(tracker),
            )

            path = (Path(tmp) / "out.txt").resolve()

        self.assertFalse(result.is_error, result.content)
        self.assertEqual(manager.pull_calls, [path])
        self.assertEqual(manager.change_calls, [])
        self.assertEqual(manager.save_calls, [])
        self.assertEqual(result.metadata["edited_paths"], [str(path)])


class NotebookEditDiagnosticWireTests(unittest.TestCase):
    def test_notebook_edit_calls_before_file_edited_and_emits_edited_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.ipynb"
            notebook = {
                "cells": [
                    {
                        "cell_type": "code",
                        "id": "c1",
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": ["print('hi')\n"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            path.write_text(json.dumps(notebook), encoding="utf-8")
            manager = _StubManager()
            tracker = DiagnosticTracker(manager, settle_timeout_ms=20)
            tool = NotebookEditTool()

            result = tool.execute(
                _call(
                    "notebook_edit",
                    notebook_path=str(path),
                    edit_mode="replace",
                    cell_id="c1",
                    new_source="print('bye')\n",
                ),
                _ctx(tracker),
            )

        self.assertFalse(result.is_error, result.content)
        self.assertEqual(manager.pull_calls, [path.resolve()])
        self.assertEqual(manager.change_calls, [])
        self.assertEqual(manager.save_calls, [])
        self.assertEqual(result.metadata["edited_paths"], [str(path.resolve())])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
