from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.runtime.tools.tool_permissions import ToolPermissionPreview
from aether.tools.builtins.file_edit import FileEditTool
from aether.tools.builtins.write_file import WriteFileTool


class FilePermissionPreviewTests(unittest.TestCase):
    def test_file_edit_preview_builds_diff_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "app.py"
            path.write_text("old\n", encoding="utf-8")
            tool = FileEditTool()
            context = TurnContext(session_id="s", iteration=1)
            call = ToolCall(
                id="c1",
                name="file_edit",
                arguments={
                    "path": str(path),
                    "old_string": "old",
                    "new_string": "new",
                },
            )

            preview = tool.build_permission_preview(call, context)

            self.assertIsInstance(preview, ToolPermissionPreview)
            assert isinstance(preview, ToolPermissionPreview)
            self.assertIn("-old", preview.diff or "")
            self.assertIn("+new", preview.diff or "")
            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")

    def test_file_edit_accept_after_preview_writes_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "app.py"
            path.write_text("old\n", encoding="utf-8")
            tool = FileEditTool()
            context = TurnContext(session_id="s", iteration=1)
            call = ToolCall(
                id="c1",
                name="file_edit",
                arguments={
                    "path": str(path),
                    "old_string": "old",
                    "new_string": "new",
                },
            )

            tool.build_permission_preview(call, context)
            result = tool.execute(call, context)

            self.assertFalse(result.is_error)
            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")

    def test_file_edit_stale_preview_refuses_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "app.py"
            path.write_text("old\n", encoding="utf-8")
            tool = FileEditTool()
            context = TurnContext(session_id="s", iteration=1)
            call = ToolCall(
                id="c1",
                name="file_edit",
                arguments={
                    "path": str(path),
                    "old_string": "old",
                    "new_string": "new",
                },
            )

            tool.build_permission_preview(call, context)
            path.write_text("changed\n", encoding="utf-8")
            result = tool.execute(call, context)

            self.assertTrue(result.is_error)
            self.assertTrue(result.metadata["stale_preview"])
            self.assertEqual(path.read_text(encoding="utf-8"), "changed\n")

    def test_write_file_preview_new_file_without_creating_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "new.txt"
            tool = WriteFileTool()
            context = TurnContext(session_id="s", iteration=1)
            call = ToolCall(
                id="c1",
                name="write_file",
                arguments={"path": str(path), "content": "hello\n"},
            )

            preview = tool.build_permission_preview(call, context)

            self.assertIsInstance(preview, ToolPermissionPreview)
            self.assertFalse(path.exists())
            self.assertFalse(path.parent.exists())
            assert isinstance(preview, ToolPermissionPreview)
            self.assertIn("+hello", preview.diff or "")

    def test_write_file_accept_after_preview_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "new.txt"
            tool = WriteFileTool()
            context = TurnContext(session_id="s", iteration=1)
            call = ToolCall(
                id="c1",
                name="write_file",
                arguments={"path": str(path), "content": "hello\n"},
            )

            tool.build_permission_preview(call, context)
            result = tool.execute(call, context)

            self.assertFalse(result.is_error)
            self.assertEqual(path.read_text(encoding="utf-8"), "hello\n")

    def test_write_file_stale_preview_refuses_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "new.txt"
            tool = WriteFileTool()
            context = TurnContext(session_id="s", iteration=1)
            call = ToolCall(
                id="c1",
                name="write_file",
                arguments={"path": str(path), "content": "hello\n"},
            )

            tool.build_permission_preview(call, context)
            path.write_text("surprise\n", encoding="utf-8")
            result = tool.execute(call, context)

            self.assertTrue(result.is_error)
            self.assertTrue(result.metadata["stale_preview"])
            self.assertEqual(path.read_text(encoding="utf-8"), "surprise\n")


if __name__ == "__main__":
    unittest.main()
