"""Sprint 3.5 / PR 3.5.2 \u2014 ``file_edit`` tool coverage.

Pins:

* arg validation (path / old_string / new_string presence + type),
* unique-match guard (zero / one / multi-match outcomes),
* ``replace_all=True`` actually replaces every occurrence,
* identity rejection (old==new),
* missing file / directory path / large-file rejection,
* spill-root protection,
* diff summary content (header line + change marker visible),
* metadata bookkeeping (change_count, bytes_before/after),
* whitespace fidelity (preserves indentation byte-for-byte).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.runtime.tools.tool_result_storage import DEFAULT_SPILL_ROOT
from aether.tools.builtins.file_edit import FileEditTool


def _ctx() -> TurnContext:
    return TurnContext(session_id="file-edit-tests", iteration=1)


def _call(**args) -> ToolCall:
    return ToolCall(id="call-file-edit", name="file_edit", arguments=args)


class ArgValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = FileEditTool()

    def test_missing_path(self) -> None:
        result = self.tool.execute(
            _call(old_string="a", new_string="b"), _ctx()
        )
        self.assertTrue(result.is_error)
        self.assertIn("'path'", result.content)

    def test_missing_old_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("x")
            result = self.tool.execute(
                _call(path=str(p), new_string="y"), _ctx()
            )
        self.assertTrue(result.is_error)
        self.assertIn("'old_string'", result.content)

    def test_empty_old_string_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("x")
            result = self.tool.execute(
                _call(path=str(p), old_string="", new_string="y"), _ctx()
            )
        self.assertTrue(result.is_error)

    def test_missing_new_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("x")
            result = self.tool.execute(
                _call(path=str(p), old_string="x"), _ctx()
            )
        self.assertTrue(result.is_error)

    def test_non_string_new_string_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("x")
            result = self.tool.execute(
                _call(path=str(p), old_string="x", new_string=123), _ctx()
            )
        self.assertTrue(result.is_error)


class UniqueMatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = FileEditTool()

    def test_unique_match_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "code.py"
            p.write_text("def foo():\n    return 1\n")
            result = self.tool.execute(
                _call(
                    path=str(p),
                    old_string="return 1",
                    new_string="return 42",
                ),
                _ctx(),
            )
            after = p.read_text()
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(after, "def foo():\n    return 42\n")
        self.assertEqual(result.metadata["change_count"], 1)
        self.assertFalse(result.metadata["replace_all"])

    def test_zero_match_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("alpha\n")
            result = self.tool.execute(
                _call(path=str(p), old_string="zeta", new_string="omega"),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("not found", result.content)

    def test_multi_match_without_replace_all_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("foo\nfoo\nfoo\n")
            result = self.tool.execute(
                _call(path=str(p), old_string="foo", new_string="bar"),
                _ctx(),
            )
            after = p.read_text()
        self.assertTrue(result.is_error)
        self.assertIn("3 places", result.content)
        self.assertEqual(result.metadata["occurrences"], 3)
        # File must NOT have been modified on a rejected edit.
        self.assertEqual(after, "foo\nfoo\nfoo\n")

    def test_replace_all_changes_every_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("foo\nfoo\nfoo\n")
            result = self.tool.execute(
                _call(
                    path=str(p),
                    old_string="foo",
                    new_string="bar",
                    replace_all=True,
                ),
                _ctx(),
            )
            after = p.read_text()
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(after, "bar\nbar\nbar\n")
        self.assertEqual(result.metadata["change_count"], 3)

    def test_unique_match_with_indentation_preserved(self) -> None:
        # Indentation of the SURROUNDING context must round-trip
        # exactly.  This is the test that catches an accidental
        # "strip whitespace before matching" bug.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "code.py"
            original = "if x:\n        return 1\n"
            p.write_text(original)
            result = self.tool.execute(
                _call(
                    path=str(p),
                    old_string="        return 1",
                    new_string="        return 99",
                ),
                _ctx(),
            )
            after = p.read_text()
        self.assertFalse(result.is_error)
        self.assertEqual(after, "if x:\n        return 99\n")


class IdentityAndNoOpTests(unittest.TestCase):
    def test_identical_old_and_new_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("a\n")
            tool = FileEditTool()
            result = tool.execute(
                _call(path=str(p), old_string="a", new_string="a"),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("identical", result.content)


class FilesystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = FileEditTool()

    def test_missing_file_errors_with_write_file_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.tool.execute(
                _call(
                    path=str(Path(tmp) / "nope.txt"),
                    old_string="a",
                    new_string="b",
                ),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("not found", result.content)
        self.assertIn("write_file", result.content)

    def test_directory_path_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.tool.execute(
                _call(path=tmp, old_string="a", new_string="b"),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("directory", result.content)

    def test_non_utf8_file_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "binary.bin"
            p.write_bytes(b"\xff\xfe\x00\x00")
            result = self.tool.execute(
                _call(path=str(p), old_string="a", new_string="b"),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("UTF-8", result.content)


class SpillRootProtectionTests(unittest.TestCase):
    def test_refuses_to_edit_files_under_spill_root(self) -> None:
        # Create a fake spill file under the canonical spill root.  We
        # must NOT use a real spill from another test \u2014 the suite
        # would become flaky.  Instead we craft a path explicitly.
        spill_session_dir = DEFAULT_SPILL_ROOT / "_file_edit_test_session"
        spill_session_dir.mkdir(parents=True, exist_ok=True)
        try:
            target = spill_session_dir / "synthetic.txt"
            target.write_text("hello\n")
            tool = FileEditTool()
            result = tool.execute(
                _call(
                    path=str(target),
                    old_string="hello",
                    new_string="bye",
                ),
                _ctx(),
            )
            self.assertTrue(result.is_error)
            self.assertIn("spill", result.content)
            # File must not have been mutated.
            self.assertEqual(target.read_text(), "hello\n")
        finally:
            try:
                target.unlink()
                spill_session_dir.rmdir()
            except OSError:
                pass


class DiffSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = FileEditTool()

    def test_diff_summary_starts_with_edited_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("alpha\n")
            resolved = p.resolve()
            result = self.tool.execute(
                _call(path=str(p), old_string="alpha", new_string="beta"),
                _ctx(),
            )
        self.assertFalse(result.is_error)
        self.assertTrue(result.content.startswith(f"edited {resolved}"))
        self.assertIn("(1 change)", result.content)
        # Diff body shows old AND new lines.
        self.assertIn("-alpha", result.content)
        self.assertIn("+beta", result.content)

    def test_replace_all_summary_uses_plural(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("x\nx\n")
            result = self.tool.execute(
                _call(
                    path=str(p),
                    old_string="x",
                    new_string="y",
                    replace_all=True,
                ),
                _ctx(),
            )
        self.assertIn("(2 changes)", result.content)

    def test_diff_truncates_for_huge_changes(self) -> None:
        # Replace_all over a 1000-line file produces 1000+ diff lines.
        # Sanity-check the elision footer kicks in.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "many.txt"
            p.write_text("\n".join("xxx" for _ in range(1000)) + "\n")
            result = self.tool.execute(
                _call(
                    path=str(p),
                    old_string="xxx",
                    new_string="yyy",
                    replace_all=True,
                ),
                _ctx(),
            )
        self.assertFalse(result.is_error)
        self.assertIn("more diff lines elided", result.content)


class MetadataTests(unittest.TestCase):
    def test_metadata_records_size_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("aaa")  # 3 bytes
            tool = FileEditTool()
            result = tool.execute(
                _call(path=str(p), old_string="aaa", new_string="zzzzz"),
                _ctx(),
            )
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["bytes_before"], 3)
        self.assertEqual(result.metadata["bytes_after"], 5)


if __name__ == "__main__":
    unittest.main()
