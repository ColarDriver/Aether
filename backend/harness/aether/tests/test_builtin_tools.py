"""Sprint 1.5 / P0-9 — bundled tool kit coverage.

Pins the contract of each :mod:`aether.tools.builtins` executor: happy
path, error path, output capping, and the JSON-schema-style
``parameters`` advertised to the LLM.  Together with
``test_phantom_synthesis`` and ``test_run_loop_phantom_tool`` this
guarantees the chain "model emits prose → engine synthesizes ToolCall
→ registry dispatches to the matching builtin → ToolResult comes
back populated" stays intact as the kit evolves.

We deliberately avoid mocking ``subprocess`` for the shell tests —
the test runner has a real ``/bin/sh`` and we'd rather catch a
regression in the real-world IO shape than hand-roll a fixture that
drifts.  All file IO happens under :class:`tempfile.TemporaryDirectory`
so the suite is hermetic.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from aether.config.schema import EngineConfig
from aether.runtime.contracts import ToolCall, TurnContext
from aether.tools.builtins import (
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    ShellTool,
    WriteFileTool,
    build_default_tool_registry,
)


def _ctx() -> TurnContext:
    return TurnContext(session_id="builtin-tests", iteration=1)


def _ctx_with_spill(spill_root: str) -> TurnContext:
    """Return a TurnContext whose engine config redirects spill files
    into the given temp dir \u2014 keeps the test hermetic and avoids
    polluting ``~/.aether/tool_results`` on the CI runner."""
    cfg = EngineConfig(tool_result_spill_dir=Path(spill_root))
    ctx = TurnContext(session_id="builtin-tests", iteration=1)
    ctx.metadata["_engine_config"] = cfg
    return ctx


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=args)


# ---------------------------------------------------------------------------
# build_default_tool_registry
# ---------------------------------------------------------------------------


class FactoryTests(unittest.TestCase):
    def test_factory_registers_canonical_tools(self) -> None:
        # Sprint 3.5 / PR 3.5.1 expanded the kit from six to nine
        # tools; ``file_edit`` / ``notebook_edit`` / ``todo_write``
        # join the original six.  Asserting the full sorted list
        # documents the contract and trips loudly if someone drops
        # a tool by accident.
        reg = build_default_tool_registry()
        names = sorted(d.name for d in reg.list_descriptors())
        self.assertEqual(
            names,
            [
                "file_edit",
                "glob",
                "grep",
                "list_dir",
                "notebook_edit",
                "read_file",
                "shell",
                "todo_write",
                "write_file",
            ],
        )

    def test_each_descriptor_advertises_object_schema_with_required(self) -> None:
        reg = build_default_tool_registry()
        for descriptor in reg.list_descriptors():
            params = descriptor.parameters
            self.assertEqual(params.get("type"), "object", descriptor.name)
            self.assertIn("properties", params, descriptor.name)
            self.assertIn("required", params, descriptor.name)
            self.assertTrue(descriptor.description.strip(), descriptor.name)


# ---------------------------------------------------------------------------
# ShellTool
# ---------------------------------------------------------------------------


class ShellToolTests(unittest.TestCase):
    def test_runs_command_and_returns_stdout(self) -> None:
        tool = ShellTool()
        result = tool.execute(_call("shell", command="echo hello"), _ctx())
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(result.metadata["exit_code"], 0)
        self.assertIn("hello", result.content)

    def test_non_zero_exit_marks_result_as_error(self) -> None:
        tool = ShellTool()
        result = tool.execute(_call("shell", command="exit 3"), _ctx())
        self.assertTrue(result.is_error)
        self.assertEqual(result.metadata["exit_code"], 3)

    def test_missing_command_returns_validation_error(self) -> None:
        tool = ShellTool()
        result = tool.execute(_call("shell", command=""), _ctx())
        self.assertTrue(result.is_error)
        self.assertIn("non-empty string", result.content)

    def test_oversize_stdout_spills_to_disk(self) -> None:
        # Sprint 3.5 / PR 3.5.1 \u2014 the head+tail truncation has been
        # replaced with disk spill.  Lower the threshold so a tiny
        # printf still trips it.  We don't care about the exact spill
        # path here (covered by test_tool_result_spill); only that the
        # content truncates with the canonical notice and metadata
        # records the fact.
        with tempfile.TemporaryDirectory() as spill_root:
            tool = ShellTool(max_result_chars=128)
            big = "abcdefgh" * 200  # 1600 bytes
            ctx = _ctx_with_spill(spill_root)
            result = tool.execute(
                _call("shell", command=f"printf '{big}'"),
                ctx,
            )
            self.assertFalse(result.is_error)
            self.assertTrue(result.metadata["truncated"])
            self.assertIn("output truncated", result.content)
            self.assertIn("use read_file to retrieve", result.content)
            self.assertEqual(ctx.metadata["tier1_spilled_count"], 1)


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------


class ReadFileToolTests(unittest.TestCase):
    def test_returns_numbered_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("alpha\nbeta\ngamma\n")
            tool = ReadFileTool(default_cwd=Path(tmp))
            result = tool.execute(_call("read_file", path="sample.txt"), _ctx())
        self.assertFalse(result.is_error)
        self.assertIn("     1| alpha", result.content)
        self.assertIn("     2| beta", result.content)
        self.assertEqual(result.metadata["lines_returned"], 3)
        self.assertEqual(result.metadata["total_lines"], 3)
        self.assertFalse(result.metadata["truncated"])

    def test_offset_and_limit_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lines.txt"
            path.write_text("\n".join(f"L{i}" for i in range(1, 11)))
            tool = ReadFileTool(default_cwd=Path(tmp))
            result = tool.execute(
                _call("read_file", path="lines.txt", offset=5, limit=2),
                _ctx(),
            )
        self.assertFalse(result.is_error)
        self.assertIn("     5| L5", result.content)
        self.assertIn("     6| L6", result.content)
        self.assertNotIn("L4", result.content)
        self.assertNotIn("L7", result.content)
        self.assertTrue(result.metadata["truncated"])

    def test_missing_file_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = ReadFileTool(default_cwd=Path(tmp))
            result = tool.execute(_call("read_file", path="nope.txt"), _ctx())
        self.assertTrue(result.is_error)
        self.assertIn("not found", result.content)

    def test_directory_path_is_error_with_use_list_dir_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = ReadFileTool(default_cwd=Path(tmp))
            result = tool.execute(_call("read_file", path="."), _ctx())
        self.assertTrue(result.is_error)
        self.assertIn("list_dir", result.content)

    def test_oversize_file_spills_to_disk(self) -> None:
        # Sprint 3.5 / PR 3.5.1 \u2014 the pre-3.5 hard 256 KB cap (which
        # raised ``is_error=True`` and refused to read) has been
        # replaced with disk spill.  A file exceeding ``max_result_chars``
        # is still **read** successfully; the response just carries a
        # truncation notice and the full output lives on disk for the
        # model to retrieve via ``read_file <spilled-path>``.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as spill_root:
            path = Path(tmp) / "big.bin"
            path.write_bytes(b"x" * 2048)
            tool = ReadFileTool(default_cwd=Path(tmp), max_result_chars=512)
            ctx = _ctx_with_spill(spill_root)
            result = tool.execute(_call("read_file", path="big.bin"), ctx)
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["size_bytes"], 2048)
        self.assertTrue(result.metadata["spilled"])
        self.assertTrue(result.metadata["truncated"])
        self.assertIn("output truncated", result.content)
        self.assertIn("use read_file to retrieve", result.content)
        self.assertEqual(ctx.metadata["tier1_spilled_count"], 1)


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------


class WriteFileToolTests(unittest.TestCase):
    def test_creates_file_with_hash_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = WriteFileTool(default_cwd=Path(tmp))
            result = tool.execute(
                _call("write_file", path="out.txt", content="hello world"),
                _ctx(),
            )
            self.assertFalse(result.is_error)
            self.assertEqual(result.metadata["size_bytes"], 11)
            self.assertEqual(len(result.metadata["sha256"]), 64)
            self.assertFalse(result.metadata["existed"])
            self.assertEqual((Path(tmp) / "out.txt").read_text(), "hello world")

    def test_overwrites_existing_file_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.txt"
            path.write_text("old")
            tool = WriteFileTool(default_cwd=Path(tmp))
            result = tool.execute(
                _call("write_file", path="out.txt", content="new"),
                _ctx(),
            )
            self.assertFalse(result.is_error)
            self.assertTrue(result.metadata["existed"])
            self.assertEqual(path.read_text(), "new")

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = WriteFileTool(default_cwd=Path(tmp))
            result = tool.execute(
                _call("write_file", path="nested/dir/out.txt", content="x"),
                _ctx(),
            )
            self.assertFalse(result.is_error)
            self.assertTrue((Path(tmp) / "nested" / "dir" / "out.txt").exists())

    def test_oversize_content_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = WriteFileTool(default_cwd=Path(tmp), max_bytes=8)
            result = tool.execute(
                _call("write_file", path="o.txt", content="too much content"),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("too large", result.content)


# ---------------------------------------------------------------------------
# ListDirTool
# ---------------------------------------------------------------------------


class ListDirToolTests(unittest.TestCase):
    def test_lists_visible_entries_and_skips_dotfiles_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "alpha.txt").write_text("a")
            (Path(tmp) / "beta").mkdir()
            (Path(tmp) / ".hidden").write_text("h")
            (Path(tmp) / "node_modules").mkdir()
            tool = ListDirTool(default_cwd=Path(tmp))
            result = tool.execute(_call("list_dir", path="."), _ctx())
        self.assertFalse(result.is_error)
        self.assertIn("alpha.txt", result.content)
        self.assertIn("beta/", result.content)
        self.assertNotIn(".hidden", result.content)
        self.assertNotIn("node_modules", result.content)
        self.assertEqual(result.metadata["entries_skipped"], 2)

    def test_include_hidden_lists_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".hidden").write_text("h")
            tool = ListDirTool(default_cwd=Path(tmp))
            result = tool.execute(
                _call("list_dir", path=".", include_hidden=True),
                _ctx(),
            )
        self.assertIn(".hidden", result.content)

    def test_path_must_be_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "f.txt"
            file_path.write_text("x")
            tool = ListDirTool(default_cwd=Path(tmp))
            result = tool.execute(_call("list_dir", path="f.txt"), _ctx())
        self.assertTrue(result.is_error)
        self.assertIn("not a directory", result.content)


# ---------------------------------------------------------------------------
# GrepTool
# ---------------------------------------------------------------------------


class GrepToolTests(unittest.TestCase):
    def _populate(self, tmp: Path) -> None:
        (tmp / "a.py").write_text("def foo():\n    return 1\n")
        (tmp / "b.py").write_text("def bar():\n    return 'foo'\n")
        (tmp / "c.txt").write_text("not relevant\n")
        sub = tmp / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("# foo bar\n")

    def test_python_walker_finds_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._populate(Path(tmp))
            tool = GrepTool(default_cwd=Path(tmp))
            # Force the python fallback regardless of whether ``rg`` is
            # actually installed on the test runner.  Setting both
            # fields short-circuits the lazy ``shutil.which`` probe.
            tool._rg_path = None
            tool._rg_resolved = True
            result = tool.execute(
                _call("grep", pattern="foo", path="."),
                _ctx(),
            )
        self.assertFalse(result.is_error)
        self.assertGreaterEqual(result.metadata["match_count"], 2)
        self.assertEqual(result.metadata["engine"], "python")

    def test_glob_filter_narrows_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._populate(Path(tmp))
            tool = GrepTool(default_cwd=Path(tmp))
            tool._rg_path = None
            tool._rg_resolved = True
            result = tool.execute(
                _call("grep", pattern="foo", path=".", glob="*.py"),
                _ctx(),
            )
        self.assertFalse(result.is_error)
        self.assertNotIn("c.txt", result.content)

    def test_invalid_regex_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = GrepTool(default_cwd=Path(tmp))
            tool._rg_path = None
            tool._rg_resolved = True
            result = tool.execute(
                _call("grep", pattern="[unclosed", path="."),
                _ctx(),
            )
        self.assertTrue(result.is_error)
        self.assertIn("invalid regex", result.content)

    def test_missing_pattern_is_error(self) -> None:
        tool = GrepTool()
        result = tool.execute(_call("grep", path="/tmp"), _ctx())
        self.assertTrue(result.is_error)


# ---------------------------------------------------------------------------
# GlobTool
# ---------------------------------------------------------------------------


class GlobToolTests(unittest.TestCase):
    def test_promotes_bare_pattern_to_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.py").write_text("x")
            sub = Path(tmp) / "sub"
            sub.mkdir()
            (sub / "b.py").write_text("y")
            tool = GlobTool(default_cwd=Path(tmp))
            result = tool.execute(_call("glob", pattern="*.py"), _ctx())
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["match_count"], 2)
        self.assertIn("a.py", result.content)
        self.assertIn("b.py", result.content)

    def test_skips_default_skip_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            git = Path(tmp) / ".git"
            git.mkdir()
            (git / "hidden.py").write_text("x")
            (Path(tmp) / "real.py").write_text("y")
            tool = GlobTool(default_cwd=Path(tmp))
            result = tool.execute(_call("glob", pattern="*.py"), _ctx())
        self.assertEqual(result.metadata["match_count"], 1)
        self.assertIn("real.py", result.content)
        self.assertNotIn("hidden.py", result.content)

    def test_missing_pattern_is_error(self) -> None:
        tool = GlobTool()
        result = tool.execute(_call("glob"), _ctx())
        self.assertTrue(result.is_error)

    def test_path_must_exist(self) -> None:
        tool = GlobTool()
        result = tool.execute(_call("glob", pattern="*.py", path="/nonexistent_xx"), _ctx())
        self.assertTrue(result.is_error)


if __name__ == "__main__":
    unittest.main()
