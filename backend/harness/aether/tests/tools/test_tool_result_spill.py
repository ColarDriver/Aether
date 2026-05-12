"""Sprint 3.5 / PR 3.5.1 \u2014 cross-tool spill behaviour.

Each upgraded builtin (``shell``, ``read_file``, ``grep``, ``glob``,
``list_dir``) routes oversized output through ``maybe_spill_for_tool``
which writes the full payload to disk under the configured spill dir
and returns ``preview + truncation_notice``.  This file pins:

* below-threshold output stays inline (no notice, no spill file),
* above-threshold output produces the canonical notice substring,
* the spilled file actually exists with the expected content,
* ``tier1_spilled_count`` on ``context.metadata`` increments,
* ``EngineConfig.tool_result_spill_enabled=False`` short-circuits the
  whole path,
* the recursion guard on ``read_file`` does NOT spill files that
  already live under the spill root.

Tests are hermetic: every spill goes into a per-test temp dir via
``EngineConfig.tool_result_spill_dir``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aether.config.schema import EngineConfig
from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.tools.builtins import (
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    ShellTool,
)


def _ctx(spill_root: str | None = None, *, enabled: bool = True) -> TurnContext:
    cfg = EngineConfig(
        tool_result_spill_enabled=enabled,
        tool_result_spill_dir=Path(spill_root) if spill_root else None,
    )
    ctx = TurnContext(session_id="spill-tests", iteration=1)
    ctx.metadata["_engine_config"] = cfg
    return ctx


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=args)


class ShellSpillTests(unittest.TestCase):
    def test_below_threshold_no_spill(self) -> None:
        with tempfile.TemporaryDirectory() as spill:
            ctx = _ctx(spill)
            tool = ShellTool(max_result_chars=10_000)
            result = tool.execute(_call("shell", command="echo small"), ctx)
            self.assertFalse(result.is_error)
            self.assertNotIn("output truncated", result.content)
            self.assertNotIn("tier1_spilled_count", ctx.metadata)
            self.assertEqual(list(Path(spill).glob("**/*")), [])

    def test_above_threshold_spills(self) -> None:
        with tempfile.TemporaryDirectory() as spill:
            ctx = _ctx(spill)
            tool = ShellTool(max_result_chars=200)
            big = "X" * 5000
            result = tool.execute(
                _call("shell", command=f"printf '{big}'"), ctx
            )
            self.assertFalse(result.is_error)
            self.assertIn("output truncated", result.content)
            self.assertIn("use read_file to retrieve", result.content)
            self.assertEqual(ctx.metadata["tier1_spilled_count"], 1)
            spilled = list(Path(spill).rglob("call-shell.txt"))
            self.assertEqual(len(spilled), 1)
            self.assertGreater(spilled[0].stat().st_size, 4000)

    def test_exit_header_survives_in_preview(self) -> None:
        # The model needs ``[exit N ...]`` even when the body spilled,
        # because that header is the only success/failure signal it
        # has after the body is replaced with a preview.  Use a small
        # command (``seq``) producing big output \u2014 mirrors realistic
        # workloads where stdout dwarfs the command string.
        with tempfile.TemporaryDirectory() as spill:
            ctx = _ctx(spill)
            tool = ShellTool(max_result_chars=500)
            result = tool.execute(
                _call("shell", command="seq 1 10000"), ctx
            )
            self.assertIn("[exit 0", result.content)
            self.assertIn("output truncated", result.content)

    def test_disabled_spill_falls_back_to_full_inline(self) -> None:
        with tempfile.TemporaryDirectory() as spill:
            ctx = _ctx(spill, enabled=False)
            tool = ShellTool(max_result_chars=100)
            result = tool.execute(
                _call("shell", command=f"printf '{'Z' * 1000}'"), ctx
            )
            # No spill notice, no spill file, content carries the full body.
            self.assertNotIn("output truncated", result.content)
            self.assertEqual(list(Path(spill).rglob("*.txt")), [])
            self.assertIn("Z" * 100, result.content)


class ReadFileSpillTests(unittest.TestCase):
    def test_below_threshold_no_spill(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as spill:
            path = Path(workspace) / "a.txt"
            path.write_text("hello\nworld\n")
            tool = ReadFileTool(default_cwd=Path(workspace))
            ctx = _ctx(spill)
            result = tool.execute(_call("read_file", path="a.txt"), ctx)
            self.assertFalse(result.is_error)
            self.assertFalse(result.metadata.get("spilled", False))
            self.assertNotIn("output truncated", result.content)

    def test_above_threshold_spills_and_records_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as spill:
            path = Path(workspace) / "big.txt"
            path.write_text("\n".join(f"line {i}" for i in range(1, 5001)))
            tool = ReadFileTool(default_cwd=Path(workspace), max_result_chars=500)
            ctx = _ctx(spill)
            result = tool.execute(_call("read_file", path="big.txt"), ctx)
            self.assertFalse(result.is_error)
            self.assertTrue(result.metadata["spilled"])
            self.assertTrue(result.metadata["truncated"])
            self.assertIn("output truncated", result.content)
            self.assertEqual(result.metadata["total_lines"], 5000)

    def test_recursion_guard_skips_spill_root(self) -> None:
        # Files living under the configured spill root must NEVER spill
        # again \u2014 otherwise reading back a spilled file enters an
        # infinite preview \u2192 read \u2192 preview loop.
        with tempfile.TemporaryDirectory() as spill:
            sess_dir = Path(spill) / "spill-tests"
            sess_dir.mkdir(parents=True)
            big = sess_dir / "previously_spilled.txt"
            big.write_text("X" * 100_000)  # well above 60k threshold
            tool = ReadFileTool(max_result_chars=500)
            ctx = _ctx(spill)
            result = tool.execute(
                _call("read_file", path=str(big)), ctx
            )
            self.assertFalse(result.is_error)
            self.assertFalse(result.metadata["spilled"])
            self.assertNotIn("output truncated", result.content)


class GrepSpillTests(unittest.TestCase):
    def _populate(self, root: Path, *, lines_per_file: int) -> None:
        for i in range(20):
            (root / f"file_{i}.txt").write_text(
                "\n".join(f"hit {j}" for j in range(lines_per_file))
            )

    def _force_python_walker(self, tool: GrepTool) -> None:
        # Make grep tests deterministic regardless of whether
        # ``rg`` exists on the runner; matches the pattern used in
        # test_builtin_tools.GrepToolTests.
        tool._rg_path = None
        tool._rg_resolved = True

    def test_below_threshold_no_spill(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as spill:
            self._populate(Path(workspace), lines_per_file=2)
            tool = GrepTool(default_cwd=Path(workspace))
            self._force_python_walker(tool)
            ctx = _ctx(spill)
            result = tool.execute(
                _call("grep", pattern="hit"), ctx
            )
            self.assertFalse(result.is_error)
            self.assertNotIn("output truncated", result.content)

    def test_above_threshold_spills_at_line_boundary(self) -> None:
        # Forcing a small threshold: the spill cut must align to the
        # last newline before max_chars so the inline preview never
        # ends mid-line.  We assert the content immediately preceding
        # the truncation notice ends with a newline character.
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as spill:
            self._populate(Path(workspace), lines_per_file=400)
            tool = GrepTool(default_cwd=Path(workspace))
            self._force_python_walker(tool)
            ctx = _ctx(spill)
            # Force the spill threshold low by patching the module
            # constant for this test (cleaner than threading another
            # ctor knob into GrepTool just for one assertion).
            from aether.tools.builtins import grep as grep_module

            original = grep_module._MAX_RESULT_CHARS
            grep_module._MAX_RESULT_CHARS = 1500
            try:
                result = tool.execute(
                    _call("grep", pattern="hit"), ctx
                )
            finally:
                grep_module._MAX_RESULT_CHARS = original
            self.assertFalse(result.is_error)
            self.assertTrue(result.metadata["spilled"])
            preview, _, _ = result.content.partition("\n\n... [output truncated")
            # Preview must end at a line break (line-boundary cut).
            self.assertTrue(preview.endswith("\n"), preview[-50:])


class GlobSpillTests(unittest.TestCase):
    def test_above_threshold_spills(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as spill:
            # 4000 files \u2014 raw paths alone overflow the 20k cap.
            for i in range(4000):
                (Path(workspace) / f"f_{i}.txt").write_text("x")
            # Bump max_results so all 4000 paths actually appear in the
            # output (default is 200).
            tool = GlobTool(
                default_cwd=Path(workspace), default_max_results=10_000
            )
            ctx = _ctx(spill)
            result = tool.execute(_call("glob", pattern="*.txt"), ctx)
            self.assertFalse(result.is_error)
            self.assertTrue(result.metadata["spilled"])
            self.assertIn("output truncated", result.content)


class ListDirSpillTests(unittest.TestCase):
    def test_above_threshold_spills(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as spill:
            for i in range(2000):
                (Path(workspace) / f"item_long_filename_{i:04d}.txt").write_text("x")
            tool = ListDirTool(
                default_cwd=Path(workspace), default_max_entries=10_000
            )
            ctx = _ctx(spill)
            result = tool.execute(_call("list_dir", path="."), ctx)
            self.assertFalse(result.is_error)
            self.assertTrue(result.metadata["spilled"])
            self.assertIn("output truncated", result.content)


class CrossSessionIsolationTests(unittest.TestCase):
    def test_two_sessions_get_separate_spill_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as spill:
            tool = ShellTool(max_result_chars=200)
            ctx_a = TurnContext(session_id="A", iteration=1)
            ctx_a.metadata["_engine_config"] = EngineConfig(
                tool_result_spill_dir=Path(spill)
            )
            ctx_b = TurnContext(session_id="B", iteration=1)
            ctx_b.metadata["_engine_config"] = EngineConfig(
                tool_result_spill_dir=Path(spill)
            )
            tool.execute(_call("shell", command=f"printf '{'A' * 1000}'"), ctx_a)
            tool.execute(_call("shell", command=f"printf '{'B' * 1000}'"), ctx_b)
            self.assertTrue((Path(spill) / "A").exists())
            self.assertTrue((Path(spill) / "B").exists())
            self.assertEqual(
                len(list((Path(spill) / "A").iterdir())), 1
            )


if __name__ == "__main__":
    unittest.main()
