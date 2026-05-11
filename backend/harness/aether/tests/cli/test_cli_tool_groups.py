"""Tests for the coalesced tool-group renderer.

Exercises:
  * category bucketing (canonical names, prefix heuristic, MCP fallback)
  * hint formatting per category (bash / search / read / web / fallback)
  * present/past tense headline generation as the iteration boundary moves
  * sink integration (renderable handed to the supplied sink callback)
"""

from __future__ import annotations

import unittest

from rich.console import Console
from rich.text import Text

from aether.cli.tool_groups import (
    ToolCategory,
    ToolGroup,
    ToolGroupTracker,
    category_for,
    hint_for_call,
)


class CategoryForTests(unittest.TestCase):
    def test_explicit_name_table_wins(self) -> None:
        self.assertEqual(category_for("read_file"), ToolCategory.READ)
        self.assertEqual(category_for("Bash"), ToolCategory.BASH)
        self.assertEqual(category_for("Grep"), ToolCategory.SEARCH)
        self.assertEqual(category_for("ls"), ToolCategory.LIST)

    def test_prefix_heuristic_for_unknown_names(self) -> None:
        self.assertEqual(category_for("read_secret_dossier"), ToolCategory.READ)
        self.assertEqual(category_for("execute_workflow"), ToolCategory.BASH)
        self.assertEqual(category_for("fetch_release_notes"), ToolCategory.WEB)
        self.assertEqual(category_for("search_documents"), ToolCategory.SEARCH)

    def test_mcp_namespaced_name_falls_through_to_specific_category(self) -> None:
        # ``mcp__filesystem__read_file`` strips to ``read_file`` first,
        # so the explicit map wins and we don't bucket it as MCP.
        self.assertEqual(
            category_for("mcp__filesystem__read_file"),
            ToolCategory.READ,
        )

    def test_mcp_namespaced_name_with_unknown_tail_is_mcp(self) -> None:
        # MCP-namespaced names that don't match the canonical name table
        # bucket as MCP, even when the tail would have matched a prefix
        # heuristic (the explicit MCP marker is a stronger signal).
        self.assertEqual(
            category_for("mcp__weather__get_today"),
            ToolCategory.MCP,
        )
        self.assertEqual(
            category_for("mcp__custom__do_something"),
            ToolCategory.MCP,
        )

    def test_unknown_name_is_other(self) -> None:
        self.assertEqual(category_for("teleport_user"), ToolCategory.OTHER)
        self.assertEqual(category_for(""), ToolCategory.OTHER)


class HintForCallTests(unittest.TestCase):
    def test_bash_hint_uses_dollar_prefix(self) -> None:
        self.assertEqual(
            hint_for_call("Bash", {"command": "ls -la"}),
            "$ ls -la",
        )

    def test_search_hint_combines_pattern_and_path(self) -> None:
        hint = hint_for_call("Grep", {"pattern": "foo", "path": "src/"})
        self.assertEqual(hint, '"foo" in src/')

    def test_search_hint_pattern_only(self) -> None:
        hint = hint_for_call("Grep", {"pattern": "foo"})
        self.assertEqual(hint, '"foo"')

    def test_read_hint_picks_path_field(self) -> None:
        hint = hint_for_call(
            "read_file",
            {"file_path": "backend/harness/aether/cli/ui.py"},
        )
        self.assertEqual(hint, "backend/harness/aether/cli/ui.py")

    def test_web_hint_returns_url(self) -> None:
        hint = hint_for_call(
            "WebFetch",
            {"url": "https://example.com/docs"},
        )
        self.assertEqual(hint, "https://example.com/docs")

    def test_long_hint_is_truncated_with_ellipsis(self) -> None:
        long_path = "/" + ("a" * 200)
        hint = hint_for_call("read_file", {"file_path": long_path})
        # 80-char default limit, with trailing ellipsis when shortened.
        self.assertLessEqual(len(hint), 80)
        self.assertTrue(hint.endswith("…"))

    def test_subagent_hint_uses_description_or_prompt(self) -> None:
        hint = hint_for_call(
            "Task",
            {"description": "explore frontend"},
        )
        self.assertEqual(hint, "explore frontend")

    def test_no_args_returns_empty_string(self) -> None:
        self.assertEqual(hint_for_call("Bash", {}), "")

    def test_hint_replaces_newlines_with_spaces(self) -> None:
        hint = hint_for_call(
            "Bash",
            {"command": "echo a\necho b"},
        )
        self.assertNotIn("\n", hint)
        self.assertEqual(hint, "$ echo a echo b")


class ToolGroupHeadlineTests(unittest.TestCase):
    @staticmethod
    def _to_plain(text: Text) -> str:
        # Strip styles for substring assertions.
        return text.plain

    def test_active_present_tense_with_one_category(self) -> None:
        group = ToolGroup()
        group.add_call("Read", {"file_path": "README.md"})
        plain = self._to_plain(group.render_headline(active=True))
        self.assertIn("Reading", plain)
        self.assertIn("1 file", plain)
        self.assertTrue(plain.endswith("…"))

    def test_active_with_multiple_categories_uses_lowercase_followons(self) -> None:
        group = ToolGroup()
        group.add_call("Grep", {"pattern": "foo"})
        group.add_call("Grep", {"pattern": "bar"})
        group.add_call("Read", {"file_path": "x"})
        group.add_call("ls", {"path": "/tmp"})
        plain = self._to_plain(group.render_headline(active=True))
        # Categories appear in enum order: SEARCH, READ, LIST.
        self.assertIn("Searching for", plain)
        self.assertIn("2 patterns", plain)
        # follow-on verbs should be lowercase
        self.assertIn(", reading 1 file", plain)
        self.assertIn(", listing 1 directory", plain)

    def test_resolved_past_tense_drops_ellipsis(self) -> None:
        group = ToolGroup()
        group.add_call("Bash", {"command": "ls"})
        group.finish_call()
        plain = self._to_plain(group.render_headline(active=False))
        self.assertIn("Ran", plain)
        self.assertIn("1 command", plain)
        self.assertNotIn("…", plain)

    def test_resolved_with_errors_appends_warning_marker(self) -> None:
        group = ToolGroup()
        group.add_call("Bash", {"command": "ls"})
        group.finish_call(is_error=True)
        plain = self._to_plain(group.render_headline(active=False))
        self.assertIn("with errors", plain)

    def test_render_hint_returns_indented_excerpt(self) -> None:
        group = ToolGroup()
        group.add_call("Bash", {"command": "ls -la"})
        hint = group.render_hint()
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("⎿", hint.plain)
        self.assertIn("$ ls -la", hint.plain)

    def test_render_hint_returns_none_when_no_hint_available(self) -> None:
        group = ToolGroup()
        group.add_call("Bash", {})  # no args → no hint
        self.assertIsNone(group.render_hint())

    def test_in_flight_tracking(self) -> None:
        group = ToolGroup()
        group.add_call("Bash", {"command": "ls"})
        group.add_call("Read", {"file_path": "x"})
        self.assertTrue(group.is_active)
        group.finish_call()
        self.assertTrue(group.is_active)
        group.finish_call()
        self.assertFalse(group.is_active)


class ToolGroupTrackerTests(unittest.TestCase):
    """Verify the iteration-boundary lifecycle: start → start → flush → start."""

    def setUp(self) -> None:
        self.printed: list[Text] = []

    def _sink(self, renderable) -> None:  # noqa: ANN001
        if isinstance(renderable, Text):
            self.printed.append(renderable)

    def test_start_call_creates_active_group(self) -> None:
        tracker = ToolGroupTracker(sink=self._sink)
        self.assertIsNone(tracker.active)
        tracker.start_call("Read", {"file_path": "x"})
        self.assertIsNotNone(tracker.active)
        active = tracker.active
        assert active is not None
        self.assertEqual(active.total_calls, 1)
        self.assertTrue(active.is_active)

    def test_finish_call_marks_resolved_but_keeps_active_group(self) -> None:
        tracker = ToolGroupTracker(sink=self._sink)
        tracker.start_call("Read", {"file_path": "x"})
        tracker.finish_call()
        # Group survives until flushed at iteration boundary.
        self.assertIsNotNone(tracker.active)
        active = tracker.active
        assert active is not None
        self.assertFalse(active.is_active)

    def test_begin_iteration_flushes_explore_tree_to_sink(self) -> None:
        # Tracker now flushes the codex-style ``● Explored`` umbrella +
        # tree of sub-calls (replaces the old ``Read 1 file`` summary
        # headline).  The middleware passes pre-computed verb/detail so
        # the tree rows read as ``<Verb> <detail>``.
        tracker = ToolGroupTracker(sink=self._sink)
        tracker.start_call("Read", {"file_path": "x"}, verb="Read", detail="x")
        tracker.finish_call()
        tracker.begin_iteration()
        self.assertIsNone(tracker.active)
        self.assertEqual(len(self.printed), 1)
        plain = self.printed[0].plain
        self.assertIn("Explored", plain)
        self.assertIn("Read", plain)
        self.assertIn("x", plain)

    def test_flush_active_with_no_group_is_noop(self) -> None:
        tracker = ToolGroupTracker(sink=self._sink)
        tracker.flush_active()
        self.assertEqual(self.printed, [])

    def test_discard_active_does_not_print_partial_group(self) -> None:
        tracker = ToolGroupTracker(sink=self._sink)
        tracker.start_call("Read", {"file_path": "x"})
        tracker.discard_active()
        self.assertIsNone(tracker.active)
        self.assertEqual(self.printed, [])

    def test_consecutive_iterations_print_two_separate_explore_trees(self) -> None:
        # Each iteration's burst becomes its own ``● Explored`` block
        # in scrollback, with the tree containing every sub-call from
        # that iteration.
        tracker = ToolGroupTracker(sink=self._sink)
        # Iteration 1: 2 reads
        tracker.start_call("Read", {"file_path": "a"}, verb="Read", detail="a")
        tracker.start_call("Read", {"file_path": "b"}, verb="Read", detail="b")
        tracker.finish_call()
        tracker.finish_call()
        # Boundary
        tracker.begin_iteration()
        # Iteration 2: 1 search
        tracker.start_call(
            "Grep", {"pattern": "foo"}, verb="Search", detail='"foo"',
        )
        tracker.finish_call()
        tracker.flush_active()
        self.assertEqual(len(self.printed), 2)
        first = self.printed[0].plain
        second = self.printed[1].plain
        # Iteration 1: explore tree with both reads
        self.assertIn("Explored", first)
        self.assertIn("Read", first)
        self.assertIn("a", first)
        self.assertIn("b", first)
        # Iteration 2: explore tree with the search
        self.assertIn("Explored", second)
        self.assertIn("Search", second)
        self.assertIn("foo", second)


if __name__ == "__main__":
    Console()  # ensure terminal init imported (avoids lazy-import order issues)
    unittest.main()
