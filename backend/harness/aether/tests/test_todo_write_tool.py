"""Sprint 3.5 / PR 3.5.3 \u2014 ``todo_write`` tool coverage.

Pins:

* arg validation (todos shape / required fields / status enum / id type),
* full-replace semantics (call N replaces call N-1's list entirely),
* terminal-status auto-clear (all completed/cancelled \u2192 store wiped),
* per-session isolation (two sessions don't see each other's todos),
* metadata counters (pending/in_progress/completed/cancelled),
* tool name normalises to a member of ``EngineConfig.cheap_tool_names``.
"""

from __future__ import annotations

import unittest

from aether.config.schema import EngineConfig
from aether.runtime.contracts import ToolCall, TurnContext
from aether.tools.builtins.todo_write import (
    TodoWriteTool,
    clear_session_todos,
    get_session_todos,
)


def _ctx(session_id: str = "todo-tests") -> TurnContext:
    return TurnContext(session_id=session_id, iteration=1)


def _call(**args) -> ToolCall:
    return ToolCall(id="call-todo", name="todo_write", arguments=args)


class ArgValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = TodoWriteTool()
        clear_session_todos("todo-tests")

    def test_todos_must_be_list(self) -> None:
        result = self.tool.execute(_call(todos="not-a-list"), _ctx())
        self.assertTrue(result.is_error)
        self.assertIn("must be a list", result.content)

    def test_each_todo_must_have_required_fields(self) -> None:
        result = self.tool.execute(
            _call(todos=[{"id": "1", "content": "x"}]), _ctx()
        )
        self.assertTrue(result.is_error)
        self.assertIn("status", result.content)

    def test_invalid_status_enum_rejected(self) -> None:
        result = self.tool.execute(
            _call(todos=[{"id": "1", "content": "x", "status": "made_up"}]),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("invalid", result.content)

    def test_empty_id_rejected(self) -> None:
        result = self.tool.execute(
            _call(todos=[{"id": "", "content": "x", "status": "pending"}]),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("non-empty string", result.content)

    def test_non_dict_item_rejected(self) -> None:
        result = self.tool.execute(_call(todos=["just-a-string"]), _ctx())
        self.assertTrue(result.is_error)
        self.assertIn("must be an object", result.content)

    def test_partial_write_does_not_mutate_store_on_error(self) -> None:
        # Pre-populate with a known list, then send a half-bad write
        # and verify the original list is intact.
        self.tool.execute(
            _call(
                todos=[
                    {"id": "1", "content": "first", "status": "pending"},
                ]
            ),
            _ctx(),
        )
        before = get_session_todos("todo-tests")
        self.tool.execute(
            _call(
                todos=[
                    {"id": "2", "content": "ok", "status": "pending"},
                    {"id": "3", "content": "bad", "status": "made_up"},
                ]
            ),
            _ctx(),
        )
        after = get_session_todos("todo-tests")
        self.assertEqual(before, after)


class ReplaceSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = TodoWriteTool()
        clear_session_todos("todo-tests")

    def test_writes_full_list(self) -> None:
        result = self.tool.execute(
            _call(
                todos=[
                    {"id": "1", "content": "first", "status": "pending"},
                    {"id": "2", "content": "second", "status": "in_progress"},
                ]
            ),
            _ctx(),
        )
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(len(get_session_todos("todo-tests")), 2)

    def test_subsequent_write_replaces_prior_list(self) -> None:
        self.tool.execute(
            _call(
                todos=[
                    {"id": "1", "content": "first", "status": "pending"},
                    {"id": "2", "content": "second", "status": "pending"},
                ]
            ),
            _ctx(),
        )
        self.tool.execute(
            _call(
                todos=[
                    {"id": "3", "content": "third", "status": "pending"},
                ]
            ),
            _ctx(),
        )
        # Old ids 1+2 are gone \u2014 the second call REPLACED, not merged.
        ids = sorted(t["id"] for t in get_session_todos("todo-tests"))
        self.assertEqual(ids, ["3"])


class TerminalAutoClearTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = TodoWriteTool()
        clear_session_todos("todo-tests")

    def test_all_completed_clears_store(self) -> None:
        result = self.tool.execute(
            _call(
                todos=[
                    {"id": "1", "content": "x", "status": "completed"},
                    {"id": "2", "content": "y", "status": "completed"},
                ]
            ),
            _ctx(),
        )
        self.assertFalse(result.is_error)
        self.assertEqual(get_session_todos("todo-tests"), [])
        self.assertTrue(result.metadata["cleared"])
        self.assertIn("[cleared", result.content)

    def test_mixed_completed_and_cancelled_also_clears(self) -> None:
        result = self.tool.execute(
            _call(
                todos=[
                    {"id": "1", "content": "x", "status": "completed"},
                    {"id": "2", "content": "y", "status": "cancelled"},
                ]
            ),
            _ctx(),
        )
        self.assertEqual(get_session_todos("todo-tests"), [])
        self.assertTrue(result.metadata["cleared"])

    def test_one_pending_keeps_list(self) -> None:
        result = self.tool.execute(
            _call(
                todos=[
                    {"id": "1", "content": "x", "status": "completed"},
                    {"id": "2", "content": "y", "status": "pending"},
                ]
            ),
            _ctx(),
        )
        self.assertFalse(result.metadata["cleared"])
        self.assertEqual(len(get_session_todos("todo-tests")), 2)

    def test_empty_input_does_not_trigger_clear_message(self) -> None:
        # An empty list is not "all done"; metadata.cleared stays False.
        result = self.tool.execute(_call(todos=[]), _ctx())
        self.assertFalse(result.is_error)
        self.assertFalse(result.metadata["cleared"])
        self.assertEqual(get_session_todos("todo-tests"), [])


class IsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = TodoWriteTool()
        clear_session_todos("sess-A")
        clear_session_todos("sess-B")

    def test_two_sessions_have_independent_lists(self) -> None:
        self.tool.execute(
            _call(
                todos=[
                    {"id": "a", "content": "A-item", "status": "pending"},
                ]
            ),
            _ctx("sess-A"),
        )
        self.tool.execute(
            _call(
                todos=[
                    {"id": "b1", "content": "B-1", "status": "pending"},
                    {"id": "b2", "content": "B-2", "status": "in_progress"},
                ]
            ),
            _ctx("sess-B"),
        )
        a = get_session_todos("sess-A")
        b = get_session_todos("sess-B")
        self.assertEqual([t["id"] for t in a], ["a"])
        self.assertEqual([t["id"] for t in b], ["b1", "b2"])


class MetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = TodoWriteTool()
        clear_session_todos("todo-tests")

    def test_status_counters(self) -> None:
        result = self.tool.execute(
            _call(
                todos=[
                    {"id": "1", "content": "a", "status": "pending"},
                    {"id": "2", "content": "b", "status": "pending"},
                    {"id": "3", "content": "c", "status": "in_progress"},
                    {"id": "4", "content": "d", "status": "completed"},
                ]
            ),
            _ctx(),
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["pending"], 2)
        self.assertEqual(result.metadata["in_progress"], 1)
        self.assertEqual(result.metadata["completed"], 1)
        self.assertEqual(result.metadata["cancelled"], 0)
        self.assertEqual(result.metadata["todo_count"], 4)


class CheapToolListTests(unittest.TestCase):
    def test_todo_write_is_in_default_cheap_tool_names(self) -> None:
        # Closes the loop opened by Sprint 3 / PR 3.2: ``todo_write``
        # must be listed in EngineConfig.cheap_tool_names so a turn
        # whose only call is ``todo_write`` does NOT consume an
        # iteration slot.  Hard-asserting here trips loudly if a
        # future refactor ever drops the entry by accident.
        cfg = EngineConfig()
        self.assertIn("todo_write", cfg.cheap_tool_names)


if __name__ == "__main__":
    unittest.main()
