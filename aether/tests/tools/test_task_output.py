"""Tests for the rewritten TaskOutputTool — PR 10.6."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.runtime.tasks import TaskRecord, TaskStatus, TaskStore
from aether.tools.builtins.task_output import TaskOutputTool


def _record(task_id: str, *, status: TaskStatus = TaskStatus.RUNNING) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        parent_session_id="s",
        subagent_type="general-purpose",
        prompt="do work",
        status=status,
        started_at=time.time(),
        last_heartbeat=time.time(),
    )


def _ctx(*, store: TaskStore | None = None) -> TurnContext:
    metadata: dict = {}
    if store is not None:
        metadata["_task_store"] = store
    return TurnContext(session_id="s", iteration=0, metadata=metadata)


def _execute(
    tool: TaskOutputTool,
    *,
    task_id: str,
    block: bool = True,
    timeout_ms: int = 30_000,
    ctx: TurnContext | None = None,
):
    ctx = ctx or _ctx()
    return tool.execute(
        ToolCall(
            id="c1",
            name="task_output",
            arguments={"task_id": task_id, "block": block, "timeout_ms": timeout_ms},
        ),
        ctx,
    )


# ---------------------------------------------------------------- input validation

class InputValidationTests(unittest.TestCase):
    def test_missing_task_id_errors(self) -> None:
        tool = TaskOutputTool()
        result = tool.execute(
            ToolCall(id="c1", name="task_output", arguments={}), _ctx()
        )
        self.assertTrue(result.is_error)
        self.assertIn("task_id", result.content)

    def test_block_must_be_bool(self) -> None:
        tool = TaskOutputTool()
        result = tool.execute(
            ToolCall(
                id="c1",
                name="task_output",
                arguments={"task_id": "t", "block": "yes"},
            ),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("block", result.content)

    def test_timeout_must_be_int(self) -> None:
        tool = TaskOutputTool()
        result = tool.execute(
            ToolCall(
                id="c1",
                name="task_output",
                arguments={"task_id": "t", "timeout_ms": "soon"},
            ),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("timeout_ms", result.content)

    def test_timeout_clamped_to_max(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            store.create(_record("t1", status=TaskStatus.COMPLETED))
            tool = TaskOutputTool(task_store=store)
            # 10x the max ceiling should still succeed (clamped silently).
            result = tool.execute(
                ToolCall(
                    id="c1",
                    name="task_output",
                    arguments={
                        "task_id": "t1",
                        "block": True,
                        "timeout_ms": 6_000_000,
                    },
                ),
                _ctx(),
            )
            self.assertFalse(result.is_error)


# ---------------------------------------------------------------- store resolution

class StoreResolutionTests(unittest.TestCase):
    def test_no_store_returns_clear_error(self) -> None:
        tool = TaskOutputTool()
        result = _execute(tool, task_id="anything")
        self.assertTrue(result.is_error)
        self.assertIn("TaskStore", result.content)

    def test_constructor_store_wins_over_context(self) -> None:
        with TemporaryDirectory() as tmp_a, TemporaryDirectory() as tmp_b:
            store_a = TaskStore(root=Path(tmp_a))
            store_b = TaskStore(root=Path(tmp_b))
            store_a.create(_record("only-in-a", status=TaskStatus.COMPLETED))
            store_b.create(_record("only-in-b", status=TaskStatus.COMPLETED))
            tool = TaskOutputTool(task_store=store_a)
            # Context supplies store_b but constructor (store_a) wins.
            result = _execute(tool, task_id="only-in-a", ctx=_ctx(store=store_b))
            self.assertFalse(result.is_error)

    def test_context_fallback_works(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            store.create(_record("from-ctx", status=TaskStatus.COMPLETED))
            tool = TaskOutputTool()  # no constructor store
            result = _execute(tool, task_id="from-ctx", ctx=_ctx(store=store))
            self.assertFalse(result.is_error)


# ---------------------------------------------------------------- non-block path

class NonBlockingSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))
        self.tool = TaskOutputTool(task_store=self.store)

    def test_unknown_task_id_errors(self) -> None:
        result = _execute(self.tool, task_id="never-existed", block=False)
        self.assertTrue(result.is_error)
        self.assertIn("unknown task_id", result.content)

    def test_running_task_returns_partial_snapshot(self) -> None:
        self.store.create(_record("t1"))
        self.store.append_output("t1", "hello world\n")
        self.store.record_progress("t1", tool_use_count_delta=2)
        result = _execute(self.tool, task_id="t1", block=False)
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["status"], "running")
        self.assertEqual(result.metadata["progress"]["tool_use_count"], 2)
        self.assertIsNone(result.metadata["duration_seconds"])
        self.assertFalse(result.metadata["timed_out"])
        self.assertIn("hello world", result.content)
        self.assertIn("Output (tail)", result.content)

    def test_completed_task_returns_summary_and_duration(self) -> None:
        self.store.create(_record("t1"))
        self.store.update_status(
            "t1", TaskStatus.COMPLETED, summary="all done"
        )
        result = _execute(self.tool, task_id="t1", block=False)
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["status"], "completed")
        self.assertEqual(result.metadata["summary"], "all done")
        self.assertIsNotNone(result.metadata["duration_seconds"])
        self.assertIn("all done", result.content)

    def test_failed_task_marks_is_error(self) -> None:
        self.store.create(_record("t1"))
        self.store.update_status("t1", TaskStatus.FAILED, error="boom")
        result = _execute(self.tool, task_id="t1", block=False)
        self.assertTrue(result.is_error)
        self.assertEqual(result.metadata["status"], "failed")
        self.assertIn("boom", result.content)

    def test_killed_task_after_recovery(self) -> None:
        self.store.create(_record("t1"))
        self.store.update_status("t1", TaskStatus.KILLED, error="restart")
        result = _execute(self.tool, task_id="t1", block=False)
        self.assertFalse(result.is_error)  # KILLED is not FAILED
        self.assertEqual(result.metadata["status"], "killed")
        self.assertIn("restart", result.content)

    def test_interrupted_task_returns_status(self) -> None:
        self.store.create(_record("t1"))
        self.store.update_status("t1", TaskStatus.INTERRUPTED)
        result = _execute(self.tool, task_id="t1", block=False)
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["status"], "interrupted")


# ---------------------------------------------------------------- blocking path

class BlockingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))
        self.tool = TaskOutputTool(task_store=self.store)

    def test_terminal_at_call_time_returns_immediately(self) -> None:
        self.store.create(_record("t1"))
        self.store.update_status("t1", TaskStatus.COMPLETED, summary="ok")
        t0 = time.monotonic()
        result = _execute(self.tool, task_id="t1", block=True, timeout_ms=10_000)
        elapsed = time.monotonic() - t0
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["status"], "completed")
        self.assertLess(elapsed, 0.5)

    def test_unblocks_when_status_changes(self) -> None:
        self.store.create(_record("t1"))

        def _flip_to_completed_after_ms(ms: int) -> None:
            time.sleep(ms / 1000.0)
            self.store.update_status("t1", TaskStatus.COMPLETED, summary="late")

        worker = threading.Thread(
            target=_flip_to_completed_after_ms, args=(400,), daemon=True
        )
        worker.start()
        try:
            t0 = time.monotonic()
            result = _execute(
                self.tool, task_id="t1", block=True, timeout_ms=5_000
            )
            elapsed = time.monotonic() - t0
        finally:
            worker.join(timeout=2.0)
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["status"], "completed")
        self.assertEqual(result.metadata["summary"], "late")
        # Should unblock soon after the 400ms flip — well under the 5s budget.
        self.assertLess(elapsed, 1.5)

    def test_timeout_returns_snapshot_with_timed_out_flag(self) -> None:
        self.store.create(_record("t1"))
        self.store.append_output("t1", "still working...\n")
        t0 = time.monotonic()
        result = _execute(self.tool, task_id="t1", block=True, timeout_ms=200)
        elapsed = time.monotonic() - t0
        self.assertFalse(result.is_error)  # timeout is not a tool error
        self.assertEqual(result.metadata["status"], "running")
        self.assertTrue(result.metadata["timed_out"])
        self.assertIn("_timed_out: true", result.content)
        # Honor the timeout window (allow ≤ 1s of jitter for the poll
        # interval + scheduling).
        self.assertLess(elapsed, 1.5)
        self.assertGreaterEqual(elapsed, 0.18)

    def test_timeout_zero_acts_like_non_block(self) -> None:
        self.store.create(_record("t1"))
        t0 = time.monotonic()
        result = _execute(self.tool, task_id="t1", block=True, timeout_ms=0)
        elapsed = time.monotonic() - t0
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["status"], "running")
        self.assertFalse(result.metadata["timed_out"])
        self.assertLess(elapsed, 0.2)


# ---------------------------------------------------------------- spill path

class SpillTests(unittest.TestCase):
    def test_large_output_is_spilled(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            store.create(_record("t1", status=TaskStatus.COMPLETED))
            # Push >MAX_RESULT_CHARS (60k) of content.
            big = ("x" * 200 + "\n") * 400  # ~80KB
            store.append_output("t1", big)
            tool = TaskOutputTool(task_store=store)
            result = _execute(tool, task_id="t1", block=False)
            self.assertFalse(result.is_error)
            # Spill machinery should have shortened content; the
            # snippet should mention the spill / preview indicator.
            self.assertLessEqual(len(result.content), tool.MAX_RESULT_CHARS + 5_000)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
