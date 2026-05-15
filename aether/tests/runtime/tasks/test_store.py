"""Tests for TaskStore — file layout, atomic writes, concurrent access."""

from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aether.runtime.tasks.contracts import TaskRecord, TaskStatus
from aether.runtime.tasks.store import TaskStore


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


class TaskStoreLazyInitTests(unittest.TestCase):
    def test_construction_does_not_create_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "tasks"
            self.assertFalse(root.exists())
            TaskStore(root=root)
            # Lazy: root only materializes on first write.
            self.assertFalse(root.exists())

    def test_create_materializes_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "tasks"
            store = TaskStore(root=root)
            store.create(_record("t1"))
            self.assertTrue(root.exists())
            self.assertTrue((root / "t1" / "task.json").exists())


class TaskStoreCRUDTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))

    def test_create_then_read_roundtrip(self) -> None:
        record = _record("t1")
        record.model = "claude-sonnet-4-6"
        record.background = True
        self.store.create(record)
        roundtrip = self.store.read("t1")
        assert roundtrip is not None
        self.assertEqual(roundtrip.task_id, "t1")
        self.assertEqual(roundtrip.status, TaskStatus.RUNNING)
        self.assertEqual(roundtrip.model, "claude-sonnet-4-6")
        self.assertTrue(roundtrip.background)

    def test_read_unknown_returns_none(self) -> None:
        self.assertIsNone(self.store.read("nope"))

    def test_update_status_terminal_sets_finished_at(self) -> None:
        self.store.create(_record("t1"))
        self.store.update_status("t1", TaskStatus.COMPLETED, summary="ok")
        record = self.store.read("t1")
        assert record is not None
        self.assertEqual(record.status, TaskStatus.COMPLETED)
        self.assertEqual(record.summary, "ok")
        self.assertIsNotNone(record.finished_at)

    def test_update_status_failed_propagates_error(self) -> None:
        self.store.create(_record("t1"))
        self.store.update_status("t1", TaskStatus.FAILED, error="boom")
        record = self.store.read("t1")
        assert record is not None
        self.assertEqual(record.error, "boom")

    def test_record_progress_accumulates(self) -> None:
        self.store.create(_record("t1"))
        self.store.record_progress("t1", tool_use_count_delta=2, input_tokens=10, output_tokens=5)
        self.store.record_progress("t1", tool_use_count_delta=1, output_tokens=3, iterations=4)
        record = self.store.read("t1")
        assert record is not None
        self.assertEqual(record.tool_use_count, 3)
        self.assertEqual(record.input_tokens, 10)
        self.assertEqual(record.output_tokens, 8)
        self.assertEqual(record.iterations, 4)

    def test_progress_on_unknown_task_is_noop(self) -> None:
        self.store.record_progress("nope", tool_use_count_delta=1)  # must not raise

    def test_heartbeat_updates_timestamp(self) -> None:
        self.store.create(_record("t1"))
        before = self.store.read("t1")
        assert before is not None
        time.sleep(0.01)
        self.store.record_heartbeat("t1")
        after = self.store.read("t1")
        assert after is not None
        self.assertGreater(after.last_heartbeat, before.last_heartbeat)


class TaskStoreStreamsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))
        self.store.create(_record("t1"))

    def test_append_output_concatenates(self) -> None:
        self.store.append_output("t1", "hello ")
        self.store.append_output("t1", "world\n")
        self.assertEqual(self.store.read_output_tail("t1"), "hello world\n")

    def test_read_output_tail_handles_missing_file(self) -> None:
        self.assertEqual(self.store.read_output_tail("nope"), "")

    def test_read_output_tail_truncates_to_max_bytes(self) -> None:
        big = "line " + ("x" * 100) + "\n"
        for _ in range(50):
            self.store.append_output("t1", big)
        # Total ~5300 bytes; tail at 200 bytes should drop a partial first line.
        tail = self.store.read_output_tail("t1", max_bytes=200)
        self.assertLess(len(tail), 250)  # bound includes the discarded line
        # Must end on a complete line (the discard semantics ensure this).
        self.assertTrue(tail.endswith("\n"))

    def test_append_output_concurrent_threads(self) -> None:
        # 10 threads × 100 appends each — total 1000 lines.
        def worker(prefix: str) -> None:
            for i in range(100):
                self.store.append_output("t1", f"{prefix}-{i}\n")

        threads = [threading.Thread(target=worker, args=(f"w{idx}",)) for idx in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        out = self.store.read_output_tail("t1", max_bytes=10**6)
        self.assertEqual(len([line for line in out.splitlines() if line]), 1000)

    def test_append_message_writes_jsonl(self) -> None:
        self.store.append_message("t1", {"role": "assistant", "content": "hi"})
        self.store.append_message("t1", {"role": "tool", "name": "ping"})
        msgs_path = Path(self._tmp.name) / "t1" / "messages.jsonl"
        lines = msgs_path.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["role"], "assistant")
        self.assertEqual(json.loads(lines[1])["name"], "ping")


class TaskStorePendingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))
        self.store.create(_record("t1"))

    def test_drain_clears_queue(self) -> None:
        self.store.enqueue_pending_message("t1", "first")
        self.store.enqueue_pending_message("t1", "second")
        self.assertEqual(self.store.drain_pending_messages("t1"), ["first", "second"])
        self.assertEqual(self.store.drain_pending_messages("t1"), [])

    def test_drain_unknown_task_returns_empty(self) -> None:
        self.assertEqual(self.store.drain_pending_messages("nope"), [])

    def test_drain_skips_malformed_lines(self) -> None:
        # Manually corrupt the pending queue.
        path = Path(self._tmp.name) / "t1" / "pending.jsonl"
        self.store.enqueue_pending_message("t1", "good")
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        self.store.enqueue_pending_message("t1", "also good")
        drained = self.store.drain_pending_messages("t1")
        self.assertEqual(drained, ["good", "also good"])


class TaskStoreResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))
        self.store.create(_record("t1"))

    def test_write_result_creates_file_and_updates_record(self) -> None:
        path = self.store.write_result("t1", {"status": "completed", "summary": "ok"})
        self.assertTrue(path.exists())
        data = json.loads(path.read_text())
        self.assertEqual(data["summary"], "ok")
        # Record should now point at the result file.
        record = self.store.read("t1")
        assert record is not None
        self.assertEqual(record.result_path, str(path))

    def test_write_result_atomic_no_partial_on_corruption(self) -> None:
        # write_result writes via .tmp + os.replace — if we manually
        # poison the .tmp it should not break the previous result.
        self.store.write_result("t1", {"status": "completed", "v": 1})
        path = Path(self._tmp.name) / "t1" / "result.json"
        original = path.read_text()
        # Drop a junk .tmp; next write should still succeed and replace.
        (Path(self._tmp.name) / "t1" / "result.json.tmp").write_text("garbage")
        self.store.write_result("t1", {"status": "completed", "v": 2})
        self.assertEqual(json.loads(path.read_text())["v"], 2)
        self.assertNotEqual(path.read_text(), original)


class TaskStoreQueriesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))

    def test_list_active_excludes_terminal(self) -> None:
        self.store.create(_record("running1"))
        self.store.create(_record("running2"))
        self.store.create(_record("completed1"))
        self.store.update_status("completed1", TaskStatus.COMPLETED)
        active_ids = sorted(r.task_id for r in self.store.list_active())
        self.assertEqual(active_ids, ["running1", "running2"])

    def test_list_recent_orders_by_started_at_desc(self) -> None:
        for i in range(3):
            r = _record(f"t{i}")
            r.started_at = 100.0 + i
            self.store.create(r)
        recent = self.store.list_recent(limit=10)
        self.assertEqual([r.task_id for r in recent], ["t2", "t1", "t0"])

    def test_corrupt_task_json_returns_none_and_logs(self) -> None:
        self.store.create(_record("t1"))
        path = Path(self._tmp.name) / "t1" / "task.json"
        path.write_text("not valid json")
        self.assertIsNone(self.store.read("t1"))
        # A corrupt record must not crash list operations either.
        self.assertEqual(self.store.list_recent(), [])

    def test_iter_records_skips_non_directories(self) -> None:
        # A stray file at the root should not break iteration.
        self.store.create(_record("t1"))
        (Path(self._tmp.name) / "stray.txt").write_text("noise")
        recent = self.store.list_recent()
        self.assertEqual([r.task_id for r in recent], ["t1"])


class TaskStoreOrphanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))

    def test_mark_orphaned_marks_stale_running(self) -> None:
        old = _record("old")
        old.last_heartbeat = time.time() - 120  # 2 minutes ago
        self.store.create(old)
        # Reset heartbeat (create updates it) then re-stale by direct write.
        record = self.store.read("old")
        assert record is not None
        record.last_heartbeat = time.time() - 120
        self.store._write_record(record)  # noqa: SLF001 — internal test hook

        fresh = _record("fresh")
        self.store.create(fresh)

        marked = self.store.mark_orphaned(stale_after=60.0)
        self.assertEqual(marked, ["old"])
        old_after = self.store.read("old")
        fresh_after = self.store.read("fresh")
        assert old_after is not None and fresh_after is not None
        self.assertEqual(old_after.status, TaskStatus.KILLED)
        self.assertIn("Gateway restarted", old_after.error or "")
        self.assertEqual(fresh_after.status, TaskStatus.RUNNING)

    def test_mark_orphaned_ignores_already_terminal(self) -> None:
        completed = _record("done")
        self.store.create(completed)
        self.store.update_status("done", TaskStatus.COMPLETED)
        # Manually backdate heartbeat.
        record = self.store.read("done")
        assert record is not None
        record.last_heartbeat = time.time() - 1000
        self.store._write_record(record)  # noqa: SLF001

        marked = self.store.mark_orphaned(stale_after=60.0)
        self.assertEqual(marked, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
