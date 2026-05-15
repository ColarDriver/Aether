"""Tests for TaskStore recovery on engine startup."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aether.runtime.tasks.contracts import TaskRecord, TaskStatus
from aether.runtime.tasks.recovery import recover_task_store
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


class RecoveryReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))

    def test_marks_old_running_as_killed(self) -> None:
        # One old RUNNING (orphan), one fresh RUNNING (alive),
        # one COMPLETED (no-op).
        old = _record("old")
        self.store.create(old)
        record = self.store.read("old")
        assert record is not None
        record.last_heartbeat = time.time() - 600
        self.store._write_record(record)  # noqa: SLF001 — test hook

        self.store.create(_record("fresh"))

        done = _record("done")
        self.store.create(done)
        self.store.update_status("done", TaskStatus.COMPLETED)

        report = recover_task_store(self.store, stale_after=60.0)

        self.assertEqual(report.orphaned_count, 1)
        self.assertEqual(report.completed_count, 1)
        self.assertEqual(report.active_count, 1)  # only "fresh" remains active

        old_after = self.store.read("old")
        assert old_after is not None
        self.assertEqual(old_after.status, TaskStatus.KILLED)

    def test_empty_store_returns_zero_report(self) -> None:
        report = recover_task_store(self.store)
        self.assertEqual(report.orphaned_count, 0)
        self.assertEqual(report.completed_count, 0)
        self.assertEqual(report.active_count, 0)

    def test_idempotent_second_call_marks_nothing_new(self) -> None:
        old = _record("old")
        self.store.create(old)
        record = self.store.read("old")
        assert record is not None
        record.last_heartbeat = time.time() - 600
        self.store._write_record(record)  # noqa: SLF001

        first = recover_task_store(self.store, stale_after=60.0)
        second = recover_task_store(self.store, stale_after=60.0)

        self.assertEqual(first.orphaned_count, 1)
        self.assertEqual(second.orphaned_count, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
