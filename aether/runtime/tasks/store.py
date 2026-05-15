"""TaskStore — directory layout under ``~/.aether/tasks/{task_id}/``.

Single-process store; concurrency is handled by per-task
``threading.Lock``.  Cross-process locking is intentionally out of scope.
See ``__init__.py`` for the file layout.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from aether.runtime.tasks.contracts import TaskRecord, TaskStatus

logger = logging.getLogger(__name__)


_DEFAULT_HEARTBEAT_STALE_SECONDS = 60.0


class TaskStore:
    """Persist subagent task state on disk.

    Construction is cheap (only creates the root directory); discovery
    happens lazily on each query.  Callers should reuse a single
    instance per gateway process — nothing in the API is process-global,
    but per-task locks live on the instance and are not shared across
    instances.
    """

    def __init__(self, root: Path | None = None) -> None:
        # Lazily resolve and create the root directory.  Creating it
        # eagerly here would surprise users with an unwanted
        # ``~/.aether/tasks/`` directory just for constructing an
        # ``AgentEngine`` (e.g. in tests that never spawn subagents).
        self.root: Path = (root or Path.home() / ".aether" / "tasks").expanduser()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self._root_created = False

    def _ensure_root(self) -> None:
        if not self._root_created:
            self.root.mkdir(parents=True, exist_ok=True)
            self._root_created = True

    # ------------------------------------------------------------------ paths
    def _task_dir(self, task_id: str) -> Path:
        return self.root / task_id

    def _lock_for(self, task_id: str) -> threading.Lock:
        with self._locks_lock:
            lock = self._locks.get(task_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[task_id] = lock
            return lock

    # ------------------------------------------------------------------ CRUD
    def create(self, record: TaskRecord) -> None:
        """Create a fresh task directory and write the initial record.

        Touches ``output.log`` / ``messages.jsonl`` / ``pending.jsonl``
        so readers attaching mid-flight don't see ``FileNotFoundError``.
        """
        self._ensure_root()
        d = self._task_dir(record.task_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "output.log").touch(exist_ok=True)
        (d / "messages.jsonl").touch(exist_ok=True)
        (d / "pending.jsonl").touch(exist_ok=True)
        record.last_heartbeat = time.time()
        with self._lock_for(record.task_id):
            self._write_record(record)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        finished_at: Optional[float] = None,
        summary: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock_for(task_id):
            record = self._read_record(task_id)
            if record is None:
                logger.warning("update_status on unknown task %s", task_id)
                return
            record.status = status
            if finished_at is not None:
                record.finished_at = finished_at
            elif status.is_terminal and record.finished_at is None:
                record.finished_at = time.time()
            if summary is not None:
                record.summary = summary
            if error is not None:
                record.error = error
            record.last_heartbeat = time.time()
            self._write_record(record)

    def record_heartbeat(self, task_id: str) -> None:
        with self._lock_for(task_id):
            record = self._read_record(task_id)
            if record is None:
                return
            record.last_heartbeat = time.time()
            self._write_record(record)

    def record_progress(
        self,
        task_id: str,
        *,
        tool_use_count_delta: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        iterations: int | None = None,
    ) -> None:
        with self._lock_for(task_id):
            record = self._read_record(task_id)
            if record is None:
                return
            record.tool_use_count += tool_use_count_delta
            record.input_tokens += input_tokens
            record.output_tokens += output_tokens
            if iterations is not None:
                record.iterations = iterations
            record.last_heartbeat = time.time()
            self._write_record(record)

    # ------------------------------------------------------------------ streams
    def append_output(self, task_id: str, text: str) -> None:
        if not text:
            return
        path = self._task_dir(task_id) / "output.log"
        with self._lock_for(task_id):
            with path.open("a", encoding="utf-8") as fh:
                fh.write(text)

    def read_output_tail(self, task_id: str, *, max_bytes: int = 100_000) -> str:
        """Read the last ``max_bytes`` of ``output.log`` for the given task.

        Discards a partial first line when seeking into the middle of
        the file so callers always get whole-line tails.
        """
        path = self._task_dir(task_id) / "output.log"
        if not path.exists():
            return ""
        try:
            size = path.stat().st_size
            with path.open("rb") as fh:
                if size > max_bytes:
                    fh.seek(size - max_bytes)
                    fh.readline()  # discard partial first line
                return fh.read().decode("utf-8", errors="replace")
        except OSError as exc:
            logger.warning("read_output_tail failed for %s: %s", task_id, exc)
            return ""

    def append_message(self, task_id: str, message: dict) -> None:
        path = self._task_dir(task_id) / "messages.jsonl"
        with self._lock_for(task_id):
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(message, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ pending
    def enqueue_pending_message(self, task_id: str, msg: str) -> None:
        path = self._task_dir(task_id) / "pending.jsonl"
        entry = {"ts": time.time(), "message": msg}
        with self._lock_for(task_id):
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def drain_pending_messages(self, task_id: str) -> list[str]:
        path = self._task_dir(task_id) / "pending.jsonl"
        with self._lock_for(task_id):
            if not path.exists():
                return []
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                logger.warning("drain_pending_messages read failed for %s: %s", task_id, exc)
                return []
            path.write_text("", encoding="utf-8")
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
                out.append(str(entry.get("message", "")))
            except json.JSONDecodeError:
                logger.warning("drain_pending_messages skipping malformed line for %s", task_id)
                continue
        return out

    # ------------------------------------------------------------------ result
    def write_result(self, task_id: str, result_dump: dict) -> Path:
        d = self._task_dir(task_id)
        d.mkdir(parents=True, exist_ok=True)
        target = d / "result.json"
        tmp = d / "result.json.tmp"
        with self._lock_for(task_id):
            tmp.write_text(
                json.dumps(result_dump, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, target)
            record = self._read_record(task_id)
            if record is not None:
                record.result_path = str(target)
                self._write_record(record)
        return target

    # ------------------------------------------------------------------ queries
    def read(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock_for(task_id):
            return self._read_record(task_id)

    def list_active(self) -> list[TaskRecord]:
        return [r for r in self._iter_records() if not r.status.is_terminal]

    def list_recent(self, limit: int = 50) -> list[TaskRecord]:
        records = sorted(
            self._iter_records(),
            key=lambda r: r.started_at,
            reverse=True,
        )
        return records[:limit]

    def mark_orphaned(
        self, *, stale_after: float = _DEFAULT_HEARTBEAT_STALE_SECONDS
    ) -> list[str]:
        """Mark stale RUNNING records as KILLED.

        Used by recovery on gateway startup.  Returns the list of
        ``task_id``s that got marked.
        """
        now = time.time()
        marked: list[str] = []
        for record in list(self._iter_records()):
            if (
                record.status == TaskStatus.RUNNING
                and (now - record.last_heartbeat) > stale_after
            ):
                self.update_status(
                    record.task_id,
                    TaskStatus.KILLED,
                    error="Gateway restarted while task was running",
                )
                marked.append(record.task_id)
        return marked

    # ------------------------------------------------------------------ internals
    def _write_record(self, record: TaskRecord) -> None:
        d = self._task_dir(record.task_id)
        d.mkdir(parents=True, exist_ok=True)
        target = d / "task.json"
        tmp = d / "task.json.tmp"
        tmp.write_text(
            json.dumps(record.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, target)

    def _read_record(self, task_id: str) -> Optional[TaskRecord]:
        path = self._task_dir(task_id) / "task.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TaskRecord.from_json(data)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("task %s corrupt: %s", task_id, exc)
            return None

    def _iter_records(self) -> Iterator[TaskRecord]:
        if not self.root.exists():
            return
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            record = self._read_record(entry.name)
            if record is not None:
                yield record


__all__ = ["TaskStore"]
