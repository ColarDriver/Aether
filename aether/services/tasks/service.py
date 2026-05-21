"""Read-only task observability service."""

from __future__ import annotations

from collections.abc import Callable

from aether.runtime.tasks import TaskRecord, TaskStore
from aether.services.common import ServiceNotFoundError, ServiceValidationError
from aether.services.tasks.contracts import TaskListResult, TaskSummary


TaskStoreFactory = Callable[[], TaskStore]


class TaskService:
    """Expose subagent task state without coupling web routes to TaskStore."""

    def __init__(
        self,
        *,
        store: TaskStore | None = None,
        store_factory: TaskStoreFactory | None = None,
    ) -> None:
        self._store = store
        self._store_factory = store_factory

    def list_tasks(
        self,
        *,
        session_id: str | None = None,
        active_only: bool = False,
        limit: int = 50,
        include_output_tail: bool = False,
        output_tail_bytes: int = 16_384,
    ) -> TaskListResult:
        if limit < 1 or limit > 200:
            raise ServiceValidationError("limit must be between 1 and 200")
        store = self._store_for_read()
        records = store.list_recent(limit=limit)
        if session_id:
            records = [record for record in records if record.parent_session_id == session_id]
        if active_only:
            records = [record for record in records if not record.status.is_terminal]
        tasks = [
            _summary_from_record(
                store,
                record,
                include_output_tail=include_output_tail,
                output_tail_bytes=output_tail_bytes,
            )
            for record in records[:limit]
        ]
        return TaskListResult(
            tasks=tasks,
            active_count=sum(1 for task in tasks if task.status not in _TERMINAL_STATUSES),
            total_count=len(tasks),
        )

    def get_task(
        self,
        task_id: str,
        *,
        include_output_tail: bool = True,
        output_tail_bytes: int = 100_000,
    ) -> TaskSummary:
        normalized = task_id.strip()
        if not normalized:
            raise ServiceValidationError("task_id is required")
        store = self._store_for_read()
        record = store.read(normalized)
        if record is None:
            raise ServiceNotFoundError("Task not found", details={"task_id": normalized})
        return _summary_from_record(
            store,
            record,
            include_output_tail=include_output_tail,
            output_tail_bytes=output_tail_bytes,
        )

    def _store_for_read(self) -> TaskStore:
        if self._store is not None:
            return self._store
        if self._store_factory is not None:
            return self._store_factory()
        return TaskStore()


_TERMINAL_STATUSES = {"completed", "failed", "interrupted", "killed"}


def _summary_from_record(
    store: TaskStore,
    record: TaskRecord,
    *,
    include_output_tail: bool,
    output_tail_bytes: int,
) -> TaskSummary:
    return TaskSummary(
        task_id=record.task_id,
        parent_session_id=record.parent_session_id,
        subagent_type=record.subagent_type,
        prompt=record.prompt,
        status=record.status.value,
        started_at=record.started_at,
        finished_at=record.finished_at,
        last_heartbeat=record.last_heartbeat,
        model=record.model,
        isolation=record.isolation,
        worktree_path=record.worktree_path,
        parent_task_id=record.parent_task_id,
        child_depth=record.child_depth,
        background=record.background,
        tool_use_count=record.tool_use_count,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        iterations=record.iterations,
        summary=record.summary,
        error=record.error,
        result_path=record.result_path,
        output_tail=(
            store.read_output_tail(record.task_id, max_bytes=output_tail_bytes)
            if include_output_tail
            else None
        ),
        metadata={
            "agent_type": record.agent_type_def_snapshot.get("name"),
            "description": record.agent_type_def_snapshot.get("description"),
        },
    )


__all__ = ["TaskService", "TaskStoreFactory"]
