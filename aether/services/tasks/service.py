"""Read-only task observability service."""

from __future__ import annotations

from collections.abc import Callable

from aether.runtime.tasks import TaskRecord, TaskStatus, TaskStore
from aether.services.common import ServiceConflictError, ServiceNotFoundError, ServiceValidationError
from aether.services.tasks.contracts import TaskChildMessageStream, TaskChildMessagesResult, TaskDeliveredMessage, TaskListResult, TaskMessage, TaskMessagesResult, TaskPendingMessage, TaskResultArtifact, TaskSendMessageResult, TaskStopResult, TaskSummary


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

    def get_task_messages(
        self,
        task_id: str,
        *,
        limit: int = 100,
    ) -> TaskMessagesResult:
        normalized = task_id.strip()
        if not normalized:
            raise ServiceValidationError("task_id is required")
        if limit < 1 or limit > 500:
            raise ServiceValidationError("limit must be between 1 and 500")
        store = self._store_for_read()
        if store.read(normalized) is None:
            raise ServiceNotFoundError("Task not found", details={"task_id": normalized})
        raw_messages = store.read_messages(normalized, limit=limit)
        raw_pending = store.read_pending_messages(normalized, limit=limit)
        raw_delivered = store.read_delivered_messages(normalized, limit=limit)
        messages = [_message_from_raw(raw) for raw in raw_messages]
        pending_messages = [_pending_message_from_raw(raw) for raw in raw_pending]
        delivered_messages = [_delivered_message_from_raw(raw) for raw in raw_delivered]
        return TaskMessagesResult(
            task_id=normalized,
            messages=messages,
            pending_messages=pending_messages,
            delivered_messages=delivered_messages,
            total_count=len(messages),
            truncated=(
                len(messages) >= limit
                or len(pending_messages) >= limit
                or len(delivered_messages) >= limit
            ),
        )

    def get_child_task_messages(
        self,
        task_id: str,
        *,
        limit: int = 50,
        per_task_limit: int = 25,
    ) -> TaskChildMessagesResult:
        normalized = task_id.strip()
        if not normalized:
            raise ServiceValidationError("task_id is required")
        if limit < 1 or limit > 100:
            raise ServiceValidationError("limit must be between 1 and 100")
        if per_task_limit < 1 or per_task_limit > 200:
            raise ServiceValidationError("per_task_limit must be between 1 and 200")
        store = self._store_for_read()
        root = store.read(normalized)
        if root is None:
            raise ServiceNotFoundError("Task not found", details={"task_id": normalized})
        records = store.list_recent(limit=1000)
        by_id = {record.task_id: record for record in records}
        by_id[root.task_id] = root
        descendants = [
            record
            for record in by_id.values()
            if record.task_id != root.task_id
            and _is_descendant_of(record, root.task_id, by_id)
        ]
        descendants.sort(key=lambda record: (record.child_depth, record.started_at, record.task_id))
        selected = descendants[:limit]
        streams = [
            _child_stream_from_record(store, record, message_limit=per_task_limit)
            for record in selected
        ]
        return TaskChildMessagesResult(
            task_id=normalized,
            streams=streams,
            total_count=len(descendants),
            truncated=len(descendants) > len(selected),
        )

    def get_task_result(self, task_id: str) -> TaskResultArtifact:
        normalized = task_id.strip()
        if not normalized:
            raise ServiceValidationError("task_id is required")
        store = self._store_for_read()
        record = store.read(normalized)
        if record is None:
            raise ServiceNotFoundError("Task not found", details={"task_id": normalized})
        result = store.read_result(normalized)
        if result is None:
            raise ServiceNotFoundError("Task result not found", details={"task_id": normalized})
        return TaskResultArtifact(
            task_id=normalized,
            result_path=record.result_path,
            result=result,
        )

    def send_task_message(
        self,
        task_id: str,
        *,
        message: str,
        summary: str | None = None,
    ) -> TaskSendMessageResult:
        normalized = task_id.strip()
        if not normalized:
            raise ServiceValidationError("task_id is required")
        text = message.strip() if isinstance(message, str) else ""
        if not text:
            raise ServiceValidationError("message is required")
        if summary is not None and not isinstance(summary, str):
            raise ServiceValidationError("summary must be a string when provided")
        store = self._store_for_read()
        record = store.read(normalized)
        if record is None:
            raise ServiceNotFoundError("Task not found", details={"task_id": normalized})
        if record.status != TaskStatus.RUNNING:
            raise ServiceConflictError(
                f"cannot send to {normalized!r}: status is {record.status.value!r} (must be 'running')",
                details={"task_id": normalized, "status": record.status.value},
            )
        store.enqueue_pending_message(normalized, text)
        return TaskSendMessageResult(
            task_id=normalized,
            queued=True,
            status=record.status.value,
            message="Queued message for the subagent's next iteration boundary.",
            queued_chars=len(text),
        )

    def stop_task(self, task_id: str, *, stopper: Callable[[str], bool]) -> TaskStopResult:
        normalized = task_id.strip()
        if not normalized:
            raise ServiceValidationError("task_id is required")
        store = self._store_for_read()
        record = store.read(normalized)
        if record is None:
            raise ServiceNotFoundError("Task not found", details={"task_id": normalized})
        if record.status.is_terminal:
            return TaskStopResult(
                task_id=normalized,
                delivered=False,
                status=record.status.value,
                message="Task is already terminal.",
            )
        delivered = bool(stopper(normalized))
        if delivered:
            store.append_message(normalized, {"role": "system", "content": "stop requested from web console"})
            return TaskStopResult(
                task_id=normalized,
                delivered=True,
                status=record.status.value,
                message="Stop signal sent to running task.",
            )
        return TaskStopResult(
            task_id=normalized,
            delivered=False,
            status=record.status.value,
            message="Task is not attached to an active runtime manager.",
        )

    def delete_session_tasks(self, session_id: str) -> int:
        normalized = session_id.strip()
        if not normalized:
            raise ServiceValidationError("session_id is required")
        return self._store_for_read().delete_session_tasks(normalized)

    def _store_for_read(self) -> TaskStore:
        if self._store is not None:
            return self._store
        if self._store_factory is not None:
            return self._store_factory()
        return TaskStore()


_TERMINAL_STATUSES = {"completed", "failed", "interrupted", "killed"}


def _child_stream_from_record(
    store: TaskStore,
    record: TaskRecord,
    *,
    message_limit: int,
) -> TaskChildMessageStream:
    messages = [_message_from_raw(raw) for raw in store.read_messages(record.task_id, limit=message_limit)]
    pending_messages = [
        _pending_message_from_raw(raw)
        for raw in store.read_pending_messages(record.task_id, limit=message_limit)
    ]
    delivered_messages = [
        _delivered_message_from_raw(raw)
        for raw in store.read_delivered_messages(record.task_id, limit=message_limit)
    ]
    return TaskChildMessageStream(
        task=_summary_from_record(
            store,
            record,
            include_output_tail=False,
            output_tail_bytes=0,
        ),
        messages=messages,
        pending_messages=pending_messages,
        delivered_messages=delivered_messages,
        total_count=len(messages),
        truncated=(
            len(messages) >= message_limit
            or len(pending_messages) >= message_limit
            or len(delivered_messages) >= message_limit
        ),
    )


def _is_descendant_of(
    record: TaskRecord,
    root_task_id: str,
    by_id: dict[str, TaskRecord],
) -> bool:
    parent_id = record.parent_task_id
    seen: set[str] = set()
    while parent_id:
        if parent_id == root_task_id:
            return True
        if parent_id in seen:
            return False
        seen.add(parent_id)
        parent_id = by_id.get(parent_id).parent_task_id if parent_id in by_id else None
    return False


def _message_from_raw(raw: dict) -> TaskMessage:
    index = _coerce_int(raw.get("index"), default=0)
    return TaskMessage(
        index=index,
        role=str(raw.get("role") or "unknown"),
        content=_optional_str(raw.get("content")),
        name=_optional_str(raw.get("name")),
        tool_call_id=_optional_str(raw.get("tool_call_id")),
        is_error=bool(raw.get("is_error")),
        iteration=_coerce_optional_int(raw.get("iteration")),
        elapsed_ms=_coerce_optional_float(raw.get("elapsed_ms")),
        error=_optional_str(raw.get("error")),
        raw=dict(raw),
    )


def _pending_message_from_raw(raw: dict) -> TaskPendingMessage:
    return TaskPendingMessage(
        index=_coerce_int(raw.get("index"), default=0),
        message=_optional_str(raw.get("message")) or "",
        ts=_coerce_optional_float(raw.get("ts")),
        raw=dict(raw),
    )


def _delivered_message_from_raw(raw: dict) -> TaskDeliveredMessage:
    return TaskDeliveredMessage(
        index=_coerce_int(raw.get("index"), default=0),
        message=_optional_str(raw.get("message")) or "",
        ts=_coerce_optional_float(raw.get("ts")),
        delivered_at=_coerce_optional_float(raw.get("delivered_at")),
        raw=dict(raw),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object, *, default: int) -> int:
    parsed = _coerce_optional_int(value)
    return parsed if parsed is not None else default


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
