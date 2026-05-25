"""Task service contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskMessage:
    index: int
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    is_error: bool = False
    iteration: int | None = None
    elapsed_ms: float | None = None
    error: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TaskPendingMessage:
    index: int
    message: str
    ts: float | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TaskDeliveredMessage:
    index: int
    message: str
    ts: float | None = None
    delivered_at: float | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TaskMessagesResult:
    task_id: str
    messages: list[TaskMessage]
    pending_messages: list[TaskPendingMessage]
    delivered_messages: list[TaskDeliveredMessage]
    total_count: int
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class TaskSummary:
    task_id: str
    parent_session_id: str
    subagent_type: str
    prompt: str
    status: str
    started_at: float
    finished_at: float | None = None
    last_heartbeat: float = 0.0
    model: str | None = None
    isolation: str | None = None
    worktree_path: str | None = None
    parent_task_id: str | None = None
    child_depth: int = 1
    background: bool = False
    tool_use_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    summary: str | None = None
    error: str | None = None
    result_path: str | None = None
    output_tail: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TaskChildMessageStream:
    task: TaskSummary
    messages: list[TaskMessage]
    pending_messages: list[TaskPendingMessage]
    delivered_messages: list[TaskDeliveredMessage]
    total_count: int
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class TaskChildMessagesResult:
    task_id: str
    streams: list[TaskChildMessageStream]
    total_count: int
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class TaskResultArtifact:
    task_id: str
    result_path: str | None
    result: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TaskListResult:
    tasks: list[TaskSummary]
    active_count: int
    total_count: int


__all__ = ["TaskChildMessageStream", "TaskChildMessagesResult", "TaskDeliveredMessage", "TaskListResult", "TaskMessage", "TaskMessagesResult", "TaskPendingMessage", "TaskResultArtifact", "TaskSummary"]
