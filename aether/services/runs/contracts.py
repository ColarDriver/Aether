"""Agent run service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from aether.services.runs.events import RunEvent


class AgentRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AgentRunOptions:
    max_iterations: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    disable_builtin_tools: bool | None = None
    system_override: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunRequest:
    session_id: str
    user_message: str
    attachments: list[dict[str, Any]] = field(default_factory=list)
    run_id: str | None = None
    cwd: str | None = None
    options: AgentRunOptions = field(default_factory=AgentRunOptions)
    approval_prompter: Any = field(default=None, repr=False, compare=False)
    tool_permission_prompter: Any = field(default=None, repr=False, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    session_id: str
    run_id: str
    final_text: str = ""
    exit_reason: str = "done"
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentRunSnapshot:
    session_id: str
    run_id: str
    status: AgentRunStatus
    started_at: float | None = None
    completed_at: float | None = None
    result: AgentRunResult | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunCancelRequest:
    session_id: str | None = None
    run_id: str | None = None
    reason: str | None = None


class RunEventSink(Protocol):
    def emit(self, event: RunEvent) -> None:
        ...


__all__ = [
    "AgentRunCancelRequest",
    "AgentRunOptions",
    "AgentRunRequest",
    "AgentRunResult",
    "AgentRunSnapshot",
    "AgentRunStatus",
    "RunEvent",
    "RunEventSink",
]
