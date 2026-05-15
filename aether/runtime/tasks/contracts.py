"""Data contracts for the on-disk task store."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskStatus(str, Enum):
    """Lifecycle status of a subagent task.

    Terminal states (``COMPLETED`` / ``FAILED`` / ``INTERRUPTED`` /
    ``KILLED``) are sticky once set; the store rejects further status
    transitions on terminal records via the in-memory lock contract.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    KILLED = "killed"           # post-mortem mark for orphaned RUNNING records

    @property
    def is_terminal(self) -> bool:
        return self in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.INTERRUPTED,
            TaskStatus.KILLED,
        }


@dataclass(slots=True)
class TaskRecord:
    """Persisted snapshot of one subagent task.

    Field semantics:

    - ``last_heartbeat`` is updated by the async lifecycle on every
      iteration boundary; recovery uses it to detect orphans (RUNNING
      records whose heartbeat is older than the configured threshold
      get marked KILLED on next gateway start).
    - ``agent_type_def_snapshot`` is the frozen dump of the
      :class:`AgentTypeDefinition` that was active when the task was
      created.  Frozen so a later registry change cannot rewrite history.
    - ``result_path`` is filled when ``write_result`` lands the final
      result.json blob; until then it stays ``None``.
    """

    task_id: str
    parent_session_id: str
    subagent_type: str
    prompt: str
    status: TaskStatus
    started_at: float                              # unix ts
    finished_at: Optional[float] = None
    last_heartbeat: float = 0.0
    agent_type_def_snapshot: dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    isolation: Optional[str] = None
    worktree_path: Optional[str] = None
    parent_task_id: Optional[str] = None
    child_depth: int = 1
    background: bool = False
    # progress counters (incrementally updated by lifecycle hooks)
    tool_use_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    # final result fields
    summary: Optional[str] = None
    error: Optional[str] = None
    result_path: Optional[str] = None              # ~/.aether/tasks/{id}/result.json

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.value
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TaskRecord":
        # Defensive copy so caller mutations don't leak into our record.
        payload = {**data}
        payload["status"] = TaskStatus(payload["status"])
        return cls(**payload)


__all__ = ["TaskRecord", "TaskStatus"]
