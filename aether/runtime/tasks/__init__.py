"""On-disk task store for subagent lifecycle persistence.

The store lives at ``~/.aether/tasks/{task_id}/`` (configurable via
:attr:`aether.config.schema.EngineConfig.task_store_path`).  Each task
gets its own directory containing:

- ``task.json``     — :class:`TaskRecord`, atomic write
- ``output.log``    — streaming text appended by lifecycle hooks
- ``messages.jsonl`` — full assistant/tool message stream, one JSON / line
- ``pending.jsonl`` — :class:`SendMessage` queue, drained at iteration boundary
- ``result.json``   — written once on terminal status

The store is single-process; multi-thread safety is provided by per-task
``threading.Lock`` instances.  Cross-process locking is intentionally not
implemented — operators that want multiple gateways must use distinct
``task_store_path`` roots.

Backs PR 10.5 (async lifecycle), PR 10.6 (TaskOutput), PR 10.7
(SendMessage + notifications), and PR 10.8 (worktree isolation
metadata).
"""

from aether.runtime.tasks.contracts import TaskRecord, TaskStatus
from aether.runtime.tasks.recovery import RecoveryReport, recover_task_store
from aether.runtime.tasks.store import TaskStore

__all__ = [
    "RecoveryReport",
    "TaskRecord",
    "TaskStatus",
    "TaskStore",
    "recover_task_store",
]
