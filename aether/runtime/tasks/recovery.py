"""Recovery for the on-disk task store.

Called once at root :class:`AgentEngine` init.  Marks any RUNNING records
whose heartbeat is older than the configured threshold as KILLED so the
gateway never reports stale "still running" tasks to the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aether.runtime.tasks.contracts import TaskStatus
from aether.runtime.tasks.store import TaskStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RecoveryReport:
    orphaned_count: int
    completed_count: int
    active_count: int


def recover_task_store(
    store: TaskStore, *, stale_after: float = 60.0
) -> RecoveryReport:
    """Reconcile on-disk task records with the now-empty in-memory state.

    Should only be called from the root engine (``delegate_depth == 0``);
    subagents share the parent's store and must not double-mark.
    """
    orphans = store.mark_orphaned(stale_after=stale_after)
    if orphans:
        logger.info(
            "task recovery: marked %d orphan(s) as KILLED: %s",
            len(orphans),
            orphans,
        )
    recent = store.list_recent(limit=10_000)
    completed = sum(1 for r in recent if r.status == TaskStatus.COMPLETED)
    active = len(store.list_active())
    return RecoveryReport(
        orphaned_count=len(orphans),
        completed_count=completed,
        active_count=active,
    )


__all__ = ["RecoveryReport", "recover_task_store"]
