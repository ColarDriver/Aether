"""Subagent manager with controlled fan-out and depth."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from aether.runtime.contracts import EngineStatus
from aether.subagents.builder import SubagentBuilder
from aether.subagents.contracts import SubagentResult, SubagentStatus, SubagentTask
from aether.subagents.default_builder import DefaultSubagentBuilder


class SubagentManager:
    """Coordinate child-agent execution with depth and concurrency limits."""

    def __init__(
        self,
        *,
        builder: SubagentBuilder | None = None,
        max_concurrent_children: int = 3,
        max_spawn_depth: int = 2,
        logger: logging.Logger | None = None,
    ) -> None:
        self.builder = builder or DefaultSubagentBuilder()
        self.max_concurrent_children = max(1, int(max_concurrent_children))
        self.max_spawn_depth = max(1, int(max_spawn_depth))
        self.logger = logger or logging.getLogger(__name__)

    def run_task(self, parent, task: SubagentTask) -> SubagentResult:
        return self.run_tasks(parent=parent, tasks=[task])[0]

    def run_tasks(
        self,
        *,
        parent,
        tasks: List[SubagentTask],
        max_concurrent_children: int | None = None,
    ) -> List[SubagentResult]:
        if not tasks:
            return []

        if parent.delegate_depth >= self.max_spawn_depth:
            raise RuntimeError(
                f"Delegation depth limit reached: depth={parent.delegate_depth}, max_spawn_depth={self.max_spawn_depth}"
            )

        child_depth = parent.delegate_depth + 1
        worker_limit = max_concurrent_children or self.max_concurrent_children
        worker_limit = max(1, min(worker_limit, len(tasks)))

        if len(tasks) == 1:
            return [self._execute_one(parent=parent, task=tasks[0], child_depth=child_depth)]

        results_by_task_id: dict[str, SubagentResult] = {}
        with ThreadPoolExecutor(max_workers=worker_limit) as executor:
            futures = {
                executor.submit(self._execute_one, parent=parent, task=task, child_depth=child_depth): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    results_by_task_id[task.task_id] = future.result()
                except Exception as exc:  # pragma: no cover - defensive fallback
                    self.logger.exception("Subagent execution failed for task %s: %s", task.task_id, exc)
                    results_by_task_id[task.task_id] = SubagentResult(
                        task_id=task.task_id,
                        status=SubagentStatus.FAILED,
                        summary=None,
                        engine_result=None,
                        error=str(exc),
                    )

        return [results_by_task_id[t.task_id] for t in tasks]

    def _execute_one(self, *, parent, task: SubagentTask, child_depth: int) -> SubagentResult:
        started_at = time.monotonic()
        child = self.builder.build_child(parent=parent, task=task, child_depth=child_depth)
        parent._register_child(child)

        try:
            child_result = child.run_loop(task.request)
            if child_result.status == EngineStatus.COMPLETED:
                status = SubagentStatus.COMPLETED
            elif child_result.status == EngineStatus.INTERRUPTED:
                status = SubagentStatus.INTERRUPTED
            else:
                status = SubagentStatus.FAILED

            summary = child_result.final_response
            return SubagentResult(
                task_id=task.task_id,
                status=status,
                summary=summary,
                engine_result=child_result,
                error=child_result.error,
                duration_seconds=time.monotonic() - started_at,
                metadata={
                    "goal": task.goal,
                    "child_depth": child_depth,
                    "subagent_id": child.subagent_id,
                },
            )
        except Exception as exc:
            self.logger.exception("Subagent task %s crashed: %s", task.task_id, exc)
            return SubagentResult(
                task_id=task.task_id,
                status=SubagentStatus.FAILED,
                summary=None,
                engine_result=None,
                error=str(exc),
                duration_seconds=time.monotonic() - started_at,
                metadata={
                    "goal": task.goal,
                    "child_depth": child_depth,
                    "subagent_id": child.subagent_id,
                },
            )
        finally:
            parent._unregister_child(child)
