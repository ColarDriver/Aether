"""Parallel execution scheduler for safe tool batches."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Literal

from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.tools.capabilities import (
    ToolCapabilities,
    capabilities_for_tool_name,
    extract_tool_scope_path,
)


ExecutionMode = Literal["sequential", "parallel"]


@dataclass(frozen=True, slots=True)
class ScheduledToolCall:
    index: int
    call: ToolCall
    capabilities: ToolCapabilities
    scope_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ToolExecutionPlan:
    mode: ExecutionMode
    reason: str
    calls: tuple[ScheduledToolCall, ...]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    index: int
    call: ToolCall
    result: ToolResult
    elapsed_ms: float = 0.0


@dataclass(slots=True)
class ToolExecutionScheduler:
    max_workers: int = 4

    def plan(
        self,
        calls: list[ToolCall],
        *,
        context: TurnContext,
        cwd: Path,
        permission_required: set[str] | None = None,
    ) -> ToolExecutionPlan:
        scheduled = tuple(
            ScheduledToolCall(
                index=index,
                call=call,
                capabilities=capabilities_for_tool_name(call.name),
                scope_path=extract_tool_scope_path(call.name, call.arguments, cwd),
            )
            for index, call in enumerate(calls)
        )
        if len(scheduled) <= 1:
            return ToolExecutionPlan("sequential", "batch-size<=1", scheduled)
        if _is_interrupted(context):
            return ToolExecutionPlan("sequential", "interrupted-before-start", scheduled)
        permission_required = permission_required or set()
        if permission_required:
            return ToolExecutionPlan("sequential", "permission-required", scheduled)
        for item in scheduled:
            caps = item.capabilities
            if not caps.parallel_safe:
                return ToolExecutionPlan("sequential", f"tool-not-parallel-safe:{item.call.name}", scheduled)
            if caps.interactive:
                return ToolExecutionPlan("sequential", f"interactive-tool:{item.call.name}", scheduled)
            if caps.requires_permission:
                return ToolExecutionPlan("sequential", f"permission-tool:{item.call.name}", scheduled)
        overlap = first_path_overlap(scheduled)
        if overlap is not None:
            return ToolExecutionPlan("sequential", f"path-overlap:{overlap[0]}:{overlap[1]}", scheduled)
        return ToolExecutionPlan("parallel", "all-safe", scheduled)

    def execute_parallel(
        self,
        plan: ToolExecutionPlan,
        *,
        context: TurnContext,
        execute: Callable[[ToolCall], ToolResult],
    ) -> list[ToolExecutionResult]:
        if plan.mode != "parallel":
            raise ValueError("execute_parallel requires a parallel plan")
        if _is_interrupted(context):
            return [
                ToolExecutionResult(item.index, item.call, _interrupted_result(item.call))
                for item in plan.calls
            ]
        max_workers = max(1, min(int(self.max_workers), len(plan.calls)))
        results: list[ToolExecutionResult] = []
        started = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aether-tool")
        future_map: dict[Future[ToolResult], ScheduledToolCall] = {}
        try:
            future_map = {
                executor.submit(_execute_one, item.call, context, execute): item
                for item in plan.calls
            }
            pending: set[Future[ToolResult]] = set(future_map)
            while pending:
                if _is_interrupted(context):
                    for future in pending:
                        future.cancel()
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    item = future_map[future]
                    if future.cancelled():
                        result = _interrupted_result(
                            item.call,
                            reason="request interrupted before worker start",
                        )
                    else:
                        try:
                            result = future.result()
                        except BaseException as exc:  # noqa: BLE001 - converted to ToolResult
                            result = _exception_result(item.call, exc)
                        if _is_interrupted(context) and not result.metadata.get("interrupted"):
                            result.metadata.setdefault("interrupted", True)
                    results.append(
                        ToolExecutionResult(
                            index=item.index,
                            call=item.call,
                            result=result,
                            elapsed_ms=(time.perf_counter() - started) * 1000.0,
                        )
                    )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        return sorted(results, key=lambda item: item.index)


def first_path_overlap(items: tuple[ScheduledToolCall, ...]) -> tuple[int, int] | None:
    scoped = [(item.index, item.scope_path) for item in items if item.scope_path is not None]
    for left_pos, (left_index, left_path) in enumerate(scoped):
        assert left_path is not None
        for right_index, right_path in scoped[left_pos + 1 :]:
            assert right_path is not None
            if paths_overlap(left_path, right_path):
                return left_index, right_index
    return None


def paths_overlap(left: Path, right: Path) -> bool:
    left_parts = left.resolve(strict=False).parts
    right_parts = right.resolve(strict=False).parts
    min_len = min(len(left_parts), len(right_parts))
    return left_parts[:min_len] == right_parts[:min_len]


def _execute_one(
    call: ToolCall,
    context: TurnContext,
    execute: Callable[[ToolCall], ToolResult],
) -> ToolResult:
    if _is_interrupted(context):
        return _interrupted_result(
            call,
            reason="request interrupted before worker start",
        )
    return execute(call)


def _exception_result(call: ToolCall, exc: BaseException) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"tool execution failed: {type(exc).__name__}: {exc}",
        is_error=True,
        metadata={
            "parallel_worker_error": True,
            "exception_type": type(exc).__name__,
        },
    )


def _interrupted_result(
    call: ToolCall,
    *,
    reason: str = "request interrupted before dispatch",
) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=f"tool execution skipped: {reason}",
        is_error=True,
        metadata={
            "interrupted": True,
            "tool_executed": False,
            "parallel_cancelled": True,
        },
    )


def _is_interrupted(context: TurnContext) -> bool:
    signal = context.interrupt_signal
    return bool(signal is not None and signal.is_aborted())


__all__ = [
    "ScheduledToolCall",
    "ToolExecutionPlan",
    "ToolExecutionResult",
    "ToolExecutionScheduler",
    "first_path_overlap",
    "paths_overlap",
]
